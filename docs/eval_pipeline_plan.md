# Evaluation 流程計劃：Guardrail service、模型拒絕率、互動式分析

> 基準版本：**LiteLLM 1.98**（wire format 已對照 1.98.1 原始碼 `litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py`）。
> 進度：**Phase 1、2 已實作**（results 契約、adapter 層、request / response 兩個方向）；Phase 3、4 尚未實作。已確認的決定見 §8。

## 1. 目標與範圍

| # | 需求 | 產出 |
|---|---|---|
| A | 寫一個 adapter，就能讀資料集、用 API 呼叫任一個 guardrail service | `ADAPTERS` registry + `http_guardrail` target |
| B | 同一份資料集，經 LiteLLM 呼叫模型，回覆交給判斷模型判定是否拒絕 | `litellm_chat` generator + 判斷模型也走 LiteLLM |
| C | 評測結束產生 results files | 有版本號的 results 契約（`run.json` + `predictions.jsonl` + `metrics.json`） |
| D | 另一個 analysis app 讀 results 並互動式視覺化 | `src/analysis/app`（Streamlit） |

三者之間只靠 **檔案契約** 耦合：評測只寫 results，app 只讀 results。

```
processed_data/*.csv ──▶ evaluation runner ──▶ results/<run>/ ──▶ analysis app
 (prompt,expected)        │                     run.json
                          ├─ task: guardrail      predictions.jsonl
                          │    └─ adapter ──HTTP──▶ guardrail service
                          └─ task: model_safety   metrics.json
                               ├─ generator ──▶ LiteLLM /v1/chat/completions ──▶ 模型
                               └─ judge ─────▶ LiteLLM /v1/chat/completions ──▶ 判斷模型
```

## 2. 現況盤點

| 元件 | 現況 | 計劃 |
|---|---|---|
| 資料集載入（`prompt,expected`，source = 檔名） | 有 | 沿用 |
| Runner（並行、重試、`--resume`） | 有 | 沿用 |
| `guardrail` task + metrics | 有，但 target 綁定本 repo 的 server（`/health`、`/v1/info`、`/v1/classify`） | 新增與 server 無關的 `http_guardrail` target |
| generic guardrail API schema | 對照 1.92.0 | 更新到 1.98（§3.1） |
| `model_safety` task + `openai_chat` generator | 有，設計上是直打 vLLM | 新增 LiteLLM 專用 generator（§4） |
| 拒絕判斷器（keyword / llm / cascade …） | 有 | 沿用；LLM judge 改走 LiteLLM |
| results 目錄 | 有，但 `predictions.jsonl` 只存 `text_preview`（前 300 字），沒有 schema 版本 | 補成自足的契約（§5） |
| 互動式分析 | 沒有（只有 `report.md`、`compare.py`） | 新增（§6） |

## 3. A：Guardrail service 評測

### 3.1 正規化格式 = LiteLLM 1.98 generic guardrail API

所有 adapter 的輸出都轉成這個形狀，後面的 metrics 與 app 只認這個形狀：

```python
@dataclass
class GuardrailResult:
    action: Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]
    texts: list[str] | None = None          # INTERVENED 時改寫後的文字（與輸入 texts 位置對應）
    blocked_reason: str | None = None
    # 以下不在 LiteLLM 協定內，adapter 能給就給，分析時才有 ROC / 類別切片
    score: float | None = None              # P(unsafe)
    categories: list[str] = ()
    raw: Any = None                         # service 原始回覆（截斷後存檔）
```

1.98 相對於 repo 目前對照的 1.92.0，需要跟上的差異：

- Response 多了 `images`、`tools`、`stream_holdback_chars`（評測只需容忍，不需使用）。
- 缺少 `action` 時 LiteLLM 視為 `NONE`。評測端要把這種情況記成 **協定錯誤**，不能默默算成放行。
- LiteLLM 設定新增 `fail_on_error`（預設 true）；`unreachable_fallback` 預設 `fail_closed`，502/503/504 視為 unreachable。
- 認證：LiteLLM 以 `x-api-key` header 送出；路徑固定為 `<api_base>/beta/litellm_basic_guardrail_api`。

### 3.2 Adapter 介面（新增）

```python
class GuardrailAdapter(abc.ABC):
    """一個 guardrail service 的翻譯層：只管 payload 怎麼組、回覆怎麼讀。"""

    def build_request(self, sample: EvalSample, input_type: str) -> AdapterRequest:
        """-> method, path, headers, json"""

    def parse_response(self, status: int, headers: Mapping[str, str], body: Any) -> GuardrailResult:
        """轉成正規化格式；無法解析 -> 丟 AdapterParseError（記為 error，不計入指標）"""

    def is_retryable(self, status: int, body: Any) -> bool: ...   # 預設 5xx / 408 / 429
    async def healthcheck(self, client) -> None: ...              # 可選
```

- 連線、逾時、重試、並行、計時、寫檔都由 `http_guardrail` target 與 runner 處理。**adapter 不碰 I/O**，所以可以用錄好的 JSON 做單元測試。
- 內建 `generic_guardrail_api` adapter：service 已經講 LiteLLM 格式時直接用，不必寫程式。送出的 request 與 LiteLLM 1.98 相同（`input_type`、`texts`、`structured_messages`、`request_data`、`litellm_call_id`、`litellm_version`、`model`）。
- service 講自己的原生 API 時：寫一個 subclass，加一行 `@ADAPTERS.register("my_service")`。
- 提供 `tests/adapter_contract.py`：新 adapter 只要附上幾組「原生回覆 → 預期 action」的樣本就能跑契約測試。

設定範例：

```yaml
task: {type: guardrail}
target:
  type: http_guardrail
  http_guardrail:
    adapter: {type: generic_guardrail_api}      # 或 my_service + 它自己的選項
    base_url_env: GUARDRAIL_SERVICE_URL
    api_key_env: GUARDRAIL_SERVICE_API_KEY
    input_type: request                          # request | response
    canary: true
```

### 3.3 評分

沿用現有規則：`block → BLOCKED`、`pass → NONE`、`mask → GUARDRAIL_INTERVENED`。

- 主指標：recall（該擋有擋）、FPR（誤擋）、precision、F1、bootstrap CI。
- `mask`：除了比對 action，另外記錄 `texts` 是否真的與輸入不同。
- adapter 有提供 `score` 時才計算 ROC、recall@FPR、threshold 分析。只有 action 時，app 會隱藏這些圖。
- 開跑前送 canary（一筆明顯該擋的 prompt）。沒被擋就中止，避免評到一個 fail-open 的 service。

### 3.4 可選：經 LiteLLM 驗證整合

直打 service 量的是 **service 的品質**。若還想確認 LiteLLM 的設定（mode、`default_on`、fail-open）沒問題，可加一個 `litellm_apply_guardrail` adapter，打 LiteLLM 1.98 的 `POST /guardrails/apply_guardrail`。這是第二優先，放在 Phase 4。

## 4. B：模型拒絕率（經 LiteLLM）

流程沿用 `model_safety` task：每筆 prompt → 產生回覆 → 判斷模型判定 → `refused / partial / complied`。

新增與調整：

1. **`litellm_chat` generator**（繼承 `openai_chat`）
   - 打 LiteLLM `/v1/chat/completions`，`model` 用 LiteLLM 的 model alias。
   - 記錄 response header：`x-litellm-call-id`、`x-litellm-model-id`、`x-litellm-version`（寫進 `run.json`，確認是 1.98）。
   - 辨識 LiteLLM 的 guardrail 攔截錯誤，記成 `blocked=true`，與模型自己的拒絕分開統計。
2. **隔離 guardrail 的影響**（最重要的一點）
   LiteLLM 上若有 `default_on` 的 guardrail，量到的會是「guardrail + 模型」，不是模型本身。計劃提供兩種模式並在 `run.json` 標明：
   - `bare`：用一把沒有掛 guardrail 的 key／或在 request 停用 guardrail。有 `blocked` 的列出現時視為設定錯誤並警告。
   - `end_to_end`：照正式流量走，`blocked` 算作被擋下。
3. **判斷模型也走 LiteLLM**：`JUDGE_BASE_URL` 指向同一個 LiteLLM，`model` 換成判斷模型的 alias；`temperature=0`、固定 prompt 版本，prompt hash 寫進 results。
4. **產生與判斷分兩階段落檔**：回覆先寫入，再判斷。換判斷模型時用現有的 `reuse_generations_from` 重判，不必重新產生。
5. **判斷模型要先驗證**：人工標 100–200 筆回覆，用現有的 `refusal_judge` task 量判斷模型的準確率，結果顯示在 app 裡。未驗證的判斷模型，拒絕率數字不可信。

指標：拒絕率（`expected=block`）、過度拒絕率（`expected=pass`）、partial 率、各 source / language 切片。

## 5. C：Results 契約

每次評測一個目錄 `results/<UTC>_<name>_<config_hash>/`。app 只依賴下面三個檔案，其餘（`config.yaml`、`data_manifest.json`、`environment.json`、`report.md`）照舊保留。

**`run.json`**（新增，app 的索引）

```json
{
  "schema_version": 1,
  "run_id": "20261111T0800Z_my-service_ab12cd",
  "task": "guardrail",                    // guardrail | model_safety
  "status": "completed",                  // running | completed | failed
  "subject": {"kind": "guardrail_service", "name": "my-service", "adapter": "generic_guardrail_api"},
  "dataset": {"name": "core_v1", "n": 5000, "sha256": "..."},
  "litellm_version": "1.98.1",
  "judge": null,                          // model_safety 才有：model、prompt_hash、已驗證準確率
  "started_at": "...", "finished_at": "...", "git_sha": "...", "config_hash": "..."
}
```

**`predictions.jsonl`**（一筆樣本一列，兩種 task 共用核心欄位）

| 欄位 | 說明 |
|---|---|
| `sample_id` | `source:hash(prompt)`，跨 run 穩定，app 用它 join |
| `prompt`、`expected`、`source`、`language` | 完整 prompt（目前只存 300 字預覽，要改） |
| `gold_positive`、`pred_positive`、`score` | 共用的二元判定 |
| `latency_ms`、`error` | |
| guardrail 專屬 | `action`、`texts`、`blocked_reason`、`categories`、`raw` |
| model_safety 專屬 | `response`、`reasoning`、`blocked`、`refusal_label`、`judge_reason`、`judge` |

**`metrics.json`**：評測結束時算好的整體與切片指標（app 首頁直接讀，不必重算）。

規則：`schema_version` 不合的 run，app 顯示提示而不是壞掉；`predictions.jsonl` 可能含機密 prompt，results 目錄不進 git。

## 6. D：互動式 analysis app

**建議用 Streamlit + Plotly**：全 Python、可直接重用 `metrics.py`、內網一行指令啟動（`streamlit run src/analysis/app/main.py -- --results results/`）。放在 `pyproject.toml` 的 `[app]` extra，不影響評測端的相依。

| 頁面 | 內容 | 互動 |
|---|---|---|
| Overview | 所有 run 的排行表：recall、FPR、F1、拒絕率、p95 latency、n、error 率 | 依 task / dataset / 日期篩選、排序、勾選 run 進比較 |
| Run 詳情 | 混淆矩陣、`expected × action` 矩陣、source / language 切片長條圖、latency 分佈 | 點切片 → 篩出該切片樣本；有 `score` 時提供 threshold 滑桿，即時重算指標與 ROC 上的位置 |
| 樣本瀏覽 | 表格 + 詳情面板：完整 prompt、action / 模型回覆、判斷理由、raw | 篩 FP / FN / error、source、language、關鍵字搜尋；匯出 CSV |
| Run 比較 | 兩個以上 run 的指標差異（含 CI）、切片差異 | **翻轉分析**：列出 A 擋到而 B 沒擋到的樣本（以及反向） |
| Guardrail × 模型 | 同一資料集上，guardrail run 與 model_safety run 以 `sample_id` join | 四象限：兩者都擋／只有 guardrail／只有模型／**都沒擋（殘餘風險）**，點進去看樣本 |

最後一頁是「同一份資料集跑兩種評測」最直接的價值：能回答「加了 guardrail 之後，還有多少有害 prompt 會得到模型的配合」。

實作要點：以 pandas 載入並用 `st.cache_data`（以檔案 mtime 為 key）；只比較 dataset sha256 相同的 run，不同時明確警告；`status=running` 的 run 也能看（顯示進度）。

## 7. 分階段實作

| Phase | 內容 | 完成標準 |
|---|---|---|
| 1 | Results 契約：`run.json`、完整 `prompt`、`schema_version`；schema 更新到 1.98 | 現有測試全過；mock 評測產出符合 §5 的目錄 |
| 2 | `ADAPTERS` registry、`http_guardrail` target、`generic_guardrail_api` adapter、契約測試工具 | 對 mock server 與真實 service 各跑通一次；新 adapter 只需一個檔案 |
| 3 | `litellm_chat` generator、bare / end_to_end 模式、判斷模型走 LiteLLM | 對 LiteLLM 1.98 跑通；換判斷模型不需重新產生；判斷模型有驗證數字 |
| 4 | Analysis app：Overview → Run 詳情 → 樣本瀏覽 → 比較 → Guardrail × 模型 | 用 Phase 2、3 的真實 results 能完成 §6 每一頁的操作 |
| 5（可選） | `litellm_apply_guardrail` adapter；response 方向（`input_type: response`）的評測 | — |

Phase 1 先做，是因為 app 與兩種評測都依賴這份契約；Phase 2 與 3 互不依賴，可以平行。

## 8. 已確認的決定

| 題目 | 決定 | 對計劃的影響 |
|---|---|---|
| Service 的格式 | 直接回 LiteLLM generic guardrail API 格式 | 用內建的 `generic_guardrail_api` adapter，不必寫程式 |
| 是否有分數 | 只回 `action` | 沒有 ROC、recall@FPR、threshold 分析；app 只顯示點估計（recall / FPR / F1 與 CI）。要比較不同嚴格度，只能請 service 換設定後各跑一次 |
| 判斷模型 | `qwen3guard_refusal`（讀 Qwen3Guard-Gen 的 `Refusal: Yes/No`） | 沒有 partial 類別，部分配合的回覆會被歸到其中一邊；仍建議抽 100–200 筆人工核對 |
| App | Streamlit 可用 | 照 §6 |
| 評測方向 | 以 request 為主，response 也要支援 | 已實作：資料多一個可選欄位 `response`，`--input-type response` 時送 `texts=[response]` |
| `bare` 與 `end_to_end` | 待決定（差異見下） | 影響 Phase 3 的預設值 |

### `bare` 與 `end_to_end` 的差異

| | `bare` | `end_to_end` |
|---|---|---|
| 請求路徑 | LiteLLM → 模型（不經過 guardrail） | LiteLLM → guardrail → 模型 →（guardrail）→ 回覆 |
| 量到的是 | 模型本身會不會拒絕 | 使用者實際上拿不拿得到有害回覆 |
| 「擋下」的來源 | 只有模型的拒絕 | guardrail 攔截 + 模型拒絕，兩者分開記錄 |
| 換 guardrail 設定後 | 數字不變 | 數字會變 |
| 需要的條件 | 一把不掛 guardrail 的 key | 正式設定即可 |
| 能回答的問題 | 「這個模型安全嗎？要不要換模型？」 | 「整個系統上線後的殘餘風險是多少？LiteLLM 的整合有沒有設對？」 |

兩者不是二選一。**建議以 `bare` 為主**：它加上 guardrail 的直打評測，就能在 app 的「Guardrail × 模型」頁以 `sample_id` 合併，離線算出 end-to-end 的結果，而且能分辨每一筆是誰擋的。`end_to_end` 則用小樣本（幾百筆）跑一次做驗證：若實測與離線合併的結果不一致，代表 LiteLLM 上的設定（mode、`default_on`、fail-open）有問題。

## 9. 使用方式（已實作的部分）

```bash
# request 方向：評測任一個講 LiteLLM 格式的 guardrail service
python -m src.evaluation --config src/config/evaluation/guardrail_service.yaml \
    --service-url http://my-guardrail:8080 --data processed_data/mixes/core_v1/ --run-name my-guardrail

# response 方向：資料為 prompt,expected,response 三欄
python -m src.evaluation --config src/config/evaluation/guardrail_service.yaml \
    --service-url http://my-guardrail:8080 --input-type response --data processed_data/responses/
```

- API key 放在 `.env` 的 `GUARDRAIL_SERVICE_API_KEY`，以 `x-api-key` 送出（與 LiteLLM 相同）。
- 開跑前會送一筆 canary；service 的政策不擋 prompt injection 時，請設 `target.http_guardrail.canary_text`。
- 其他 service：在 `src/evaluation/adapters.py` 仿照 `GenericGuardrailAPIAdapter` 寫一個 subclass 並註冊，範例見 `tests/test_adapters.py::test_custom_adapter_is_one_class`。
