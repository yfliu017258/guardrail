# Evaluation 流程計劃：Guardrail service、模型拒絕率、互動式分析

> 基準版本：**LiteLLM 1.98**（wire format 已對照 1.98.1 原始碼 `litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py`）。
> 進度：**Phase 1–4 已實作**，並以真實的 LiteLLM 1.98.1 proxy 做過整套測試（§10）。已確認的決定見 §8，用法見 §9。

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
| 判斷模型 | `qwen3guard_refusal`：直接讀 Qwen3Guard-Gen 的 Yes / No（`Refusal: Yes/No` 或只回 `Yes` / `No` 都接受），不要求 logprobs | 沒有 partial 類別，部分配合的回覆會被歸到其中一邊；仍建議抽 100–200 筆人工核對 |
| App | Streamlit 可用 | 照 §6 |
| 評測方向 | 以 request 為主，response 也要支援 | 已實作：資料多一個可選欄位 `response`，`--input-type response` 時送 `texts=[response]` |
| `bare` 與 `end_to_end` | 只量 `bare`（模型本身，不經過 guardrail） | `model_safety_litellm.yaml` 設 `on_blocked: error`：請求若被 LiteLLM 上的 guardrail 擋下，該筆記為錯誤並在結果中警告，不會算成拒絕 |

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

### Guardrail 擋下時的 HTTP 狀態（LiteLLM 1.98.1，已對照原始碼並實測）

| 哪一段 | 擋下時的回覆 |
|---|---|
| Guardrail service → LiteLLM | **HTTP 200**，body 為 `{"action": "BLOCKED", "blocked_reason": ...}`。service 若回非 2xx，LiteLLM 會當成 guardrail 故障（依 `fail_on_error` / `unreachable_fallback` 處理），不是攔截 |
| LiteLLM → 呼叫端（`/v1/chat/completions`） | **HTTP 400**，`error.message` 為 `blocked_reason`（`GuardrailRaisedException`，`status_code=400`） |

評測直打 service，走的是第一段，所以 adapter 只在 2xx 時讀 `action`，不必處理 400。

第二段只出現在模型拒絕率的評測（經 LiteLLM 呼叫模型）。LiteLLM 的 guardrail 攔截和其他 400（模型名稱錯誤、超過 context、格式錯誤）沒有 header 或錯誤類型可以區分，差別只在 `error.message`：攔截時是 guardrail 的 `blocked_reason` 原文，其他情況帶有 LiteLLM 自己的錯誤字樣。`litellm_chat` 依此判斷，並可用 `blocked_markers` 填入你的 `blocked_reason` 文字做精確比對。被擋下的列如何計算由 `task.model_safety.on_blocked` 決定：

| `on_blocked` | 被 400 擋下的請求 | 量到的是 |
|---|---|---|
| `error`（預設） | 不計入拒絕率，另外記錄筆數並警告 | 模型本身（只看真正送到模型的請求） |
| `refusal` | 算作拒絕 | 呼叫端實際的體驗（guardrail + 模型） |
| `compliance` | 算作未拒絕 | — |

不論選哪一種，`metrics.json` 都會記錄 `n_blocked`，app 的 Run 詳情也會顯示被擋下的筆數。

## 9. 使用方式

```bash
uv pip install -e ".[dev,app]"

# 1) Guardrail service（request 方向）
python -m src.evaluation --config src/config/evaluation/guardrail_service.yaml \
    --service-url http://my-guardrail:8080 --data processed_data/mixes/core_v1/ --run-name my-guardrail

# 2) 模型拒絕率（bare，經 LiteLLM；判斷模型為 Qwen3Guard，也經 LiteLLM）
#    .env：LITELLM_BASE_URL=http://litellm:4000/v1、LITELLM_API_KEY=<不掛 guardrail 的 key>
python -m src.evaluation --config src/config/evaluation/model_safety_litellm.yaml \
    --set task.model_safety.generator.model=<模型 alias> --set task.model_safety.judge.model=<qwen3guard alias> \
    --data processed_data/mixes/core_v1/ --run-name <模型 alias>

# 3) Guardrail service（response 方向）：先把模型 run 的回覆轉成 prompt,expected,response
python -m src.analysis.responses results/<模型 run> -o processed_data/responses/<模型>.csv
python -m src.evaluation --config src/config/evaluation/guardrail_service.yaml \
    --service-url http://my-guardrail:8080 --input-type response --data processed_data/responses/<模型>.csv

# 4) 互動式分析
streamlit run src/analysis/app.py -- --results results/
```

- Guardrail 的 API key 放在 `.env` 的 `GUARDRAIL_SERVICE_API_KEY`，以 `x-api-key` 送出（與 LiteLLM 相同）。
- 開跑前會送一筆 canary；service 的政策不擋 prompt injection 時，請設 `target.http_guardrail.canary_text`。
- 第 3 步的標記是推定的：「該拒絕的 prompt 而模型沒有拒絕」的回覆標為 `block`，其餘為 `pass`。沒拒絕不一定等於內容有害，`block` 的列建議人工看過。
- 換判斷模型不必重新產生回覆：`--set task.model_safety.reuse_generations_from=results/<先前的 run>`。
- 其他 service：在 `src/evaluation/adapters.py` 仿照 `GenericGuardrailAPIAdapter` 寫一個 subclass 並註冊，範例見 `tests/test_adapters.py::test_custom_adapter_is_one_class`。

### App 的頁面

| 頁面 | 回答的問題 |
|---|---|
| 總覽 | 所有 run 的 recall、FPR、錯誤率、p50 / p95 latency（含 95% 信賴區間） |
| Run 詳情 | 單一 run 的 expected × action 矩陣、source / language 切片、可篩選的樣本表與完整內容 |
| Run 比較 | **不同 guardrail（或不同模型）的差異**：指標並排、latency 分佈、兩兩之間「只有 A 擋到／只有 B 誤擋」的樣本 |
| 模型 × Guardrail | **特定模型加上某個 guardrail 的收益與代價**：殘餘風險下降多少、正常請求可用率變化、端到端 latency 增加多少，並可點進各類樣本 |
| Latency | 各 run 的 p50 / p90 / p95 / p99、累積分佈、依判定結果與 prompt 長度的拆解 |

「模型 × Guardrail」是把兩份 run 以 `sample_id` 離線合併：被 guardrail 擋下的請求不會送到模型，latency 只計 guardrail；其餘為 guardrail + 模型。這個 latency 是兩段時間相加，不含 LiteLLM 呼叫 guardrail 的額外開銷。

## 10. 整套測試的紀錄

環境：真實的 LiteLLM 1.98.1 proxy；proxy 後面是一個假的 OpenAI 相容上游（一個會拒絕部分 prompt 的模型，以及一個照 Qwen3Guard 格式回答的判斷模型）；兩個設定不同的本 repo mock guardrail server（B 少了幾類規則、多一條會誤擋的規則、並加 120 ms 延遲）。資料為 `processed_data/samples/smoke.csv`（27 筆）。

| 檢查項目 | 結果 |
|---|---|
| Guardrail A、B 的 request 方向評測 | 通過；B 因為不擋預設的 canary 而被中止，改設 `canary_text` 後通過（canary 機制如預期運作） |
| 經 LiteLLM 的模型拒絕率 + Qwen3Guard 判斷 | 通過；`run.json` 記錄到 `litellm_version: 1.98.1` |
| bare 模式的保護 | 強制在請求上掛 guardrail 後，12 筆被擋的請求記為錯誤並出現警告，沒有被算成拒絕 |
| 真實 LiteLLM 呼叫本 repo 的 guardrail server | 通過（`guardrails: [...]` 的請求回 HTTP 400 與 `blocked_reason`） |
| 模型回覆 → response 方向資料 → 兩個 guardrail 的 response 評測 | 通過 |
| App 五個頁面 | 以瀏覽器實際開啟並截圖檢查，無例外 |

沒有驗證到的部分：真實的 guardrail service、真實的 Qwen3Guard（判斷品質與 logprobs 經 LiteLLM 是否可用）、大量資料下 app 的速度。

## 11. 批次執行：新增 guardrail／模型／資料集

一份清單檔列出所有評測對象，一個指令補齊還沒有結果的 run。

```bash
cp src/config/evaluation/suite.example.yaml suite.yaml   # 填入你的 service、模型 alias、資料集
scripts/run_suite.sh                                     # 只跑還沒有完成結果的組合
scripts/run_suite.sh suite.yaml --limit 50               # 每個資料集只取 50 筆，快速確認功能
scripts/run_suite.sh suite.yaml --only my-new-guardrail  # 只跑指定的 guardrail／模型／資料集
scripts/run_suite.sh suite.yaml --dry-run                # 只列出會跑哪些
scripts/app.sh                                           # 開啟分析 app
```

```yaml
# suite.yaml
results: results
datasets:
  core_v1: processed_data/mixes/core_v1/
  answers_qwen: {path: processed_data/responses/qwen.csv, input_type: response}   # response 方向
guardrails:
  my-guardrail: {url: "http://my-guardrail:8080", api_key_env: GUARDRAIL_SERVICE_API_KEY}
models:
  qwen3.6-27b: {}                  # 名稱即 LiteLLM 的 model alias
judge: {model: qwen3guard}
```

- **新增 guardrail**：在 `guardrails` 加一行，重跑同一個指令。它會對每個資料集各跑一次。
- **新增資料集**：在 `datasets` 加一行，重跑。每個 guardrail 和模型都會補跑這個資料集。
- **新增模型**：在 `models` 加一行，重跑。模型只跑 request 方向的資料集。
- 每個 run 命名為 `<對象>@<資料集>`。同名、資料相同（檔案內容與筆數上限都相同）且狀態為 completed 的 run 會跳過；資料改了或加 `--force` 才會重跑。
- 某個 service 連不上時，該 run 記為 failed，其他組合照常執行；指令最後列出每個組合的狀態，有失敗時結束碼為 1。
- `suite.yaml` 已列入 `.gitignore`（內含內部網址與模型名稱）。

### 放著跑、中斷後續跑、筆數上限

```bash
scripts/run_experiments.sh start                                   # suite.yaml 裡還沒完成的全部實驗，背景執行
scripts/run_experiments.sh start my-guardrail qwen3.6-27b@core_v1   # 只跑指定的實驗
scripts/run_experiments.sh status                                  # 是否還在跑、目前進度、最後的總表
scripts/run_experiments.sh logs                                    # 跟著看 log
scripts/run_experiments.sh stop                                    # 停止；之後 start 會從停下的地方接續
MAX_SAMPLES=200 RETRIES=5 RETRY_WAIT=300 scripts/run_experiments.sh start
```

- **實驗的指定方式**：guardrail／模型名稱、資料集名稱，或 `<對象>@<資料集>`，名稱取自 `suite.yaml`。
- **背景執行**：用 `nohup` 啟動，關掉終端機不會中斷；log 在 `logs/experiments_<時間>.log`，同時間只允許一個在跑。
- **Resume**：每筆結果寫完就落檔。行程被中斷、機器重開或 service 中途掛掉後，再次 `start`（或直接跑 `python -m src.evaluation.suite`）會在原本的 run 目錄接續：已完成的列保留，只送還沒做的列，最後一列若寫到一半會自動修掉。設定改過（例如換了網址）就無法接續，會開新的 run。
- **自動重試**：某個實驗失敗時，其他實驗照跑；全部跑完一輪後等 `RETRY_WAIT` 秒，把失敗的再試一次（接續而非重來），最多 `RETRIES` 輪。
- **單筆的連線失敗**：每筆請求先重試 3 次；仍失敗的列記為錯誤、不計入指標。對已完成的 run 加 `EXTRA="--retry-errors"` 可以只補送這些列。
- **筆數上限（max data）**：`suite.yaml` 的 `max_samples`（全部資料集）、資料集自己的 `max_samples`，或 `MAX_SAMPLES=` / `--max-samples`（優先）。抽樣固定 seed 並依 `expected` 分層，所以每個 guardrail 和模型測到的是同一批樣本。上限不同就視為不同的資料，會產生新的 run。
- 還沒送出任何請求就失敗的 run（service 連不上、key 錯）不會留下空目錄，錯誤訊息在 log 和總表裡。

單獨一個 run 也能手動接續：`python -m src.evaluation --resume results/<run 目錄>`。

## 12. Result 的儲存方式

所有結果放在一個目錄（預設 `results/`，不進 git），每次評測一個子目錄，彼此獨立。App 只讀這個目錄。

```
results/
  20261111-164010_my-guardrail-core_v1_53e2aebe75a4/     # <UTC 時間>_<run 名稱>_<設定 hash>
    run.json             # 索引：schema_version、狀態、評測對象、資料集識別碼、LiteLLM 版本、判斷模型、起訖時間
    predictions.jsonl    # 逐筆結果，一列一筆樣本（評測中持續寫入，可 --resume）
    metrics.json         # 整體與切片指標、信賴區間、latency
    config.yaml          # 完整設定快照（secrets 已遮蔽）
    data_manifest.json   # 每個資料檔的 sha256、使用筆數、抽樣 seed
    task.json            # 評測對象的細節（adapter、網址、generator、judge）
    environment.json     # Python 與套件版本
    report.md  slices.csv  errors.csv      # 給人看的報表、切片、誤判樣本
```

| 檔案 | 誰會讀 | 內容重點 |
|---|---|---|
| `run.json` | app（列出與篩選 run） | `status`：running / completed / failed；`subject`：guardrail 或模型的名稱、方向、adapter；`dataset.sha256`：兩個 run 是否用同一份資料 |
| `predictions.jsonl` | app（所有圖表與樣本表都由它算出） | 共用欄位：`sample_id`、`prompt`、`expected`、`source`、`language`、`gold_positive`、`pred_positive`、`latency_ms`、`error` |
| | | Guardrail 專屬：`action`、`texts`、`texts_changed`、`blocked_reason`、`input_type`、`raw` |
| | | 模型專屬：`response`（模型回覆全文）、`reasoning`（thinking 內容）、`finish_reason`、`usage`、`refusal_label`、`judge`、`judge_reason`、`raw`（判斷模型的原始輸出）、`judge_extra`、`blocked`、`block_message`、`gen_latency_ms`、`judge_latency_ms`、`gen_fingerprint`、`response_truncated` |
| `metrics.json` | 人、其他工具 | 評測結束時算好的數字；app 不依賴它，會從 `predictions.jsonl` 重算 |

### Guardrail service 的資訊

設定 `info_path`（例如 `/v1/info`）後，每個 guardrail run 會向 service 的資訊端點取一次 JSON，原樣存進結果：

```yaml
guardrails:
  my-guardrail: {url: "http://my-guardrail:8080", info_path: /v1/info}
```

| 存在哪裡 | 內容 |
|---|---|
| `run.json` → `subject.service_info` | 評測開始時 service 回報的資訊（app 讀這份） |
| `run.json` → `subject.service_changed_during_run` | 評測前後的資訊是否不同 |
| `task.json` → `server_info`、`server_info_end`、`server_identity` | 開始與結束時各一份；`server_identity` 是去掉啟動時間、計數器等會自然變動的欄位後的內容 |

- **格式不限**：service 回什麼 JSON 就存什麼，不需要固定欄位。名稱像 key、token、secret 的欄位在寫檔時會被遮蔽。
- **評測中途換版**：開始和結束的資訊不同時，結果會帶警告，app 的 Run 詳情也會顯示。
- **續跑時的保護**：中斷的 run 要接續時，若 service 的資訊已經和當初不同，就不會接續，而是開新的 run，避免一個 run 混到兩個版本。
- **端點讀不到**：預設只記錄錯誤並警告，評測照跑；設 `info_required: true` 則直接中止。
- **哪些欄位算「會自然變動」**：預設忽略 `started_at`、`uptime`、`stats` 等，可用 `info_ignore_keys` 調整。
- App：Run 詳情可展開完整資訊；Run 比較列出各 run 之間不同的設定（例如模型、門檻、版本）。

設計上的幾個決定：

- **一個 run 一個目錄、不覆寫**：重跑會產生新目錄，舊結果保留；app 以開始時間排序。
- **`sample_id` = `<資料集檔名>:<prompt 的 hash>`**：同一筆樣本在每個 run 的 id 相同，所以不同 guardrail、guardrail 與模型之間可以逐筆對齊。
- **結果自足**：`predictions.jsonl` 存完整 prompt 與回覆，分析時不需要原始資料集。也因此目錄可能含敏感內容，不進 git，搬移時整個目錄一起搬即可。
- **`schema_version`**：格式改版時遞增；app 遇到不認得的版本會略過並提示，不會壞掉。
- **JSONL 逐筆追加**：評測中斷後可續跑，進行中的 run 也能在 app 裡看到目前為止的結果。

## 13. 目前使用的三個 prompt injection 資料集

```bash
uv pip install -e ".[data]"
python -m src.converters --config src/config/data_pi_three.yaml     # 下載並轉成 prompt,expected
```

| 輸出檔（= 評測時的 source） | 來源 | 取用的部分 | 標籤對應 |
|---|---|---|---|
| `deepset_prompt_injections.csv` | `deepset/prompt-injections` | test，116 列（en / de） | `label` 1 → block，0 → pass |
| `neuralchemy_pi.csv` | `neuralchemy/Prompt-injection-dataset`，`core` config | test，942 列 | `label` 1 → block，0 → pass |
| `eric12132_pi_en.csv` | `Eric12132/prompt-injections-benchmark-chinese` | test，91 列，`prompt` 欄（英文） | Yes → block，No → pass |
| `eric12132_pi_zhtw.csv` | 同上 | test，91 列，`taiwan_translation` 欄 | 同上 |

之後在 `suite.yaml` 的 `datasets` 各加一行即可。轉換時空白的文字、無法辨識的標籤會被丟棄並記在 `<name>.manifest.json`，同一檔內重複的 prompt 只留一筆。三個資料集的品質問題與建議的清理方式見 `docs/prompt_injection_data.md` §1。
