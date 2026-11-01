# 架構與設計決定

## 1. LiteLLM 1.92.0 串接契約（已對照原始碼）

來源：`litellm/proxy/guardrails/guardrail_hooks/generic_guardrail_api/generic_guardrail_api.py`、
`litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py`、
`litellm/llms/openai/chat/guardrail_translation/handler.py`。

| 項目 | 行為 |
|---|---|
| Endpoint | `POST {api_base}/beta/litellm_basic_guardrail_api`（路徑自動補上） |
| Auth | `litellm_params.api_key` → header `x-api-key` |
| Request（pre_call） | `texts` = **所有 message 的文字**（含 system / 歷史，除非 `skip_system_message_in_guardrail: true`）；`structured_messages` = 完整對話；`request_data` = user_api_key_* metadata；`request_headers`（已過濾）；`litellm_call_id`、`model`、`tools` |
| Request（post_call） | `texts` = 模型輸出的各 choice 文字；**沒有 prompt 上下文** |
| Response | `{"action": "NONE"}` 放行；`{"action": "BLOCKED", "blocked_reason": "..."}` → proxy 拋 `GuardrailRaisedException`（client 收到錯誤）；`{"action": "GUARDRAIL_INTERVENED", "texts": [...]}` → 依序替換文字（遮罩） |
| 錯誤處理 | `unreachable_fallback: fail_open|fail_closed`（連不上 / 502/503/504）；`fail_on_error: false` → 任何錯誤都放行 |
| Streaming | 預設每 `streaming_sampling_rate`（5）個 chunk 檢查一次；`streaming_end_of_stream_only: true` 只在結尾檢查。**streaming 時只有 BLOCKED 有效，`GUARDRAIL_INTERVENED` 的遮罩不會套用**（1.92 會原樣送出 chunk），所以 streaming 的部署要在 regex stage 設 `response_mask_action: block`；短於取樣間隔的回覆在擋下之前就已送達。`streaming_buffer_until_moderated` 不會傳給 `generic_guardrail_api` |
| 測試端點 | `POST /guardrails/apply_guardrail {guardrail_name, text, input_type}` → 不呼叫 LLM 直接跑 guardrail（evaluation 的 `litellm_apply_guardrail` target） |

## 1.1 Admin API（不改 config file 時的管理方式）

| Endpoint（PROXY_ADMIN） | 用途 | 備註 |
|---|---|---|
| `GET /v2/guardrails/list` | DB + 記憶體中的 guardrails | `api_key` 會遮罩 → 無法 diff，`--force` 可強制重送 |
| `POST /guardrails` `{"guardrail": {...}}` | 建立 | 立即在該 pod 初始化；設定錯誤會 rollback 回 400 |
| `PUT /guardrails/{id}` | 整筆取代 | `src.litellm_admin apply` 使用 |
| `PATCH /guardrails/{id}` | 合併 litellm_params | |
| `DELETE /guardrails/{id}` | 刪除 | 其他 pod 在下次 DB 同步時移除 |
| `POST /team/update` / `POST /key/update` `{"guardrails": [...]}` | 掛到 team / key | **整個取代**清單 |
| `POST /guardrails/apply_guardrail` | 不呼叫 LLM 直接測 | evaluation target 與 `litellm_admin test` 使用 |

- `litellm_params.api_key` / `api_base` 若為 `os.environ/NAME`，在 proxy 端初始化時以 `get_secret` 解析（`guardrail_registry.py`）。
- 存 DB 時，litellm_params 會以 `model_dump()`（含所有預設值）序列化成 JSON；所以 diff 只比對 spec 中**有寫的 key**。
- 其他 pod 透過 `_init_non_llm_objects_in_db`（DB polling，受 `general_settings.supported_db_objects` 控制）同步。

## 2. 為什麼選 `generic_guardrail_api` + 獨立服務

| 選項 | 優點 | 缺點 |
|---|---|---|
| **generic_guardrail_api → 自建 FastAPI 服務**（採用） | proxy 只需透過 admin API 註冊，不必改 config、不必部署程式碼到 proxy；guard 服務獨立部署/擴充/換模型；可在 proxy 外單獨評測；多個 guardrail 名稱可做 A/B / shadow | 多一跳 HTTP（同機房 ~1ms） |
| proxy 內 custom guardrail（`guardrail: module.Class`） | 可拿到完整 request（含 post_call 的 prompt） | 需要把 Python 檔放進 proxy image / config（**沒有 config 權限時不可行**）；跑在 proxy 進程內，升級 LiteLLM 容易壞 |
| SaaS（Lakera、Azure…） | 免維運 | 資料外送、延遲、無法客製 zh-TW / 公司 policy、成本 |

## 3. Guardrail 服務內部

```
LiteLLM request ──► build_inputs()  (request_scope: last_user | all_user | all_texts)
                        │
                        ▼
                 GuardPipeline.run()
      sequential: rules ─► classifier ─► LLM guard   (第一個 block 就停)
      parallel / run_all: 全部同時跑（evaluation 用，收集 per-stage 結果）
                        │
           每個 stage：Verdict(unsafe, score, categories, error, latency)
           on_error: fail_open | fail_closed（per stage）
           shadow: true → 只記錄、不影響決策（新模型上線前觀察）
                        │
                        ▼
     action = BLOCKED | GUARDRAIL_INTERVENED (masked texts) | NONE
```

- **會判斷哪些內容**（每項都是一個額外的 `GuardInput`，帶有 `origin` 欄位）：
  - 最新的 user 訊息（依 `request_scope`）。
  - client 的 **system / developer 訊息**（`pipeline.judge_system_messages`，預設 true；shared proxy 上 client 可以自己寫 system prompt）。
  - 最後一則 user 訊息之後的 **tool 輸出與 assistant 文字**（`pipeline.judge_tool_messages`，預設 true；indirect prompt injection 就藏在這裡）。
  - **tool_calls**（函式名稱 + 參數，request 與 response 兩側都判斷；任何 rule 命中都回 BLOCKED，因為 tool call 沒辦法遮罩）。
- **限制**：stage 名稱不能重複；每個 input type 最多只能有一個會遮罩的 stage（LiteLLM 依位置替換 texts，兩組遮罩無法合併）；shadow stage 在 LiteLLM 路由上以背景工作執行，不佔請求延遲，`/v1/classify` 的 `run_all` 會等它跑完。
- **Context 上限**：`openai_guard` 依 `max_model_len`（必須等於 vLLM 的 `--max-model-len`）估算 token 預算，並保留頭尾截斷。估算時數字與標點各算 1 個 token，因為 Qwen 會逐位切數字。伺服器回 context 過長時，依「實際送出量 × 伺服器回報的超出比例」縮減，最多重試 3 次；仍失敗就回 error，由 `on_error` 決定。
- **`request_scope: last_user`**（預設）：只判斷最新一則 user 訊息，之前的對話當 context。否則多輪對話中一句舊的可疑內容會讓之後每一輪都被擋。若要抓「分多輪拼湊的攻擊」，改用 `all_user`。
- **分數**：`openai_guard` 從 vLLM logprobs 算 P(unsafe)，只在「anchor 之後、模型實際產生 label token」的位置讀，不讀空白、引號，也不讀 `<think>` 內的位置。有分數時由 `threshold` 決定，hard label 保留在 `extra.hard_label_unsafe`。parser 因選項而推翻 label 時（例如 ignored category），分數會改成和判定一致，原始值存在 `extra.label_score`。rule stage（regex）不產生假機率，分數是 `None`；但 rule 擋下的列，pipeline 分數固定為 1.0，re-threshold 時才不會把它當成放行。
- **Log**：預設不記錄 prompt 原文（`server.log_text: false`），只記錄決策、類別、分數、延遲。

## 4. Evaluation 可重現性

| 機制 | 位置 |
|---|---|
| Config 合併順序：YAML（`_base_` 繼承）→ CLI args（只有明確傳入的才覆蓋）→ `--set a.b=c` | `src/config/config.py::load_config` |
| `.env` → `config["envs"]`；輸出到 `results/` 時 **自動遮蔽** KEY/TOKEN/SECRET/PASSWORD | `redact()`、`dump_config()` |
| `config_hash`（排除 meta / envs）寫進資料夾名稱與 report | `config_hash()` |
| `meta`：git commit / dirty、argv、套件版本、host、時間 | `build_meta()` |
| 資料：每個 CSV 的 sha256、取樣種子、使用列數 | `results/<run>/data_manifest.json` |
| 固定 eval mix（版本化，不要原地修改） | `src/converters/mix.py` |
| 環境：完整 `pip freeze` | `results/<run>/environment.json` |
| 中斷續跑：`predictions.jsonl` 逐筆寫入；transport 失敗的列會在 `--resume` 時重跑 | `runner.py` |
| 只重算指標（換 threshold）不重跑模型 | `--rescore <run_dir> --set metrics.threshold=0.3` |
| 推論決定性：`temperature: 0`、可設 `seed`；vLLM batch 不同仍可能有極小差異 → 用 CI 判斷差異是否顯著 | server preset |

## 5. 擴充點：Registry + 類別階層

所有可擴充的元件都用同一個 `Registry`（`src/common/registry.py`）：`@REG.register("name")` 註冊類別，`REG.build(cfg)` 依 `cfg["type"]` 建立物件（類別有 `from_config` 時呼叫 `from_config`）。plug-in 模組採 lazy import：某個模組 import 失敗不會連帶影響其他元件，查詢不到名稱時，錯誤訊息會一併列出可用名稱與 import 失敗的原因。

| Registry | 位置 | base class | 由 config 的哪個 key 選擇 | 列出 |
|---|---|---|---|---|
| `CONVERTERS` | `src/converters/base.py` | `BaseConverter` | `data.yaml` 的 `converter:`（預設為 key 名稱） | `python -m src.converters --list` |
| `BACKENDS` | `src/server/backends/base.py` | `GuardBackend` | stage `type:` | — |
| `PARSERS` | `src/server/backends/parsers.py` | `OutputParser` | stage `parser:` + `parser_options:` | — |
| `PROMPT_BUILDERS` | `src/server/backends/prompts.py` | `PromptBuilder` | stage `prompt_mode:` | — |
| `TASKS` | `src/evaluation/tasks/base.py` | `EvalTask` | `task.type:` | `python -m src.evaluation --list` |
| `TARGETS` | `src/evaluation/targets.py` | `Target` | `target.type:`（guardrail task） | 同上 |
| `GENERATORS` | `src/evaluation/generation.py` | `Generator` | `task.model_safety.generator.type:` | 同上 |
| `REFUSAL_JUDGES` | `src/refuse_judge/base.py` | `RefusalJudge` | `judge.type:` | `python -m src.refuse_judge --list` |

### 5.1 Dataset converters：資料來源 × 對應樣板

```
BaseConverter            template method: run() -> records(split) -> convert(record) -> GuardrailSample(s)
├─ 資料來源（records）    HFDatasetSource · HFFilesSource · MultiConfigHFSource · LocalFileSource
└─ 對應樣板（convert）    TextLabelConverter ─┬─ UnsafeOnlyConverter / SafeOnlyConverter
                                              └─ _SpanPIIConverter
                          PromptResponseConverter ─┬─ _AegisLike (aegis2, nemotron_safety_v3)
                                                   ├─ _WildGuardLike (wildguardtest, polyguard_prompts)
                                                   └─ beavertails, rtp_lx, xstest_responses
具體資料集 = 來源 + 樣板，例如  class Aegis2(HFDatasetSource, _AegisLike)
```

新增一個資料集：

```python
@register("my_injection_set")
class MyInjectionSet(HFDatasetSource, TextLabelConverter):
    """One line: what it is, why it is here, caveats."""
    hf_id = "org/dataset"; license = "cc-by-4.0"
    text_field = "text"; label_field = "label"; positive_values = {"1"}; negative_values = {"0"}
    category = "prompt_injection"; attack_type = "direct"
    # 只在欄位對應表達不了時才 override：keep() / categories() / lang() / meta() / expand()
```

接著在 `src/config/data.yaml` 加一筆，測試時 patch `conv.records`（參考 `tests/test_converters.py`）。

### 5.2 Guard stage：backend × prompt builder × parser

```
GuardBackend [BACKENDS]: regex · hf_classifier · mock · openai_guard
openai_guard = PromptBuilder [PROMPT_BUILDERS] (chat_template · policy · chat_user_template · completion)
             + OutputParser  [PARSERS] (llama_guard · qwen3guard · granite_guardian · wildguard · nemotron ·
                                         nemotron35 · aprielguard · json_policy · yes_no)
```

新的模型家族通常只需要一個 `OutputParser` subclass（實作 `_parse`，宣告 `OPTIONS` / `CATEGORY_MAP` / `SCORE_SPEC`）。如果 prompt 形狀也特殊，再加一個 `PromptBuilder`，`openai_guard` 本身不用改。

### 5.3 Evaluation：task × (target | generator + judge)

```
EvalTask [TASKS]
├─ guardrail      -> Target [TARGETS]: inprocess · guardrail_server · litellm_apply_guardrail · litellm_chat
├─ model_safety   -> Generator [GENERATORS]: openai_chat · mock   +  RefusalJudge  (+ optional harm GuardPipeline)
└─ refusal_judge  -> RefusalJudge [REFUSAL_JUDGES]: keyword · llm · qwen3guard_refusal · hf_classifier · cascade · vote
```

task 決定三件事：`select()`（哪些樣本適用）、`gold()` 與 `predict()`（什麼算 positive）、`extra_metrics()`。runner、metrics、report、resume / rescore、contamination 都是共用的。詳細說明見 `docs/model_safety.md`。
