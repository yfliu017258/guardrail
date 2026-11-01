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
| Streaming | 預設每 `streaming_sampling_rate`（5）個 chunk 檢查一次；`streaming_end_of_stream_only: true` 只在結尾檢查 |
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

- **`request_scope: last_user`**（預設）：只判斷最新一則 user 訊息，之前的對話當 context。否則多輪對話中一句舊的可疑內容會讓之後每一輪都被擋。若要抓「分多輪拼湊的攻擊」，改用 `all_user`。
- **分數**：`openai_guard` 從 vLLM logprobs 在標籤 token 位置（例如 `Safety:` 之後）算 P(unsafe)；有分數時由 `threshold` 決定，hard label 保留在 `extra.hard_label_unsafe`。
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
