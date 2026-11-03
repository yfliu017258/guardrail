# OJT — Guardrail Server：服務與可重現評測

本專案的目標是**設定好 guardrail server**，並用可重現的方式評測它。所有測試與評測都直接打 server。

- **`src/server`**：FastAPI + uvicorn 的 guardrail server，採 generic guardrail API（`POST /beta/litellm_basic_guardrail_api`）。
  - pipeline 依序是 regex 規則 → 輕量 classifier → vLLM 上的 LLM guard。
  - 也提供 `/v1/classify`（每個 stage 的細節）、`/health`、`/ready`、`/v1/info`。
  - 驗收工具：`python -m src.server.check`。
- **`src/evaluation`**：自動、可重現的評測 CLI，預設直接打正在跑的 server。
  - 輸出 metrics（含 CI、ROC、recall@FPR、每個 stage 的指標）、切片、誤判樣本，以及 config snapshot。
  - 也能評測**裸模型**（Qwen3.6-27B、GLM-5.3-Flash）的拒答行為，和拒答判斷器本身的準確度。
- **`src/refuse_judge`**：判斷模型回覆是否拒答，有 keyword、llm、qwen3guard_refusal、hf_classifier、cascade、vote 六種。
- **`src/converters`**：把公開或內部資料集轉成 **只有 `prompt,expected` 兩欄**的 CSV（`processed_data/<資料集>.csv`；`expected` = `block` / `pass` / `mask`），附 manifest，並能凍結版本化的 eval mix（資料夾，每個來源一個 CSV）。

文件：[`docs/server.md`](docs/server.md)（**server 設定指南**）· [`docs/refactor/README.md`](docs/refactor/README.md)（**重構指南**：目標架構、config 系統、遷移步驟）· [`docs/plan.md`](docs/plan.md) · [`docs/datasets.md`](docs/datasets.md) · [`docs/prompt_injection_data.md`](docs/prompt_injection_data.md)（**PI 評測資料建議**） · [`docs/models.md`](docs/models.md) · [`docs/architecture.md`](docs/architecture.md)（§5：Registry 與類別階層）· [`docs/model_safety.md`](docs/model_safety.md)

所有可擴充元件（converters、guard backends / parsers / prompt builders、evaluation tasks / targets / generators、refusal judges）都是註冊在同一個 `Registry` 的類別。新增功能 = 寫一個 subclass + 一行 `@REG.register("name")`。

## 目錄

```
src/
  config/
    config.py            # YAML 合併(_base_) → args → --set；.env → config["envs"]；dump 時只記錄 env 名稱、遮蔽 secrets
    server/              # default.yaml, mock.yaml, models/*.yaml（各 guard 模型 preset）, pipelines/, policies/
    evaluation/          # default / smoke / model_safety / smoke_model_safety .yaml
    models/              # 裸模型 generator preset：qwen3_6_27b.yaml, glm_5_3_flash.yaml
    refuse_judge/        # keyword / llm_judge / qwen3guard / cascade .yaml
    data.yaml            # converters + 固定 eval mix
  common/registry.py     # 共用 Registry（register / get / build / describe, lazy plug-ins）
  data_schema/           # Sample(prompt, expected)：CSV 格式唯一定義處；taxonomy 對照表；語言偵測
  converters/            # base.py（BaseConverter + CONVERTERS）, sources.py, templates.py, datasets/*.py, mix.py
  server/                # app.py, pipeline.py, schemas.py, check.py（驗收）, testing.py（背景 server）,
                         # backends/{base,regex,hf_classifier,openai_guard,prompts,parsers,mock}.py
  evaluation/            # tasks/{base,guardrail,refusal}.py, targets.py, generation.py, runner.py, metrics.py, report.py, compare.py
  refuse_judge/          # base.py（RefusalJudge + REFUSAL_JUDGES）, keyword / llm / guard / classifier / ensemble, __main__.py
deploy/server/             # run_server.sh, Dockerfile
deploy/vllm/serve_guard.sh # 在 MIG slice 上啟動 guard 模型（vLLM）
processed_data/samples/smoke.csv   # 27 筆中英小樣本（prompt,expected；測試/CI）
tests/                             # 不需 GPU / 網路；整合測試會自己起一個真的 server（mock backend）
```

## 快速開始

```bash
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"            # + ".[data]" 轉換 HF 資料集；+ ".[hf]" 本地 classifier
cp .env.example .env

pytest -q                             # 全部離線可跑

# 1) 不需 GPU：mock server + 冒煙評測（打 server）
python -m src.server --config src/config/server/mock.yaml --port 8080 &
python -m src.server.check --url http://localhost:8080
python -m src.evaluation --config src/config/evaluation/smoke.yaml

# 2) 資料
python -m src.converters --list
python -m src.converters --config src/config/data.yaml                 # 或 --dataset aegis2 xstest
python -m src.converters.mix --config src/config/data.yaml --only core_v1

# 3) guard 模型（MIG 上）+ server
MIG_UUID=MIG-xxxx MODEL=Qwen/Qwen3Guard-Gen-4B deploy/vllm/serve_guard.sh
GUARD_MODEL_BASE_URL=http://localhost:8000/v1 CONFIG=src/config/server/models/qwen3guard_gen_4b.yaml deploy/server/run_server.sh
python -m src.server.check --url http://localhost:8080 --load 200     # 驗收：決策、/ready、延遲

# 4) 評測：直接打 server
python -m src.evaluation --server-url http://localhost:8080 --data processed_data/mixes/core_v1/ --run-name qwen3guard-4b
python -m src.evaluation --server-url http://localhost:8080 --endpoint guardrail_api --data ...    # 走正式 endpoint

# 4b) 裸模型安全（見 docs/model_safety.md）
python -m src.evaluation --config src/config/evaluation/model_safety.yaml src/config/models/qwen3_6_27b.yaml
python -m src.evaluation --list                                                        # 所有 task / target / generator / judge

# 5) 比較（換模型 = 用另一個 preset 重啟 server，再評測一次）
python -m src.evaluation.compare results/*/ --sort f1 -o results/leaderboard.md
```

## Config 規則

```bash
python -m src.evaluation \
  --config src/config/evaluation/default.yaml my_overrides.yaml \  # 依序合併
  --concurrency 32 \                                                # 明確給的 args 才覆蓋（預設為 None）
  --set target.inprocess.overrides='[pipeline.stages.0.threshold=0.7]' \
  --set metrics.fpr_targets='[0.01,0.02]'                           # 值以 YAML 解析
```

- `_base_: other.yaml` 用於繼承；list 是**整個取代**，不會 append。
- `.env` 的值會放進 `config["envs"]`。程式一律用 `env(cfg, "NAME")` 取值；config 只寫環境變數的**名稱**（`*_env` 欄位）。
- 寫入 `results/<run>/config.yaml` 時，名稱含 KEY/TOKEN/SECRET/PASSWORD 的值會被遮蔽。

## 每次評測的輸出（`results/<UTC>_<name>_<config_hash>/`）

| 檔案 | 內容 |
|---|---|
| `config.yaml` | 完整解析後 config（含 meta：git sha、argv、套件版本）＋ config_hash |
| `data_manifest.json` | 每個資料檔的 sha256、使用列數、取樣 seed、標籤/語言分佈 |
| `environment.json` | `pip freeze` |
| `target.json` | 被評測的 pipeline（各 stage、模型、threshold、server config） |
| `predictions.jsonl` | 逐筆結果（含 per-stage 分數、延遲、raw 輸出），可 `--resume` |
| `metrics.json` / `report.md` | precision/recall/F1/FPR（含 bootstrap 95% CI）、AUROC/AUPRC、best-F1 threshold、recall@FPR、per-stage、切片、latency |
| `slices.csv` | 依 source（檔名）/ language（偵測）的指標 |
| `errors.csv` | **最有把握的誤擋 / 漏擋**，人工分析用 |
