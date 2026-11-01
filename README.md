# OJT — LiteLLM Guardrail：服務與可重現評測

為內部 LiteLLM proxy（v1.92.0）設計 input/output guardrail 的 OJT 專案參考實作：

- **`src/server`**：FastAPI + uvicorn 的 guardrail 服務，實作 LiteLLM 內建 `generic_guardrail_api` 的協定；多 stage pipeline，依序為 regex 規則、輕量 classifier、vLLM 上的 LLM guard。
- **`src/evaluation`**：自動、可重現的評測 CLI。可評測 in-process pipeline、guardrail 服務，或真正的 LiteLLM proxy。輸出 metrics（含 CI、ROC、recall@FPR）、切片、錯誤樣本，以及 config snapshot。
- **`src/litellm_admin`**：沒有 proxy config file 權限時，透過 LiteLLM admin API 以宣告式方式管理 guardrail（plan / apply / attach / test），每次變更都留 audit。
- **`src/converters`**：把公開或內部資料集轉成統一 CSV（`processed_data/`），附 manifest，並能凍結版本化的 eval mix。

計劃、調查與設計：[`docs/plan.md`](docs/plan.md) · [`docs/datasets.md`](docs/datasets.md) · [`docs/models.md`](docs/models.md) · [`docs/architecture.md`](docs/architecture.md)

## 目錄

```
src/
  config/
    config.py            # YAML 合併(_base_) → args → --set；.env → config["envs"]；dump 時遮蔽 secrets
    server/              # default.yaml, mock.yaml, models/*.yaml（各模型 preset）, pipelines/, policies/
    evaluation/          # default.yaml, smoke.yaml
    data.yaml            # converters + 固定 eval mix
  data_schema/           # GuardrailSample（CSV 格式唯一定義處）、taxonomy 對照表、語言偵測
  converters/            # base.py（registry/manifest）, datasets/*.py, mix.py, __main__.py
  server/                # app.py, pipeline.py, schemas.py, backends/{regex,hf_classifier,openai_guard,mock,parsers}.py
  evaluation/            # dataset.py, targets.py, runner.py, metrics.py, report.py, compare.py
  litellm_admin/         # client.py（admin API）, __main__.py（list/plan/apply/delete/attach/test/export）
deploy/litellm/guardrails.yaml     # 期望的 guardrails（admin API 用，進 git 版控）
deploy/litellm/proxy_config.yaml   # config-file 寫法參考（本專案不需要改）
deploy/vllm/serve_guard.sh         # 在 MIG slice 上啟動 vLLM
processed_data/samples/smoke.csv   # 30 筆中英小樣本（測試/CI）
tests/                             # 70 個測試，不需 GPU / 網路
```

## 快速開始

```bash
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"            # + ".[data]" 轉換 HF 資料集；+ ".[hf]" 本地 classifier
cp .env.example .env

pytest -q                             # 全部離線可跑

# 1) 不需 GPU 的端到端 smoke（mock backend）
python -m src.evaluation --config src/config/evaluation/smoke.yaml

# 2) 資料
python -m src.converters --list
python -m src.converters --config src/config/data.yaml                 # 或 --dataset aegis2 xstest
python -m src.converters.mix --config src/config/data.yaml --only core_v1

# 3) 模型（MIG 上）
MIG_UUID=MIG-xxxx MODEL=Qwen/Qwen3Guard-Gen-4B deploy/vllm/serve_guard.sh

# 4) 評測：in-process（最快的迭代迴圈）
python -m src.evaluation --server-config src/config/server/models/qwen3guard_gen_4b.yaml \
    --data processed_data/mixes/core_v1.csv --run-name qwen3guard-4b

# 5) 比較
python -m src.evaluation.compare results/*/ --sort f1 -o results/leaderboard.md

# 6) 服務 + 透過 admin API 註冊到 LiteLLM（不需改 proxy config）
python -m src.server --config src/config/server/pipelines/two_stage.yaml
python -m src.litellm_admin plan && python -m src.litellm_admin apply --yes
python -m src.litellm_admin test ojt-guard-pre
python -m src.litellm_admin attach --team <pilot_team_id> --guardrails ojt-guard-pre ojt-guard-post --yes
python -m src.evaluation --target litellm_apply_guardrail --data processed_data/mixes/core_v1.csv
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
| `slices.csv` | 依 source / language / input_type / attack_type 的指標 |
| `errors.csv` | **最有把握的誤擋 / 漏擋**，人工分析用 |
