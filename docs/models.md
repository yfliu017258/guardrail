# 模型調查：H200 MIG 1g.35gb（2026-11-01 於 HF Hub 查證）

> 記憶體估算是推算值（bf16 ≈ 2 bytes/param），**沒有實機測過**。1g slice 只有 1/7 的 SM，瓶頸通常是 **prefill 算力** 而不是 VRAM：
> 限制 `--max-model-len`（8–16K）、截斷長輸入、guard 輸出要短（`max_tokens` 16–64），並在真實併發下量 p95。

## 1. 生成式 guard（主 stage）

| 模型 | 大小 / VRAM | 中文 | 自訂 policy | 輸出格式 → parser | 建議 | preset |
|---|---|---|---|---|---|---|
| **Qwen/Qwen3Guard-Gen-4B** | 4.4B / ~8.8GB | ✅ 119 langs | ❌ 固定 taxonomy | `Safety: Safe/Unsafe/Controversial\nCategories: …` → `qwen3guard` | **首選 baseline**（延遲/品質平衡） | `qwen3guard_gen_4b.yaml` |
| Qwen/Qwen3Guard-Gen-8B | 8.2B / ~16.4GB | ✅ | ❌ | 同上 | 4B 不夠時 | `qwen3guard_gen_8b.yaml` |
| Qwen/Qwen3Guard-Gen-0.6B | 0.75B / ~1.5GB | ✅ | ❌ | 同上 | cascade 第一層 / smoke | `qwen3guard_gen_0_6b.yaml` |
| **nvidia/Nemotron-3.5-Content-Safety** (2026-06) | 4.3B / ~8.6GB | ✅（繁簡未說明） | ✅ `chat_template_kwargs.custom_policy` | `User Safety: …\nResponse Safety: …\nSafety Categories: …` → `nemotron35` | **公司自訂規則首選** | `nemotron35_content_safety.yaml` |
| nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 | 8B / ~16GB | ✅ Mandarin | ✅（taxonomy 在 prompt） | JSON → `nemotron` | 第二個多語比較對象 | `nemotron_safety_guard_8b_v3.yaml` |
| openai/gpt-oss-safeguard-20b | 21B MoE (3.6B active) MXFP4 / ~14GB | 未說明 | ✅ 完整 policy | 依 policy（JSON）→ `json_policy` | 離線標註、policy 原型、第二層；**推理 token 慢** | `gpt_oss_safeguard_20b.yaml` |
| meta-llama/Llama-Guard-3-8B | 8B / ~16GB | ❌ | 部分 | `safe` / `unsafe\nS1,S2` → `llama_guard` | 英文業界參考值（gated） | `llama_guard_3_8b.yaml` |
| ibm-granite/granite-guardian-4.1-8b | 8.4B / ~16.8GB | ❌ en only | ✅ criteria | `<score>yes</score>` → `granite_guardian` | 英文 / RAG groundedness 軌 | `granite_guardian_4_1_8b.yaml` |
| openguardrails/OpenGuardrails-Text-2510 | 14.8B GPTQ int4 / ~10GB | ✅ | 平台開關 | `unsafe\nS#` → `llama_guard` + `category_map` | ⚠️ S2/S3 為 PRC 政治類別；需合規審查 | （可自行加） |
| ServiceNow-AI/AprielGuard | 7.9B / ~15.8GB | 未說明 | ❌ | `unsafe-O14\nadversarial` → `aprielguard` | 同時給 harm + adversarial | （可自行加） |
| 略過 | Llama-Guard-3-1B、Llama-Guard-4-12B（無中文、VRAM 緊）、ShieldGemma、WildGuard、Granite 3.3、DynaGuard | | | | | |

## 2. 輕量 classifier（第一層，ms 級）

| 模型 | 大小 | 中文 | label | 備註 | preset |
|---|---|---|---|---|---|
| **Horizon-Labs/prompt-injection-guard-base** | 308M | ✅（訓練語言含 zh） | `SAFE`/`INJECTION` | direct + indirect；內建 NFKC / zero-width 正規化；0.5 時 benign 觸發詞誤報 12.2% → **要校準 threshold** | `horizon_pi_guard_base.yaml` |
| Horizon-Labs/prompt-injection-guard-small | 141M | ✅ | 同上 | 更快 | 改 `model` 即可 |
| patronus-studio/wolf-defender-prompt-injection | 308M | 部分（含 Mandarin 訓練） | `benign`/`injection`（小寫） | Qualifire / 規避攻擊表現佳；indirect 較弱 | 改 `model` + `unsafe_labels` |
| **Horizon-Labs/content-safety-guard-small** | 141M | ✅ PolyGuard-zh F1 0.808 | 16 個 sigmoid（`unsafe` + 15 類） | 由 Qwen3Guard-Gen-8B 蒸餾；XSTest 誤擋 0.22 → 校準 | `horizon_content_safety_small.yaml` |
| meta-llama/Llama-Prompt-Guard-2-86M | 279M | ❌ 未評測 | **config 無 id2label** → 用 index 1 | 只當 baseline | `prompt_guard_2_86m.yaml` |
| 略過 | protectai v2（archived、英文）、Prompt-Guard-2-22M（英文詞表）、deepset、PIGuard、NemoGuard-JailbreakDetect、sheltron（NC） | | | | |

## 3. 建議實驗順序

1. **Baseline**：`llm_judge_baseline.yaml`（重現現有 SaaS/prompt 方式）→ 取得要超越的數字。
2. **單模型掃描**：Qwen3Guard-Gen-0.6B / 4B / 8B、Nemotron-3.5、Nemotron-8B-v3、Llama-Guard-3-8B（英文參考）。
3. **Classifier**：Horizon PI guard、content-safety-guard-small、Prompt-Guard-2（baseline）。
4. **組合**：`pipelines/two_stage.yaml`；用 eval 的 `per_stage` 與 ROC 選 threshold。
5. **自訂 policy**：Nemotron-3.5 + `policies/enterprise_policy.md`，對比內建 taxonomy，看公司特有類別（confidential）是否改善。
6. Phase 2：streaming（Qwen3Guard-Stream，非 vLLM，需自寫服務）、indirect injection（RAG / tool 輸出）。

## 4. 部署筆記

```bash
nvidia-smi -L                                   # 取得 MIG UUID
MIG_UUID=MIG-xxxx MODEL=Qwen/Qwen3Guard-Gen-4B MAX_LEN=8192 deploy/vllm/serve_guard.sh
python -m src.server --config src/config/server/models/qwen3guard_gen_4b.yaml
```

- `--max-logprobs` 要 ≥ `top_logprobs`，否則 vLLM 會拒絕帶 logprobs 的請求（腳本預設 20）。
- encoder 類（<1.2GB）可與 4B/8B LLM guard 共用同一個 slice；vLLM 用 `--gpu-memory-utilization` 留空間。
- gpt-oss-safeguard 需 `--reasoning-parser openai_gptoss` 讓 reasoning 與答案分開（parser 只看 `content`）。
