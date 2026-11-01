# 資料集調查（2026-11-01 於 HF Hub 查證）

> 欄位、license、筆數多數是讀 dataset card / viewer 確認的；標「未驗證」的項目是 gated 或 viewer 失效，取得權限後要再確認。
> 各 repo 在 2026 年仍有更新，**正式報告前務必用 `revision` 釘住版本**（`src/config/data.yaml`）。

## 1. 建議組合

| 用途 | 資料集 | converter | License | 備註 |
|---|---|---|---|---|
| 內容安全（en，prompt+response） | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` test 1,964 | `aegis2` | CC-BY-4.0 ✅ | 人工標註、可商用；丟掉 `REDACTED` 列 |
| 內容安全（zh-CN，可轉 zh-TW） | `nvidia/Nemotron-Safety-Guard-Dataset-v3` `zh/test.jsonl` | `nemotron_safety_v3` | CC-BY-4.0 ✅ | 與 Aegis 2.0 同 id，**不要跨語言 train/test**；機器翻譯 |
| 內容安全（zh-CN + en，含 response/refusal） | `ToxicityPrompts/PolyGuardPrompts` test | `polyguard_prompts` | CC-BY-4.0 ✅ | 會過濾「翻譯器拒答」被誤標成 harmful 的列 |
| 內容安全（en，最佳 response/refusal） | `allenai/wildguardmix` (WildGuardTest 1,725) | `wildguardtest` | ODC-BY，**gated** | 取得權限後確認 config 名稱 |
| 真實流量 FPR | `lmsys/toxic-chat` toxicchat0124 | `toxic_chat` | **CC-BY-NC** ⚠️ | 預設關閉，需法務確認 |
| Prompt injection（一站式） | `Horizon-Labs/prompt-injection-eval-suite`（16 configs） | `pi_eval_suite` | 各 config 不同 | 含 over-defense、direct、indirect（BIPIA / PIArena / LLMail）；`synthetic_*` 是發布者自家分佈 |
| Injection 邊界對 | `3nesdeniz/agentic-prompt-injection-boundary-pairs` | `boundary_pairs` | CC-BY-4.0 | 同詞彙的 benign/attack 成對 → 測關鍵字誤擋 |
| Over-defense | `leolee99/NotInject`（339） | `notinject` | MIT | 全部 safe，量 FPR |
| Jailbreak（in-the-wild） | `TrustAIRLab/in-the-wild-jailbreak-prompts` | `in_the_wild_jailbreak` | MIT | regular_* 可當 role-play hard negative；需去重 |
| Over-refusal | `Paul/XSTest`（450）、`bench-llm/or-bench` hard-1k / toxic | `xstest`、`or_bench` | CC-BY-4.0 | 英文 |
| Harm recall（unsafe-only） | `JailbreakBench/JBB-Behaviors`、`swiss-ai/harmbench`、`DAMO-NLP-SG/MultiJail` | `jbb_behaviors`、`harmbench`、`multijail` | MIT | 只能算 recall；MultiJail 有人工翻譯 zh |
| **原生 zh-TW 內容安全** | `ToxicityPrompts/RTP-LX` `data/ZH_Hant-*.parquet`（約 1K 筆/語系，另有 ZH_Hans、EN） | `rtp_lx` | ODC-BY | 人工標註 1–5 分；毒性續寫（RealToxicityPrompts 系），不是對話請求、偏 toxic；先確認是台灣用語而非香港 |
| 原生 zh-TW（台灣特有風險） | `mtkresearch/TS-Bench`（GitHub，400 筆：詐騙、金融、醫療錯誤資訊、歧視、政治操弄） | `local_file` | 未確認 | Breeze-Guard 作者所建，對 Breeze-Guard 有利 |
| PII（zh-TW） | `lianghsun/tw-PII-bench`（910） | `tw_pii_bench` | Apache-2.0 | 唯一查證過的 zh-TW PII（身分證/健保卡/統編…）；LLM 生成，僅 5–10% 人工檢查 |
| PII + secrets | `cjhwong/seatbelt-pii-tagger-data` heldout | `seatbelt_pii_heldout` | Apache-2.0 | zh-Hant / zh-Hans / en + regex 產生的 secrets |
| 內部資料 | 紅隊、SaaS 方案已標註的 log | `local_file` | internal | **不要 commit 原始檔** |

## 2. 重要發現

1. **原生 zh-TW 資料很少**（critic 二次查證後修正）：只有 RTP-LX `ZH_Hant`（毒性續寫）、TS-Bench（GitHub，400 筆台灣特有風險）、tw-PII-bench（PII）。沒有 zh-TW 的 prompt-injection / over-refusal 資料集。
   → 作法：以上述三者為錨點；`opencc: s2twp` 轉換（converter 內建選項，會標 `meta.converted`）+ 人工審；內部手寫公司情境 golden set。
2. **PRC 政治規範會污染標註**：`SUSTech/ChineseSafe` 的「政治错误」把六四、法輪功相關、批評中共標成違規；TC260 系列依中國標準。
   → `chinesesafe` converter 預設丟棄該 subject；不要用 TC260 系列。
3. **資料品質陷阱**：PolyGuardPrompts 的 zh 列有翻譯器拒答被標 harmful、`<SOURCE_TEXT>` 標記外洩；ChineseSafe 有明顯錯標；deepset/prompt-injections 標註可疑且被大量模型訓練過（只當 sanity check）。
4. **兩條標註軸要分開**：「是否是攻擊（injection/jailbreak）」vs「內容是否有害」。很多 PI 資料集把「單純有害但沒有 override 企圖」的請求標成 benign。
   → schema 中 `attack_type` 與 `categories` 分開；報告分 slice 看。
5. **PII 是「偵測」不是「阻擋」**：PII 出現在 prompt 中該擋、遮罩還是放行是 policy 決定；PII 資料集主要用來驗證 `regex` / mask stage。

6. **資料污染（contamination）**：模型在「自己的訓練資料家族」上分數會虛高，比較時要排除：

   | 模型 | 訓練資料（與 eval 重疊） |
   |---|---|
   | Nemotron-3.5-Content-Safety、Llama-3.1-Nemotron-Safety-Guard-8B-v3 | Aegis 2.0、Nemotron-Safety-Guard-Dataset-v3（v3 的 zh test = Aegis test 的翻譯，id 相同） |
   | Horizon content-safety-guard | Nemotron v3、Salad-Data、JBB-Behaviors |
   | Horizon prompt-injection-guard | neuralchemy、S-Labs、llmail-inject、TrustAIRLab、agentic-5k |
   | PolyGuard | PolyGuardMix（≈ PolyGuardPrompts 同源） |
   | YuFeng-XGuard | WildGuardMix |
   | Breeze-Guard-26 | TS-Bench（同作者） |

   server preset 的 stage 可寫 `trained_on: [aegis2, "nemotron_safety_v3*"]`，或在 eval config 寫 `metrics.contaminated_sources`；report 會另外列出排除這些 source 後的 **clean** 指標。

## 3. 統一格式（`src/data_schema/schema.py`）

| 欄位 | 說明 |
|---|---|
| `sample_id` | `source:sha256(text)[:16]`，跨 run 穩定；用於 resume 與去重 |
| `input_type` | `request`（pre_call，判斷 prompt）/ `response`（post_call，判斷回覆；`prompt` 只是上下文） |
| `label` | 被判斷那一側的 ground truth：`safe` / `unsafe` |
| `categories` | 統一 taxonomy（`Category`），`|` 分隔 |
| `language` | `en` / `zh-TW` / `zh-CN` / `zh` / `mixed` …（簡易繁簡偵測，可換 opencc / fasttext） |
| `attack_type` | `direct` / `indirect` / `jailbreak` / `roleplay` / `obfuscation` … |
| `meta` | 原始標籤、子類別等（JSON） |

## 4. 自建 zh-TW 測試集建議

- 300–1000 筆起步，**safe 與 unsafe 各半**，每筆標 `category` 與 `attack_type`。
- 一定要有 **公司情境的 hard negatives**：製程 / 良率術語（killer defect、kill ratio、strip、etch、化學 attack、shot 等字面看起來危險的詞）、工程用語（`kill -9`、exploit、attack surface、penetration test）、正常的政治 / 兩岸新聞討論、醫療自助問題。
- 一定要有 **公司情境的 unsafe**：把機密資料改寫成可外流格式、要求繞過資料分級、在貼上的文件 / email 中藏指令（indirect injection）。
- 兩人標註 + 計算一致性（Cohen's κ）；有爭議的列保留在 `meta.disagreement`，報告時可排除或單獨看。
