# 期中報告投影片：給本地模型（GLM-5.3）的指導書

這份文件分兩部分：**第一部分給你（使用者）看**，說明怎麼準備材料、怎麼把工作交給模型；**第二部分整段貼給模型**，是它要遵守的規則和每張投影片的規格。

目的只有一個：投影片上的每一句話都要能在程式碼或評測結果裡找到依據。模型不知道的事要寫「待填」，不能猜。

---

## 第一部分：使用方式

### 1. 準備事實資料

```bash
python -m src.analysis.report_context --results results/ -o report_context.md
```

產出的 `report_context.md` 內含：所有 run 的指標與信賴區間、各 run 的設定與警告、模型 × guardrail 的合併結果、誤判樣本。**投影片上的數字只能來自這個檔。**

還沒有真實的評測結果時也可以先產生大綱：數字的位置模型會標成「待填」。

### 2. 選擇餵給模型的方式

| 你的模型環境 | 做法 |
|---|---|
| 能讀 repo 檔案（coding agent、IDE 外掛） | 把第二部分貼給它，並告訴它 repo 路徑和 `report_context.md` 的位置 |
| 只能貼文字（聊天介面） | 加上 `--with-code` 重新產生：`python -m src.analysis.report_context --results results/ -o report_context.md --with-code`，把關鍵程式碼一起附在檔案裡（約 16 萬字元），連同第二部分一起貼上 |
| context 放不下 16 萬字元 | 分兩輪：第一輪貼第二部分 + 不含程式碼的 `report_context.md`，產生第 5–8 節（結果）；第二輪貼第二部分 + 個別程式檔，產生第 3–4 節（方法） |

### 3. 你要自己補的內容

以下資訊不在程式碼和結果裡，模型無從得知。請在交給模型時一併提供，否則它會標成「待填」：

- 報告對象、時間長度、頁數上限。
- 選用 Qwen3Guard 的理由，以及用哪個尺寸、部署在哪裡。
- 公司內部的背景：目前 SaaS 做法的實際問題、上線時程、latency 要求。
- guardrail service 的實作細節（它不在這個 repo 裡）。

### 4. 檢查模型的產出

模型交回的每張投影片都附「依據」欄。抽幾張對照：

- 依據寫的是檔案路徑和函式名稱 → 打開確認內容相符。
- 依據寫的是 `report_context.md` 的某一節 → 確認數字一字不差。
- 依據寫「待填」→ 由你補上，或刪掉那一點。
- 沒有依據的句子 → 要求模型補依據或刪除。

---

## 第二部分：給模型的指示（整段貼上）

你要為一份技術期中報告撰寫投影片內容。主題是「guardrail 評測流程的建置與初步結果」。你會拿到一個程式碼庫和一份事實資料檔 `report_context.md`。

### 你的工作方式

1. **先讀，再寫。** 依「閱讀順序」讀完程式碼和 `report_context.md`，之後才開始寫投影片。
2. **只寫有依據的內容。** 每一個論點都要對應到下列其中一種來源，並在「依據」欄寫明：
   - 程式碼：檔案路徑加上函式或類別名稱，例如 `src/evaluation/targets.py` 的 `HTTPGuardrailTarget.predict`。
   - 設定檔：檔案路徑加上欄位名稱。
   - 評測結果：`report_context.md` 的節號與列。
   - 使用者另外提供的說明：標「使用者提供」。
3. **數字只能抄，不能算、不能估。** 所有百分比、筆數、毫秒數都要和 `report_context.md` 完全一致。需要的數字不在檔案裡時，寫「待填」。
4. **不知道就寫「待填：＜缺什麼＞」。** 不要用常識或訓練時的印象補上。特別是：選模型的理由、公司背景、guardrail service 的內部實作、上線時程。
5. **程式碼和文件不一致時，以程式碼為準**，並在「待確認事項」列出不一致之處。
6. **不要誇大。** 樣本數少時必須提到信賴區間；測試用的假資料不能寫成真實結果。`report_context.md` 第 1 節的 run 名稱若含 `mock`、`smoke`、`test-model`，就是功能測試，不是正式結果。

### 閱讀順序

先讀前五項；其餘在寫到對應投影片時再讀。

| 順序 | 檔案 | 要弄清楚的事 |
|---|---|---|
| 1 | `report_context.md` | 有哪些 run、各自的數字、警告 |
| 2 | `src/evaluation/suite.py`（檔頭說明、`build_jobs`、`run_job`） | 一次評測怎麼組成、什麼情況會跳過／接續／重跑 |
| 3 | `src/evaluation/adapters.py` | adapter 的介面、送出的 request 格式、怎麼解讀 `action` |
| 4 | `src/evaluation/targets.py` 的 `HTTPGuardrailTarget` | canary、錯誤處理、service 資訊的擷取 |
| 5 | `src/evaluation/tasks/refusal.py` 的 `ModelSafetyTask` | 模型拒絕率怎麼量、被擋下的請求怎麼算 |
| 6 | `src/evaluation/generation.py` 的 `LiteLLMChatGenerator` | 怎麼呼叫模型、怎麼分辨 guardrail 的 400 |
| 7 | `src/refuse_judge/guard.py` | Qwen3Guard 的輸出怎麼讀 |
| 8 | `src/evaluation/runner.py`、`src/evaluation/dataset.py` | 重試、續跑、抽樣 |
| 9 | `src/analysis/loader.py` 的 `summarize`、`combine`、`combo_summary` | 指標和「模型 × guardrail」的算法 |
| 10 | `src/analysis/app.py` 的 `PAGES` 與各 `page_*` 函式 | 分析介面有哪些頁面、每頁有什麼 |
| 11 | `src/config/evaluation/*.yaml` | 預設值 |
| 12 | `docs/prompt_injection_data.md` §1（若存在） | 資料集的已知問題 |
| 13 | `docs/plan.md`（若存在） | 專案目標與時程 |

### 投影片規格

共 12–14 張。每張的「必須回答」是那張投影片存在的理由；答不出來就標「待填」，不要用別的內容充數。

| # | 標題方向 | 必須回答 | 主要依據 |
|---|---|---|---|
| 1 | 標題頁 | 報告主題、報告人、日期 | 使用者提供 |
| 2 | 背景與問題 | 為什麼需要 guardrail？現有做法缺什麼？ | `docs/plan.md` §1；使用者提供 |
| 3 | 目標與進度 | 四個目標各自做到哪裡？ | `docs/plan.md` §2；程式碼中實際存在的功能 |
| 4 | 評測流程總覽 | 從資料集到分析介面，資料怎麼流動？ | 閱讀順序 2–5、9 |
| 5 | Guardrail 評測怎麼做 | 送什麼給 service？怎麼判定對錯？哪些情況不計分？ | `adapters.py`、`HTTPGuardrailTarget`、`tasks/guardrail.py` |
| 6 | 模型拒絕率怎麼做 | 怎麼呼叫模型？誰判斷有沒有拒絕？被 guardrail 擋下的請求怎麼處理？ | `ModelSafetyTask`、`LiteLLMChatGenerator`、`guard.py` |
| 7 | 可信度的設計 | 做了哪些事讓結果可重現、不會被誤讀？ | canary、續跑、固定抽樣、service 資訊、協定錯誤不計分（各自的程式位置） |
| 8 | 資料集 | 用了哪些？規模？已知問題對數字的影響？ | `report_context.md` §1；`docs/prompt_injection_data.md` §1 |
| 9 | 模型選擇：Qwen3Guard | 用在哪些地方？為什麼選它？ | `guard.py`、`model_safety_litellm.yaml`；理由為「使用者提供」 |
| 10 | 結果：guardrail | 各資料集的 recall 與 FPR（含信賴區間） | `report_context.md` §1 |
| 11 | 結果：模型 + guardrail | 殘餘風險降低多少？正常請求受多少影響？latency 增加多少？ | `report_context.md` §3 |
| 12 | 誤判案例 | 一個漏擋、一個誤擋，可能的原因 | `report_context.md` §4（原因若無依據，標「推測」） |
| 13 | 限制 | 目前的數字有哪些不能下的結論？ | `report_context.md` §2 的警告；信賴區間；`combine` 的 latency 算法 |
| 14 | 下一步 | 後半段要做什麼？ | `docs/plan.md` §4；使用者提供 |

### 寫第 5、6、7、13 張時必須查證的事實

這些是最容易寫錯的地方。寫之前請回到程式碼確認，並在依據欄註明你看的位置。

- guardrail 評測是直接呼叫 service，還是經過 LiteLLM？（看 `HTTPGuardrailTarget.predict` 送往哪裡）
- 判定「擋下」的條件是什麼？`GUARDRAIL_INTERVENED` 算不算擋下？（看 `predict` 最後如何設定 `unsafe`）
- service 的回覆缺少 `action` 時怎麼處理？（看 `GenericGuardrailAPIAdapter.parse_response`）
- canary 沒被擋下時會發生什麼事？（看 `_canary`）
- 模型拒絕率預設把「被 LiteLLM guardrail 回 400 的請求」算成什麼？（看 `model_safety_litellm.yaml` 的 `on_blocked` 和 `ModelSafetyTask.predict`）
- 程式如何分辨「guardrail 的 400」和「其他 400」？這個方法的限制是什麼？（看 `LiteLLMChatGenerator.is_blocked`）
- Qwen3Guard 的判定有沒有「部分拒絕」這個類別？（看 `guard.py` 檔頭說明與 `parse_reply`）
- 「模型 × guardrail」的 latency 是實測還是相加？被擋下的請求怎麼計？（看 `loader.combine`）
- 信賴區間用什麼方法？（看 `loader.wilson`）
- 中斷後續跑時，什麼情況下不會接續而是開新的 run？（看 `suite.run_job` 與 `__main__._check_resume`）

### 輸出格式

每張投影片用以下格式。投影片文字要短；細節放講者備註。

```
## 投影片 N：＜標題，一句完整的結論，不是名詞＞

要點（最多 4 點，每點一行）：
- ...

建議的圖表：＜圖表類型＞，資料來自＜report_context.md 的哪一節，或分析介面的哪一頁＞
（沒有適合的圖表就寫「無」）

講者備註（3–6 句，口語）：
...

依據：
- ＜論點＞ ← ＜檔案路徑＋函式名稱，或 report_context.md 節號，或「使用者提供」，或「待填」＞
```

標題要寫成結論。例如寫「Guardrail 評測直接呼叫 service，不經過 LiteLLM」，不要寫「評測方法」。

全部投影片之後，再輸出兩份清單：

```
## 待填清單
- 投影片 N：＜缺什麼資訊＞ ← 需要誰提供

## 待確認事項
- ＜程式碼與文件不一致之處、你不確定的解讀、依據薄弱的論點＞
```

### 交出前的自我檢查

逐項確認，並在輸出最後列出檢查結果：

1. 每個數字都能在 `report_context.md` 找到完全相同的值。
2. 每個要點都有依據；依據是「待填」的，已列入待填清單。
3. 沒有把功能測試的 run 當成正式結果。
4. 樣本數小於 100 的結果，都提到了信賴區間或樣本數。
5. 第 13 張（限制）至少涵蓋：資料集品質、樣本數、latency 是估計值、判斷模型未經人工驗證（若 `report_context.md` 沒有相反的證據）。
6. 沒有出現程式碼裡不存在的功能。不確定某功能是否存在時，回去搜尋程式碼。
