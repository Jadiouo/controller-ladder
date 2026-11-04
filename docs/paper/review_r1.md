# 閉門會議審查 第一輪（r1）：workshop 論文骨架

日期：2026-11-02
審查對象：`docs/paper/skeleton.md`、`docs/paper/claims_evidence.md`（只讀，未修改）
審查目的：寫正文之前，判斷 (1) 主張站不站得住；(2) 還缺哪些實驗；(3) 投 ICRA／RSS 2027 workshop 夠不夠。

**說明：這是模擬的審查觀點，不是真的聯絡了專家，委員之間也不是彼此獨立的評分。分數代表目前材料的說服力，不是錄取率或任何真實結果的預測。**

核對方式：除了讀兩份文件，另外實際核對了 `results/.test_access.log`（109 行，分布 demo_students 47、mjx_eval_arena 56、spike05/06/07/09b 各 1、spike10b 2，與文件一致）、`results/spike13b/table.md` 的 R0/R1（69.2 → 86.0，+16.8 [12.0, 21.6]，一致）、`results/spike13a/table.md` 的 gate 行（log-std0 -3.5：-3.0 [-8.5, 2.5]；-2：+7.8；-1.5 且 1.5M 步：+9.0；-1.5 且 3M 步：+10.2 [4.8, 15.6]，一致）、`results/spike14a/table.md` 的 L-dec4 與 ensemble（70.3、74.0，一致）。以上抽查沒有發現數字抄錯。問題不在數字，在詮釋與措辭。

「已定案、不重開」的決定：框架採「benchmark + 公平協定」（docs/progress.md:210、docs/03:78-79）；主問題是「學生要多什麼才能贏過古典老師」；不寫 "first"。本審查只檢查這些決定有沒有被寫過頭，不重開。

---

## 第一部分：委員與分數

| 委員 | 背景與偏好 | 分數 | 主要扣分理由 |
|---|---|---|---|
| A 古典控制／system ID | 最在意古典基線是不是被公平調過，也最在意模擬是否過度偏袒古典 | 66 | 理想化 whip 古典佔盡便宜（模型類內、無雜訊，估計誤差 < 1e-6）；古典拿到分布當先驗；但肯定 T-adapt3 與 T-oracle 的差距量測得扎實 |
| B 強化學習 | 最在意 RL 有沒有真的學會；零結果必須有正對照 | 55 | 主結論的 RL 部分多半「沒學會」（S5、S6、S7）；gate 只過 0.2 pp 且是 8 次裡挑最好的；S5/S6 用的是後來證實有問題的探索尺度 |
| C 模仿學習與生成式 policy | 最在意學生有沒有被公平放大（資料量、DAgger、調參） | 62 | K4 只有 BC、一個資料量、NEW 種子；Flow-1 是事後挑的；但「老師 canonicalise 比換生成式學生更重要」這個方向有意思 |
| D 實驗設計與統計 | 最在意預先註冊的真實性、多重比較、等效性 | 52 | 「預先註冊」是同一 repo 的 markdown；HOLD 被反覆重用還拿來選古典；跨 spike 沒有多重比較校正；「沒贏」被寫成「打平」，沒有等效檢定 |
| E benchmark workshop 審稿人 | 最在意貢獻是否可重用、協定是否清楚、結論是否被數據撐住 | 68 | 協定與稽核日誌是亮點；但單模擬器、單任務有 headroom；4 頁塞不下 |
| F 懷疑論者 | 專找過度宣稱 | 48 | 「每個設定古典都 matched or exceeded」與自己的例外和失敗模式不符；摘要用 4-7% 卻略掉同表的 29%；「equal tools」「pre-registered」「fair」三個詞都過頭 |
| G 可重現性與稽核 | 最在意日誌、種子、偏離表 | 60 | 稽核日誌只涵蓋 arena evaluate（spike01-04 不在內，109 行中 56 行是一次 MJX 評估），放進主文容易被讀成「所有鎖定集存取」；learn_demo 已過期 |
| H 召集委員（workshop 組織者視角） | 負責整體判斷：夠不夠上、該怎麼投 | 62 | 方向對、誠實度高，但現在是「一堆很誠實的弱證據」，缺一個站得住的強主張；頁數預算不可行 |

平均約 59。分數紀律：這一輪沒有新增成果，所以不是拿來評文字，而是評「骨架現在的主張強度」。沒拿到的分數裡，措辭可以修回來的約 10 分（F、D、G 的大部分），其餘只能靠補實驗（B、A、D 的大部分）。

---

## 第二部分：辯論重點

**議題 1：K2 到底是「古典贏」、「打平」還是「學習器沒學會」？**
- B：8 個設定裡，S5（juggling）、S6（理想 whip）、S7（真實 whip）的 PPO 都標為 not learned（C-F1、C-F2），所以「沒贏」跟「古典比較好」無法分開。S5、S6 的 PPO 還是用 log-std0 -3.5／-4.5，而 spike13a 已證明這個尺度會讓 PPO 學不會。
- A：S7 用了估計器（不是 RL），L-est 在 TEST-B 上 +0.9 [-3.5, 5.4]，那個是乾淨的零結果。
- B 反駁：L-est 是用模擬器標籤、in-distribution，估計器本身沒學好參數（C-D2）。
- D：兩邊都對，但摘要寫 "matched or exceeded" 就是在說等效，而沒有任何等效檢定。
- 結論：K2 只能寫成「沒有任何學習方法以 +10 pp 的門檻贏過最強古典；其中 N 個設定學習器本身沒有通過學會檢查」。N 要在 Table 1 實際數出來。**已有共識**。

**議題 2：「equal tools／fair protocol」站不站得住？**
- E：這是論文的賣點，且 hook（untuned 老師被 RL 贏，兩參數重調後消失）很有說服力。
- A：公平是雙向破功的：spike03a 古典用了約 7.7 倍的模擬步數；spike13a、14a 學習側用了 7～17 倍的 episode 預算；古典側拿到真實分布當 bounds／priors，學習側拿模擬器標籤。
- F：標題寫 "Giving Classical Control Equal Tools"，內文卻說「不是每個軸都相等，我們只是報告軸」。標題與內文互相打架。
- E 被說服：標題 1 的 "Equal"、"Fair" 要降級成「審計式」「附預算揭露」。
- 結論：**共識**，列為必改 M3。

**議題 3：headroom 階梯（K3）是不是因果歸因？**
- D：階梯是固定順序，而 C-C4 自己承認非加性（H_in 與 H_out 兩個方向數字不同），+16.8 只是這個順序下的邊際貢獻。R2-R1 在 NEW 上是 +7.5 [+1.0, +14.0]，在 HOLD 上不顯著，表示「其他區塊不顯著」可能只是統計力不足。
- A：總 headroom 有鎖定集的獨立對應（TEST-B +23.6），而且 R1 的方向在 HOLD 與 NEW 一致，不是單點。
- D 部分被說服：總量可信；但「參數估計是主要缺口」要寫成「在這個階梯順序、HOLD+NEW 上的最大單一區塊」，並同時放 leave-one-out（H_out）方向。
- 未完全解決：HOLD 被拿來選過古典，R0 在 HOLD 上偏樂觀（69.7 對 TEST-B 65.6），是否讓缺口被低估？委員傾向「會讓缺口被低估，所以方向保守」，但沒有數據證明。標為**未解決**，靠實驗 2 回答。

**議題 4：「預先註冊」這個詞能不能用？**
- G：卡片是同 repo 的 markdown，有幾張是看過先前 spike 的 TEST 結果之後才寫的（spike02 的 W1/W2 強度、learn_demo 超參數）；偏離表很長；外部沒有登記。
- E：workshop 讀者多半能接受「版本控制的預先指定」，只要說清楚。
- F：標題 2 與摘要寫 "pre-registered" 會被一句話打爆。
- 結論：改用 "pre-specified in time-stamped, version-controlled hypothesis cards"，並附 git 時間序驗證。**共識**（M2）。

**議題 5：K4 要不要放進 4 頁？**
- C：K4 有科學上有趣的訊息，但只有 BC、一個資料量、NEW 種子，Flow-1 還是事後挑的；放進主文會稀釋 K3。
- H：4 頁預算裡 Limitations 草稿就已約 600 字（約 25%），再放 Fig 4 不可行。
- 備註：作者偏好：「所有選定的項目都保留，只排序不砍」的偏好。因此建議不是刪除，而是「主文 2～3 句 + 附錄完整一張圖」，**由作者決定**。K4 已經在骨架中被標為次要，這裡只是把它的位置寫清楚。

**議題 6：古典的脆弱性結果（C-C6）該不該進主文？**
- F：actuator noise x4 時 T-adapt3-full 掉 12.5 pp，L-est 不掉（差 -16.5 [-28.5, -5.0]）。這是唯一一個「學習側比古典穩」的證據，卻被丟進附錄 C。
- A：這是 n=200、frozen 方法、未重調，只是弱證據。
- F：弱歸弱，但它與「古典 matched or exceeded」直接相反，選擇性放附錄就是選擇性報告。
- A 讓步：至少要在討論段放一句並指向附錄。**共識**（M9）。

**議題 7：gate 算不算正對照？**
- B：+10.2 對門檻 +10，是 8 次裡最好的，用 HOLD n=500 而非預先登記的 n=200，且預先登記的「零殘差 ≤ 40%」沒達成（改用 b=6，零殘差 51–58%，可用 headroom 約 20 pp），SAC 沒過。這是「勉強通過」。
- E：骨架已把這些寫在 Limitations，態度誠實。
- B：誠實放在限制段不等於主張強度合理；K1 的 gate 主張要同步降級。
- 結論：寫成 "passed marginally"，並在新種子上重複（實驗 3）。**共識**（M12）。

被說服的紀錄：E 被 D 說服，recovered fraction 必須附 CI（+1.1 [-3.3, 5.7] 除以 16.8，比例的 CI 大約從 -20% 跨到 +34%，「4–7%」不是可以獨立引用的點估計）。A 被 F 說服，把脆弱性結果放進討論。

---

## 共識

**能站住的（保留）**
1. 總 headroom：最強古典與 oracle 差 +23.6 pp [18.8, 28.4]（TEST-B，鎖定，一次性），HOLD／NEW 的階梯總量一致。這是全篇最硬的數字。
2. 「參數估計是最大的單一區塊」在 HOLD+NEW 上方向一致（+16.8 [12.0, 21.6]；+13.0 [4.5, 21.5]），可以當 K3 主張，但要附前提（見 M7）。
3. 學習到的估計器降低參數誤差卻沒有提升成功率（C-D1，預先註冊判決 "no side wins"），加上「成功率對 theta_hat 是近乎混沌的函數，浮點變化就移動 3.6 pp」（C-D3）。後者是很誠實、很實用的 benchmark 副產品。
4. 開頭的 hook（untuned 老師被 RL 贏、兩參數重調後消失）與稽核日誌、偏離表的整體誠實度。

**站不住或要降級的**
1. 「每個設定古典都 matched or exceeded」。
2. 「equal tools」「fair」「pre-registered」三個詞的字面強度。
3. K2 的 RL 部分（多數沒學會）不能當「古典較好」的證據。
4. K4 的「生成式學生至多 +5 pp」；那是點估計，CI 到 +10.2，且 Flow-1 是事後挑的。

**總評**：K1（協定）中等、K2（跨設定零結果）要大幅改寫措辭才中等、K3（headroom 分解）中等偏強但缺鎖定集確認、K4 弱到中等。

---

## 第三部分：必改項

每條格式：問題 → 對應檔案段落 → 怎麼驗證。

**M1　摘要與 §3.1 的 "in every setting ... matched or exceeded" 說得太滿**
- 位置：`skeleton.md` §1 摘要第 4 句；§3.1 Contributions；§2 K2 列。
- 問題：與自己的例外（C-B6 +9.3 [2.7, 17.3]；C-B8c OOD +5.9 [1.1, 10.8]；dose x2 +7.3 [-0.0, 14.7]）、與「多個設定學習器沒學會」互相牴觸；"matched" 暗示等效，但沒有等效檢定。
- 怎麼驗證：改後的句子要能在 `claims_evidence.md` 一對一對到 C-B10 的計數規則（learned ≥ +10 pp 且 CI > 0 的設定數為 0），同時在 Table 1 加一欄「學習器是否通過 ≥ 5 pp 學會檢查」，數出 N；摘要引用 N。建議替換（N 於 Table 1 填入）：
  「在 8 個設定中，沒有任何學習方法以預先指定的 +10 pp 門檻贏過最強的同工具古典基線；其中 N 個設定的學習器本身沒有通過學會檢查，因此那些設定不能證明古典較好。」

**M2　"pre-registered" 一詞過度**
- 位置：標題候選 2、摘要第 2 句與第 3 句、§3.3 第 4 點。
- 問題：沒有外部登記；部分卡片在看過先前 spike 的 TEST 後才寫（spike02、learn_demo）。
- 怎麼驗證：全文 grep "pre-regist"，逐處改成 "pre-specified in time-stamped, version-controlled hypothesis cards"；用 `git log --follow docs/03_hypotheses.md` 與各 spike 結果檔的 commit 時間，確認每個宣稱「預先」的卡片版本確實早於對應結果；找不到的 spike 在偏離表標示「卡片晚於結果」。

**M3　標題 1 與 K1 的 "Equal Tools"、"Fair"**
- 位置：標題 1；§3.3 第 2、3 點；Limitations (6)。
- 問題：預算超出 2x 上限的情況在 6 個 spike 中有 3 個（含學習側 7～17 倍）；spike03a 是古典側多用 7.7 倍；古典側拿到分布當先驗。標題與 Limitations (6) 自相矛盾。
- 怎麼驗證：標題不含未加限定的 "Equal"／"Fair"（例如改成 "An Audited, Budget-Disclosed Benchmark ..."）；Table 1 加「預算比（學習側／古典側）」欄，數字取自 App B（`s03a/README.md:23-30`、`s13a/README.md:99-102`、`s14a/README.md:50-59`）。

**M4　K2 被 RL 沒學會與舊探索尺度污染**
- 位置：§3.4 E1（S5、S6）；Limitations (1)；`claims_evidence.md` C-A4、C-B5、C-B7、C-F2。
- 問題：S5/S6 的 PPO 用 log-std0 -3.5／-4.5（spike06/07、09b），而 spike13a 證明 -3.5 學不到簡單修正（-3.0 [-8.5, 2.5]，改 -2.0 變 +7.8 [3.0, 12.8]）。目前只在 Limitations 寫「可能受影響」。
- 怎麼驗證：二選一。(a) 跑實驗 3，用修正後的探索尺度重跑 S5、S6 的 PPO 並過 gate；(b) 在 Table 1 把 S5/S6 的 PPO 列標為「uninterpretable」，不計入 K2 的證據數。無論哪個，主文要出現一句話交代。

**M5　S3（learn_demo）已過期且受污染，卻列在 Table 1**
- 位置：§3.4 E1 的 S3；`claims_evidence.md` C-B3、I-6。
- 問題：表格產生於 2026-10-25，早於 2026-10-30 的參考角度 wrap 修正；超參數看過一次 TEST 後才選；progress.md 自己寫「要整個重做」。
- 怎麼驗證：重跑後 `results/learn_demo/table.md` 的時間晚於 wrap 修正，且 `.test_access.log` 有新紀錄；否則從 Table 1 移除，K2 由「8 個設定」改為「7 個設定」，並同步改 C-B10。

**M6　摘要的數字挑選與不確定性**
- 位置：摘要第 5～6 句；§3.4 E3、E5；`claims_evidence.md` C-D2、C-D5、C-E2、C-E3。
- 問題：(a) "recover only 4-7%" 沒附 CI（分子 CI 跨 0，比例 CI 約 -20% 到 +34%）；(b) 同表的 3 模型 ensemble 回復 29%（+4.8 [-0.8, 10.4]，事後）被略掉，讀者會以為學習側上限就是 4–7%；(c) "halve length and damping error" 只對這兩項成立，質量 -29%、摩擦 -21%（I-13，已有處理）；(d) "generative students added at most 5 pp" 是點估計，Flow-1 CI 到 +10.2，且 Flow-1 是事後挑的。
- 怎麼驗證：每一個摘要數字能在 claims 表找到同列且 CI 一併出現。建議替換：
  「A learned estimator roughly halves length and damping error (not mass or friction error) yet recovers 4–7% of the gap (single models; a post hoc 3-model ensemble recovers 29%, CI includes 0).」
  「...whereas the best generative student added +5.0 pp (95% CI [-0.3, 10.2]) over an MLP, below the pre-registered bar but not an equivalence result.」

**M7　K3 只有 HOLD+NEW，歸因順序相依**
- 位置：§2 K3；§3.4 E2；標題候選 3；`claims_evidence.md` C-C1～C-C4、第 2 節。
- 問題：HOLD 曾用來選古典（R0 在 HOLD 69.7 對 TEST-B 65.6，樂觀），NEW n=200 CI 很寬；階梯是固定順序，C-C4 自己說非加性；R2-R1 在 NEW 顯著而 HOLD 不顯著，「其他區塊不顯著」不等於不存在。標題 3「Where the Headroom Is」比證據強。
- 怎麼驗證：Fig 2 同時畫 ladder（由下往上）與 leave-one-out（H_out，由上往下）兩個方向；文字用「在固定順序階梯下最大的單一區塊」；標題 3 在實驗 2 完成前不得採用；理想上實驗 2 在新的鎖定種子上重現 R0、R1、R4 與 L-dec4 的判決。

**M8　統計段落：等效性與跨 spike 多重比較**
- 位置：§3.3 第 4 點；Limitations (7)(9)；`claims_evidence.md` 第 4 節第 10 項。
- 問題：沒有等效檢定就不能說 "matched"；全文約數十個 CI，只在單一假說內做 Holm；少數正向例外（OOD +5.9 [1.1, 10.8]）在任何跨 spike 校正下都會變得不顯著。
- 怎麼驗證：不需新實驗。對 Table 1 每個 learned-minus-best-classical 的 CI，計算它與預先指定等效邊界（例如 ±5 pp 與 ±10 pp 兩個）的關係，附一張小表；並在文中明講「跨 spike 未校正，正向例外視為探索性」。檢查：表格中每個宣稱 "matched" 的列都有 CI 完全落在邊界內，否則用詞改成 "no detectable difference at the +10 pp bar"。

**M9　古典脆弱性結果被藏在附錄**
- 位置：§3.4 附錄 C 指向；§3.5 討論；`claims_evidence.md` C-C6。
- 問題：這是唯一一個「古典 SysID 在 actuator noise x4 掉 12.5 pp、學習估計器不掉」的證據，與主張方向相反，不能只放附錄。
- 怎麼驗證：主文 §3.5 至少一句話，含數字與 n=200／未重調兩個前提，並指向 App C。

**M10　稽核日誌的涵蓋範圍**
- 位置：§3.3 第 8 點、Fig 1；`claims_evidence.md` C-A1、I-15。
- 問題：骨架只提「spike05/06 追補」，沒提 spike01、02、03a、03b、04 的 TEST 評估根本不在日誌；109 行中 56 行是一次 MJX 評估，47 行是 demo_students。把 "109 行" 單獨寫出，會讓讀者以為鎖定種子被碰了 109 次或全部有紀錄。
- 怎麼驗證：Fig 1 說明用「日誌涵蓋的 arena 評估」；另列一欄「未入日誌的 spike」；`awk -F'\t' '{print $2}' results/.test_access.log | sort | uniq -c` 的結果與圖一致（本輪已核對，一致）。

**M11　相關工作未讀全文，且最近鄰論文尚未區分**
- 位置：§3.2；所有 "(to verify)"。
- 問題：`docs/02_lit_check.md` 只讀摘要；2609.29424（混合手臂，PID/MPC/RL/蒸餾的三方比較）被標為最接近但全文未讀，差異段落是佔位；2506.17832 同。
- 怎麼驗證：投稿前 grep 不到 "(to verify)"；兩篇各寫一句「它做了什麼、我們不同在哪」，數字取自 PDF 本文。

**M12　gate 的描述要在主文降級**
- 位置：§2 K1；§3.4 E4；摘要 "a sanity gate that a learner must pass"。
- 問題：通過是 +10.2 對 +10（餘裕 0.2 pp），8 次 gate 中最好的一次，評估集與預先登記不同（n=500 而非 n=200），預先登記的零殘差 ≤ 40% 未達成（可用 headroom 約 20 pp），SAC 沒過。
- 怎麼驗證：主文用 "passed marginally"，並列出上述 4 點；若實驗 3 的新種子重複驗證通過，則可改回 "passed"。

**M13　頁數預算不可行**
- 位置：§3 開頭預算；§5 Limitations 草稿。
- 問題：4 頁約 2,400 字，卻計畫 4 張圖、2 張表、約 600 字的九點 Limitations。
- 怎麼驗證：先確認 CFP 頁數（目前未公布，骨架已註明是推論）；依實際頁數做逐節字數表並加總不超過上限；Limitations 壓成條列，細節移到附錄的偏離表。

---

## 建議項（選改）

1. 標題用候選 1 的改寫版（見 M3）；候選 3 留到實驗 2 之後。
2. I-10 的 L-est-full 在 TEST-B 與 HOLD 都是 66.5%，投稿前確認是巧合而不是複製貼上。
3. Table 1 加入每個設定的 n 與「CI 半寬」，讓讀者看到 TEST n=50 的比較只有約 ±15 pp 的解析度。
4. 對 TEST 在 juggling 被 spike05/06/07 重複使用的事，在 Table 1 腳註標示。
5. 討論段加一句「這個零結果不能告訴我們什麼」（骨架 §3.5 已提到，建議給具體例子：不能推論真實硬體、不能推論更強的 RL 配方）。
6. 在 Fig 3 把 post hoc 的 R1+noise 診斷以不同符號標示為事後分析。

## 可以不改的

- 「benchmark + 公平協定」的整體框架（作者定案）。
- 不寫 "first"，只寫 "to our knowledge ... search limited to 2026-10"。
- 「Reported as not supported, never refuted for all designs」的用語。
- K1 與 gate 教訓合併、(c)(d) 合併成 K3 的決定。
- 把 catch/toss/Push-T 與古典階梯降為動機材料，不計入 Table 1。
- Limitations 的自我揭露程度（九點都有數字出處），只需要壓縮。
- TEST-B 只用一次，且不為了「看起來好」重用。

---

## 還需要的實驗（依重要性排序）

| 序 | 實驗 | 成本 | 補哪個主張 | 備註 |
|---|---|---|---|---|
| 1 | **真實條件 whip 上的學習側正對照**：至少一個學習器在這個 +23.6 pp 的缺口上展現出實際回收一部分（候選：特權觀測殘差 RL、BC→RL 微調、大規模 DAgger），並通過同一個 gate | 高（需 GPU、多組設定、數個 seed；本機走 a GPU job queue，單工作 ≤10GB 與 ≤6 小時在預先授權內，12M 步以上要先問） | K2 的核心（「沒贏」能否與「沒學會」分開），以及主問題「學生需要什麼額外成分」 | spike13b README:65 的特權觀測探針計畫過但沒跑；沒有這一項，K2 只能寫成範圍受限的零結果 |
| 2 | **K3 的鎖定種子確認**：新 frozen card（v8）、全新鎖定集（TEST-B 已用掉），跑 R0、R1、R4 與 L-dec4／古典 C0 的估計器對決，單一分析路徑 | 中（CPU 模擬為主，500 種子 × 約 4～5 個條件；受 free -h 與 MemoryMax 限制，序列執行） | K3（參數估計缺口）、M7、M6 的 recovered fraction | 順便讓標題 3 有資格使用 |
| 3 | **用修正探索尺度重跑 S5、S6 的 PPO，並過 gate；同時在新種子上重複 gate；重跑 S3（learn_demo）** | 中（GPU；PPO 3M 步級別，每組數小時；learn_demo 為 CPU 小工作） | K2 的 N 值（M4、M5、M12）、K1 的 gate 主張 | 若時間不夠，改採 M4(b) 的降級寫法，這項就變成選做 |
| 4 | **估計器對決的等量努力古典側**：MAP 先驗權重調參、探針激發最佳化、多重擬合集成，對上 9～17 倍的學習側預算 | 中 | K3 第二半（C-D1、C-D6）；回答「對古典不公平」 | s14a/README.md:65 因 CPU 競爭放棄，現已可補 |
| 5 | **「真實條件」量級的合理性**：感測器／致動器雜訊、摩擦、質量散佈與硬體或公開手臂資料對照，並做一條不依賴我們選擇的靈敏度掃描；dose 曲線在鎖定前重跑、學習器逐級重訓 | 中到高（要找資料；掃描為 CPU） | 所有真實 whip 主張（S7、K3）的外部效度；Limitations (3) | |
| 6 | **統計重算（零新模擬）**：recovered fraction 的 CI、TOST／邊界表、跨 spike 多重比較說明 | 低 | M6、M8；K2 的「matched」措辭 | 這項不需要排隊，建議最先做 |
| 7 | **第二個有 headroom 的任務與（或）第二個模擬器**（或單一硬體驗證） | 高 | K2／K3 的普遍性；Limitations (4)(5) | workshop 不是必要條件；若不做，必須把主張範圍寫成「在這個模擬器與這個任務」並移除標題 3 |
| 8 | **放大學生的 K4 實驗**：多個資料量、DAgger、擴散模型調參、更多種子 | 中 | K4（C-E3 目前是統計力不足的零結果） | 若 K4 只放附錄一張圖，可延後 |
| 9 | **先驗不對稱的消融**：古典不給分布 bounds；學習不給模擬器標籤 | 中 | K1 的「公平」主張（M3 的另一半） | 選做 |
| 10 | **PPO 配置多 seed（≥ 5）與多演算法** | 中到高 | M4／C-F1 | 若實驗 1、3 做了，這項併入 |

**序 6 的註記**：它不需要 GPU 或排隊，是這一輪最便宜、對分數幫助最大的一步，建議在動任何模擬之前先做。

---

## 結論：夠不夠投 ICRA／RSS 2027 workshop

- **能不能投**：能，但**要先完成必改 M1～M13，並至少完成實驗 6、2；實驗 1 與 3 做不到時，必須用 M4(b) 的降級寫法並把 K2 的標題權重降下來**。時間上，CFP 預估 2027-03～05 截止（推論），距今有餘裕做實驗 6、2、3；實驗 1 視 GPU 排程與結果風險而定。
- **適合的 workshop 類型**：benchmark／評估方法／可重現性／負面結果類型。**不適合**期待新方法或硬體展示的 workshop。
- **目前版本若直接寫成正文投出**：預期會被抓到的第一批問題是 M1、M2、M3、M4（懷疑論者與統計委員都會抓）。
- 本輪停止條件：M1～M13 逐條處理並在 `docs/progress.md` 紀錄採納或不採納的理由；實驗 6 的結果進入 claims 表；之後再開第二輪（同一批委員，分數標成「上一版 → 這一版」）。若第二輪仍有必改項，依規則停下來問作者。
