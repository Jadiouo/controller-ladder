# 閉門會議審查 第二輪（r2）：workshop 論文骨架 v3（交付前最後一關）

日期：2026-11-03
審查對象：`docs/paper/skeleton.md`（v3）、`docs/paper/claims_evidence.md`（v3）、`docs/paper/stats/equivalence.md`、`docs/paper/prespec_audit.md`
上一輪：`docs/paper/review_r1.md`（M1-M13）
本輪規則：只讀；沒有跑模擬或測試；只寫這個檔。

**說明：這是模擬的審查觀點，不是真的聯絡了專家，委員之間也不是彼此獨立的評分。分數代表目前材料的說服力，不是錄取率或任何真實結果的預測。委員與 r1 完全相同（A-H 八位），分數才能比較。**

**這是依「複審」精神加上完整辯論給的分數：本輪有大量新增成果（兩個鎖定集、三個新 spike），所以分數可以移動較多；沒有新增成果的部分沒有加分。**

## 0. 本輪實際核對了什麼（可重跑的）

| 項目 | 指令／來源 | 結果 |
|---|---|---|
| 稽核日誌行數與分布 | `wc -l results/.test_access.log`；`awk -F'\t' '{print $2}' ... \| sort \| uniq -c` | 111 行：demo_students 47、mjx_eval_arena 56、spike05/06/07/09b 各 1、spike10b 2、spike15a 1、spike15e 1（加總 111，與 skeleton §3.3 第 8 點一致） |
| 摘要字數 | 對 skeleton 第 28 行 `wc -w` | 183，與標示一致 |
| 摘要數字抽查（14 個，要求 8 個） | 開原檔對照 | 全部對上，見第 1 節 |
| Limitations 草稿字數 | `sed -n 146p skeleton.md \| wc -w` | **935 字**（預算表只給 250 字） |
| §3.5 討論字數 | 第 120-122 行 | 335 字（預算 180） |
| K2＋K3 兩列字數 | 第 38-39 行 | 459 字 |
| 殘留 "(to verify)" | `grep -c "(to verify)" skeleton.md` | 3 行仍有 |
| 殘留 "pre-regist" | `grep -n -i "pre-regist" skeleton.md` | 第 146 行（Limitations 9）仍有 "pre-registered tests"；claims_evidence 有 10 處舊列用語（該檔第 5 行聲明其意為 pre-specified） |
| spike16a 在 git 裡的狀態 | `git show --stat 9fe5682` | 結果已提交（48 檔），但 skeleton 開頭、第 67 行、§6、claims C-H16、prespec_audit 末段仍寫 TODO／running |
| spike15f 在 git 裡的狀態 | `git log`、讀 `results/spike15f/README.md` | 已提交（a8b4895），但 skeleton Limitations (12) 與 claims C-H14 仍寫「尚未分離」 |

## 1. 摘要數字回溯抽查（要求至少 8 個，實際 14 個）

| # | 摘要／正文數字 | 原檔 | 結果 |
|---|---|---|---|
| 1 | +17.6 pp [12.6, 22.6]（R1-R0，TEST-C） | `results/spike15a/table.md` 第 7 行 | 對 |
| 2 | 0.59 [0.36, 0.82]（C-ens） | `results/spike15e/table.md`「Success rate per method」 | 對 |
| 3 | 0.90 [0.68, 1.16]（B） | 同上 | 對 |
| 4 | +5.6 [0.6, 10.6]；90% CI [1.4, 9.8]；Holm p 0.0324；H16b p 0.0432 | `spike15e/table.md` verdict 表 | 對 |
| 5 | 66.4 / 84.4 / 77.0 / 82.6（R0/R1/C-ens/B） | 同上；我自己算 77.0-66.4=10.6、(10.6/18.0)=0.59、(16.2/18.0)=0.90 | 對 |
| 6 | -17.6 [-20.3, -14.9]（S6）、-23.8 [-26.9, -20.7]（S5） | `spike15b/table.md` pooled 列 | 對 |
| 7 | gate +15.4 [10.2, 20.6]、+63.2 [58.8, 67.4]，單 seed | `spike15b/table.md` gate 行 | 對 |
| 8 | gate +10.2 [4.8, 15.6]（8 次挑 1） | `spike13a/table.md` 第 24、56 行 | 對 |
| 9 | K4 +5.0 [-0.3, 10.2] | `spike12a/table.md` 第 25、73 行 | 對 |
| 10 | T-oracle 89.2、總 headroom 23.6 [18.8, 28.4] | `spike10b/table.md` 第 16、59 行 | 對 |
| 11 | 實際 TEST-C 梯度 65.6→83.2→88.6→93.0 與 H13b +4.4 [0.8, 7.8] | `spike15a/table.md` | 對 |
| 12 | M9 反證：x4 T-adapt3-full -12.5 [-21.5, -3.5]；L-est +4.0 | `spike13b/table.md` 第 66-80 行 | 對 |
| 13 | 同上的差 -16.5 [-28.5, -5.0] | **只在 `spike13b/README.md` 第 76 行，不在 table.md**；skeleton 引 `s13b/table.md:43-81` | 引用位置不對（必改 R2-M8） |
| 14 | spike16a catch/Push-T/toss 數字、gate 失敗 4 次 | `spike16a/table.md` | 對；但「bit-identical」略過頭（R2-M7） |

結論：沒有發現抄錯的數字。問題在敘述、版本同步和少數引用位置，不在數字本身。

---

## 2. 評分表（r1 → r2）

| 委員 | 背景與偏好 | r1 → r2 | 哪裡變好 | 哪裡還沒解決／新扣分 |
|---|---|---|---|---|
| A 古典控制／system ID | 最在意古典是否被公平調過 | 66 → 73 | spike15d 真的替古典做了同規則的對照（C-ens，同池、同聚合器、CAL 挑 20 個對 B 的 22 個），而且在 TEST-D 上贏過 R0 +10.6，這是對古典最公平的一步 | Laplace 先驗用的是生成分布的 sd；沒有 Laplace 寬度敏感度、沒有等量努力（只試三個強化變體）；多起點家族退化（439/500 沒有第二解）所以「古典對照」實際只有一個家族 |
| B 強化學習 | RL 零結果要有正對照 | 55 → 59 | S5/S6 PPO 用正確探索尺度重跑；approach A 的 gate 失敗有如實寫 | 仍然沒有任何 control-level 學習器在有 headroom 的設定裡學會；gate 全是單 seed、S5 gate 很簡單、S6 在天花板；「退化 17.6-23.8 pp」更像配方問題而不是學習的性質。主問題的 control-level 半邊仍是空的 |
| C 模仿學習／生成式 | 學生有沒有被公平放大 | 62 → 63 | S3 重跑乾淨（holdout，五個學生都 MATCH） | K4 不變（BC、單一資料量、NEW 種子、事後挑 Flow-1）；沒有 DAgger 或大規模版本 |
| D 實驗設計與統計 | 預先指定的真實性、多重比較、等效 | 52 → 67 | v8/v8.1/v9 卡片各自單獨先提交（可從 git 驗證）；Holm 家族明寫；H13b、H14、H16b 的失敗都照規則文字報告；equivalence.md 補了 TOST | v9 卡片自身矛盾（見議題 2）；spike15e 程式碼在開跑時未提交、中途 agent 被中斷由指揮官接手（只寫在 claims C-H14，skeleton K1 只寫 "committed alone"）；三個 spike（15b/15c/15d）仍是 README-only；equivalence.md 是兩支腳本手工合併，重跑會覆蓋 |
| E workshop 審稿人（benchmark／可重現性） | 貢獻可重用、協定清楚、結論撐得住 | 68 → 72 | 協定＋鎖定集＋稽核日誌＋誠實的失敗報告；Table 1 的規格現在清楚 | 這份「骨架」本身遠超 4 頁預算（Limitations 935 字對 250）；M11、M13 沒做；還沒有一張真的表或圖 |
| F 懷疑論者 | 專找過度宣稱 | 48 → 62 | "matched or exceeded"、"equal"、"pre-registered" 都處理；「bottleneck」也被審視過 | 見議題 1、3：「瓶頸在計畫選擇」仍是詮釋；C-ens 是在 B 贏了之後才建的；摘要第一句和第三句互相拉扯；「gate-validated」這個詞太強 |
| G 可重現性與稽核 | 日誌、種子、偏離表 | 60 → 66 | 111 行與分布逐項對得上；每列有 commit 與 rerun key；spike16a 做了本機對學校機器的數值比對 | 同一個事實在五處版本不同步（spike16a、spike15f）；`equivalence.py` 與 `equivalence_r2.py` 手工合併；spike16a 的「bit-identical」不精確 |
| H 召集委員 | 整體判斷 | 62 → 70 | 方向一致、誠實度高，現在有兩個鎖定集的硬數字，K3 從「HOLD 弱證據」變成「鎖定集中等偏強」 | 要把「骨架」變成 4 頁稿還有相當的工作；K2 的 8 個設定裡真正有 headroom 的只有 1-2 個，要在摘要說出來 |

平均：r1 約 59.1 → r2 約 66.8。沒拿到的分數裡，文字能修回來的約 6-8 分（版本同步、措辭、字數）；其餘只能靠補實驗（control-level 正對照、第二個有 headroom 的任務或模擬器、等量努力的古典對照）。

---

## 3. r1 必改項 M1-M13 逐條核對

| 項 | r1 要求 | r2 判定 | 驗證／備註 |
|---|---|---|---|
| M1 | 刪 "matched or exceeded"，改 N 設定寫法 | **已解決**，但解決後出現新的措辭問題（見 R2-M2） | skeleton 第 28 行摘要、K2 列已改為「+10 pp bar」寫法，N=3 來自 C-B10；`grep -n "matched or exceeded"` 不再命中摘要 |
| M2 | 不用 "pre-registered" | **大部分解決**；殘留 | skeleton 第 20 行、第 88 行改好；第 146 行 Limitations (9) 仍寫 "pre-registered tests"；claims_evidence 有 10 處舊列用語（檔頭有聲明） → R2-M9 |
| M3 | 標題不用 Equal／Fair；加預算比欄 | **已解決（文字）**；Table 1 的預算比欄僅在計畫中，表還沒生成 | skeleton 第 19 行；§3.4 E1 欄位清單含 budget ratio |
| M4 | S5/S6 PPO 重跑或降級 | **已解決**（spike15b 重跑）；**但 gate 為單 seed**，不能說已完全排除配方問題 | `spike15b/table.md`；摘要用 "gate-validated" 太強 → R2-M3 |
| M5 | S3 重跑或移除 | **已解決**（spike15b S3，holdout＋fresh，TEST 未動） | `spike15b/table.md` 第 76-92 行；equivalence.md S3 列 |
| M6 | 摘要數字附 CI，不挑最好的 | **已解決** | 摘要 14 個數字抽查全部同列有 CI |
| M7 | K3 要鎖定集重現，順序相依要處理 | **已解決**：TEST-C（v8）＋反向順序；標題 3 已劃掉 | `spike15a/table.md` 第 7、63-66 行；git 序見 prespec_audit |
| M8 | 等效檢定與跨 spike 多重比較說明 | **已解決**（`equivalence.md`），可重現性有瑕疵（兩支腳本手工合併，檔頭自己承認重跑會覆蓋 round-2 列）→ 建議項 S2-5 | |
| M9 | 脆弱性結果進主文 | **已解決**，數字對，引用位置有一處錯（-16.5 在 README） → R2-M8 | skeleton 第 122 行 |
| M10 | 稽核日誌涵蓋範圍 | **已解決** | 111 行分布逐項核對一致；Fig 1 說明已寫「arena evaluations logged」與未入日誌的 spike |
| M11 | 讀 2609.29424 與 2506.17832 全文，清掉 "(to verify)" | **未解決** | 3 行仍有 "(to verify)"；skeleton 第 78 行自己標「待讀」 |
| M12 | gate 寫成 "passed marginally" | **已解決（措辭）**；新種子重複 gate 未做（skeleton 第 115 行自己說未做） | |
| M13 | 4 頁預算可行 | **未解決，且惡化** | CFP 頁數仍未知；預算表加總 1,770 字，但實際草稿：Limitations 935 字（預算 250）、§3.5 335 字（預算 180）、K2+K3 兩列 459 字 → R2-M5 |

r1 未解決項：**M11、M13**（以及 r1 實驗 1：control-level 學習器正對照，實驗 4：等量努力古典，實驗 7：第二任務或模擬器，這三項仍然沒做，已被 Limitations 與 Discussion 如實承認，不算新的必改）。

---

## 4. 閉門會議：辯論

### 議題 1：「瓶頸在計畫選擇」是證據，還是詮釋？
- **F（懷疑論者）**：摘要寫 "true parameters add +17.6 pp, yet the bottleneck is plan selection"。+17.6 說的是「參數」是最大的一階；然後說「瓶頸是計畫選擇」，兩句話在同一句裡指向兩個不同的東西。證據只有一個對比：C-ens（不改善參數）在 TEST-D 上回收 0.59。這個證據支持的是「有很大一部分缺口可以不靠更好的參數就拿回來」，不是「瓶頸」。
- **A（古典控制）**：同意方向，補一個混淆。多計畫評分把「單一 θ̂ 下近乎混沌的成功率」取平均（R1+noise 診斷只剩 65%）。也就是說，C-ens 的增益可能是變異縮減，而不是「對不確定性穩健」。spike15d 只有一個控制：`R0 + c11/centre LAP`（只擴候選池、不做多計畫評分）= +3.6 [0.4, 6.8]，回收約 0.20。這個控制排除了「只是候選變多」，但沒排除「隨便 4 個鄰近計畫取平均也行」。
- **D（統計）**：另外，「瓶頸」暗示反事實（如果把計畫選擇修好，其他東西就不再是瓶頸）。TEST-D 上 B9 = 84.4 = R1，但沒有 R4（oracle）；TEST-C 上 R4 = 93.0，比 R1 高 9.8 pp。所以計畫選擇＋學習式估計器最多回到 R1，剩下的 R1→R4 完全沒碰到。摘要的 "the gap" 沒有定義，讀者會讀成「總 headroom」。
- **B（RL）**：反方：這個用詞在 skeleton 裡其實很小心，§3.5 寫的是「decision rule over an uncertainty set」，而且 K3 的 strength 欄已經寫「詮釋 medium」。
- **D 回**：小心在 K3 欄，不在摘要。摘要和標題候選 4 才是讀者先看的。
- **被說服**：B 被 A 的「變異縮減」論點說服（至少要標成未排除的替代解釋）。F 被 B 說服，承認 K3 欄的 strength 標示是誠實的；但仍堅持摘要必須同步。
- **結論**：**部分解決**。詮釋放在摘要、標題 4 就要降級成「不改善參數也能回收大部分缺口」；「瓶頸」留到討論段並明說替代解釋。缺的控制實驗（隨機鄰近計畫平均）列為選做。

### 議題 2：v9 卡片內部不一致（H16b 雙邊 95% CI 規則 vs Holm 單邊 p），處理夠不夠誠實？
- **D**：做法是對的：卡片規則文字是 binding，H16b 失敗（上界 10.6 > 10），同時把「單邊 p 0.0432 會過 Holm」寫出來。方向是選了對自己的主張不利的那個，這是正確的保守方向。三處揭露：K3 列、§3.3 第 4 點、Limitations (9)。
- **E（workshop 審稿人）**：夠誠實，但主文只有一句話的空間。現在 K3 列有兩個版本的上界（95% 的 10.6 與 90% 的 9.8），讀者會被兩個數字弄糊塗。建議主文只留一句：「The pre-specified rule (two-sided 95% upper bound < 10) fails (10.6); the card's own one-sided Holm p would pass (0.043); we report the rule text.」
- **F**：我挑另一個毛病：卡片事前寫死的解讀句是「學習式優勢可能 ≥10 pp，論文照實改寫 K2」。目前 K2 的句子是「a small, real advantage ... whose upper end does not exclude 10 pp」，開頭先說「small, real」，把預先寫死的方向（可能 ≥10）放到後面。這是強調順序問題，不是事實錯誤。
- **D 同意 F**：讀者會看到 "small, real" 就停。K2 該先寫「ahead by +5.6 [0.6, 10.6]」並把「≥10 pp not excluded」放在同一個分句。
- **未解決**：沒有。**共識**：處理方式誠實且足夠；加一個強調順序的修改（R2-M2 內）。

### 議題 3：C-ens 是不是夠強的古典對照？
- **A**：它比我預期的強：CAL 挑 20 個變體，B 挑 22 個；同池同聚合器；TEST-D 贏 R0 +10.6。
- **B**：但時間順序：先有 B（spike15c）贏 R0 +17，然後才建 C-ens（spike15d）。這代表「最強古典」是看到學習式贏了之後才升級的。K2 說「no learned method beat the strongest classical」，這句話在 R0（升級前的最強古典）上是**不成立**的：B−R0 = +16.2 [11.0, 21.4]，下界已經 > 10。
- **F**：這是這一輪最重要的一條。摘要沒有說 C-ens 是事後升級的。I-21 在 claims 裡有寫，但摘要和 K2 沒有。
- **A 反駁**：升級古典本身是協定的目的（K1：古典也要延伸工具），不是作弊。問題只在揭露。
- **C（IL）**：還有：學習式側的 3 個估計器是用模擬器標籤、13,120 個 episode 訓練的；古典側的 Laplace 先驗吃到生成分布的 sd。兩邊都不是「同樣東西」，而且古典側沒有做寬度敏感度。
- **被說服**：A 同意揭露必須進摘要；B 同意 C-ens 作為對照已盡力，缺的是等量努力（未做 spike15d 的 Laplace×2）。
- **結論**：**共識**。必改（R2-M1）：摘要與 K2 要說「古典基線是在學習式側贏過 R0 之後，用同一個選擇規則升級的」。「學習式估計器用了模擬器標籤」要進摘要（短語即可）。
- **仍未解決**：C-ens 是否「夠強」無法由現有證據回答；只能說「在這個規則與這個生成分布下已合理努力」。這是研究／執行後才會知道的事，不是送出前必須回答的。

### 議題 4：RL gate 多為單 seed／最佳挑選，摘要怎麼寫才不過頭？
- **B**：摘要 "with a gate-validated exploration scale it degraded the base by 17.6-23.8 pp" 的 "gate-validated"：S5 gate 是簡單 gate（init 0.0% → 63.2%），S6 gate +15.4 單 seed，S6 的 base 在天花板（96.8%）。這個詞暗示「RL 配方已被驗證可學習」；實際上驗證的是「在一個已知偏置上能學」。
- **E**：同意改，但這不是推翻。K1 與 Limitations (1) 已誠實列出。
- **C**：退化 17.6-23.8 pp 還有一個讀法：殘差 PPO 在有雜訊的探索下輸出非零殘差，base 本來就接近最佳，殘差只會傷害。這支持「RL 配方不適合」而不是「學習不行」。摘要已用 "residual-RL learner did not learn"，OK。
- **結論**：**共識**。"gate-validated" → "an exploration scale that passed a single-seed gate"（R2-M3）。

### 議題 5：spike16a 只當廣度（附錄 E2）是否恰當？
- **E**：恰當，而且骨架自己的措辭已經對：catch、Push-T 在天花板（classical 99.2/100.0%，+10 pp 算術上不可能），toss gate 失敗 4 次（最大 +1.3），沒有判決。這三張小表增加的是廣度，不是 K2 的證據。
- **B**：一個提醒：catch 的 learned−classical 是 -0.5 [-1.1, +0.1]，Push-T 只有 1 個訓練 seed。這兩個「TIE」在天花板上沒有資訊量，附錄的標題要避免出現 "tie" 這個詞被讀成等效。
- **D**：另一點：spike16a README 第 0 節的主張寫成「可以被支持或被反駁」，是單邊框架（只有 learned win 會算反駁）。這沒問題，但 Limitations 不要把這三個任務加進 "8 settings" 的計數。
- **G（可重現性）**：數值比對很好（本機對學校機器 30/30 outcome 相同、qpos 差 0.0）；但 Push-T 的 smooth 項差 7.8e-16，所以「bit-identical」不精確（R2-M7）。
- **結論**：**共識**，只當廣度是對的；不要升進主文。同時，版本同步（R2-M4）是必改。

### 議題 6：K2 的「8 個設定」是不是在數票？
- **F**：S3 本來就沒有 headroom（老師是精確的）；S6 base 在 96.8% 天花板；S4 是 pre-lesson recipe 的 pilot；S1/S2 追蹤；S8 BC-only。8 個設定裡真正有 headroom 且學習器有機會的，只有 S7（whip realistic）。「在 8 個設定都沒贏」會被讀成 8 個獨立證據。
- **H（召集）**：骨架 Limitations (5) 有寫，摘要末尾 "mostly one task" 也有。但摘要中段的 "8 settings" 沒有和 headroom 連起來。
- **A**：可以在摘要加半句「only the whip has headroom」。這比降低 8 的權重更誠實。
- **結論**：**共識**（併入 R2-M2 的摘要修改）。

### 議題 7：這份文件能不能算「能投」？（召集委員總結）
- **H**：證據層面可以：K3 已經是鎖定集中等偏強，協定與誠實度是賣點。稿件層面還不行：文字未達 4 頁，正文還沒寫，M11 沒讀完。
- **未解決**：CFP 頁數未知，無法判斷哪些要移附錄。

---

## 5. 共識：送出前最該處理的五件事

1. 摘要與 K2 揭露「最強古典是在學習式側贏過 R0 之後才升級的」（R2-M1、R2-M2）。
2. 摘要「瓶頸在計畫選擇」降成可由證據支持的句子；定義 "gap" 是 R0→R1，不是總 oracle headroom（R2-M2）。
3. 把 spike16a、spike15f 的結果同步到 skeleton、claims、prespec_audit（R2-M4、R2-M6），避免同一事實有 TODO 與數字並存。
4. Limitations（935 字）與 §3.5（335 字）壓到預算（R2-M5），並讀完兩篇最近鄰論文清掉 "(to verify)"（M11）。
5. 殘留 "pre-registered" 與引用位置錯誤清乾淨（R2-M8、R2-M9）。

---

## 6. 必改項（must-fix）

只列「照做會出錯／數字或版本對不上／違反作者規則」的項目，每條附位置、驗證與改法。

**R2-M1　C-ens 是事後升級的古典，摘要與 K2 沒說，K2 第一句在 R0 上不成立**
- 位置：`skeleton.md` §1 摘要；§2 K2 列；`claims_evidence.md` I-21 與 C-B10。
- 驗證：`results/spike15e/table.md` verdict 表：B−R0 = +16.2 [+11.0, +21.4]，下界 > 10；spike15c 日期（11-02 18:07）早於 spike15d（20:31）。
- 改法：摘要加「a rule built first for the learned side and then given to the classical side」（見作業單 W1）；K2 加一句：「The whip classical baseline was upgraded (C-ens) after the learned method had beaten R0 by +16.2 [11.0, 21.4]; against R0 alone the learned side clears the +10 pp bar.」（W2）

**R2-M2　摘要「瓶頸在計畫選擇」、"gap" 未定義、K2 強調順序**
- 位置：`skeleton.md` §1 摘要；§2 K2 列、K3 列（"and the bottleneck is plan selection:"）；§3.5。
- 問題：(a) 證據是「不改善參數也能回收 0.59」，不是「瓶頸」；(b) 多計畫評分可能只是變異縮減，沒有控制；(c) 回收分數的分母是 R1−R0（TEST-D 18.0），但 TEST-C 上 R4−R0 = 27.4，R1→R4 的 +9.8 完全沒被碰到，讀者會把 "the gap" 讀成總 headroom；(d) 8 個設定裡只有 whip 有 headroom；(e) v9 卡片預先寫死的解讀是「學習式優勢可能 ≥10 pp」，K2 卻先寫 "small, real"。
- 改法：整段摘要換成作業單 W1；K2、K3 照 W2、W3 改；§3.5 加一句替代解釋：「An untested alternative is variance reduction: averaging the model score over four nearby plants may smooth a success rate that is near-chaotic in theta_hat.」

**R2-M3　"gate-validated" 太強**
- 位置：`skeleton.md` §1 摘要（"with a gate-validated exploration scale"）；§3.4 E1 S5/S6 兩行；`claims_evidence.md` M4 列（第 226 行）。
- 驗證：`spike15b/table.md` gate 行：兩個 gate 都是 1 個 seed；S5 gate init 0.0%；S6 base 在 96.8%。
- 改法：W1 已改摘要為「an exploration scale that passed a single-seed gate」；E1 兩行把 "gate-validated" 換成 "single-seed-gate" 即可。

**R2-M4　spike16a 的狀態在五個地方不同步**
- 位置：(1) `skeleton.md` 第 3 行變更紀錄「Appendix slot E2 reserved for spike16a (TODO, no numbers)」；(2) 第 67 行「spike16a mini tables ... TODO, results not available」；(3) §6 第 149 行「spike16a (appendix E2) pending」；(4) `claims_evidence.md` C-H16 列（TODO，「No number is entered here」）與第 189 行「In progress」；(5) `prespec_audit.md` 末段「spike16a (running) ... will be checked」。同時第 141 行（App E2 列）卻已有完整數字。
- 驗證：`git show --stat 9fe5682`：spike16a 結果已提交；skeleton 第 141 行有數字。
- 改法：(1)(2)(3) 見 W5-W7；(4) 把 C-H16 改成實際數字列：catch hold 0.992→0.987（-0.5 [-1.1, +0.1]）、new 1.000→0.993（-0.7 [-1.8, +0.0]）；push_t hold +0.4 [-0.4, +1.2]、new -1.5 [-3.5, 0.0]；toss gate 4 次失敗（最大 +1.3 [0.0, 3.0]），僅描述 +0.6 [-1.2, +2.4]／-1.0 [-3.0, +1.0]；Holm p 0.975／0.672；標 [PRE-README]（README 與結果在同一 commit，且 spike16a 程式碼在開跑時未提交）；(5) 在 prespec_audit 加一列 spike16a：規則 README §0，與結果同一 commit 9fe5682，UNPROVABLE。

**R2-M5　頁數預算：Limitations 935 字、§3.5 335 字，遠超預算；M13 未解**
- 位置：`skeleton.md` 第 146 行（Limitations 草稿）、第 120-122 行（§3.5）、第 38-39 行（K2、K3）。
- 驗證：`sed -n 146p skeleton.md | wc -w` = 935；預算表第 64 行 250 字。
- 改法：Limitations 主文版保留 5 條（RL 沒學會、HOLD 重用、自訂真實條件、單模擬器＋單任務、預算與多重比較），每條不超過 50 字；其餘 (7)(8)(9)(10)(11)(12) 移到附錄 A 的偏離表；§3.5 壓到 180 字；K2、K3 兩列是表格內的註解，不進正文。同時在頁數預算表加「實際草稿字數」一欄，總和要 ≤ CFP 上限；CFP 頁數出來之前，以 4 頁為準。

**R2-M6　spike15f 已完成，Limitations (12) 與 C-H14 仍說「尚未分離」**
- 位置：`skeleton.md` Limitations (12)；`claims_evidence.md` C-H14（含「untracked」「its result is not used」）。
- 驗證：`results/spike15f/README.md`：holdout 1100-1199（n=100），警告 92-98% 來自 planner 的模型 rollout，plant 警告 0，執行的 episode 0/100 發散；但沒有在 holdout 500 與 TEST-D 上重跑。
- 改法：W4 取代 Limitations (12)；C-H14 同步改（commit a8b4895）。注意要保留「n=100，只在 holdout，TEST-D 全量未重檢」這個限制。

**R2-M7　spike16a「30/30 bit-identical」不精確**
- 位置：`skeleton.md` App E2 列（第 141 行）。
- 驗證：`results/spike16a/cmp_summary.txt`：outcome 與 episode 長度 10/10 相同，qpos 差皆 0.0，但 push_t 的 smooth 差 7.8e-16。
- 改法：W8。

**R2-M8　一個主文數字引用位置錯誤；K1 對 spike15e 的「committed alone」會被過度解讀**
- 位置：(a) `skeleton.md` 第 122 行：「-16.5 pp [-28.5, -5.0]」引 `s13b/table.md:43-81`，但該差值只在 `results/spike13b/README.md` 第 76 行；(b) §2 K1 列「15e (v9, committed alone)」。
- 驗證：`grep -rn "28.5" results/spike13b/*.md` 只命中 README；`results/spike15e/README.md` 第 5 行「code in spike15e/ was uncommitted at run time」，第 3 節「The original agent was interrupted ... the commander ran the gated TEST-D step」。
- 改法：(a) 引用改為 `s13b/README.md:76`；(b) W9：卡片單獨先提交，但程式碼在開跑時未提交，並由指揮官接手完成受閘門限制的步驟。

**R2-M9　殘留 "pre-registered"（r1 M2 的尾巴）**
- 位置：`skeleton.md` 第 146 行 Limitations (9)「Most pre-registered tests had」；`claims_evidence.md` C-D1、C-D5、C-F6、I-16、第 136 行及第 236、238 行。
- 驗證：`grep -n -i "pre-regist" skeleton.md claims_evidence.md`。
- 改法：skeleton 照 W10；claims_evidence 的舊列在引用進正文時一律改成 "pre-specified"（檔頭第 5 行的聲明可以保留，但表格內文最好也改，避免被逐字複製進論文）。

---

## 7. 建議項（選改，不採用也不算硬傷）

- **S2-1**（對應議題 1）選做一個控制：在 TEST-D 以外的 holdout 上，用「R0 估計值加隨機擾動」的 4 個計畫做同一個多計畫評分，量變異縮減能拿回多少。成本低（CPU、holdout n=500），能把「瓶頸在計畫選擇」從詮釋升到有控制的說法。
- **S2-2** 在 spike15d 加 Laplace×2 寬度敏感度（README 自己說沒做），可補 A 的疑慮；描述性即可。
- **S2-3** Table 1 的 S7 列同時給 R0 與 C-ens 兩個對比（skeleton E1 已要求，生成時別漏）。
- **S2-4** 主文把 H16b 的矛盾壓成一句（見議題 2 的 E 建議句）；細節留附錄。
- **S2-5** 把 `equivalence.py` 與 `equivalence_r2.py` 合成一支，一次重生整份 `equivalence.md`（目前檔頭承認兩支腳本手工合併，重跑會覆蓋 round-2 列）。
- **S2-6** App E2 的標題不要用 "tie"，改 "no learned win; catch and Push-T at ceiling; toss gate failed"，避免被讀成等效。
- **S2-7** TEST-D 上補報 R4（oracle）會讓「回收分數」有第二個分母；但那要動鎖定集，**不建議**。改為在摘要明講 "gap = R0→R1"（已含在 W1）。
- **S2-8** 把 H14 的 table.md 預設輸出「SUPPORTED（unadjusted）」在正文引用時一律加上「Holm p 0.094，未確認」，skeleton 已做到，生成表格時別漏。

---

## 8. 逐句修改作業單

下列「原文」均已用 `grep -c -F` 確認在 `docs/paper/skeleton.md` 中恰好命中 1 次（W1 為整段，以段首唯一）。本輪**沒有修改**任何檔案，作業單交由主 session 套用。

### W1｜必改｜摘要整段重寫（R2-M1、M2、M3）
原文定位：`skeleton.md` §1 摘要，第 28 行，整段（以 `Learning-based controllers are often reported` 開頭，唯一命中）
問題與處理：揭露 C-ens 事後升級、定義 gap 範圍、降級「瓶頸」、"gate-validated" 改實、加入標籤來源、說明只有 whip 有 headroom；字數控制在 200 以內（經 `wc -w` 計 199）。
完整替換文字：

    Learning-based controllers are often reported to beat classical ones, yet comparisons rarely give the classical side extended tooling, a disclosed budget, and rules fixed before testing. We propose an audited protocol for classical-versus-learned arm control in simulation: extended classical tooling, three-axis budgets, rules pre-specified in time-stamped hypothesis cards, a sanity gate for learners, and an access log. Across 8 tracking, juggling and whip settings, no learned method beat the strongest classical baseline by the pre-specified +10 pp bar, but only the whip has headroom; in 3 settings the residual-RL learner did not learn, and with a single-seed-gated exploration scale it degraded the base by 17.6-23.8 pp where tested. On realistic whip, locked sets (n=500) show true parameters add +17.6 pp [12.6, 22.6], yet most of that gap is recovered without better parameters: scoring candidate swings across several plausible plants, a rule built first for the learned side and then given to the classical side, recovers 0.59 [0.36, 0.82] of it with a classical Laplace ensemble and 0.90 [0.68, 1.16] with estimators trained on simulator labels. Under this rule the learned side is ahead by +5.6 pp [0.6, 10.6]; 10 pp cannot be excluded. One simulator, mostly one task.

篇幅：原 183 字 → 新 199 字；淨差 +16（仍 ≤ 200）
對應委員：F、B、A、D

### W2｜必改｜K2 句子：揭露升級、先寫數字再寫「不排除 10 pp」（R2-M1、M2）
原文定位：`skeleton.md` §2 K2 列
原文（精確替換範圍）：

    and on whip the learned estimator has a small, real advantage under robust plan selection (+5.6 pp [0.6, 10.6]) whose upper end (10.6) does not exclude 10 pp.*

問題與處理：預先寫死的解讀是「可能 ≥10 pp」；K2 先寫 "small, real" 把方向顛倒。同時揭露 C-ens 升級順序。
完整替換文字：

    and on whip the learned side is ahead of the upgraded classical (C-ens, built after the learned method had beaten R0 by +16.2 [11.0, 21.4]) by +5.6 pp [0.6, 10.6] under the same robust plan selection; an advantage of 10 pp or more cannot be excluded (H16b not supported).*

篇幅：原 29 字 → 新 47 字；淨差 +18（K2 列在表內，超出部分在 R2-M5 的壓縮中一併扣）
對應委員：B、F、D

### W3｜必改｜K3 句子：「bottleneck」降級（R2-M2）
原文定位：`skeleton.md` §2 K3 列
原文（精確替換範圍）：

    and the bottleneck is plan selection: scoring

問題與處理：證據只支持「不改善參數也能回收大部分缺口」。
完整替換文字：

    and a large share of that gap is recoverable by plan selection alone (the claim 'the bottleneck is plan selection' is an interpretation; variance reduction by averaging over nearby plants is not ruled out): scoring

篇幅：原 7 字 → 新 31 字；淨差 +24（同上，列於 R2-M5 壓縮）
對應委員：F、A、D

### W4｜必改｜Limitations (12) 同步 spike15f（R2-M6）
原文定位：`skeleton.md` §5 Limitations 草稿 (12)
原文（精確替換範圍）：

    (12) *Numerical warnings.* The TEST-D log has 125,555 MuJoCo 'simulation unstable' warnings (4,621 on holdout); no non-finite outcome appears in any executed episode record, but whether the warnings come from planner model rollouts or the executed plant has not been separated yet.

完整替換文字：

    (12) *Numerical warnings.* The TEST-D log has 125,555 MuJoCo 'simulation unstable' warnings. A re-run on 100 holdout episodes (spike15f) attributes 92-98% of the warnings to planner model rollouts, none to the executed plant, with 0/100 diverged executed episodes for R0, L-ens, B and C-ens; the full-n holdout and TEST-D check was not repeated.

篇幅：原 54 字 → 新 59 字；淨差 +5（此條將隨 R2-M5 移到附錄）
對應委員：G、D

### W5｜必改｜變更紀錄的 spike16a 狀態（R2-M4）
原文定位：`skeleton.md` 第 3 行 v3 變更紀錄
原文（精確替換範圍）：

    Appendix slot E2 reserved for spike16a (TODO, no numbers)

完整替換文字：

    Appendix slot E2 filled from spike16a (commit 9fe5682; breadth only)

篇幅：原 9 字 → 新 11 字；淨差 +2
對應委員：G、H

### W6｜必改｜預算表的 spike16a 狀態（R2-M4）
原文定位：`skeleton.md` §3 "Moved to appendix" 段（第 67 行）
原文（精確替換範圍）：

    spike16a mini tables (catch, toss, Push-T; TODO, results not available)

完整替換文字：

    spike16a mini tables (catch, toss, Push-T; results in `results/spike16a/table.md`, breadth only)

篇幅：原 9 字 → 新 12 字；淨差 +3
對應委員：G

### W7｜必改｜§6 開放項目的 spike16a 狀態（R2-M4）
原文定位：`skeleton.md` §6 第 149 行
原文（精確替換範圍）：

    spike16a (appendix E2) pending

完整替換文字：

    spike16a (appendix E2) done (breadth only); claims row C-H16 and `prespec_audit.md` still to be synchronised

篇幅：原 4 字 → 新 15 字；淨差 +11（同步完成後把後半句刪掉）
對應委員：G

### W8｜必改｜「bit-identical」（R2-M7）
原文定位：`skeleton.md` App E2 列（第 141 行）
原文（精確替換範圍）：

    machine-A vs machine-B 30/30 episodes bit-identical

完整替換文字：

    machine-A vs machine-B 30/30 episodes with identical outcomes and qpos (Push-T smooth term differs by 7.8e-16)

篇幅：原 7 字 → 新 15 字；淨差 +8
對應委員：G

### W9｜必改｜K1 對 spike15e 的 "committed alone"（R2-M8）
原文定位：`skeleton.md` §2 K1 列
原文（精確替換範圍）：

    15e (v9, committed alone)

完整替換文字：

    15e (v9 card committed alone; the 15e code was uncommitted at run time and the gated step was finished by the orchestrator with unchanged code)

篇幅：原 5 字 → 新 24 字；淨差 +19
對應委員：D、G

### W10｜必改｜殘留 "pre-registered"（R2-M9）
原文定位：`skeleton.md` Limitations (9)
原文（精確替換範圍）：

    Most pre-registered tests had

完整替換文字：

    Most pre-specified tests had

篇幅：原 4 字 → 新 4 字；淨差 0
對應委員：D、F

### W11｜必改｜-16.5 的引用位置（R2-M8a）
原文定位：`skeleton.md` §3.5 反證段（第 122 行）。該行有 "App C, s13b/table.md:43-81"，請在該行把 `s13b/table.md:43-81` 改成 `s13b/table.md:66-80 (drops) and s13b/README.md:76 (the -16.5 difference)`。這一行的同一個字串在檔案其他處也出現（Fig/App C 列），套用時只改第 122 行；其餘位置的引用是整張表，維持原狀。
問題與處理：-16.5 [-28.5, -5.0] 不在 table.md。
篇幅：淨差 +6
對應委員：G

### 非 skeleton 的同步項目（無法用單一唯一字串給替換，列定位）
- `claims_evidence.md`：C-H16 列（以 `| C-H16 | **TODO (spike16a, appendix E-bis):**` 開頭，唯一命中）整列換成 R2-M4 (4) 的數字列；第 189 行「In progress on the second machine」改為完成；C-H14 列（含 "a follow-up instrumentation script"）依 W4 改。
- `prespec_audit.md`：末段 "spike16a (running) states its design"（唯一命中）改為 R2-M4 (5) 的表列與一句結論；表格新增 spike15f（無規則，診斷）與 spike16a 兩列。

### 套用後檢查
- 摘要字數 ≤ 200（`sed -n 28p skeleton.md | wc -w`）。
- `grep -n -i "pre-regist" skeleton.md` 不再命中。
- `grep -n "TODO" skeleton.md` 只剩 "M11"、"M13" 相關；`grep -n "pending" skeleton.md` 不含 spike16a。
- 重新核對 `results/.test_access.log` 仍為 111 行。

## 9. 修改後字數表（摘要與正文）

| 文件部分 | 現況 | 只套必改（W1-W11） | 全部套用（含 R2-M5 壓縮） | 限制 |
|---|---|---|---|---|
| 摘要 | 183 | 199 | 199 | ≤ 200 |
| Limitations 草稿 | 935 | 約 980 | 約 250（主文）＋附錄 | 預算 250 |
| §3.5 討論 | 335 | 約 340 | 約 180 | 預算 180 |

R2-M5 的壓縮需要主 session 實際改寫，這裡不給逐字替換（內容是刪減與搬到附錄，沒有新增事實）。

## 10. 仍有爭議與需要本人決定的事

**需要確認的事實**
1. spike16a 的 `README` 第 0 節與結果在同一個 commit（9fe5682），程式碼開跑時也未提交：確認標成 [PRE-README] 沒有問題（本審查認為沒有問題）。
2. 兩篇最近鄰論文（2609.29424、2506.17832）的全文差異（M11）：需要你或指派的 agent 讀 PDF。

**需要選擇的方向**
1. 摘要要不要放「only the whip has headroom」（W1 已放）；若你認為會削弱 K2 的重量，替代方案是只在 Table 1 與 Discussion 寫。
2. S2-1（變異縮減控制）要不要在投稿前做：做了會把「瓶頸」留在摘要；不做就照 W1／W3 降級。
3. CFP 出來後，是否需要把 spike16a 升進主文（本審查建議不要）。

**研究或執行後才會知道的事（不是送出前必須回答的）**
1. C-ens 是否「夠強」的古典對照（Laplace 寬度、等量努力）。
2. 是否存在 control-level 學習器能在有 headroom 的設定裡學會（r1 實驗 1，沿用至今）。
3. 第二個模擬器或第二個有 headroom 的任務是否重現 K3。

## 11. 結論：能不能投

- **證據面**：可以。K2/K3 在改完 W1-W3 後，措辭與證據相稱；v9 卡片矛盾的處理誠實且足夠；spike16a 只當廣度是對的。
- **稿件面**：還不行。必改項 R2-M1 至 R2-M9 都是文字／同步／字數，不需要新模擬；M11（讀全文）與 M13（頁數）要在投稿前完成。
- **輪數**：這是第 2 輪。R2-M1 至 R2-M9 套用並重新驗證後，若仍有必改項，依規則停下來交給你決定，不開第 3 輪。
- **記錄**：每條必改的採納結果與理由請寫進 `docs/progress.md`。
