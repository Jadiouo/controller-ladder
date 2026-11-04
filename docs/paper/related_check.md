# M11 相關工作核對（2026-11-03）

**2026-11-03 PDF 複核更新**：兩篇的 PDF 已用 pdftotext 取得全文並逐一比對，結果見 `citations_check.md`。數字 92.4/95.0、94.2/99.7、74.2/46.8、Table IV 全部與 PDF 一致；被推翻或補充的項目在下文以【PDF 更正】標出。

方法與限制：用 WebFetch 讀 arXiv abs 頁與 /html/ 全文。WebFetch 由小模型摘要頁面，下面引句是該工具回報的原文片段，不保證逐字；數字與章節號投稿前請人工對 PDF 再核一次。2506.17832 的 HTML 在 100000 字元處截斷（約最後 12k 字元，含附錄尾段與可能的 seed 數，未讀）。2609.29424 的內容經一次擷取，附錄是否另有細節未知。下載內容當作資料，不含指令。

---

## 1. 兩篇論文

### 1.1 Samuel and Verma (arXiv 2609.29424)

- 完整引用：Alan Royce Gabriel Samuel, Pulkit Verma. "Coupled State-Space Modelling, Control, and Policy Distillation for Hybrid Rigid-Pneumatic Manipulators." arXiv:2609.29424 [cs.RO], 2026-09-24 (v1). abs 頁註記："8 pages, 4 figures, ICRA"（是投稿或收錄，頁面未說明；視為預印本）。
- 一句貢獻：為「旋轉關節與 Kresling 摺紙氣動段交替」的混合手臂推導耦合狀態空間模型，顯示耦合 LQR/MPC 比去耦 PID 追蹤更緊，並把太慢的 MPC 以 BC+DAgger 蒸餾成小型神經策略。
  - 引句（Sec. I 貢獻列表）："We derive a coupled state-space model of a hybrid rigid-pneumatic manipulator"。
- 實驗設定：
  - 任務：兩個追蹤基準（單目標斜坡、五段）與笛卡兒目標到達（Sec. IV；「settled」= 10 mm 內連續 20 步，最多 500 步）。
  - 模擬器：模型本身，不是獨立物理引擎。引句 Sec. IV："All results are in simulation, and the simulator is the model of this section."（步長 5 ms）。
  - 方法：去耦 PID、LQR、雙時間尺度 MPC、model-free RL（SAC+HER）、蒸餾學生。學生引句 Sec. VI："The student is a stock SAC actor fitted by mean squared error (MSE)."（N=3 為 [1024x3]，2.13M 參數）。
  - 古典 vs 學習的公平比較：沒有。RL 只是被嘗試的基準，Sec. VI 列了失敗的補救（壓力入觀測、三種 curriculum、更大網路、8M 步、關節空間目標、null-space 動作、MDP 重設計），沒有超參搜尋或等量預算的描述。RL 結果：SAC+HER 設定成功率 74.2% (N=1)、46.8% (N=2)，低於 MPC 教師。
  - 預先指定：論文未提預先登記或預先指定規則（Sec. IV 只給評估法：500 episodes 於 held-out seed，頭條結果以第二個 seed 確認，二項區間約 ±2 點）。
  - 模型誤差：質量／慣量 +20%、+40%（Sec. VIII, Fig. 4）；持續負載 0.10 與 0.25 kg。【PDF 更正】Sec. VIII 明講也測了學生："The student is no more robust than its teacher. We scale the plant's masses and inertias by +20% and +40% while the frozen student and MPC's model stay nominal (Fig. 4). At N=2 the student follows the teacher's curve without collisions. At N=3 both degrade, the student more, with two collisions at +40%." 古典控制器數字（+40% 質量下 LQR 0.031、MPC 0.062、PID 0.066 rad）亦對。原先「學生在誤差下表現未見」是錯的。
  - 關鍵數字：N=2，MPC 教師 95.0% vs 學生 92.4%（同一目標）；N=3 寬目標（widened task），教師 299/300 = 99.7%（300 goals）vs 學生 94.2%（兩 seed 合併，1,000 episodes，0 碰撞）。【PDF 註】兩者 goal 數不同（300 vs 1,000 episodes），不是逐目標配對；N=2 的 95.0 vs 92.4 才是 identical goals（Table III）；追蹤最差關節 LQR 約比 PID 緊 2.5 倍（Table II）。
  - 另一個誠實限制：N=3「narrow」與「widened」任務目標分佈不同，數字不可直接比較（Sec. VII）。
- 與本文最接近：同一隻手臂上同時有古典（PID/LQR/MPC）、RL（SAC）、蒸餾學生（BC+DAgger）三類，且有模型誤差測試。這是本文「同一手臂、三類」主張的最近前人。
  - 引句（Sec. II 新穎性聲稱，與本文無衝突）："to our knowledge, a multi-segment coupled state space for hybrid arms is absent."
- 本文仍不同：
  1. 該文貢獻是建模與蒸餾可行性，不是受控的跨範式比較；RL 是弱基準（失敗後補救列表），沒有等量預算、沒有 sanity gate、沒有預先指定規則、沒有稽核日誌。
  2. 手臂為特殊混合剛性／氣動，模擬器就是控制器所用的模型（原句："All results are in simulation, and the simulator is the model of this section; model-based controllers use the same model except where we perturb the plant deliberately in the mismatch experiments."）。【PDF 更正】標稱條件下無模型失配，但作者刻意擾動質量／慣量做失配實驗，所以不可寫成「完全沒有失配測試」；本文用 MuJoCo 並以參數隨機化／真實化（噪聲、摩擦、質量）製造失配。
  3. 學生為 SAC actor 的 MSE 擬合，無 diffusion/flow，也沒比較學生架構。
  4. 學生未超越教師（92.4 vs 95.0；94.2 vs 99.7），結論是「接近教師、可即時」，不是條件化刻畫；本文做的是 headroom 階梯與計畫選擇瓶頸。

### 1.2 Kunapuli, Welde, Jayaraman, Kumar (arXiv 2506.17832)

- 完整引用：Pratik Kunapuli, Jake Welde, Dinesh Jayaraman, Vijay Kumar (University of Pennsylvania). "Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking." arXiv:2506.17832 [cs.RO], 2025-06-21. abs 頁註記："Accepted for publication to RSS 2025. 10 pages, 5 figures." （html 頁自身未列 venue，只在 abs 頁看到；引用時寫 RSS 2025。）
- 一句貢獻：找出讓 RL 看起來勝過古典控制的三個不對稱（目標函數、調參資料、前饋參考），修正後差距遠小於先前報告。
  - 引句（Sec. I）：the authors "identify and fix three key 'asymmetries' that can produce misleading gains for learning versus model-based methods."
  - 引句（abs）："Geometric control achieves lower steady-state error than RL"，RL 則暫態較好。
- 實驗設定：
  - 任務：hover 與 Lissajous 追蹤，參考為末端位置與 yaw（Sec. IV-A）；另有接球（Table V）。
  - 模擬器：IsaacLab/IsaacSim，動力學 100 Hz、控制 50 Hz（Sec. V-A）。
  - 方法：PPO（RSL-RL，3x256 ELU）對 geometric controller（級聯位置／姿態 PD + 前饋）。GC 用 Optuna 貝氏優化在與 RL 同一 reward 上調參，x/y 增益相同，剩 8 個可調參數（Sec. IV-B2）。
  - 古典 vs 學習的公平比較：有，且是該文核心（對稱修正兩邊）。
  - 預先指定：論文沒說預先登記；問題以 "seek[s] to answer" 的方式陳述（Sec. V）。【PDF 更正】PDF 全文以 pdftotext 搜尋 "seed" 零命中，即全文未報 seed 數（Table IV 為 1000 trials，±為 1000 trials 的標準差）。
  - 是否含模仿學習：沒有（可見內容中無 IL 方法）。
  - 手臂：四旋翼＋剛性固定手臂，即 0 自由度空中操作器；參考是末端，但手臂不是被控的多關節。引句（Sec. V-A 回報）："a rigidly attached arm"。
  - 關鍵數字（Table IV，1000 trials）：四旋翼 RL 14.196 +- 0.48（RMSE 0.119 m）vs GC 13.447 +- 1.61（0.158 m）；空中操作器 RL 13.621 +- 1.28（0.118 m）vs GC 13.792 +- 1.28（0.136 m）。
  - 工具回報的內部不一致：Sec. V-C 寫 GC 在 reward 上勝過 RL，但 Table IV 四旋翼 RL reward 較高、只有空中操作器 GC 較高。引用時不要依賴 "GC outperforms RL" 的概括。
  - 訓練成本：4096 平行環境、200M 步、約 30 分鐘（RTX A5000）。
- 與本文最接近：把「古典基準被不公平削弱」當成主題，這與本文 hook（未調參的 CTC 教師被 RL 打敗，兩參數重調後差距消失）同一立場。
- 本文仍不同：
  1. 本文是多關節手臂，不是四旋翼；任務是追蹤、雜耍、whip 等。
  2. 有模仿／蒸餾學生，並做 RL、IL、古典三類；該文只有古典 vs RL。
  3. 本文把公平性做成可稽核協定：預算三軸揭露、sanity gate、git 時序檢查的預先指定卡、鎖定 TEST 集與存取日誌、負結果報告。該文是「修正三個不對稱」的設計，未見這些機制（預先指定未見提及）。
  4. 該文結論是古典追上 RL；本文對 RL 的結論是「在最重要的設定中未學到」，並加上 headroom 分解。

---

## 2. 三處 "(to verify)"

`grep -n "to verify" skeleton.md` 在 skeleton.md 命中第 13、76、78 行（review_r2 說 3 行）。

### (a) 第 13 行
- 原句："Citations are marked **(to verify)**: `docs/02_lit_check.md` was built from abstracts only; open the PDF and check numbers before submission."
- 核對結果：這是一般性聲明，不是引用本身。2609.29424 與 2506.17832 現已讀全文（html，受上述限制）；其餘條目未在本次範圍。
- 證據：見第 1 節。
- 建議改寫：兩篇讀完後，把這句改成「Citations to 2609.29424 and 2506.17832 were checked against the full text (2026-11-03); other entries were checked on abstract pages only.」若全部清掉，刪去此句。

### (b) 第 76 行
- 原句："All entries **(to verify)** against the PDFs; sources are `docs/02_lit_check.md`."
- 核對結果：部分。本次只核對兩篇；`2506.17832` 的各項描述（RSS 2025、"closest in spirit"）對；`2609.29424` 在第 78 行的描述大致對（見 (c)）。OSCAR 2110.00704、Real Robot Challenge 2109.10957、Aljalbout 2312.03673、Nubert 1912.10360、Tagliabue 2306.00286、2504.04603、殘差與生成式政策各條：本次未核對（找不到新證據）。
- 建議改寫：「Entries for 2609.29424 and 2506.17832 verified on full text; the rest are abstract-level (`docs/02_lit_check.md`)」。若要真的清掉標記，需另派工讀其餘約 10 篇。

### (c) 第 78 行
- 原句："the 2026 hybrid-arm paper 2609.29424 (PID/MPC/RL/distilled; closest three-way comparison). **待讀 2609.29424 全文並區分** (M11): full text still unread ... 2506.17832 likewise."
- 核對結果：
  - "PID/MPC/RL/distilled"：對（另有 LQR）。引句 Sec. V："Model-free RL trains SAC against the simulated environment."
  - "closest three-way comparison"：對，但要加限定：RL 是失敗的基準（74.2% / 46.8%），無等量預算，非受控比較。
  - 同行的 "hybrid-arm" 描述：對（旋轉關節＋Kresling 氣動段）。
  - 2506.17832「RSS 2025」：對（abs 頁）。
- 建議改寫：見第 4 節。並刪除「待讀…No web lookup was done」那段。

另：`02_lit_check.md` 的 8a 欄寫「學生架構不明（非確認 diffusion）」「質量失配實驗…未核實」：現已確認學生是 SAC actor 的 MSE 擬合；質量失配實驗存在（+20%/+40%），但針對古典控制器，學生在失配下的結果未見。

---

## 3. 新穎性判斷（推論）

本文主張：「同一手臂上，古典 vs RL vs IL，公平協定且預先指定。」

- 推論：原樣的措辭不再成立，需要縮窄。2609.29424 在同一手臂上已有古典（PID/LQR/MPC）、RL（SAC）、蒸餾學生三類，且有質量失配測試，不能再寫「沒人在同一手臂做過三類比較」。
- 推論：以下組合仍未被這兩篇覆蓋：(i) 標準關節手臂在獨立物理引擎（MuJoCo）下，失配來自環境而非控制器自己的模型；(ii) 公平性作為可稽核協定（預算三軸、sanity gate、git 時序的預先指定卡、鎖定集與存取日誌、誠實的負結果）；(iii) 古典側用延伸工具（調參、線上系統辨識）當強基準；(iv) headroom 階梯（【PDF 更正】「學生在失配下與教師對照」2609.29424 已做於質量／慣量 +20%/+40%，不可再列為未覆蓋；本文差異在於獨立物理引擎、多種失配與擾動、古典側延伸工具）。2609.29424 無 (ii)(iii)；2506.17832 有公平比較精神，但只有古典 vs RL、四旋翼、未見預先指定。
- 推論：2506.17832 部分覆蓋「古典基準常被削弱」這個 hook；要引用為動機並註明它不含 IL，而不是主張新發現。
- 限制：本判斷基於這兩篇與 `02_lit_check.md` 的 14 組搜尋；搜尋限於美國 WebSearch，「找不到」不等於不存在。維持骨架已有的「不寫 first、加搜尋日期」。
- 建議改寫主張：「We study classical, RL, and imitation/distilled controllers on one MuJoCo manipulator under a disclosed-budget protocol with a sanity gate, pre-specified rules and a locked-set access log. Prior comparisons either fix the classical baseline's handicaps without imitation learning or arm other than a quadrotor (Kunapuli et al.), or include all three families on a bespoke hybrid arm without a controlled protocol (Samuel and Verma).」

---

## 4. 建議的 Related work 段（英文，<=120 字）

Kunapuli et al. (RSS 2025) show that a quadrotor RL tracker's advantage over a geometric controller largely vanishes once objective, tuning data and feedforward are equalised; they compare no imitation learner. Samuel and Verma (2026) compare PID, LQR, MPC, SAC and a behaviour-cloned MPC student on a hybrid rigid-pneumatic arm in a simulator that is the controller's own model, with a weak RL baseline and no pre-specified protocol. We study classical, RL and distilled controllers on one MuJoCo manipulator under disclosed budgets, a sanity gate, pre-specified rules and a locked-set log.

（約 100 字。）
