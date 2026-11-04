# 假設卡（2026-10-24 定，跑 spike02 前寫死）

## 論文問題（作者同意的新題目）
學生需要「什麼額外的東西」才能超越古典老師？老師的哪些弱點會讓學生有機會贏？

## 指標（事前寫死）
- 主指標：追蹤 RMSE（rad），整條軌跡，含初始暫態。
- 次指標：穩態 RMSE（去掉前 0.5 s）、力矩平方積分、力矩飽和時間比例。
- 「贏」的定義：學生和老師在同一組 50 條測試軌跡上做配對比較，差值的 95% CI 完全在 0 以下，而且平均改善至少 10%。

## 假設
H1（對照，spike01 已支持）：純 BC 或 BC+DART 在任何條件下都不會贏老師。
H2：老師有「沒建模的弱點」（力矩飽和、致動器延遲 10 ms、沒建模的 Coulomb 摩擦）時，在老師上面做 residual RL，至少在一種弱點下會贏老師 ≥10%。
  否證：三種弱點下 residual RL 都沒達到「贏」的定義。
H3：老師沒有弱點（spike01 的條件）時，residual RL 的改善 < 5%。（預測學生的增益來自老師的弱點，而不是 RL 本身。）
  否證：沒有弱點時 residual RL 也贏 ≥10%。
H4：DAgger 不會讓學生贏（DAgger 只能更貼近老師）。

## 公平協定
- 老師的增益只調一次（在標稱模型上），之後固定。學生的超參數調參次數上限寫在結果 README。
- RL 的環境互動步數寫明。BC 類方法的資料量用相同軌跡數。
- 所有方法共用同一組測試軌跡、擾動和種子。

## 備用題目（H2 被否證時）
「蒸餾保留了多少古典老師的穩健性？要付出多少代價？」把對照組和負結果做成系統性刻畫。

---
# v2 修訂（2026-10-25，spike02 之後、白方第 2 輪之後；在看到 spike03 結果前寫死）

## 修訂紀錄
- H2 的判定改成相對於「W0 下 RL 的表現」：額外貢獻 = (Teacher_Wk − Teacher_W0) − (RL_Wk − RL_W0)。spike02 的 H2「支持」降級為「待重判」。
- spike02 的 W1 力矩上限（27→32 Nm）和 W2 延遲（10→30 ms）是看過測試集後決定的，屬於事後調整。正式實驗改成只用校準集（seed 1000–1049）決定，測試集（seed 0–49）鎖定。
- 指標（事後增加，明確標註）：主指標維持含暫態 RMSE，但每張表都要並列穩態 RMSE（去掉前 0.5 s）、整定時間、脈衝恢復時間、飽和比例、力矩能量。
- 學生觀測要對齊：RL 多出來的誤差特徵，BC 類學生也要拿到。

## 預算公平（三軸一起報）
模擬步數、牆鐘時間、可調自由度。古典調參只用校準集，目標函數和主指標一致。

## 古典對手分層
C0：原老師（不調）。C1：Kp/Kd 或 Q/R 調過。C2：C1 加積分項或擾動觀測器，輸出夾在和 RL 殘差相同的 ±16 Nm 內。

## H5
在 W0 與 W3 下，若 C2 的含暫態 RMSE 與 residual RL 的差距 ≤ 10%（或 C2 更好），就判定「RL 在簡單追蹤任務的優勢大部分來自古典沒調好」。若 RL 仍贏 > 10%，且配對 CI 不含 0，則判定 RL 有古典調參拿不到的增益。

## H6（新任務：顛球）
顛球任務中，mirror-law 老師在球的恢復係數或球拍質量有誤差時，連續顛球數會明顯下降；在老師上做 residual RL 的學生，在至少一種誤差下中位數連續顛球數比老師多 ≥ 30%。
否證：任何誤差下都沒達到。若連顛球都沒有學生勝出 → 停下來回報作者，改走備用題目。

---
# v3 新增（2026-10-25，spike03b 之後；在 spike05 結果出來前寫死）
背景：spike02/03a/03b 都出現同一個模式——只要針對已知的誤差條件重調古典控制器的一兩個參數，就能追平甚至贏過 residual RL。但這種重調等於「事先知道是哪一種誤差」。
## H8（未知且變動的誤差）
每個 episode 的誤差從一個分佈抽樣，控制器不知道是哪一種（例如顛球：恢復係數 U[0.4,0.85]、球質量 ×U[1,5]、感測延遲 U[0,150] ms）。比較：
- T-nom：標稱老師
- T-robust：用 CMA-ES 在校準集的同一個分佈上調一組固定參數（古典的穩健調參）
- T-adapt：古典自適應（線上估計恢復係數或延遲，再代入 mirror law），一定要給古典這個機會
- R-DR：residual RL＋domain randomization，沒有記憶
- R-DR-hist：residual RL＋domain randomization＋觀測歷史（可以隱式辨識系統）
預測：R-DR-hist 的中位數顛球數比 T-robust 和 T-adapt 中較好的那一個高 ≥30%，配對 bootstrap CI 不含 0。
否證：達不到。達不到 → 結論改成「在這些任務上，古典的穩健或自適應調參就足以追平學習」，這也是可以投稿的結果。
次要：用更高的顛球上限（N=300），避免截尾掩蓋差異。

---
# v4 新增（2026-10-30，spike05 診斷之後；在 spike06 結果出來前寫死）
## spike05 的修正
- 診斷結果：R-DR-hist 輸給 T-nom，原因是 PPO 的探索噪聲太大（σ≈1.6 Nm，光這個噪聲就讓老師的平均顛球數從 51 掉到 5），加上學習率太高（3e-4），不是程式 bug。spike05 中「RL 輸了」這句撤回，改成「該 RL 設定不足」。
- 「古典在 H8 分佈下飽和於 300」仍然成立。但正因為飽和，H8 不可能被推翻，所以 H8 不具鑑別力。
## H8b（更難、不會飽和的未知誤差分佈）
- 誤差分佈只用校準集加寬（恢復係數下限、延遲上限、質量上限），規則是讓 T-robust 和 T-adapt 中較好的那一個，在 CAL 上的成功率（達到上限 300 的比例）落在 20–60%。
- 古典在新分佈上重新調參：T-robust、T-adapt 都用 CMA-ES，只用校準集，預算記錄下來。
- RL 用修正後的配方：log-std 初值 ≤ −3.5 並設下限，lr ≤ 1e-4，epoch 4。超參數掃描只用 CAL，總模擬步數不超過古典調參步數的 2 倍。殘差疊在 T-robust 和 T-adapt 中較強的那一個上（R-on-best），有歷史和沒歷史各一組。
- 主指標：達到上限的比例（成功率）。顛球數分佈是雙峰的，中位數不穩定，所以改用平均數和成功率，另外並列中位數。
- 預測：R-on-best（有歷史）的成功率比最好的古典高 ≥10 個百分點，配對 bootstrap CI 不含 0。
- 否證：達不到。達不到 → 結論是「同等調參預算下，古典穩健或自適應控制在顛球上不輸 residual RL」。

---
# v5 新增（2026-10-30，spike06 與白方審查之後；在 spike07 任何結果出來前寫死）
## 框架（作者決定）
混合方案：「benchmark 加公平協定」論文。arena 與公平協定是貢獻，主張限定在「這套協定下」。不論學習贏或輸都要寫得成，所以負結果需要 oracle 上限和強 RL 撐住。

## 至今的偏離表（照實列出，論文附錄要放）
| 實驗 | 偏離 | 是否看過 TEST |
|---|---|---|
| spike02 | W1、W2 強度在看過 TEST 後決定 | 是 |
| learn_demo | 超參數在看過一次 TEST 後才在 CAL 上選 | 是（要重跑） |
| spike05 | RL 配方有缺陷（探索噪聲 σ≈1.6 Nm），結論撤回 | 否 |
| spike06 | 第二輪才加「用 CAL 選 checkpoint」；重調後古典的 CAL 成功率 76%，超出 20–60% 區間；T-robust kp 撞到上限 2500 | 否 |
| spike05/06 | 腳本繞過 TEST 記錄，事後補記 | — |

## spike07（顛球，沿用 spike06 的分佈 C，results/spike06/dist.json；TEST seeds 0–49 只跑一次）
- **古典**：T-robust 放寬 kp 上限（2500 → 10000）在 CAL 上重調，預算與 spike06 相同（81 次評估）。若新的 T-robust 在 CAL 上比舊的好，best-classical 就改用新的。
- **RL 先修穩定性（只用 CAL 和訓練 seed）**：診斷 spike06 的後期崩潰，候選修正包括：cap 處 bootstrap value（不當作 terminal）、更大 batch、KL 或 target-KL early stop、BC 正則（往老師動作拉）。通過標準：在 CAL 上，最終 iterate 不輸初始化，訓練曲線不會掉到 <50%。修好之前不跑 TEST。
- **方法**（每個 3 個訓練 seed）：
  - R-PPO-fixed：修好的 PPO 殘差，加在 best-classical 上，有歷史觀測
  - R-SAC：SAC 殘差，加在 best-classical 上，有歷史觀測
  - R-oracle：同 R-PPO-fixed，但觀測加上真實的 e、質量、延遲（量上限，不算公平方法）
- **選模型**：所有 RL 一律在 CAL 上選 checkpoint（固定在 7 個等距 checkpoint 裡選），不再取最後一次迭代。
- **預算**：每個 RL 方法的模擬步數（含 CAL 評估）不超過古典調參步數的 2 倍。牆鐘時間和可調自由度一起報。

## H9（headroom）
R-oracle 的 TEST 成功率比 best-classical 高 ≥10 pp，而且配對 bootstrap CI 不含 0。
- 成立 → 這個分佈上學習有空間；接著看公平的 RL 方法拿回了多少。
- 不成立 → 在這個分佈上，就算知道真實參數，殘差學習也幾乎沒有增益。這是強負結果的證據。

## H10（公平 RL）
R-PPO-fixed 或 R-SAC 中，至少一個的 TEST 成功率比 best-classical 高 ≥10 pp，而且 CI 不含 0。兩個比較都做 Holm 校正。
- 否證：兩個都達不到。
- 附帶要報：殘差範數（證明 RL 真的有學東西），以及 CAL 的訓練曲線。

## 之後（不在 spike07 裡，已排入清單）
甩鞭任務（老師天生弱）也跑同樣的 H9/H10；spike03b 用修正後的配方重跑；learn_demo 重做。

---
# v6 新增（2026-10-30，spike09a 之後；在 spike09b 任何 RL 或 TEST 結果出來前寫死）
## 背景
顛球：在 spike06 的分佈上，古典穩健調參接近可達上限（spike08_diag：格內重調沒有增益），所以殘差 RL 沒有可學的空間。作者決定把 H9/H10 移到甩鞭。spike09a 顯示甩鞭有 headroom：知道鏈長的格內最佳在短鏈格比 T-robust 高 33 pp（holdout n=125）。整體 T-robust 在 holdout 上沒有贏過 T-nom。

## spike09b 設計（甩鞭，分佈 = results/spike09a/dist.json，w=0.2）
- **古典**
  - T-nom
  - T-robust：沿用 spike09a 的參數
  - T-adapt：古典系統辨識。開頭用一段短的探測動作，以標稱模型模擬一組鏈長、質量、阻尼候選，用最小平方對上關節力矩或狀態，挑出最接近的參數交給規劃器。探測時間算在 episode 內。這是給古典的公平機會，一定要做。
  - **best-classical**：在 T-nom、T-robust、T-adapt 中，用 holdout seeds 1100–1599 選（這些不是 TEST）。
- **參考上限（不算公平方法）**
  - T-oracle：規劃器直接拿到真實的鏈長、質量、阻尼
  - R-oracle：殘差 PPO 加在 best-classical 上，觀測含真實參數
- **公平的學習方法**（各 3 個訓練 seed）
  - R-PPO-hist：殘差 PPO 加在 best-classical 上，觀測含歷史
  - L-est：學出來的估計器。從和 T-adapt 相同的探測段歷史，回歸出鏈長、質量、阻尼，再交給古典規劃器。訓練資料只用訓練 seeds（≥500000）。這是「學習加古典」的混合方法，對照 T-adapt 的差別只在估計器是學出來的還是模型擬合的。
- **選模型**：RL 一律在 CAL 上從 7 個等距 checkpoint 中選。
- **「有學到東西」的檢查**：CAL 上選中的 checkpoint 要比初始化高 ≥5 pp，否則表中標「未學到」。殘差範數一定要報。
- **預算**：以 episode 數和牆鐘為主軸。每個學習方法的訓練 episode 數（含掃描和 CAL 評估）不超過古典調參 episode 數（T-robust 9050 加 T-adapt 的調參）的 2 倍；若 PPO 需要更多，要明寫成偏離，並另報「2 倍預算時」的 checkpoint。自由度也要報。
- **TEST**：seeds 0–49 只跑一次，所有方法一起跑，寫進 results/.test_access.log。

## H9w（headroom，甩鞭）
R-oracle 或 T-oracle 的 TEST 成功率比 best-classical 高 ≥10 pp，而且配對 bootstrap CI 不含 0。這是後面公平比較有意義的前提。

## H10w（公平學習，甩鞭）
R-PPO-hist 或 L-est 中，至少一個的 TEST 成功率比 best-classical 高 ≥10 pp，而且 CI 不含 0（兩個比較做 Holm 校正）。
- 次要比較：L-est 對 T-adapt，也就是學出來的估計器對模型擬合，同樣做配對 CI。
- 否證：兩個都達不到。

---
# v7 新增（2026-10-31，spike10a 之後；在 spike10b 任何學習方法或 TEST 結果出來前寫死）
## 背景
在理想模擬下（spike09b），古典系統辨識（T-adapt）幾乎拿走了甩鞭的所有 headroom（TEST 98%）。作者決定加入真實條件再比一次。spike10a 的條件為：感測雜訊、逐節質量不均、鏈關節摩擦、致動器噪聲，全開時 T-adapt 掉到 56%，完整 oracle 是 91.8%（holdout）。

## spike10b 設計（甩鞭，分佈 = results/spike09a/dist.json 加上 results/spike10a/dist.json 的全開條件）
- **前置檢查**：先確認 MuJoCo NaN／unstable 警告的來源。若會影響任何方法的指標，先修再往下做。
- **古典**（都在 CAL 上設計或調參，預算要記錄）
  - T-nom、T-robust、T-adapt、T-adapt2：沿用 spike10a
  - T-adapt3：擴充模型類別的古典系統辨識。擬合參數加入鏈關節摩擦和逐節質量（或低維參數化，例如線性質量梯度），用同樣的探測段。調參預算約 1500 episode。這是給古典的最強機會，一定要做。
  - **best-classical**：在 holdout 1100–1599（n≥300）上，從上面所有古典方法中選出。
- **參考上限**：T-oracle（完整），不算公平方法。
- **公平的學習方法**（各 3 個訓練 seed，訓練資料只用 seeds ≥500000）
  - L-est-full：學出來的估計器。從和 T-adapt 系列相同的探測段（含雜訊）回歸出完整 plant 參數（鏈長、逐節質量或其低維參數化、阻尼、摩擦），再交給和 T-oracle 相同的規劃器。這是主要的學習方法，因為 headroom 的本質是「辨識」。
  - R-PPO-hist：殘差 PPO 加在 best-classical 上。依前三次的經驗，可能學不起來。
- **選模型**：所有學習方法的超參數和 checkpoint 只在 CAL 上選。
- **「有學到東西」的檢查**：PPO 在 CAL 上選中的 checkpoint 要比初始化高 ≥5 pp，否則表中標「未學到」。
- **預算**：學習方法的訓練加評估 episode 數，不超過古典調參 episode 數（T-robust、T-adapt2、T-adapt3 合計）的 2 倍。探測段 episode 依時間長度折算。
- **TEST**：seeds 0–49 只跑一次，所有方法一起跑，寫進 results/.test_access.log。另外在 holdout（n=500）上也要報同一張表，因為 TEST 的 n=50 統計力不足。

## H11（學出來的辨識 vs 古典辨識）
L-est-full 的 TEST 成功率比 best-classical 高 ≥10 pp，而且配對 bootstrap CI 不含 0。
- 次要：在 holdout n=500 上做同樣的比較，並報 L-est-full 對 T-adapt3 的差。
- 否證：達不到。達不到時，結論是「在這組真實條件下，擴充模型類別的古典辨識不輸學出來的辨識」。

## H12（殘差 RL）
R-PPO-hist 的 TEST 成功率比 best-classical 高 ≥10 pp，而且 CI 不含 0。H11、H12 一起做 Holm 校正。

## v7.1 修訂（2026-10-31，白方審查之後、spike10b 開跑前寫死）
白方提出 6 條必改項。逐條判斷如下：
1. **劑量曲線（採納）**：真實條件強度分 ×0.5、×1、×2 三級（三個因子同比例縮放），另外單獨開 a、b、c 各一次。這些都在 holdout 上報，每級 n≥200。白方說找不到單因子結果，這點部分不成立：results/spike10a/cal_base.json 有單因子的 CAL 結果（n=50），但沒有 holdout 的版本，所以仍然要補。解讀規則先寫死：若學習只在 ×1 勝出，只能寫成「單點結果」，不得寫成一般性結論。
2. **先驗揭露與分佈外對照（採納）**：論文表格要明列 L-est-full 的訓練分佈和評估分佈相同，而且用了模擬標籤。另外加兩個版本：
   - L-est-OOD：用 ×0.5 和 ×2 的條件訓練，在 ×1 上評估。
   - L-est-eff：只回歸有效參數（鏈長、總質量、阻尼、摩擦），不使用逐節質量標籤。
   古典也做對稱的 T-adapt3-OOD：在 ×0.5 或 ×2 上調，在 ×1 上評估。
3. **T-adapt3 的可辨識性（採納）**：在 CAL 上先報參數估計誤差。同時做「完整逐節質量」和「低維梯度」兩種參數化，兩種都列為 best-classical 候選。調參預算提高到約 3000 episode。放棄準則：兩種參數化都在 CAL 上不比 T-adapt2 好時，照實報告，不再追加。
4. **主端點改用新的、從未用過的種子（採納）**：TEST-B = seeds 20000–20499（n=500），只跑一次，寫進 results/.test_access.log。原本的 TEST 0–49 也跑一次，並列報告。H11、H12 的判定以 TEST-B 為準。holdout 1100–1599 已被重複使用很多次，只用來選模型。
5. **預算三軸並列（採納）**：episode 數、牆鐘、標籤來源。L-est 系列要用模擬器的全參數標籤，古典不用，這點要明文寫出來。
6. **NaN 通過標準（採納）**：每個方法發散（NaN 或 unstable）的 episode 比例要 <0.5%，並且逐方法報告。超過就先修再往下做。

## v8（2026-11-02，閉門審查第 1 輪之後、spike15a 開跑前寫死）

目的：讓 K3（headroom 階梯與估計器對決）有鎖定 seed 的證據。

- **鎖定集 TEST-C：seed 30000–30499（n=500）**，之前從未使用。只能跑一次，每次存取都要寫進 results/.test_access.log。
- 條件：甩鞭真實條件（spike10a/10b dist，SUBF=4），和 spike13b、spike14a 相同。
- 方法全部凍結，不重新調參或訓練：R0 = T-adapt3-full；R1 = R0 流程換真實參數；R2 = R1 再換真實鏈狀態；R4 = T-oracle；R5 = 規範化 T-oracle；L-est-full（spike10b 的 3 個模型）；L-dec4（spike14a 的 3 個 seed，逐 seed 和集成都報）；C-MAP（spike14a 的 MAP＋多起點）。
- **H13（估計主導 headroom）**：R1 − R0 ≥ +10 pp，而且配對 bootstrap 95% CI 下界 > 0。
- **H13b（探測成本小）**：R4 − R2 的 95% CI 上界 < +5 pp。
- **H14（估計器對決等效）**：L-dec4（3 seed 合併）− R0 的 90% CI 落在 ±10 pp 內（TOST）。另外報告 L-ens 和 C-MAP 對 R0 的差，只描述，不判定。
- 回收比例 (X − R0)/(R1 − R0) 報配對 bootstrap CI。
- 多重比較：H13、H13b、H14 三個判定用 Holm 校正。
- 階梯的歸因順序：另外報告反向順序（先換鏈狀態，再換參數），只描述。

### v8.1 修訂（2026-11-02，TEST-C 開跑前寫死）
- spike14a 沒有存 L-dec4 的權重，所以 v8「不重新訓練」做不到。處理方式：用完全相同的配方和 seed 0/1/2 重訓，並存下權重（experiments/spike15a/retrain_dec4.py）。
- 接受條件（只用 holdout 1100–1599 比對重訓模型和 spike14a 儲存的預測，不看 TEST-C）：3 seed 合併的 holdout 成功率差在 ±2 pp 內，而且逐 episode 成功與否一致率 ≥90%。達到就視為同一個凍結模型。
- 沒達到時，L-dec4 仍然用重訓版本跑 TEST-C，但 H14 標示為「重訓模型」，並在論文寫明。不可以為了達標反覆重訓。
- Holm 校正用單邊 bootstrap p 值，主要門檻 α=0.05（family-wise）。

## v9（2026-11-02，spike15d 之後、TEST-D 開跑前寫死；作者選「開新鎖定集確認」）

目的：讓「穩健選計畫」這個新發現，以及學習式估計器在同一規則下的優勢，都有鎖定 seed 的證據。

- **鎖定集 TEST-D：seed 40000–40499（n=500）**，之前從未使用。只跑一次，存取寫進 results/.test_access.log（預期從 110 行變 111 行）。
- 條件：甩鞭真實條件，和 spike15a/15c/15d 相同（SUBF=4）。
- 方法全部凍結，不重新選、不重訓：
  - R0 = T-adapt3-full；R1 = R0 換真參數；
  - L-ens = 3 個凍結 L-dec4（results/spike15c/models/model_r15dec4_s{0,1,2}.pt）平均 θ；
  - **B** = c11/M3c/max（spike15c 第 0 節的 CAL 選擇）；
  - **C-ens** = LAP/c11/M3c/hit06（spike15d 的 CAL 選擇），Laplace 抽樣的亂數種子規則照 spike15d 程式；
  - 描述性：B9（c11/M9/mean，spike15c 事後擴充）、R0+c11/center（LAP 家族）。
- **H15（穩健選擇對古典有效）**：C-ens − R0 的 95% CI 下界 > 0。
- **H15b（穩健選擇＋學習式估計器吃掉 headroom）**：B − R0 ≥ +10 pp，而且 95% CI 下界 > 0。
- **H16a（學習式優勢）**：B − C-ens 的 95% CI 下界 > 0。
- **H16b（學習式優勢小於 10 pp）**：B − C-ens 的 95% CI 上界 < +10 pp。
  - 解讀先寫死：H16a 和 H16b 都成立 → 「學習式估計器有小而真實的優勢，未達本文 10 pp 門檻」；只有 H16b 成立 → 「未偵測到學習式優勢」；H16b 不成立 → 「學習式優勢可能 ≥10 pp」，論文照實改寫 K2。
- 多重比較：H15、H15b、H16a、H16b 四個判定用 Holm 校正（單邊 bootstrap p 值，α=0.05 family-wise）。
- 另報：回收比例 (X−R0)/(R1−R0) 附 CI；B 與 C-ens 每 episode 的 CPU 秒數。
- 開跑前檢查（只能用 holdout，不能看 TEST-D）：凍結的 B 與 C-ens 在 holdout 1100–1599 上重跑，成功率要和 spike15c/15d 的報告值差在 ±1 pp 內。不符就停下來回報，不跑 TEST-D。
