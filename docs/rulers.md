# 這個專案的尺

> 補件（2026-11-03）：本檔彙整既有結果檔裡已量到的東西，沒有新實驗。「未量」表示專案裡沒有人量過；「衍生」表示數字是我從結果檔的逐 seed 值算的，原檔沒有寫。出處一律用 `results/…` 或 `docs/…` 路徑。
> 尺在事後才整理成檔，不是閘門前事前宣告；事前宣告的部分是 docs/03_hypotheses.md 的各版本卡（commit 順序見 paper/support/prespec_audit.md）。

專案型態：research＋rl（見 docs/project_type.md）

## 切法與 seed 集合

分組鍵：episode seed（每個 seed 決定初始條件、植物誤差、感測噪聲序列）。有效 N = episode 數，且全部來自同一個植物家族、同一個模擬器；沒有更高層的組（沒有跨機器人、跨模擬器的組）。

| 集合 | seed | n | 用途 | 狀態 | 出處 |
|---|---|---|---|---|---|
| CAL | 1000-1049（官方）；whip／spike13a 起加 2000-2149 | 50；200 | 調參、挑 checkpoint、挑變體 | 反覆使用 | docs/03_hypotheses.md；results/spike13a/README.md:9 |
| HOLD（holdout） | 1100-1599（spike10b 只用 1100-1399） | 500（300） | 次要評估，也是 best-classical 的選擇集 | 反覆使用，非封存 | paper/support/claims_evidence.md 開頭 seed 說明 |
| TEST | 0-49 | 50 | 早期 spike（01-07、09b）的鎖定測試 | spike05/06/07 對同一個任務共用同一組 | results/.test_access.log；claims_evidence C-B5 |
| TEST-B | 20000-20499 | 500 | spike10b 主端點 | 封存，只用一次 | results/spike10b/table.md |
| TEST-C | 30000-30499 | 500 | spike15a（卡 v8／v8.1） | 封存，只用一次（log 2026-11-02T13:04:07） | results/spike15a/README.md:56 |
| TEST-D | 40000-40499 | 500 | spike15e（卡 v9） | 封存，只用一次（log 2026-11-02T21:54:34） | results/spike15e/README.md:11 |
| NEW／fresh | 700000+ 各區間，各 n=200 | 200 | 確認用，每區間只做評估一次 | 見下表 | claims_evidence.md 開頭 |
| DEMO | 5000-5019 | 5-20 | demo、pilot | 不用於結論 | claims_evidence C-G1, C-G2 |

fresh 區間：700000-700199（spike11a）、700200-700399（spike12a）、700400-700599（spike13a）、700600-700799（spike13b）、701000-701199（spike14a）、702000-702499 gate 與 702500-702699 fresh（spike15b）、703000-703199（spike15c）、704000-704199（spike15d）、705000-705199 與 gate 706000-706299（spike16a）。出處：claims_evidence.md 開頭 seed 說明、各 results/spikeNN/README.md。

存取記錄：results/.test_access.log 共 111 行（demo_students 47、mjx_eval_arena 56、spike05 1、spike06 1、spike07 1、spike09b 1、spike10b 2、spike15a 1、spike15e 1）。spike01-04 的 TEST 評估不在 log 內，所以 111 不等於鎖定集存取次數（claims_evidence C-A1、I-15）。

test 是否封存：TEST-B/C/D 是封存後只跑一次；TEST（0-49）沒有封存，被多個 spike 重用。

## 決策尺

- **決策尺 = 成功率差（pp），配對（同 seed）percentile bootstrap 95% CI**，在鎖定集上判定。追蹤任務用 RMSE 相對差（%）。出處：docs/03_hypotheses.md 各 H 的判定句；results/spike15e/table.md:7-14。
- 判定規則（事前寫在假設卡）：「贏」＝比 best-classical 高 ≥ 10 pp 且 CI 不含 0；追蹤任務為 RMSE 改善 ≥ 10% 且 CI 不含 0（docs/03_hypotheses.md:6-8）。
- **10 pp 門檻是專案自訂的**，不是由雜訊量出來的 2×sd。claims_evidence C-B10 弱點欄也寫「The 10 pp threshold was chosen by us」。與下面量到的雜訊比較：HOLD n=500 的單一模型重算差最大 3.6 pp，CAL n=200 的標準誤約 3.2 pp，所以 10 pp 大於這些已量到的雜訊（約 2.8 倍於 3.6），但沒有以 2×sd 推導（見「門檻」）。
- 等效（TOST）：邊界 ±10 pp（追蹤任務 ±10% RMSE），另列 ±5 pp；以 90% 配對 bootstrap CI 完全落在邊界內為「等效」。這是事後算的（沒有新模擬），跨 spike 未校正。出處：paper/support/stats/equivalence.md:26。鎖定集上事先寫在卡裡的 TOST 只有 H14（v8，±10）。
- 多重比較：每張卡自己的判定用 Holm（v8：H13、H13b、H14；v9：H15、H15b、H16a、H16b，單邊 bootstrap p）。跨 spike 沒有校正。出處：docs/03_hypotheses.md:194, 220。
- 已知的卡內不一致：v9 H16b 的規則用雙邊 95% CI 上界，Holm 步驟用對應 90% CI 的單邊 p；以規則文字為準，H16b 不成立（claims_evidence I-17）。

## 尺 1：鎖定集成功率差（TEST-B／C／D）

- **怎麼算的**：每個方法在同一組 seed 上跑一次，成功率差配對 bootstrap；指令見 docs/experiments.tsv 的 spike10b、spike15a、spike15e。
- **量的母體**：whip（甩鞭）任務，單一植物家族、單一模擬器（MuJoCo 3.17）、n=500 episode。
- **雜訊**：
  - seed sd：訓練 seed 的 sd 多為 3 個 seed，見「已量到的雜訊」表。
  - 跨機偏移：machine-A 與 machine-B 機器各 30 個 episode 結果相同（見下表）。
  - 解析度 = 1/500 = 0.2 pp。
  - 有效 N = 500 個 episode seed，組間 sd 未量（沒有更高層的組）。
- **門檻**：2×sd 未算出；以自訂 10 pp 代用，方向：越高越好（學習式減古典）。
- **最笨 baseline**：零殘差（init）與 T-nom；例如 HOLD 上 R0（T-adapt3-full）69.2%、TEST-C 65.6%、TEST-D 66.4%（results/spike15a/table.md、results/spike15e/table.md:20）。
- **已知盲區**：只有一個植物家族；n=500 的 CI 約 ±4 pp；學習式列是 3 個模型的平均，CI 較窄（claims_evidence I-11）。

## 尺 2：HOLD／NEW 成功率差（n=500／200）

- **量的母體**：同尺 1，但 HOLD 被反覆使用，且是 best-classical 的選擇集，所以偏樂觀（69.7% HOLD vs 65.6% TEST-B，results/spike10b/table.md:123）。
- **用途**：選擇與確認，不是最終判定。claims_evidence 第 2 節列出只靠 HOLD/NEW 的主張。
- **雜訊**：同下表；n=200 的 CI 半寬約 ±6.5 pp（claims_evidence C-C6）。
- **門檻**：未算（同尺 1）。

## 尺 3：CAL 成功率（n=50／100／200，只用於選擇）

- **雜訊**：n=50 標準誤約 7 pp（results/spike13a/README.md:9）；results/spike15b/README.md:12 另寫 n=50 的 SE 約 6 pp（與前者的 7 pp 不一致，皆為估計）；n=200 標準誤約 3.2 pp（results/spike13a/README.md:83）。
- **盲區**：在 CAL 上挑出的 checkpoint 與變體會有贏者詛咒；例：S6-X1 在 CAL 上看起來 +5.0、+11.5、+5.0 pp，換不重疊的 seed 就消失（results/spike15b/README.md:42；claims_evidence C-F5）。

## 尺 4：追蹤 RMSE（rad）

- **怎麼算的**：50 條 TEST 軌跡（seed 0-49），含初始暫態，配對差 95% CI（早期 spike 為 t 區間）；holdout 版 n=500（spike15b S3）。
- **雜訊**：RL 5 個 seed 的平均；個別 seed sd 未單獨彙整。
- **盲區**：標稱植物上老師是精確的，所以學生沒有改進空間（claims_evidence C-B3b）。

## 已量到的雜訊

| 項目 | 數字 | 出處 | 備註 |
|---|---|---|---|
| 同模型浮點重算的 holdout 成功率變動 | 最多 3.6 pp／模型（spike13b 65.6/65.6/62.8% vs spike14a 重跑 67.0/69.2/63.4%；3 模型平均 64.7→66.5%） | results/spike14a/table.md:19-20, 32-37；claims_evidence C-D3 | 原因（float32 vs float64 特徵）是作者自述，未獨立驗證 |
| CAL 標準誤，n=200 | 約 3.2 pp | results/spike13a/README.md:83 | 由 p≈0.7、n=200 的二項式估計 |
| CAL 標準誤，n=50 | 約 7 pp（另一處寫 6 pp） | results/spike13a/README.md:9；results/spike15b/README.md:12 | 兩處依成功率不同，數字有出入，皆為估計 |
| RL gate 的 seed 間差 | spike15c 方法 A：seed 0 +7.2 [+2.0, +12.4]，seed 1 +1.2 [-4.0, +6.4]，門檻 +10 | results/spike15c/README.md:36；claims_evidence C-F6 | 同配方兩個 seed，差 6.0 pp，兩個都低於門檻 |
| RL 訓練 seed 間（S6 real，holdout） | 每 seed −15.0、−16.8、−21.0 pp；sd 約 3.1 pp（衍生） | results/spike15b/table.md（holdout 逐 seed 列）；claims_evidence C-F3 | 3 個 seed；sd 為我算的 |
| RL 訓練 seed 間（S5 real，holdout） | 每 seed −29.2、−14.4、−27.8 pp；sd 約 8.2 pp（衍生） | results/spike15b/table.md:49-51 | 3 個 seed；sd 為我算的 |
| RL 訓練 seed 間（spike13a 確認，HOLD／NEW） | HOLD −3.0、−0.2、−3.0（sd 約 1.6，衍生）；NEW +4.0、−4.5、−3.5（sd 約 4.6，衍生） | results/spike13a/table.md:51-53；claims_evidence C-F1 | 3 個 seed |
| 同一組凍結 L-ens 在兩個鎖定集 | TEST-C +8.0 [+2.6, +13.6]，TEST-D +4.4 [−1.4, +10.0] | results/spike15a/table.md:48-57；results/spike15e/table.md:22 | 不同 seed 集，差 3.6 pp，不是配對 |
| 同一 L-ens 在不同 spike 的 holdout | 74.0%（spike14a）vs 71.0%（spike15c/15d/15e，CPU 重新推論） | claims_evidence I-20；results/spike15c/README.md:42 | theta 最多差 0.005 |
| 跨機偏移（machine-A vs machine-B） | catch、toss、Push-T 各 10 episode 結果、長度、qpos 逐項相同（共 30/30）；numpy 2.4.6 vs 2.2.6 | results/spike16a/cmp_summary.txt；claims_evidence C-H16 | Push-T max_cov 摘要差 7.8e-16；只涵蓋這三個古典＋學習配對，whip 沒比 |
| 解析度 | n=50：2 pp；n=200：0.5 pp；n=500：0.2 pp | 1/N 計算 | |
| 有效 N（組數） | = episode 數；組間 sd 未量 | | 沒有更高層的組 |
| 訓練 seed 數 | 多數 3，spike03a／04／02 為 5（PPO），早期 BC 為 10 | 各 results/spikeNN/README.md | 低於 ml-rulers 預設的 5 |
| 評估 episode 數 | 50（TEST 早期）、200、500 | 同上 | |

沒量過（寫「未量」）：
- 追蹤任務（RMSE）以外所有尺的「訓練 seed sd」：只有上表列出的 3-seed 逐值，沒有 ≥5 seed 的 sd，**未量**。
- 跨機偏移在 whip 任務：**未量**（只比了 catch、toss、Push-T）。
- 同一方法在不同鎖定集（TEST-B/C/D）的 seed 集間 sd：只有 R0 的 65.6／65.6／66.4（不同方法版本，非嚴格重複量測），不是專門量的雜訊，**未量**。
- 評估 episode 的 bootstrap 重抽樣次數與重抽樣隨機性對 CI 邊界的影響（例如 H16b 的 10.6 對 10）：**未量**。
- 逐 PPO 配置的跨 seed sd（spike13a sweep 每配置只有 1 個訓練 seed）：**未量**。

## 門檻

2×sd 的正式門檻：**未算**，因為沒有 ≥5 個 seed 的 seed sd（上表只有 3-seed 逐值）。專案用的是自訂的 10 pp（「贏」＝ ≥ +10 pp 且 CI 不含 0），事前寫在假設卡。唯一的雜訊對照：RL 訓練 seed 間 sd 約 1.6-8.2 pp（衍生，3 seed），2×sd 約 3-16 pp，所以 10 pp 落在這個範圍內，不能說它比 2×sd 保守。

## RL 專用

- 環境版本／commit：MuJoCo 3.17.0；arena/ 的 commit 隨各 spike（見 docs/experiments.tsv 的 commit 欄）。換過版本的地方：reference-angle wrap fix（2026-10-30，影響 learn_demo，results/learn_demo 已過期）。
- seed 下限：本專案實際 3（PPO 訓練 seed）；ml-rulers 預設 5、最少 3。spike13a 掃描每配置 1 seed，spike15b gate 每環境 1 seed，spike15c gate 2 seed，都已在 claims_evidence 第 7 節標 **[單 seed／最佳挑選]**。
- sanity gate：注入已知可學的常數力矩偏差，檢查 PPO 管線能學到（門 +10 pp；spike13a 通過 +10.2，餘裕 0.2 pp，8 次裡挑最好；spike15b 兩環境通過；spike15c 失敗）。出處：claims_evidence C-A3、C-F3、C-F4、C-F6。
- eval episode 數：500（HOLD、鎖定集）或 200（NEW、fresh）；訓練與評估 seed 分開（訓練 ≥ 500000）。
- 報告方式：mean＋配對 bootstrap 95% CI；不用 IQM。逐 seed 值有列在 table.md。

## 決策規則

- 候選要：所有已宣告的門皆不退、至少一門過（早期 spike 為單一門）。
- 差距小於門檻不補 seed；本專案以「CI 含 0」或「TOST 不在邊界內」描述，措辭用「本尺無法分辨」，不寫「無效」。
- 最終 seed：鎖定集一次性使用，不重跑（spike15a、spike15e 的 README 都寫了 do NOT rerun）。
- **決策尺**：上面「決策尺」一節（配對 bootstrap，鎖定集上的 pp 差）。事前宣告日期：各卡的 commit 日期（v8 d167ed6 2026-11-02 10:58；v9 800de6e 2026-11-02 20:32；其餘見 paper/support/prespec_audit.md）。
