# spike15d: B (learned L-dec4, c11/M3c/max) vs classical control

C-ens (CAL-selected) = **LAP/c11/M3c/hit06** (CAL 81.0 %); 22 variants incl. 2 controls.


## holdout 1100-1599 n=500

| method | success % [95% CI] | vs R0 pp | recovered (X-R0)/(R1-R0) [CI] |
|---|---|---|---|

**Verdict (pre-registered rule): UNCERTAIN.** holdout B - C-ens +7.0 (95 % [+2.6, +11.4], 90 % [+3.4, +10.6]); new +4.0 (95 % [-3.0, +11.0], 90 % [-2.0, +10.0]). Both 90 % CIs inside +-5 pp: False.

| R0 T-adapt3-full | 69.2 [65.2, 73.2] | - |  |
| R1 true theta | 86.0 [83.0, 89.0] | +16.8 [+12.0, +21.6] |  |
| L-ens (3 L-dec4 nets, base pick) | 71.0 [67.0, 75.0] | +1.8 [-3.8, +7.4] | 0.11 [-0.28, 0.38] |
| **B c11/M3c/max (learned, fixed)** | 86.6 [83.6, 89.4] | +17.4 [+12.6, +22.4] | 1.04 [0.82, 1.32] |
| **C-ens LAP/c11/M3c/hit06 (CAL-selected)** | 79.6 [76.0, 83.2] | +10.4 [+5.6, +15.0] | 0.62 [0.39, 0.86] |
| R0 + c11/centre/mean, family MS (no ensemble) | 70.0 [66.0, 74.0] | +0.8 [+0.0, +1.8] | 0.05 [0.00, 0.11] |
| R0 + c11/centre/mean, family LAP (no ensemble) | 73.6 [69.8, 77.4] | +4.4 [+1.4, +7.4] | 0.26 [0.09, 0.43] |

**B - C-ens = +7.0 pp, 95 % CI [+2.6, +11.4], 90 % CI [+3.4, +10.6]**; C-ens - R0 = +10.4 [+5.6, +15.0]; B - R0 = +17.4 [+12.6, +22.4]. Base candidate == R0 outcome fraction: {'MS': 1.0, 'LAP': 1.0}. C-ens changes the R0 pick in 80 % of episodes.

| variant | success % |
|---|---|
| MS/c8/M3c/mean | 69.8 |
| MS/c8/M3c/cmean | 69.8 |
| MS/c8/M3c/hit04 | 69.6 |
| MS/c8/M3c/hit06 | 69.8 |
| MS/c8/M3c/max | 69.2 |
| MS/c11/M3c/mean | 71.0 |
| MS/c11/M3c/cmean | 71.0 |
| MS/c11/M3c/hit04 | 70.2 |
| MS/c11/M3c/hit06 | 71.2 |
| MS/c11/M3c/max | 70.0 |
| LAP/c8/M3c/mean | 76.0 |
| LAP/c8/M3c/cmean | 76.4 |
| LAP/c8/M3c/hit04 | 74.0 |
| LAP/c8/M3c/hit06 | 75.8 |
| LAP/c8/M3c/max | 73.4 |
| LAP/c11/M3c/mean | 80.2 |
| LAP/c11/M3c/cmean | 80.4 |
| LAP/c11/M3c/hit04 | 79.6 |
| LAP/c11/M3c/hit06 | 79.6 |
| LAP/c11/M3c/max | 80.0 |
| MS/c11/center/mean | 70.0 |
| LAP/c11/center/mean | 73.6 |

LAP M9c (descriptive, 9 members) c11:

| aggregator | success % | B - this pp [95% CI] |
|---|---|---|
| mean | 82.2 | +4.4 [+0.0, +8.8] |
| cmean | 82.0 | +4.6 [+0.2, +9.0] |
| hit04 | 81.0 | +5.6 [+1.2, +10.0] |
| hit06 | 81.6 | +5.0 [+0.8, +9.2] |
| max | 79.8 | +6.8 [+2.4, +11.2] |

## new seeds 704000-704199 n=200

| method | success % [95% CI] | vs R0 pp | recovered (X-R0)/(R1-R0) [CI] |
|---|---|---|---|
| R0 T-adapt3-full | 63.0 [56.0, 69.5] | - |  |
| R1 true theta | 81.0 [75.5, 86.0] | +18.0 [+10.5, +25.5] |  |
| L-ens (3 L-dec4 nets, base pick) | 68.0 [61.5, 74.5] | +5.0 [-3.5, +13.5] | 0.28 [-0.23, 0.70] |
| **B c11/M3c/max (learned, fixed)** | 82.5 [77.0, 87.5] | +19.5 [+11.0, +28.0] | 1.08 [0.70, 1.67] |
| **C-ens LAP/c11/M3c/hit06 (CAL-selected)** | 78.5 [72.5, 84.0] | +15.5 [+8.0, +23.0] | 0.86 [0.51, 1.33] |
| R0 + c11/centre/mean, family MS (no ensemble) | 63.0 [56.0, 69.5] | +0.0 [-2.0, +2.0] | 0.00 [-0.12, 0.12] |
| R0 + c11/centre/mean, family LAP (no ensemble) | 69.0 [62.5, 75.5] | +6.0 [+1.5, +10.5] | 0.33 [0.10, 0.64] |

**B - C-ens = +4.0 pp, 95 % CI [-3.0, +11.0], 90 % CI [-2.0, +10.0]**; C-ens - R0 = +15.5 [+8.0, +23.0]; B - R0 = +19.5 [+11.0, +28.0]. Base candidate == R0 outcome fraction: {'MS': 1.0, 'LAP': 1.0}. C-ens changes the R0 pick in 81 % of episodes.

| variant | success % |
|---|---|
| MS/c8/M3c/mean | 65.0 |
| MS/c8/M3c/cmean | 65.0 |
| MS/c8/M3c/hit04 | 65.0 |
| MS/c8/M3c/hit06 | 65.0 |
| MS/c8/M3c/max | 65.0 |
| MS/c11/M3c/mean | 63.5 |
| MS/c11/M3c/cmean | 63.5 |
| MS/c11/M3c/hit04 | 63.0 |
| MS/c11/M3c/hit06 | 63.0 |
| MS/c11/M3c/max | 64.0 |
| LAP/c8/M3c/mean | 71.5 |
| LAP/c8/M3c/cmean | 70.5 |
| LAP/c8/M3c/hit04 | 68.5 |
| LAP/c8/M3c/hit06 | 69.0 |
| LAP/c8/M3c/max | 71.5 |
| LAP/c11/M3c/mean | 79.0 |
| LAP/c11/M3c/cmean | 79.0 |
| LAP/c11/M3c/hit04 | 76.0 |
| LAP/c11/M3c/hit06 | 78.5 |
| LAP/c11/M3c/max | 76.0 |
| MS/c11/center/mean | 63.0 |
| LAP/c11/center/mean | 69.0 |

LAP M9c (descriptive, 9 members) c11:

| aggregator | success % | B - this pp [95% CI] |
|---|---|---|
| mean | 83.0 | -0.5 [-7.0, +6.5] |
| cmean | 84.0 | -1.5 [-8.0, +5.5] |
| hit04 | 80.0 | +2.5 [-4.5, +10.0] |
| hit06 | 81.5 | +1.0 [-6.0, +8.0] |
| max | 80.5 | +2.0 [-5.0, +9.5] |

## CAL n=200 (selection set)

| method | success % [95% CI] | vs R0 pp | recovered (X-R0)/(R1-R0) [CI] |
|---|---|---|---|
| R0 T-adapt3-full | 72.0 [66.0, 78.0] | - |  |
| R1 true theta | 78.0 [72.0, 83.5] | +6.0 [-2.5, +14.5] |  |
| L-ens (3 L-dec4 nets, base pick) | 77.5 [71.5, 83.0] | +5.5 [-3.0, +13.5] | 0.92 [-2.00, 4.50] |
| **B c11/M3c/max (learned, fixed)** | 91.5 [87.5, 95.0] | +19.5 [+12.5, +26.5] | 3.25 [-13.00, 18.50] |
| **C-ens LAP/c11/M3c/hit06 (CAL-selected)** | 81.0 [75.5, 86.0] | +9.0 [+2.0, +16.0] | 1.50 [-4.50, 7.67] |
| R0 + c11/centre/mean, family MS (no ensemble) | 73.0 [67.0, 79.0] | +1.0 [-1.5, +4.0] | 0.17 [-1.00, 1.50] |
| R0 + c11/centre/mean, family LAP (no ensemble) | 76.5 [70.5, 82.0] | +4.5 [+0.0, +9.0] | 0.75 [-2.67, 4.50] |

**B - C-ens = +10.5 pp, 95 % CI [+4.5, +16.5], 90 % CI [+5.5, +15.5]**; C-ens - R0 = +9.0 [+2.0, +16.0]; B - R0 = +19.5 [+12.5, +26.5]. Base candidate == R0 outcome fraction: {'MS': 1.0, 'LAP': 1.0}. C-ens changes the R0 pick in 80 % of episodes.

| variant | success % |
|---|---|
| MS/c8/M3c/mean | 74.0 |
| MS/c8/M3c/cmean | 73.5 |
| MS/c8/M3c/hit04 | 72.5 |
| MS/c8/M3c/hit06 | 73.5 |
| MS/c8/M3c/max | 73.5 |
| MS/c11/M3c/mean | 74.0 |
| MS/c11/M3c/cmean | 74.0 |
| MS/c11/M3c/hit04 | 74.0 |
| MS/c11/M3c/hit06 | 74.0 |
| MS/c11/M3c/max | 74.5 |
| LAP/c8/M3c/mean | 72.5 |
| LAP/c8/M3c/cmean | 73.0 |
| LAP/c8/M3c/hit04 | 72.0 |
| LAP/c8/M3c/hit06 | 74.5 |
| LAP/c8/M3c/max | 71.0 |
| LAP/c11/M3c/mean | 79.0 |
| LAP/c11/M3c/cmean | 80.0 |
| LAP/c11/M3c/hit04 | 78.5 |
| LAP/c11/M3c/hit06 | 81.0 |
| LAP/c11/M3c/max | 79.0 |
| MS/c11/center/mean | 73.0 |
| LAP/c11/center/mean | 76.5 |