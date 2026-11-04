"""Round-2 additions to the TOST table (zero simulation; reads existing per-episode arrays only).
Rows: S7-PPO (spike13a, 3 seeds, HOLD), S3 (spike15b IL students, holdout), S5/S6 (spike15b PPO, holdout).
Same conventions as equivalence.py (paired percentile bootstrap, 10000 resamples) but its own RNG seed (1).
Run: .venv/bin/python docs/paper/stats/equivalence_r2.py > /dev/null ; output is printed as markdown table rows.
"""
import json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[3] / "results"
B = 10000
rng = np.random.default_rng(1)

def boot(d, lo, hi):
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    return d.mean(), np.percentile(m, lo), np.percentile(m, hi)

def verdict(lo, hi, m):
    if lo > -m and hi < m: return "EQUIV"
    if hi <= -m: return "L worse"
    if lo >= m: return "L better"
    return "no"

def row(id_, setting, cls, lrn, c, L, flag, rel=False):
    c = np.asarray(c, float); L = np.asarray(L, float)
    d = (c - L) if rel else (L - c)       # positive = learned better
    scale = 100.0 / c.mean() if rel else 100.0
    m, l90, h90 = boot(d, 5, 95); _, l95, h95 = boot(d, 2.5, 97.5)
    cs, ls = (f"{c.mean():.5f}", f"{L.mean():.5f}") if rel else (f"{100*c.mean():.1f}", f"{100*L.mean():.1f}")
    print(f"| {id_} | {setting} | {cls} | {lrn} | {len(c)} | {cs} | {ls} | {scale*m:+.1f} [{scale*l90:+.1f}, {scale*h90:+.1f}] | [{scale*l95:+.1f}, {scale*h95:+.1f}] | {verdict(scale*l90, scale*h90, 10)} | {verdict(scale*l90, scale*h90, 5)} | {flag} |")

# S7-PPO from spike13a (3 seeds, corrected exploration scale), HOLD 1100-1599, init = T-adapt3-full
init = np.load(R / "spike13a/init_holdout_b0_cap12.npy")
ppo = np.load(R / "spike13a/eval_conf_ppo_holdout.npy").mean(0)
row("S7-PPO (13a)", "whip realistic, HOLD 1100-1599 (reused set, not locked)", "T-adapt3-full", "R-PPO-hist, 3 seeds x 3M steps, log-std0 -1.5/-2", init, ppo, "NOT learned (sweep: none >= +5 pp on CAL); gate passed marginally")
# S3 from spike15b
h = json.load(open(R / "spike15b/s3/final_holdout.json"))
t = np.asarray(h["teacher"]["rmse"], float)
for name in ("bc", "mip", "diffusion (DDIM-10)", "flow (10 steps)", "flow (1 step)"):
    L = np.mean([h[f"{name}|{i}"]["rmse"] for i in range(3)], 0)
    row(f"S3-{name.split()[0]}", "tracking clean, holdout 1100-1599 (fixed reference)", "CTC+LQR teacher", f"{name} (3 seeds)", t, L, "yes (same data as teacher; no headroom)", rel=True)
# S5/S6 from spike15b
for tag, nm, ini, ev, base in [("S5", "juggling unknown wide error", "s5_init_holdout_b0.npy", "s5_eval_real_ls-2.0_holdout.npy", "T-robust(spike06)"),
                              ("S6", "whip idealised", "s6_init_holdout_b0.npy", "s6_eval_real_ls-1.5_holdout.npy", "T-adapt")]:
    c = np.load(R / "spike15b" / ini); L = np.load(R / "spike15b" / ev).mean(0)
    ls0 = "-2.0" if tag == "S5" else "-1.5"
    row(f"{tag}-PPO (15b)", f"{nm}, holdout 1100-1599", base, f"R-PPO-hist, 3 seeds, log-std0 {ls0}", c, L, "gate passed; DEGRADES" + (" (ceiling: +5 pp impossible)" if tag == "S6" else ""))
