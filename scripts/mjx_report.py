"""Aggregate MJX-PPO arena evaluations (results/mjx/eval_<tag>_<W>.pkl) and compare with spike02 RL and spike03a C0/C1.
  python scripts/mjx_report.py tag1 tag2 ... -> results/mjx/ppo_table.md"""
import os, pickle, sys, glob
import numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tags = sys.argv[1:]
sp3 = pickle.load(open(f"{ROOT}/results/spike03a/eval.pkl", "rb"))

def ci(x):
    x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), h

def paired(a, b):   # b better if mean(a-b) > 0; a,b (50,)
    d = np.asarray(a) - np.asarray(b); m, h = ci(d); return m, h, m / np.mean(a)

L = ["# MJX-trained residual PPO, evaluated on the CPU arena (TEST_SEEDS 0-49, split='test')", "",
     "RMSE in rad, transient included, +- = 95% t-CI over the 50 trajectories (policy value = per-trajectory mean over its PPO seeds). "
     "Paired rows: improvement = mean(A-B)/mean(A), positive = B better; CI over the 50 paired trajectories; WIN = CI of (A-B) > 0 and improvement >= 10% (protocol).", ""]
for W in ("W0", "W3"):
    ref = sp3[W]; sp2 = pickle.load(open(f"{ROOT}/results/spike02/raw_rl_{W}.pkl", "rb"))
    base = {"C0 (ctc_lqr, arena)": None}
    rows = {}
    rows["spike02 RL (CPU torch, 2M x5)"] = {c: np.mean([sp2[("rl", s)][c]["rmse"] for s in range(5)], 0) for c in ("clean", "dist")}
    rows["spike03a C0 (untuned CTC)"] = {c: ref["C0"][0][c]["rmse"] for c in ("clean", "dist")}
    rows["spike03a C1 (tuned gains)"] = {c: ref["C1"][0][c]["rmse"] for c in ("clean", "dist")}
    seeds_info = {}
    for tag in tags:
        f = f"{ROOT}/results/mjx/eval_{tag}_{W}.pkl"
        if not os.path.exists(f): continue
        d = pickle.load(open(f, "rb")); ss = sorted(s for (k, s) in d if k == "rl")
        rows[f"MJX PPO {tag} ({len(ss)} seeds)"] = {c: np.mean([d[("rl", s)][c]["rmse"] for s in ss], 0) for c in ("clean", "dist")}
        seeds_info[tag] = {c: [float(d[("rl", s)][c]["rmse"].mean()) for s in ss] for c in ("clean", "dist")}
        c0diff = max(float(np.abs(d[("ctc_lqr", 0)][c]["rmse"] - ref["C0"][0][c]["rmse"]).max()) for c in ("clean", "dist"))
        seeds_info.setdefault("_c0diff", c0diff)
        L_extra = None
    L += [f"## {W}", "", "| method | clean RMSE | dist RMSE |", "|---|---|---|"]
    for n, r in rows.items():
        L.append(f"| {n} | {ci(r['clean'])[0]:.4f} +- {ci(r['clean'])[1]:.4f} | {ci(r['dist'])[0]:.4f} +- {ci(r['dist'])[1]:.4f} |")
    L += ["", "Per-seed mean RMSE (clean / dist): " + "; ".join(f"{t}: " + ", ".join(f"{a:.4f}/{b:.4f}" for a, b in zip(v['clean'], v['dist'])) for t, v in seeds_info.items() if t[0] != "_"),
          f"arena `ctc_lqr` (this repo, today) vs spike03a C0 per-trajectory RMSE: max |diff| = {seeds_info.get('_c0diff', float('nan')):.1e}", ""]
    L += ["| A | B | cond | mean(A-B) | 95% CI | improvement | WIN |", "|---|---|---|---|---|---|---|"]
    mj = [n for n in rows if n.startswith("MJX PPO")]
    for n in mj:
        for A, B in (("spike03a C0 (untuned CTC)", n), ("spike02 RL (CPU torch, 2M x5)", n), ("spike03a C1 (tuned gains)", n)):
            for c in ("clean", "dist"):
                m, h, imp = paired(rows[A][c], rows[B][c]); win = (m - h > 0) and imp >= 0.10
                L.append(f"| {A} | {B} | {c} | {m:+.5f} | [{m-h:+.5f}, {m+h:+.5f}] | {imp*100:+.1f}% | {'YES' if win else 'no'} |")
    L.append("")
open(f"{ROOT}/results/mjx/ppo_table.md", "w").write("\n".join(L)); print("\n".join(L))
