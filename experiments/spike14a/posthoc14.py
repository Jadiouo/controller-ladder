"""Post-hoc (not pre-registered) episode-level overlap on the holdout / new seeds: where does each method convert or lose episodes relative to R0, and the R1+noise diagnostic."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
out = []
for st in ('hold', 'new'):
    f = os.path.join(OUT14, f'pipe_{st}.json'); P = json.load(open(f)) if os.path.exists(f) else {}
    g = lambda c: np.array(P[c]['success'], float) if c in P else None
    ms = {'dec4 (mean of 3 seeds, rounded to majority)': (lambda: (np.mean([g(f'dec4_a0.9_e38_s{k}') for k in range(3)], 0) >= 0.5).astype(float)), 'dec4 seed 0': lambda: g('dec4_a0.9_e38_s0'), 'ens_dec4': lambda: g('ens_dec4'), 'lmlp1 s0': lambda: g('lmlp1_s0'), 'gru4 s0': lambda: g('gru4_s0'), 'C_map1_ms': lambda: g('C_map1_ms'), 'HYB': lambda: g('C_hyb')}
    r0, r1 = g('C0'), g('R1'); n_gap = int(((r1 == 1) & (r0 == 0)).sum()); n_r1f = int((r1 == 0).sum())
    out += [f'### {st}: episode-level overlap with R0 (C0) and R1 (n={len(r0)})', '', f'| method | success pct | gained vs R0 (R0 fail -> success) | lost vs R0 (R0 success -> fail) | converted of the R1-only episodes (R1 success and R0 fail, n={n_gap}) | success pct when R1 fails (n={n_r1f}) |', '|---|---|---|---|---|---|']
    for nm, fn in ms.items():
        try: v = fn()
        except Exception: v = None
        if v is None: continue
        gain = int(((r0 == 0) & (v == 1)).sum()); lost = int(((r0 == 1) & (v == 0)).sum()); gap = int(((r1 == 1) & (r0 == 0)).sum()); conv = int(((r1 == 1) & (r0 == 0) & (v == 1)).sum())
        out.append(f'| {nm} | {100*v.mean():.1f} | {gain} | {lost} | {conv}/{gap} = {100*conv/max(gap,1):.0f}% | {100*v[r1 == 0].mean():.0f}% |')
    out.append('')
P = json.load(open(os.path.join(OUT14, 'pipe_hold.json'))); r1 = np.mean(P['R1']['success']) * 100
out += ['### R1 + noise diagnostic (holdout, n=500; post hoc): TRUE theta plus iid Gaussian noise of the size of the selected learned estimator\'s holdout error (ls 0.016, per-link log-mass 0.094, log-damping 0.320, friction 0.50e-3 Nm; noise x1 and x0.5)', '', '| plant parameters given to the planner | success % |', '|---|---|']
for k, nm in (('R1', 'true theta'), ('R1n05', 'true theta + noise x0.5'), ('R1n1', 'true theta + noise x1'), ('C0', 'R0 fit'), ('dec4_a0.9_e38_s0', 'learned dec4 (seed 0)')):
    if k in P: out.append(f"| {nm} | {100*np.mean(P[k]['success']):.1f} |")
open(os.path.join(OUT14, 'posthoc.md'), 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
