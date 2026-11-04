"""Report for spike09b: reads results/spike09b/{test_raw.pkl, holdout.json, ppo_*.pkl, lest_train.json, adapt_probe_cal.json, budget.json} -> table.md."""
import sys, os, json, pickle, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); OUT = os.path.join(ROOT, 'results/spike09b'); W = 0.2
R = pickle.load(open(os.path.join(OUT, 'test_raw.pkl'), 'rb')); best = R['_best']; HO = json.load(open(os.path.join(OUT, 'holdout.json'))); NB = 20000
def boot(x, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed); m = r.integers(0, len(x), (NB, len(x))); b = x[m].mean(1); return b
def ci(x, seed=0): b = boot(x, seed); return float(np.quantile(b, .025)), float(np.quantile(b, .975))
def pval(d, seed=1):
    b = boot(d, seed); return float(min(1.0, max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0 / NB)))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(o): run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(run, 1.0)
    return adj
def agg(prefix, key='success'):
    ks = sorted(k for k in R if k.startswith(prefix + '#')); return np.mean([R[k][key] for k in ks], 0) if ks else None
M = {}   # name -> per-seed success (seed-averaged over training seeds for multi-seed methods)
for n in ('T-nom', 'T-robust', 'T-adapt', 'T-oracle'): M[n] = R[n]['success']
for n in ('L-est', 'R-PPO-hist', 'R-oracle'): M[n] = agg(n)
B = M[f'T-{best}']; seeds = np.arange(50)
def pp(x): return f'{100*x:.1f}'
def row(name, key):
    out = {}
    for k in ('success', 'min_dist', 'hit_speed', 'energy'): out[k] = agg(name, k) if name in ('L-est', 'R-PPO-hist', 'R-oracle') else R[name][k]
    res = agg(name, 'res_mean') if name in ('R-PPO-hist', 'R-oracle') else None
    return out, res
L = []; A_ = L.append
A_(f'# spike09b TEST results (whip, seeds 0-49, single locked run). best-classical = T-{best} (selected on holdout 1100-1599)\n')
A_('## Main table (multi-seed methods: per-test-seed mean over the 3 training seeds, then statistics over the 50 test seeds)\n')
A_('| method | success % [95% CI] | closest approach (m) | hit speed (m/s, hits only) | energy (Nm^2/step) | residual norm (Nm, mean over swing) |'); A_('|---|---|---|---|---|---|')
for name in ('T-nom', 'T-robust', 'T-adapt', 'T-oracle', 'L-est', 'R-PPO-hist', 'R-oracle'):
    o, res = row(name, None); lo, hi = ci(o['success']); hs = o['hit_speed']; hsm = np.nanmean(hs) if np.isfinite(hs).any() else float('nan')
    A_(f"| {name}{' (= best-classical)' if name == 'T-' + best else ''} | {pp(o['success'].mean())} [{pp(lo)}, {pp(hi)}] | {np.mean(o['min_dist']):.4f} | {hsm:.2f} | {np.mean(o['energy']):.1f} | {'-' if res is None else f'{res.mean():.3f}'} |")
A_('\nPer training seed (success %):\n'); A_('| method | seed 0 | seed 1 | seed 2 |'); A_('|---|---|---|---|')
for m in ('L-est', 'R-PPO-hist', 'R-oracle'):
    A_(f"| {m} | " + ' | '.join(pp(R[f'{m}#{s}']['success'].mean()) for s in range(3)) + ' |')
A_('\n## Paired differences vs best-classical (percentage points; paired percentile bootstrap over the 50 TEST seeds, %d resamples)\n' % NB)
A_('| method | diff pp | 95% CI | p (boot) |'); A_('|---|---|---|---|')
for name in ('T-nom', 'T-robust', 'T-adapt', 'T-oracle', 'L-est', 'R-PPO-hist', 'R-oracle'):
    d = M[name] - B; lo, hi = ci(d, 2); A_(f'| {name} | {pp(d.mean())} | [{pp(lo)}, {pp(hi)}] | {pval(d):.4f} |')
def hyp(title, names, strict=True):
    ds = [M[n] - B for n in names]; ps = np.array([pval(d) for d in ds]); adj = holm(ps); res = []
    A_(f'\n### {title}\n'); A_('| candidate | diff pp | 95% CI | p | Holm p | >=10 pp and CI > 0 and Holm p < .05 |'); A_('|---|---|---|---|---|---|')
    for n, d, p_, a_ in zip(names, ds, ps, adj):
        lo, hi = ci(d, 2); ok = bool(d.mean() >= 0.10 and lo > 0 and (a_ < 0.05 or not strict)); res.append(ok); A_(f'| {n} | {pp(d.mean())} | [{pp(lo)}, {pp(hi)}] | {p_:.4f} | {a_:.4f} | {"yes" if ok else "no"} |')
    return any(res)
h9 = hyp('H9w (headroom): T-oracle or R-oracle vs best-classical (Holm-strict)', ['T-oracle', 'R-oracle']); h9l = hyp('H9w spec-literal (no Holm requirement)', ['T-oracle', 'R-oracle'], strict=False); h10 = hyp('H10w (fair learning): L-est or R-PPO-hist vs best-classical', ['L-est', 'R-PPO-hist'])
A_(f'\n**H9w: strict {"SUPPORTED" if h9 else "NOT SUPPORTED"}, spec-literal {"SUPPORTED" if h9l else "NOT SUPPORTED"}.  H10w: {"SUPPORTED" if h10 else "REFUTED (neither reaches +10 pp with CI > 0)"}.**\n')
ceil = 1.0 - B.mean(); A_(f'Ceiling note: best-classical TEST success = {pp(B.mean())} %, so the largest conceivable gain is {pp(ceil)} pp; a +10 pp win is {"arithmetically impossible" if ceil < 0.10 else "possible"} on TEST.\n')
d = M['L-est'] - M['T-adapt']; lo, hi = ci(d, 3); A_(f'### Secondary: L-est (learned estimator) vs T-adapt (model fit): {pp(d.mean())} pp, 95% CI [{pp(lo)}, {pp(hi)}], p = {pval(d, 3):.4f}\n')
# chain-length groups
ch = R['T-nom']['chain'][:, 0] - 1; edges = [-W / 2, 0.0, W / 2]; cell = np.searchsorted(edges, ch); names = ['len -20..-10 %', 'len -10..0 %', 'len 0..+10 %', 'len +10..+20 %']
A_('## Success % by true chain-length error quartile (n per cell in header; ~12 each, wide CIs)\n'); A_('| method | ' + ' | '.join(f'{names[c]} (n={int((cell == c).sum())})' for c in range(4)) + ' |'); A_('|---|---|---|---|---|')
for name in ('T-nom', 'T-robust', 'T-adapt', 'T-oracle', 'L-est', 'R-PPO-hist', 'R-oracle'): A_(f'| {name} | ' + ' | '.join(pp(M[name][cell == c].mean()) for c in range(4)) + ' |')
# learned check
A_('\n## "Did it learn?" check (CAL 1000-1049: selected checkpoint minus initialisation, need >= +5 pp)\n'); A_('| run | init CAL succ | selected ckpt CAL succ | diff pp | learned | residual norm at selected ckpt (Nm) |'); A_('|---|---|---|---|---|---|')
for f, nm in (('ppo_hist.pkl', 'R-PPO-hist'), ('ppo_oracle.pkl', 'R-oracle')):
    P = pickle.load(open(os.path.join(OUT, f), 'rb'))
    for s, v in sorted(P.items()): A_(f"| {nm}#{s} | {pp(v['init']['succ'])} | {pp(v['selected']['succ'])} (ckpt it {v['selected']['it']}) | {pp(v['selected']['succ'] - v['init']['succ'])} | {'yes' if v['learned'] else 'NOT LEARNED'} | {v['selected']['res_mean']:.3f} |")
A_('\n## Holdout (1100-1599) used for best-classical selection\n'); A_('| method | success % | CI |'); A_('|---|---|---|')
for n in ('nom', 'robust', 'adapt', 'oracle'):
    x = np.array(HO[n]['success']); lo, hi = ci(x); A_(f'| T-{n} | {pp(x.mean())} | [{pp(lo)}, {pp(hi)}] |')
BG = json.load(open(os.path.join(OUT, 'budget.json'))) if os.path.exists(os.path.join(OUT, 'budget.json')) else {}
L.append('\n## Budget (episodes)\n```\n' + json.dumps(BG, indent=1) + '\n```')
open(os.path.join(OUT, 'table.md'), 'w').write('\n'.join(L)); print('\n'.join(L))
