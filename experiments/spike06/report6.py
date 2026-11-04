"""table.md for spike06 (H8b). Primary: success = reaching the 300-hit cap. Paired bootstrap over the 50 locked test episodes (5000 resamples)."""
import pickle, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env6 as W; DIST = W.use_file()
R = pickle.load(open('results/spike06/test_raw.pkl', 'rb')); best = R['_best']; rng = np.random.default_rng(0); NB = 5000; idx = rng.integers(0, 50, (NB, 50))
fam = lambda n: n.split('#')[0]
def grp(name): ks = sorted(k for k in R if k != '_best' and fam(k) == name); return ks
def succ(name): return np.array([(R[k]['hits'] >= 300) for k in grp(name)], float).mean(0)      # per-episode success (mean over seeds for RL)
def hits(name): return np.array([R[k]['hits'] for k in grp(name)], float)
def ci(x): return np.percentile(x, [2.5, 97.5])
def pdiff(a, b, sub=None):
    d = a - b; s = np.arange(50) if sub is None else np.where(sub)[0]; ix = s[rng.integers(0, len(s), (NB, len(s)))]; bs = d[ix].mean(1); return d[s].mean(), ci(bs)
def sci(a, sub=None):
    s = np.arange(50) if sub is None else np.where(sub)[0]; ix = s[rng.integers(0, len(s), (NB, len(s)))]; return a[s].mean(), ci(a[ix].mean(1))
meth = ['T-nom', 'T-robust(spike05)', 'T-adapt(spike05)', 'T-robust', 'T-adapt', 'R-dr', 'R-hist']; meth = [m for m in meth if grp(m)]
lab = {'R-dr': 'R-on-best (no history)', 'R-hist': 'R-on-best-hist', 'T-robust(spike05)': 'T-robust (spike05 params, not retuned)', 'T-adapt(spike05)': 'T-adapt (spike05 params, not retuned)', 'T-robust': 'T-robust (retuned)', 'T-adapt': 'T-adapt (retuned)'}
pp = R['T-nom']['params']; e = np.array([p['e'] for p in pp]); mb = np.array([p['mb'] for p in pp]); dl = np.array([p['delay_ms'] for p in pp])
em, mm, dm = np.mean(DIST['e']), np.mean(DIST['mb']), np.mean(DIST['delay'])
L = [f'# spike06 table: H8b (wide, non-saturating unknown error), ball juggling, cap N=300, locked test seeds 0-49\n',
     f'Error distribution (dist.json): restitution e~U{DIST["e"]}, ball mass x U{DIST["mb"]}, sensor delay U{DIST["delay"]} ms (5 ms steps), 4 mm noise. best-classical (chosen on CAL only) = **{best}**.',
     'Success = reached the 300-hit cap. RL rows: per-episode success = mean over the 3 training seeds; CIs = paired bootstrap over the 50 test episodes (5000 resamples).\n',
     '## 1. Main table\n', '| method | #seeds | success [95% CI] | mean hits | median hits | apex err (m) | torque energy (Nm^2) |', '|---|---|---|---|---|---|---|']
for m in meth:
    ks = grp(m); s, (lo, hi) = sci(succ(m)); h = hits(m); ap = np.nanmean([R[k]['apex'] for k in ks]); te = np.mean([R[k]['tau2'] for k in ks])
    L.append(f'| {lab.get(m, m)} | {len(ks)} | {s*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}] | {h.mean():.1f} | {np.median(h):.0f} | {ap:.3f} | {te:.0f} |')
ps_ = lambda m: [str(round(float((R[k]['hits'] >= 300).mean() * 100))) + '%' for k in grp(m)]
L.append('\nper-seed success: ' + '; '.join(lab[m] + ': ' + str(ps_(m)) for m in ('R-dr', 'R-hist') if grp(m)))
L.append('\n## 2. Paired comparisons of success rate (A minus B, percentage points)\n'); L.append('| A vs B | diff [95% CI] | per-seed diff of A | mean hits diff [95% CI] |'); L.append('|---|---|---|---|')
bn = best
def hmean(m): return hits(m).mean(0)
verd = None
for A in ('R-hist', 'R-dr'):
    if not grp(A): continue
    for B in dict.fromkeys((bn, 'T-robust', 'T-adapt', 'T-nom')):
        d, (lo, hi) = pdiff(succ(A), succ(B)); dh, (hl, hh) = pdiff(hmean(A), hmean(B)); ps = ', '.join('%+.0f' % (((R[k]['hits'] >= 300).mean() - succ(B).mean()) * 100) for k in grp(A))
        L.append(f'| {lab[A]} vs {B}{" (best)" if B == bn else ""} | {d*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] | {ps} | {dh:+.1f} [{hl:+.1f}, {hh:+.1f}] |')
        if A == 'R-hist' and B == bn: verd = (d, lo, hi)
d, (lo, hi) = pdiff(succ('T-adapt'), succ('T-robust')); L.append(f'| T-adapt vs T-robust | {d*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] | | |')
L.append('\n## 3. Subgroups (success [95% CI]; split at the midpoint of each range)\n')
G = {f'e low (<{em:.2f})': e < em, f'e high': e >= em, f'delay low (<{dm:.0f} ms)': dl < dm, 'delay high': dl >= dm, f'ball mass low (<{mm:.1f}x)': mb < mm, 'ball mass high': mb >= mm}
hdr = [m for m in meth if m in ('T-nom', 'T-robust', 'T-adapt', 'R-dr', 'R-hist')]
L.append('| group | n | ' + ' | '.join(lab.get(m, m) for m in hdr) + ' | R-on-best-hist minus best-classical [CI] |'); L.append('|---|---|' + '---|' * (len(hdr) + 1))
for g, s in G.items():
    cells = []
    for m in hdr: v, (lo, hi) = sci(succ(m), s); cells.append(f'{v*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}]')
    dd = ''
    if grp('R-hist'): d, (lo, hi) = pdiff(succ('R-hist'), succ(bn), s); dd = f'{d*100:+.0f} [{lo*100:+.0f}, {hi*100:+.0f}]'
    L.append(f'| {g} | {s.sum()} | ' + ' | '.join(cells) + f' | {dd} |')
L.append('\n## 4. H8b verdict\n')
if verd:
    d, lo, hi = verd; ok = d >= 0.10 and lo > 0
    L.append(f'Rule (docs/03_hypotheses.md v4): R-on-best-hist success >= best-classical + 10 pp and paired bootstrap CI excludes 0. Observed vs {bn}: {d*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}] -> **H8b {"SUPPORTED" if ok else "REFUTED (not supported)"}**.')
open('results/spike06/table.md', 'w').write('\n'.join(L)); print('\n'.join(L))
