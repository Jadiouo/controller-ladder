import pickle, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env5 as V
R = pickle.load(open('results/spike05/test_raw.pkl', 'rb')); rng = np.random.default_rng(0); NB = 5000
def fam(n): return n.split('#')[0]
def hits_of(name):
    ks = [k for k in R if fam(k) == name]; return np.array([R[k]['hits'] for k in sorted(ks)], float), sorted(ks)
meth = ['T-nom', 'T-robust', 'T-adapt', 'R-dr', 'R-hist']; label = {'R-dr': 'R-DR', 'R-hist': 'R-DR-hist'}
H = {m: hits_of(m)[0] for m in meth if hits_of(m)[0].size}
pp = R['T-nom']['params']; e = np.array([p['e'] for p in pp]); mb = np.array([p['mb'] for p in pp]); dl = np.array([p['delay_ms'] for p in pp])
idx = rng.integers(0, 50, (NB, 50))
def med(h, ix=None): return np.median(h if ix is None else h[:, ix])
def ci(x): return np.percentile(x, [2.5, 97.5])
def medci(h, sub=None):
    s = np.arange(50) if sub is None else np.where(sub)[0]
    if len(s) == 0: return float('nan'), (float('nan'),) * 2
    ix = s[rng.integers(0, len(s), (NB, len(s)))]; m = np.array([np.median(h[:, i]) for i in ix]); return float(np.median(h[:, s])), tuple(ci(m))
L = ['# spike05 table: H8 (unknown, per-episode random plant error), ball juggling, cap N=300, locked test set seeds 0-49\n',
     'Error distribution (per episode, from seed): restitution e~U[0.4,0.85], ball mass x U[1,5], sensor delay U[0,150] ms (rounded to 5 ms steps), 4 mm position noise.',
     'RL rows pool all (seed, episode) pairs; CI = cluster bootstrap over the 50 episodes (5000 resamples). Medians of 300 are the cap (censored).\n',
     '## 1. Main table\n', '| method | #seeds | median hits [95% CI] | mean hits | frac reaching cap 300 | apex err (m) | hit x-drift (m) | torque energy (Nm^2) |', '|---|---|---|---|---|---|---|---|']
for m, h in H.items():
    ks = hits_of(m)[1]; mm, (lo, hi) = medci(h)
    ap = np.nanmean([R[k]['apex'] for k in ks]); dr = np.nanmean([R[k]['drift'] for k in ks]); te = np.mean([R[k]['tau2'] for k in ks])
    L.append(f'| {label.get(m, m)} | {len(ks)} | {mm:.0f} [{lo:.0f}, {hi:.0f}] | {h.mean():.1f} | {(h>=300).mean():.2f} | {ap:.3f} | {dr:.4f} | {te:.0f} |')
if 'R-dr' in H: L.append('\nper-seed medians: ' + '; '.join(f'{label[m]}: {[int(np.median(x)) for x in H[m]]}' for m in ('R-dr', 'R-hist') if m in H))
cl = [m for m in ('T-robust', 'T-adapt') if m in H]; best = max(cl, key=lambda m: np.median(H[m])) if cl else None
L.append(f'\nStronger classical opponent on the test median (conservative choice, against H8): **{best}**.\n')
L.append('## 2. Paired comparisons (relative gain of median, paired cluster bootstrap over episodes)\n'); L.append('| A vs B | median A | median B | rel. gain [95% CI] | seed-level rel. gains |'); L.append('|---|---|---|---|---|')
def rel(a, b):
    ma, mb_ = np.median(a), np.median(b); r = np.array([(np.median(a[:, i]) - np.median(b[:, i])) / max(np.median(b[:, i]), 1e-9) for i in idx]); lo, hi = ci(r)
    return ma, mb_, (ma - mb_) / max(mb_, 1e-9), lo, hi
verd = {}
for A in ('R-hist', 'R-dr'):
    for B in ['T-nom'] + cl + ([best + ' (best)'] if best else []):
        if A not in H: continue
        Bn = B.replace(' (best)', ''); ma, mb_, r, lo, hi = rel(H[A], H[Bn]); sl = ', '.join(f'{(np.median(x)-mb_)/max(mb_,1e-9)*100:+.0f}%' for x in H[A])
        L.append(f'| {label[A]} vs {B} | {ma:.0f} | {mb_:.0f} | {r*100:+.0f}% [{lo*100:+.0f}, {hi*100:+.0f}] | {sl} |'); verd[(A, B)] = (r, lo, hi)
if best and 'R-hist' in H and 'T-robust' in H:
    L.append('\n| T-adapt vs T-robust | ' + ' | '.join([f'{x:.0f}' for x in (np.median(H["T-adapt"]), np.median(H["T-robust"]))]) + ' | ' + '%+.0f%% [%+.0f, %+.0f]' % tuple(100 * np.array(rel(H['T-adapt'], H['T-robust'])[2:])) + ' | |') if 'T-adapt' in H else None
L.append('\n## 3. Subgroups (median hits [95% CI], episodes in group)\n')
groups = {'e low (<0.625)': e < 0.625, 'e high (>=0.625)': e >= 0.625, 'delay low (<75 ms)': dl < 75, 'delay high (>=75 ms)': dl >= 75, 'ball mass low (<3x)': mb < 3, 'ball mass high (>=3x)': mb >= 3,
          'e low & delay low': (e < 0.625) & (dl < 75), 'e low & delay high': (e < 0.625) & (dl >= 75), 'e high & delay low': (e >= 0.625) & (dl < 75), 'e high & delay high': (e >= 0.625) & (dl >= 75)}
L.append('| group | n | ' + ' | '.join(label.get(m, m) for m in H) + ' | winner (highest median) |'); L.append('|---|---|' + '---|' * (len(H) + 1))
for g, s in groups.items():
    cells = []; meds = {}
    for m, h in H.items(): mm, (lo, hi) = medci(h, s); cells.append(f'{mm:.0f} [{lo:.0f}, {hi:.0f}]'); meds[m] = mm
    L.append(f'| {g} | {s.sum()} | ' + ' | '.join(cells) + f' | {label.get(max(meds, key=meds.get), max(meds, key=meds.get))} |')
L.append('\nRank correlation (Spearman) of hits with plant parameters, per method (pooled over seeds):')
from scipy.stats import spearmanr
for m, h in H.items(): hh = h.mean(0); L.append(f'- {label.get(m, m)}: e {spearmanr(e, hh)[0]:+.2f}, ball mass {spearmanr(mb, hh)[0]:+.2f}, delay {spearmanr(dl, hh)[0]:+.2f}')
if 'T-adapt' in R:
    ee = R['T-adapt'].get('e_est'); 
    if ee: 
        ef = np.array([np.mean(x[-3:]) if len(x) else np.nan for x in ee]); L.append(f'\nT-adapt estimator quality on test episodes: corr(final e_hat, true e) = {np.corrcoef(ef[~np.isnan(ef)], e[~np.isnan(ef)])[0,1]:.2f}, mean abs err {np.nanmean(np.abs(ef-e)):.3f}; delay: corr(mean d_obs[ms], true delay) = ' +
                 f'{np.corrcoef([np.mean(x)*5 if len(x) else np.nan for x in R["T-adapt"]["d_est"]], dl)[0,1]:.2f}' if all(len(x) for x in R['T-adapt']['d_est']) else '')
L.append('\n## 4. H8 verdict\n')
if best and 'R-hist' in H:
    r, lo, hi = verd[('R-hist', best + ' (best)')]; ok = r >= 0.3 and lo > 0
    L.append(f'Rule (pre-registered in docs/03_hypotheses.md v3): R-DR-hist median >= 1.3 x the better of T-robust/T-adapt, paired bootstrap CI excludes 0. Observed vs {best}: {r*100:+.0f}% [{lo*100:+.0f}, {hi*100:+.0f}] -> **H8 {"SUPPORTED" if ok else "REFUTED (falsified)"}**.')
open('results/spike05/table.md', 'w').write('\n'.join(L)); print('\n'.join(L))
