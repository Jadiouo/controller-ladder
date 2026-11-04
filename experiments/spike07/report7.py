"""table.md for spike07 (H9 oracle headroom, H10 fair RL). Success = reached 300-hit cap. Paired bootstrap over the 50 locked TEST episodes (5000 resamples). RL episode success = mean over the 3 training seeds."""
import pickle, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env7 as W; DIST = W.use_file()
R = pickle.load(open('results/spike07/test_raw.pkl', 'rb')); best = R['_best']; rng = np.random.default_rng(0); NB = 5000; IDX = rng.integers(0, 50, (NB, 50))
FINAL = json.load(open('results/spike07/final_methods.json'))
fam = lambda n: n.split('#')[0]
def grp(name): return sorted(k for k in R if k != '_best' and fam(k) == name)
def succ(name): return np.array([(R[k]['hits'] >= 300) for k in grp(name)], float).mean(0)
def hits(name): return np.array([R[k]['hits'] for k in grp(name)], float)
ci = lambda x: np.percentile(x, [2.5, 97.5])
def pdiff(a, b, sub=None):
    d = a - b; s = np.arange(50) if sub is None else np.where(sub)[0]; ix = s[rng.integers(0, len(s), (NB, len(s)))]; bs = d[ix].mean(1); return d[s].mean(), ci(bs), float((bs <= 0).mean())
def sci(a, sub=None):
    s = np.arange(50) if sub is None else np.where(sub)[0]; ix = s[rng.integers(0, len(s), (NB, len(s)))]; return a[s].mean(), ci(a[ix].mean(1))
meth = ['T-nom', 'T-robust(spike06)', 'T-robust(kp10000)', 'T-adapt(spike06)', 'R-ppo', 'R-sac', 'R-oracle']; meth = [m for m in meth if grp(m)]
lab = {'R-ppo': 'R-PPO-fixed (hist)', 'R-sac': 'R-SAC (hist)', 'R-oracle': 'R-oracle (PPO-fixed + true e, mass, delay)'}
L_ = lambda m: lab.get(m, m) + (' **[best-classical]**' if m == best else '')
pp = R['T-nom']['params']; e = np.array([p['e'] for p in pp]); mb = np.array([p['mb'] for p in pp]); dl = np.array([p['delay_ms'] for p in pp])
em, mm, dm = np.mean(DIST['e']), np.mean(DIST['mb']), np.mean(DIST['delay'])
L = ['# spike07 table: H9 (oracle headroom) and H10 (fair RL), ball juggling, cap N=300, locked TEST seeds 0-49 (evaluated once)\n',
     f'Distribution C (results/spike06/dist.json): restitution e~U{DIST["e"]}, ball mass x U{DIST["mb"]}, sensor delay U{DIST["delay"]} ms. best-classical (chosen on CAL only) = **{best}**. All RL residuals are stacked on it (cap 12 Nm per joint).',
     'Success = reached the 300-hit cap. RL rows: per-episode success = mean over the 3 training seeds; CIs = paired bootstrap over the 50 test episodes (5000 resamples).\n',
     '## 1. Main table\n', '| method | #seeds | success [95% CI] | mean hits | median hits | apex err (m) | torque energy (Nm^2) | residual L2 norm mean / max (Nm, TEST) |', '|---|---|---|---|---|---|---|---|']
for m in meth:
    ks = grp(m); s, (lo, hi) = sci(succ(m)); h = hits(m); ap = np.nanmean([R[k]['apex'] for k in ks]); te = np.mean([R[k]['tau2'] for k in ks])
    rn = f"{np.mean([R[k]['res_mean'] for k in ks]):.2f} / {max(R[k]['res_max'] for k in ks):.2f}" if 'res_mean' in R[ks[0]] else '-'
    L.append(f'| {L_(m)} | {len(ks)} | {s*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}] | {h.mean():.1f} | {np.median(h):.0f} | {ap:.3f} | {te:.0f} | {rn} |')
ps_ = lambda m: [str(round(float((R[k]['hits'] >= 300).mean() * 100))) + '%' for k in grp(m)]
L.append('\nper-seed success (TEST): ' + '; '.join(lab[m] + ': ' + str(ps_(m)) for m in ('R-ppo', 'R-sac', 'R-oracle') if grp(m)))
L.append('\nper-seed residual norm mean/max (Nm, TEST): ' + '; '.join(lab[m] + ': ' + str([f"{R[k]['res_mean']:.2f}/{R[k]['res_max']:.2f}" for k in grp(m)]) for m in ('R-ppo', 'R-sac', 'R-oracle') if grp(m)))
L.append('\n## 2. Paired comparisons of success rate (A minus B, percentage points)\n'); L.append('| A vs B | diff [95% CI] | one-sided p (A<=B) | per-seed diff of A | mean hits diff [95% CI] |'); L.append('|---|---|---|---|---|')
hmean = lambda m: hits(m).mean(0); V_ = {}
for A in ('R-ppo', 'R-sac', 'R-oracle'):
    if not grp(A): continue
    for B in dict.fromkeys((best, 'T-robust(spike06)', 'T-adapt(spike06)', 'T-nom')):
        d, (lo, hi), p = pdiff(succ(A), succ(B)); dh, (hl, hh), _ = pdiff(hmean(A), hmean(B)); ps = ', '.join('%+.0f' % (((R[k]['hits'] >= 300).mean() - succ(B).mean()) * 100) for k in grp(A))
        L.append(f'| {lab[A]} vs {B}{" (best)" if B == best else ""} | {d*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] | {p:.4f} | {ps} | {dh:+.1f} [{hl:+.1f}, {hh:+.1f}] |')
        if B == best: V_[A] = (d, lo, hi, p)
if best != 'T-robust(spike06)' and grp('T-robust(kp10000)'): d, (lo, hi), p = pdiff(succ('T-robust(kp10000)'), succ('T-robust(spike06)')); L.append(f'| T-robust(kp10000) vs T-robust(spike06) | {d*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] | {p:.4f} | | |')
L.append('\n## 3. Subgroups (success [95% CI]; split at the midpoint of each range)\n')
G = {f'e low (<{em:.2f})': e < em, 'e high': e >= em, f'delay low (<{dm:.0f} ms)': dl < dm, 'delay high': dl >= dm, f'ball mass low (<{mm:.1f}x)': mb < mm, 'ball mass high': mb >= mm}
hdr = [m for m in meth if m in (best, 'T-nom', 'R-ppo', 'R-sac', 'R-oracle')]
L.append('| group | n | ' + ' | '.join(L_(m) for m in hdr) + ' | R-oracle minus best [CI] |'); L.append('|---|---|' + '---|' * (len(hdr) + 1))
for g, s in G.items():
    cells = []
    for m in hdr: v, (lo, hi) = sci(succ(m), s); cells.append(f'{v*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}]')
    dd = ''
    if grp('R-oracle'): d, (lo, hi), _ = pdiff(succ('R-oracle'), succ(best), s); dd = f'{d*100:+.0f} [{lo*100:+.0f}, {hi*100:+.0f}]'
    L.append(f'| {g} | {s.sum()} | ' + ' | '.join(cells) + f' | {dd} |')
L.append('\n## 4. Verdicts (rules from docs/03_hypotheses.md v5)\n')
if 'R-oracle' in V_:
    d, lo, hi, p = V_['R-oracle']; ok = d >= 0.10 and lo > 0
    L.append(f'**H9** (R-oracle >= best-classical + 10 pp, paired-bootstrap 95% CI excludes 0): observed {d*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}] -> **H9 {"SUPPORTED" if ok else "NOT SUPPORTED"}**.')
c = [(A, V_[A]) for A in ('R-ppo', 'R-sac') if A in V_]
if c:
    order = sorted(c, key=lambda t: t[1][3]); m_ = len(order); adj = {}; run_ = 0
    for i, (A, v) in enumerate(order): run_ = max(run_, min(1.0, (m_ - i) * v[3])); adj[A] = run_
    L.append('\n**H10** (at least one of R-PPO-fixed / R-SAC >= best-classical + 10 pp, CI excludes 0; two comparisons, Holm-corrected one-sided bootstrap p, alpha 0.05):\n'); L.append('| comparison | diff pp [95% CI] | raw p | Holm-adjusted p | >=10 pp and CI>0 | passes with Holm (adj p<0.05) |'); L.append('|---|---|---|---|---|---|')
    anyok = False
    for A, (d, lo, hi, p) in c:
        ok1 = d >= 0.10 and lo > 0; ok2 = ok1 and adj[A] < 0.05; anyok |= ok2
        L.append(f'| {lab[A]} vs {best} | {d*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] | {p:.4f} | {adj[A]:.4f} | {"yes" if ok1 else "no"} | {"yes" if ok2 else "no"} |')
    L.append(f'\n-> **H10 {"SUPPORTED" if anyok else "REFUTED (not supported)"}**.')
# CAL training-curve summary
L.append('\n## 5. CAL training-curve summary (checkpoints at 1/7 ... 7/7 of training; CAL = seeds 1000-1049, mean action, success = reached 300 hits; selected checkpoint marked *)\n')
bc = json.load(open('results/spike07/best.json'))['cal']; L.append(f'(initialisation = best-classical alone, CAL success {bc["frac_cap"]*100:.0f}%, mean hits {bc["mean"]:.1f})\n')
L.append('Note: the train-success column is the rolling success of the last 60 completed training episodes; it is biased low early (only failures complete first) and uninformative for 128 envs x 2.5M steps; the CAL curve is the criterion.\n')
L.append('| method / seed | CAL success at the 7 checkpoints (%) | selected (CAL success %, mean hits) | min CAL ckpt (%) | train success (last 60 train eps) first -> min -> last (%) | residual L2 mean at CAL ckpts (Nm) | train sim steps | ckpt-eval sim steps | train wall (s) |'); L.append('|---|---|---|---|---|---|---|---|---|')
for m, path in FINAL.items():
    Rr = pickle.load(open(path, 'rb'))
    for s, v in sorted(Rr.items()):
        ck = v['ckpts']; sel = v['selected']; cs = ' '.join(('*' if c['it'] == sel['it'] else '') + f"{c['frac_cap']*100:.0f}" for c in ck); tr = [x[2] for x in v['log'] if x[2] == x[2]]; rs = ' '.join('%.2f' % c['res_mean'] for c in ck)
        L.append(f"| {m}#{s} | {cs} | {sel['frac_cap']*100:.0f}%, {sel['mean']:.1f} | {min(c['frac_cap'] for c in ck)*100:.0f} | {tr[0]*100:.0f} -> {min(tr)*100:.0f} -> {tr[-1]*100:.0f} | {rs} | {v['env_steps_exec']/1e6:.1f}M | {v['ckpt_eval_steps']/1e6:.1f}M | {v['train_time']:.0f} |")
open('results/spike07/table.md', 'w').write('\n'.join(L)); print('\n'.join(L))
