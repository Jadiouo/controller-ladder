"""-> results/spike04/table.md"""
import sys, json, pickle; sys.path.insert(0, 'spike04'); import numpy as np
from scipy import stats
from lib4 import settle_time, recovery_time
E = pickle.load(open('results/spike04/eval.pkl', 'rb')); CS = json.load(open('results/spike04/calib_sigma.json')); SIG = E['sigma']
TUN = {c: json.load(open(f'results/spike04/tuned_{c}.json')) for c in ('N1', 'N2')}
def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = stats.t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); return m, m-h, m+h
def metr(runs, cond):
    g = lambda f: np.mean([f(r[cond]) for r in runs], 0)
    d = dict(rmse=g(lambda c: c['rmse']), ss=g(lambda c: c['ss_rmse']), en=g(lambda c: c['energy']), sat=g(lambda c: c['sat']))
    d['settle'] = g(lambda c: settle_time(c['emax'])) if cond == 'clean' else None
    d['rec'] = g(lambda c: recovery_time(c['emax'], c['dist'])) if cond == 'dist' else None
    return d
cs = lambda n: 'C1F' if TUN[n]['Cstar'] == 'C1F' else 'C1new'
M = {(n, c, m): metr(E[n][m], c) for n in ('N1', 'N2') for c in ('clean', 'dist') for m in E[n]}
NAME = {'C0': 'C0', 'C1old': 'C1-old', 'C1new': 'C1-new', 'C1F': 'C1-new+LPF', 'RL_C0': 'RL(base C0)', 'RL_C1STAR': 'RL(base C1*)'}
L = []; A = L.append; f = lambda x, p=4: '%.*f +- %.*f' % (p, ci(x)[0], p, ci(x)[0]-ci(x)[1])
A('# spike04 results: do tuned classical gains still tie residual RL once the sim has sensor noise + delay? (H7)\n')
A(f"Test set = seeds 0-49 (locked). sigma_q = {SIG:g} rad (measured velocity = finite difference of noisy angles, sigma_qd = {np.sqrt(2)*SIG/0.005:.3f} rad/s). N1 = noise + 30 ms delay, N2 = noise only. RMSE in rad vs TRUE state, transient included. +- = 95% t-CI over 50 trajectories (RL rows = per-trajectory mean over 5 PPO seeds). settle = clean only, 4.0 = never (censored); recov = dist only. C* (the classical comparator and RL base) is chosen on calibration only: N1 -> {TUN['N1']['Cstar']}, N2 -> {TUN['N2']['Cstar']}.\n")
A('## 0. Noise-strength calibration (calibration seeds 1000-1049, C1-old = spike03a W0 gains Kp x9.24, Kd x4.43)\n')
A('| condition | sigma_q | C0 RMSE clean/dist (ratio vs no noise) | C1-old RMSE clean/dist (ratio vs no noise) |\n|---|---|---|---|')
for cn in ('N2', 'N1'):
    b = CS['cond'][cn]['base']; A(f"| {cn} | 0 | {b['C0'][0]:.4f} / {b['C0'][1]:.4f} | {b['C1old'][0]:.4f} / {b['C1old'][1]:.4f} |")
    for s in CS['grid']:
        r = CS['cond'][cn]['rows'][str(s)]; A(f"| {cn} | {s:g}{' **(chosen)**' if s == SIG and cn == 'N2' else ''} | {r['C0']['rmse'][0]:.4f} / {r['C0']['rmse'][1]:.4f} ({r['C0']['ratio'][0]:.2f}x / {r['C0']['ratio'][1]:.2f}x) | {r['C1old']['rmse'][0]:.4f} / {r['C1old']['rmse'][1]:.4f} ({r['C1old']['ratio'][0]:.2f}x / {r['C1old']['ratio'][1]:.2f}x) |")
A('\nRule (H7.md + Amendment 1): smallest sigma with C1-old >= 1.3x its no-noise RMSE in both clean and dist, in N2. In N1 the delay alone already breaks C1-old (and makes C0 near-unstable: any noise lifts C0 from 0.014 to ~0.22), so the ratio is not usable there.\n')
A('## 1. Tuning / training budget (three axes)\n')
A('| cond | method | free DoF | evals / policies | plant control steps | wall-clock | calib objective | tuned values |\n|---|---|---|---|---|---|---|---|')
for cn in ('N1', 'N2'):
    t = TUN[cn]; A(f"| {cn} | C0 | 0 | 0 | 0 | 0 | {t['C0_calib_obj']:.5f} | spike02 teacher |")
    A(f"| {cn} | C1-old | 0 (spike03a) | 0 | 0 | 0 | {t['C1old_calib_obj']:.5f} | Kp x9.24, Kd x4.43 |")
    c = t['C1']; A(f"| {cn} | C1-new | 2 | {c['evals']} | {c['sim_steps']:,} | {c['wall']:.0f} s | {c['calib_obj']:.5f} | Kp x{c['sp']:.2f}, Kd x{c['sd']:.2f} |")
    c = t['C1F']; A(f"| {cn} | C1-new+LPF | 3 | {c['evals']} | {c['sim_steps']:,} | {c['wall']:.0f} s | {c['calib_obj']:.5f} | Kp x{c['sp']:.2f}, Kd x{c['sd']:.2f}, fc {c['fc']:.1f} Hz |")
    for m in ('RL_C0', 'RL_C1STAR'):
        if m not in E[cn]: A(f'| {cn} | {NAME[m]} | - | NOT RUN (time budget rule, see README) | - | - | - | - |'); continue
        rl = E[cn][m]; A(f"| {cn} | {NAME[m]} | ~19k actor weights, 0 tuned hyperparameters | {len(rl)} seeds | {len(rl)*rl[0]['steps']:,} (2M per policy) | {np.mean([r['train_time'] for r in rl]):.0f} s per seed (loaded machine) | - | residual bound 16 Nm |")
A('\n## 2. Per-condition metrics (test set)\n')
A('| cond | eval | method | RMSE | ss-RMSE (t>0.5s) | settle (s) | recov (s) | sat. frac | energy |\n|---|---|---|---|---|---|---|---|---|')
for n in ('N1', 'N2'):
    for c in ('clean', 'dist'):
        for m in [m for m in NAME if m in E[n]]:
            d = M[(n, c, m)]; A(f"| {n} | {c} | {NAME[m]}{' (=C*)' if m == cs(n) else ''} | {f(d['rmse'])} | {f(d['ss'])} | {f(d['settle'],2) if c=='clean' else '-'} | {f(d['rec'],2) if c=='dist' else '-'} | {f(d['sat'])} | {f(d['en'],0)} |")
A('\n## 3. Paired differences (50 trajectories). impr = mean(A-B)/mean(A): positive = B better than A. CI is on RMSE(B)-RMSE(A)\n')
A('| cond | eval | A -> B | RMSE diff [95% CI] | impr | ss-RMSE impr | energy rel diff |\n|---|---|---|---|---|---|---|')
def pair(n, c, a, b):
    x, y = M[(n, c, a)], M[(n, c, b)]; dd = y['rmse'] - x['rmse']; m, lo, hi = ci(dd)
    return m, lo, hi, -m/x['rmse'].mean(), 1 - y['ss'].mean()/x['ss'].mean(), y['en'].mean()/x['en'].mean() - 1
PAIRS = lambda n: [('C0', 'C1old'), ('C0', 'C1new'), ('C1old', 'C1new'), ('C1new', 'C1F'), ('C1new', 'RL_C1STAR'), ('C0', 'RL_C0'), ('C0', 'C1STAR'), ('C0', 'RL_C1STAR'), ('CSTAR', 'RL_C1STAR'), ('CSTAR', 'RL_C0')]
for n in ('N1', 'N2'):
    for c in ('clean', 'dist'):
        for a, b in PAIRS(n):
            a = cs(n) if a == 'CSTAR' else a; b = cs(n) if b == 'C1STAR' else b
            if a == b or a not in E[n] or b not in E[n]: continue
            m, lo, hi, im, ss, en = pair(n, c, a, b); A(f"| {n} | {c} | {NAME[a]} -> {NAME[b]} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {im*100:+.1f}% | {ss*100:+.1f}% | {en*100:+.1f}% |")
A('\n## 4. H7 verdict (rule in H7.md)\n')
A('WIN cell = RL(base C1*) beats C1* by >= 10% (mean RMSE) with paired CI of (RL - C1*) entirely < 0. Primary = N1 (YES if both N1 cells WIN, PARTIAL if one, else NO); N2 judged the same way (secondary).\n')
A('| cond | eval | C1* | RMSE C1* | RMSE RL(base C1*) | impr | CI of RL-C1* | cell |\n|---|---|---|---|---|---|---|---|')
V = {}
for n in ('N1', 'N2'):
    cells = []
    if 'RL_C1STAR' not in E[n]: V[n] = 'not run'; continue
    for c in ('clean', 'dist'):
        a = cs(n); m, lo, hi, im, _, _ = pair(n, c, a, 'RL_C1STAR'); win = im >= 0.10 and hi < 0; cells.append(win)
        A(f"| {n} | {c} | {NAME[a]} | {M[(n,c,a)]['rmse'].mean():.4f} | {M[(n,c,'RL_C1STAR')]['rmse'].mean():.4f} | {im*100:+.1f}% | [{lo:+.4f}, {hi:+.4f}] | {'WIN' if win else ('RL worse (CI>0)' if lo > 0 else 'no win')} |")
    V[n] = 'YES' if all(cells) else ('PARTIAL' if any(cells) else 'NO')
A(f"\n**H7 primary (N1, noise + delay): {V['N1']}.  N2 (noise only, secondary): {V['N2']}.**\n")
A('Per-seed RL RMSE (RL base C1*, test, clean / dist) for the spread across PPO seeds:\n')
for n in ('N1', 'N2'):
    for m in [m for m in ('RL_C0', 'RL_C1STAR') if m in E[n]]:
        A(f"- {n} {NAME[m]}: " + ', '.join(f"{r['clean']['rmse'].mean():.4f}/{r['dist']['rmse'].mean():.4f}" for r in E[n][m]))
open('results/spike04/table.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[-12:]))
