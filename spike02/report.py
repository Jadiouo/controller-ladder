"""Builds results/spike02/table.md and rmse_by_W.png from raw_*.pkl"""
import pickle, os, glob, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = 'results/spike02/'; WS = ['W0', 'W1', 'W2', 'W3', 'W4']; METH = ['dart', 'dagger', 'rl']; NAME = dict(teacher='Teacher', dart='BC+DART', dagger='DAgger', rl='ResidualPPO')
raw = {}
for W in WS:
    d = {}
    for f in (R + f'raw_bc_{W}.pkl', R + f'raw_rl_{W}.pkl'):
        if os.path.exists(f): d.update(pickle.load(open(f, 'rb')))
    if d: raw[W] = d
def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return float('nan')
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
def seeds_of(d, m): return sorted(s for (mm, s) in d if mm == m)
def metric(d, m, cond, key):
    """per-seed array of trajectory-mean metric, plus (seeds,50) matrix"""
    if m == 'teacher': M = d[('teacher', 0)][cond][key][None]
    else: M = np.array([d[(m, s)][cond][key] for s in seeds_of(d, m)])
    return M
def paired(d, m, cond, key='rmse'):
    S = metric(d, m, cond, key); Tt = metric(d, 'teacher', cond, key)[0]
    diff = S.mean(0) - Tt                        # per-trajectory, student averaged over seeds
    mean_d, h = diff.mean(), ci(diff); rel = -mean_d / Tt.mean()
    win = (diff.mean() + h < 0) and rel >= 0.10
    seed_d = S.mean(1) - Tt.mean()               # per-seed mean difference
    return dict(n_seeds=len(S), diff=mean_d, lo=mean_d - h, hi=mean_d + h, rel=rel, win=win, wf=(diff < 0).mean(),
                wf_pooled=(S < Tt[None]).mean(), seed_win=(seed_d < 0).mean(), seed_ci_hi=seed_d.mean() + ci(seed_d) if len(S) > 1 else float('nan'))
L = ['# spike02 results', '', 'All numbers on the same 50 test trajectories (seeds 0-49). clean = no disturbance, dist = 3 torque pulses/episode. RMSE in rad. +- = 95% CI (t): over the 50 trajectories for the Teacher, over seeds for students (different meaning, as in spike01).', '']
L += ['## 1. Per-condition metrics', '', '| W | cond | method | #seeds | RMSE | ss-RMSE (t>0.5s) | energy (Nm^2 s) | sat. frac |', '|---|---|---|---|---|---|---|---|']
for W, d in raw.items():
    for cond in ['clean', 'dist']:
        for m in ['teacher'] + METH:
            if m != 'teacher' and not seeds_of(d, m): continue
            row = []
            for key in ['rmse', 'ss_rmse', 'energy', 'sat']:
                M = metric(d, m, cond, key)
                v = M.mean(1) if m != 'teacher' else M[0]   # seeds -> per-seed mean ; teacher -> per traj
                f = '%.4f +- %.4f' if key != 'energy' else '%.0f +- %.0f'
                if key == 'sat': f = '%.4f +- %.4f'
                row.append(f % (v.mean(), ci(v) if (m != 'teacher' or True) else 0))
            L.append(f"| {W} | {cond} | {NAME[m]} | {1 if m=='teacher' else len(seeds_of(d,m))} | " + ' | '.join(row) + ' |')
L += ['', '## 2. Paired differences vs Teacher (RMSE; student averaged over seeds per trajectory, diff CI over 50 trajectories)', '',
      '"win" = CI entirely < 0 AND mean improvement >= 10% (hypothesis card). traj-win = share of the 50 trajectories where the seed-averaged student beats the teacher; pooled = share over all (seed, trajectory) pairs; seed-win = share of seeds whose mean RMSE < teacher mean; seed-CI-hi = upper end of the 95% CI of the per-seed mean difference (seed-level robustness, <0 required for a robust win).', '',
      '| W | cond | method | mean diff | 95% CI | improvement | WIN | traj-win | pooled-win | seed-win | seed-CI-hi |', '|---|---|---|---|---|---|---|---|---|---|---|']
P = {}
for W, d in raw.items():
    for cond in ['clean', 'dist']:
        for m in METH:
            if not seeds_of(d, m): continue
            p = P[(W, cond, m)] = paired(d, m, cond)
            L.append(f"| {W} | {cond} | {NAME[m]} | {p['diff']:+.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] | {100*p['rel']:+.1f}% | {'YES' if p['win'] else 'no'} | {p['wf']:.2f} | {p['wf_pooled']:.2f} | {p['seed_win']:.2f} | {p['seed_ci_hi']:+.4f} |")
L += ['', '### Paired differences, steady-state RMSE and energy (same conventions)', '', '| W | cond | method | ss-RMSE diff [CI] (impr.) | energy diff [CI] (rel) |', '|---|---|---|---|---|']
for W, d in raw.items():
    for cond in ['clean', 'dist']:
        for m in METH:
            if not seeds_of(d, m): continue
            a_, b_ = paired(d, m, cond, 'ss_rmse'), paired(d, m, cond, 'energy')
            Te = metric(d, 'teacher', cond, 'energy')[0].mean()
            L.append(f"| {W} | {cond} | {NAME[m]} | {a_['diff']:+.4f} [{a_['lo']:+.4f},{a_['hi']:+.4f}] ({100*a_['rel']:+.1f}%) | {b_['diff']:+.1f} [{b_['lo']:+.1f},{b_['hi']:+.1f}] ({100*b_['diff']/Te:+.2f}%) |")
# verdicts
def anywin(ms, Ws, conds=('clean', 'dist')):
    return [(W, c, m, P[(W, c, m)]) for W in Ws for c in conds for m in ms if (W, c, m) in P and P[(W, c, m)]['win']]
L += ['', '## 3. Hypothesis verdicts (rules fixed in docs/03_hypotheses.md; operationalisation below)', '']
have = lambda m, Ws: all(any((W, c, m) in P for c in ('clean', 'dist')) for W in Ws)
def best(m, Ws): 
    xs = [(P[(W, c, m)]['rel'], W, c) for W in Ws for c in ('clean', 'dist') if (W, c, m) in P]
    return max(xs) if xs else None
w1 = anywin(['dart'], WS)
L.append(f"**H1** (BC+DART never wins in any condition; plain BC not rerun here, spike01 covers it): " + ('SUPPORTED' if not w1 and have('dart', [w for w in WS if w in raw]) else 'FALSIFIED' if w1 else 'UNDECIDABLE') +
         f". Best DART improvement over all W x cond: {best('dart', WS)}. Wins: {[(a,b) for a,b,_,_ in w1]}")
w2 = anywin(['rl'], ['W1', 'W2', 'W3'])
L.append(f"**H2** (residual RL wins >=10% under at least one single weakness W1-W3): " + ('SUPPORTED' if w2 else 'FALSIFIED' if have('rl', ['W1', 'W2', 'W3']) else 'UNDECIDABLE') +
         f". Wins: {[(W,c,'%.1f%%' % (100*p['rel']),'seed-CI-hi %+.4f' % p['seed_ci_hi']) for W,c,_,p in w2]}; best (impr, W, cond) = {best('rl', ['W1','W2','W3'])}")
if 'W0' in raw and seeds_of(raw['W0'], 'rl'):
    r0 = [(c, P[('W0', c, 'rl')]['rel'], P[('W0', c, 'rl')]['win']) for c in ('clean', 'dist')]
    if any(x[2] for x in r0): v = 'FALSIFIED'
    elif all(x[1] < 0.05 for x in r0): v = 'SUPPORTED'
    else: v = 'UNDECIDABLE (improvement between 5% and 10%, or 5-10% without a CI-clean win)'
    L.append(f"**H3** (no weakness: residual RL improvement < 5%): {v}. W0 residual RL improvement: " + ', '.join(f"{c}: {100*r:+.1f}% (win={wn})" for c, r, wn in r0))
else: L.append('**H3**: UNDECIDABLE (no W0 RL data)')
w4 = anywin(['dagger'], WS)
L.append(f"**H4** (DAgger never wins): " + ('SUPPORTED' if not w4 and have('dagger', [w for w in WS if w in raw]) else 'FALSIFIED' if w4 else 'UNDECIDABLE') + f". Best DAgger improvement (impr, W, cond): {best('dagger', WS)}. Wins: {[(a,b) for a,b,_,_ in w4]}")
open(R + 'table.md', 'w').write('\n'.join(L) + '\n')
# figure
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True); cols = dict(teacher='k', dart='tab:blue', dagger='tab:green', rl='tab:red')
for ax, cond in zip(axs, ['clean', 'dist']):
    for j, m in enumerate(['teacher'] + METH):
        xs, ys, es = [], [], []
        for i, W in enumerate(WS):
            if W in raw and (m == 'teacher' or seeds_of(raw[W], m)):
                M = metric(raw[W], m, cond, 'rmse'); v = M[0] if m == 'teacher' else M.mean(1)
                xs.append(i + (j - 1.5) * 0.2); ys.append(v.mean()); es.append(ci(v))
        ax.bar(xs, ys, 0.2, yerr=es, color=cols[m], label=NAME[m], capsize=2)
    ax.set_xticks(range(len(WS))); ax.set_xticklabels(WS); ax.set_yscale('log'); ax.set_title(f'{cond}'); ax.set_ylabel('tracking RMSE (rad)'); ax.grid(axis='y', alpha=.3)
axs[0].legend(); fig.suptitle('spike02: teacher vs students per weakness (error bars: 95% CI; teacher over trajectories, students over seeds)')
fig.tight_layout(); fig.savefig(R + 'rmse_by_W.png', dpi=130)
print('\n'.join(L[-8:]))
