import pickle, json, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
res = pickle.load(open('results/spike01/raw.pkl', 'rb'))
ERRS = sorted({k[0] for k in res}); SEEDS = sorted({k[2] for k in res if k[1] == 'bc'})
def ci(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), (stats.t.ppf(0.975, n-1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
def stat(e, m, cond, metric):
    if m == 'teacher':   # deterministic: one run; CI over the 50 test trajectories
        return ci(res[(e,'teacher',0)][cond][metric])
    seeds = [s for s in SEEDS if (e,m,s) in res]   # CI over training seeds (each seed value = mean over 50 test trajs)
    return ci([res[(e,m,s)][cond][metric].mean() for s in seeds])
fmt = lambda v, p: f"{v[0]:.{p}f} ± {v[1]:.{p}f}"
L = ["# Spike01 results (mean ± 95% CI)\n",
     "Teacher CI: t-interval over the 50 test trajectories (teacher is deterministic). Student CI: t-interval over training seeds "
     f"(n={len(SEEDS)}; each seed's value = mean over the same 50 test trajectories). e = nominal payload mass error relative to the true 1.5 kg.\n"]
for cond, name in [('clean', 'No disturbance'), ('dist', 'With random torque pulses')]:
    L.append(f"\n## {name}\n\n| e | Teacher RMSE (rad) | BC RMSE | BC+DART RMSE | Teacher energy (Nm²s) | BC energy | BC+DART energy |\n|---|---|---|---|---|---|---|")
    for e in ERRS:
        L.append(f"| {e:+.0%} | " + " | ".join([fmt(stat(e,m,cond,'rmse'),4) for m in ['teacher','bc','dart']] + [fmt(stat(e,m,cond,'energy'),1) for m in ['teacher','bc','dart']]) + " |")
L.append("\n## Paired difference: student RMSE − teacher RMSE (same 50 test trajs), mean ± 95% CI over seeds; and fraction of (seed,traj) pairs where student < teacher\n\n| cond | e | method | diff | frac student better |\n|---|---|---|---|---|")
for cond in ['clean', 'dist']:
    for e in ERRS:
        tr = res[(e,'teacher',0)][cond]['rmse']
        for m in ['bc','dart']:
            seeds = [s for s in SEEDS if (e,m,s) in res]
            d = [res[(e,m,s)][cond]['rmse'].mean() - tr.mean() for s in seeds]
            fr = np.mean([(res[(e,m,s)][cond]['rmse'] < tr).mean() for s in seeds])
            L.append(f"| {cond} | {e:+.0%} | {m} | {fmt(ci(d),4)} | {fr:.2f} |")
L.append("\n## Training diagnostics (mean over seeds): final normalized MSE, rmse of the data-collection rollouts\n\n| e | method | final loss | collection RMSE |\n|---|---|---|---|")
for e in ERRS:
    for m in ['bc','dart']:
        ss = [s for s in SEEDS if (e,m,s) in res]
        L.append(f"| {e:+.0%} | {m} | {np.mean([res[(e,m,s)]['final_loss'] for s in ss]):.4f} | {np.mean([res[(e,m,s)]['train_rmse'] for s in ss]):.4f} |")
open('results/spike01/table.md', 'w').write("\n".join(L) + "\n")
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, cond, name in zip(axs, ['clean','dist'], ['No disturbance','With torque pulses']):
    for m, lab, c in [('teacher','Teacher','k'),('bc','BC','tab:blue'),('dart','BC+DART','tab:orange')]:
        v = [stat(e,m,cond,'rmse') for e in ERRS]
        ax.errorbar(np.array(ERRS)*100, [x[0] for x in v], yerr=[x[1] for x in v], marker='o', capsize=3, label=lab, color=c)
    ax.set_yscale('log'); ax.set_xlabel('nominal payload error e (%)'); ax.set_ylabel('tracking RMSE (rad)'); ax.set_title(name); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig('results/spike01/rmse_vs_error.png', dpi=130)
print("\n".join(L))
