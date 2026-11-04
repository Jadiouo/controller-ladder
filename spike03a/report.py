"""-> results/spike03a/table.md"""
import sys, json, pickle; sys.path.insert(0, 'spike03a'); import numpy as np
from scipy import stats
from lib import settle_time, recovery_time
cal = json.load(open('results/spike03a/calib_W.json')); tun = json.load(open('results/spike03a/tuned_W0_W3.json')); E = pickle.load(open('results/spike03a/eval.pkl', 'rb'))
def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = stats.t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); return m, m-h, m+h
def metr(runs, cond):
    """list of eval dicts (seeds) -> dict of per-trajectory arrays (seed-averaged)"""
    g = lambda f: np.mean([f(r[cond]) for r in runs], 0)
    d = dict(rmse=g(lambda c: c['rmse']), ss=g(lambda c: c['ss_rmse']), en=g(lambda c: c['energy']), sat=g(lambda c: c['sat']))
    d['settle'] = g(lambda c: settle_time(c['emax'])) if cond == 'clean' else None
    d['rec'] = g(lambda c: recovery_time(c['emax'], c['dist'])) if cond == 'dist' else None
    return d
M = {(w, c, m): metr(E[w][m], c) for w in E for c in ('clean', 'dist') for m in E[w]}
L = []; A = L.append
A('# spike03a results: tuned classical opponents vs residual PPO (H5), and H2 re-judged under the v2 formula\n')
A('Test set = seeds 0-49 (locked), same refs/init errors/pulses for every method. RMSE in rad (transient included). +- = 95% t-CI over the 50 trajectories (RL = per-trajectory mean over its 5 PPO seeds). settle = first time after which max-joint error stays < 0.02 rad (clean only; 4.0 s = never, censored). recov = mean over pulses of time from pulse end until error stays < 0.02 rad (<=1 s window, dist only; 0 = never left the band). sat = fraction of control steps with any joint at the cap. energy = int tau^2 dt (Nm^2 s).\n')
A('## 0. Frozen protocol: W1/W2 strength re-decided on the calibration set only (seeds 1000-1049)\n')
A('Rule: mildest strength where the untuned teacher C0 has RMSE >= 1.3x its W0 value in BOTH clean and dist (calibration W0: clean %.4f, dist %.4f).\n' % tuple(cal['W0']))
A('| weakness | spike02 value (chosen after looking at test seeds) | new value (calibration only) | calib ratio clean / dist at new value |\n|---|---|---|---|')
nc, nd = str(cal['new_cap']), str(cal['new_delay_steps'])
A(f"| W1 torque cap | 32 Nm (first 27, then raised after a teacher-only look at test) | **{cal['new_cap']} Nm** | {cal['cap'][nc][0]:.2f} / {cal['cap'][nc][1]:.2f} ({int(cal['cap'][nc][2])}/50 calib trajectories RMSE>0.1; cap 28 gives only {cal['cap']['28'][0]:.2f}/{cal['cap']['28'][1]:.2f}, a cliff) |")
A(f"| W2 delay | 30 ms (6 steps; spec 10 ms did not hurt) | **{int(cal['new_delay_steps'])*5} ms ({cal['new_delay_steps']} steps)** | {cal['delay'][nd][0]:.2f} / {cal['delay'][nd][1]:.2f} (5 steps = 25 ms: {cal['delay']['5'][0]:.2f}/{cal['delay']['5'][1]:.2f}) |")
A('\nW3 (Coulomb 1.5 Nm + 0.3 viscous) unchanged from spike02 (not re-decided; the task only asks to record W1/W2). Students were NOT re-run for the new W1 value, so W1/W2 are not evaluated in this spike.\n')
A('## 1. Tuning budget (calibration set only, objective = 0.5*RMSE_clean + 0.5*RMSE_dist, transient included)\n')
A('| W | method | free params | evals | plant control steps (1 eval = 100 rollouts x 800 = 80k) | wall-clock | calib objective | tuned values |\n|---|---|---|---|---|---|---|---|')
for w in ('W0', 'W3'):
    t = tun[w]; A(f"| {w} | C0 | 0 | 0 | 0 | 0 | {t['C0_calib_obj']:.5f} | Kp=400, Kd (LQR default) |")
    c = t['C1']; A(f"| {w} | C1 | 2 | {c['evals']} | {c['sim_steps']:,} | {c['wall']:.0f} s | {c['calib_obj']:.5f} | Kp x{c['sp']:.2f}, Kd x{c['sd']:.2f} |")
    c = t['C2']; A(f"| {w} | C2 | 3 (+ the same 2 gains re-tuned) | {c['evals']} | {c['sim_steps']:,} | {c['wall']:.0f} s | {c['calib_obj']:.5f} | Kp x{c['sp']:.2f}, Kd x{c['sd']:.2f}, Ki={c['ki']:.0f} 1/s^2, extra torque clamp +-16 Nm |")
    rl = E[w]['RL']; A(f"| {w} | residual PPO (spike02) | ~19k net weights, 0 tuned hyperparameters | - | 2,000,000 x 5 seeds (10,000,000 total; 2M per policy) | {np.mean([r['train_time'] for r in rl]):.0f} s per seed (spike02 log, loaded machine) | - | - |")
A('\n## 2. Per-condition metrics (test set)\n')
A('| W | cond | method | RMSE | ss-RMSE (t>0.5s) | settle (s) | recov (s) | sat. frac | energy |\n|---|---|---|---|---|---|---|---|---|')
f = lambda x, p=4: '%.*f +- %.*f' % (p, ci(x)[0], p, ci(x)[0]-ci(x)[1])
for w in ('W0', 'W3'):
    for c in ('clean', 'dist'):
        for m in ('C0', 'C1', 'C2', 'RL'):
            d = M[(w, c, m)]; A(f"| {w} | {c} | {m if m!='RL' else 'ResidualPPO'} | {f(d['rmse'])} | {f(d['ss'])} | {f(d['settle'],2) if c=='clean' else '-'} | {f(d['rec'],2) if c=='dist' else '-'} | {f(d['sat'])} | {f(d['en'],0)} |")
A('\n## 3. Paired differences (50 trajectories). impr = mean(A-B)/mean(A): positive = B better than A\n')
A('| W | cond | A -> B | RMSE diff (B-A) [95% CI] | impr | ss-RMSE impr | energy rel diff |\n|---|---|---|---|---|---|---|')
cell = {}
for w in ('W0', 'W3'):
    for c in ('clean', 'dist'):
        for a, b in (('C0', 'C1'), ('C1', 'C2'), ('C0', 'C2'), ('C0', 'RL'), ('C1', 'RL'), ('C2', 'RL')):
            x, y = M[(w, c, a)], M[(w, c, b)]; dd = y['rmse'] - x['rmse']; m, lo, hi = ci(dd); imp = -m / x['rmse'].mean()
            cell[(w, c, a, b)] = (m, lo, hi, imp)
            A(f"| {w} | {c} | {a} -> {b} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {imp*100:+.1f}% | {-(y['ss']-x['ss']).mean()/x['ss'].mean()*100:+.1f}% | {(y['en']-x['en']).mean()/x['en'].mean()*100:+.1f}% |")
A('\n## 4. H5 verdict (rule in docs/03_hypotheses.md v2)\n')
A('Rule operationalisation (fixed before looking at test numbers): gap = (RMSE_C2 - RMSE_RL)/RMSE_C2 per cell, paired CI over 50 trajectories. Cell = "TUNING" if gap <= 10% (or C2 better); "RL-GAIN" if gap > 10% and CI of RL-C2 entirely < 0; else "unclear". H5 supported if all four cells (W0/W3 x clean/dist) are TUNING; refuted if all four are RL-GAIN; otherwise mixed (reported per cell).\n')
A('| W | cond | RMSE C2 | RMSE RL | gap | CI of RL-C2 | cell |\n|---|---|---|---|---|---|---|')
lab = []
for w in ('W0', 'W3'):
    for c in ('clean', 'dist'):
        m, lo, hi, imp = cell[(w, c, 'C2', 'RL')]
        l = 'TUNING' if imp <= 0.10 else ('RL-GAIN' if hi < 0 else 'unclear'); lab.append(l)
        A(f"| {w} | {c} | {M[(w,c,'C2')]['rmse'].mean():.4f} | {M[(w,c,'RL')]['rmse'].mean():.4f} | {imp*100:+.1f}% | [{lo:+.4f}, {hi:+.4f}] | {l} |")
v5 = 'SUPPORTED (RL advantage mostly = untuned teacher)' if all(l == 'TUNING' for l in lab) else ('REFUTED (RL keeps a gain tuned classical control cannot reach)' if all(l == 'RL-GAIN' for l in lab) else 'MIXED (see per-cell labels)')
A(f"\n**H5: {v5}.** Cells: {lab}\n")
A('For context: what tuning alone removes from the spike02 "RL win" over C0: C0->C2 impr vs C0->RL impr (RMSE):\n')
A('| W | cond | C0->C1 | C0->C2 | C0->RL |\n|---|---|---|---|---|')
for w in ('W0', 'W3'):
    for c in ('clean', 'dist'):
        A(f"| {w} | {c} | {cell[(w,c,'C0','C1')][3]*100:+.1f}% | {cell[(w,c,'C0','C2')][3]*100:+.1f}% | {cell[(w,c,'C0','RL')][3]*100:+.1f}% |")
# ---- H2 v2 ----
A('\n## 5. H2 re-judged with the v2 formula (spike02 numbers: teacher and RL per-trajectory RMSE, 5 RL seeds averaged; no new runs)\n')
A('extra contribution of weakness k = (Teacher_Wk - Teacher_W0) - (RL_Wk - RL_W0), per trajectory (same seeds), paired over 50. Operationalisation (mine; v2 gives the formula but no threshold): H2 "weakness-attributable win" in cell if extra contribution CI entirely > 0 AND mean extra >= 10% of mean Teacher_Wk RMSE. W4 (combined) is shown for information, not counted (card says single weaknesses). W1 uses the spike02 cap 32 Nm and W2 30 ms (students were not re-trained on the new calibration values).\n')
A('| W | cond | Teacher_Wk - Teacher_W0 | RL_Wk - RL_W0 | extra [95% CI] | extra / Teacher_Wk | counts? | pass |\n|---|---|---|---|---|---|---|---|')
S = {w: pickle.load(open(f'results/spike02/raw_rl_{w}.pkl', 'rb')) for w in ('W0', 'W1', 'W2', 'W3', 'W4')}
def tr(w, c):
    t = S[w][('teacher', 0)][c]['rmse']; rl = np.mean([S[w][('rl', s)][c]['rmse'] for s in range(5)], 0); return t, rl
passes = []
for w in ('W1', 'W2', 'W3', 'W4'):
    for c in ('clean', 'dist'):
        t0, r0 = tr('W0', c); tk, rk = tr(w, c); ex = (tk - t0) - (rk - r0); m, lo, hi = ci(ex); frac = m / tk.mean()
        ok = lo > 0 and frac >= 0.10
        if w != 'W4': passes.append(ok)
        A(f"| {w} | {c} | {(tk-t0).mean():+.4f} | {(rk-r0).mean():+.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {frac*100:+.1f}% | {'yes' if w!='W4' else 'info'} | {'YES' if ok else 'no'} |")
A(f"\n**H2 (v2): {'SUPPORTED' if any(passes) else 'FALSIFIED'}** (passing single-weakness cells: {sum(passes)}/6).\n")
A('Spike02 original verdict for comparison: SUPPORTED (RL beat teacher by >=10% in 5 of 6 single-weakness cells, but it also beat it by 17%/45% at W0, which is what v2 corrects for).\n')
open('results/spike03a/table.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
