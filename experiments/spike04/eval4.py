"""Step 4: locked test set (seeds 0-49), N1 and N2: C0, C1-old, C1-new, C1-new+LPF (all re-evaluated here) + residual PPO (per-seed test evals saved by run_rl4.py at the end of training). Also checks that the measurement stream is identical across methods."""
import sys, json, pickle; sys.path.insert(0, 'experiments/spike04'); from lib4 import *
SIG = json.load(open('results/spike04/calib_sigma.json'))['sigma']; old = json.load(open('results/spike03a/tuned_W0_W3.json'))['W0']['C1']
# measurement check: what a controller sees must be true state + the per-seed noise array, the same array for every method (nz depends only on the trajectory seed)
w = COND['N1']; P = sample_refs(TEST); ie = sample_init_err(TEST); nz = make_noise(TEST, SIG); e = NoisyEnv(50, w); e.reset(P, ie, None, nz); mx = 0.0
for k in range(60):
    qm, qdm = e.state(); qt, _ = e.state_true(); mx = max(mx, np.abs(qm - qt - nz[:, k]).max()); e.step(np.zeros((50, 2)))
print('max |(q_meas - q_true) - noise array| over 60 steps:', mx, '; noise std', nz.std(), flush=True); assert mx < 1e-12
out = dict(sigma=SIG)
import os
for cn in ('N1', 'N2'):
    w = COND[cn]; t = json.load(open(f'results/spike04/tuned_{cn}.json')); R = {}
    R['C0'] = [evaluate4(lambda: Ctrl4(), w, TEST, SIG)]
    R['C1old'] = [evaluate4(lambda: Ctrl4(old['sp'], old['sd']), w, TEST, SIG)]
    R['C1new'] = [evaluate4(lambda: Ctrl4(t['C1']['sp'], t['C1']['sd']), w, TEST, SIG)]
    R['C1F'] = [evaluate4(lambda: Ctrl4(t['C1F']['sp'], t['C1F']['sd'], t['C1F']['fc']), w, TEST, SIG)]
    for b, nm in (('C0', 'RL_C0'), ('C1STAR', 'RL_C1STAR')):
        fn = f'results/spike04/raw_rl_{cn}_{b}.pkl'
        if not os.path.exists(fn): continue
        raw = pickle.load(open(fn, 'rb'))
        R[nm] = []
        for k in sorted(raw):
            v = raw[k]; r = {c: v[c] for c in ('clean', 'dist')}; r['train_time'] = v['train_time']; r['steps'] = v['steps']; r['maxres'] = v.get('maxres'); R[nm].append(r)
    out[cn] = R; print(cn, 'done', {m: len(v) for m, v in R.items()}, flush=True)
pickle.dump(out, open('results/spike04/eval.pkl', 'wb'))
