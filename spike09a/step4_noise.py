"""Step 4 (like spike08_diag expB): T-robust under iid gaussian torque noise added to the commanded torque every control step (then clipped to the torque limit).
sigma in {0, 0.1, 0.35, 1.0} Nm per joint; 10 noise reps per sigma>0, per-episode success averaged over reps; CAL 1000-1049 over the whole dist. Paired CI vs sigma=0."""
import time, multiprocessing as mp; from common import *
SIG = [0, 0.1, 0.35, 1.0]; REPS = 10
def run_rep(a):
    seed, W, P, sg, rep = a; return run_one((seed, W, P, sg, rep))
if __name__ == '__main__':
    W = json.load(open(os.path.join(OUT, 'dist.json')))['w']; PR = json.load(open(os.path.join(OUT, 'robust.json')))['params']; S = np.arange(1000, 1050); guard(S); pool = mp.Pool(8); t0 = time.time(); res = {}
    for sg in SIG:
        reps = [0] if sg == 0 else list(range(REPS)); H = []
        for rp in reps:
            P = dict(PR); P['_rep'] = rp; H.append(peval(pool, S, W, P, noise=sg)[0])
        H = np.array(H); res[sg] = H.mean(0); print(sg, res[sg].mean(), f'{time.time()-t0:.0f}s', flush=True)
    rows = [dict(sigma=sg, succ=float(res[sg].mean()), succ_ci=boot_ci(res[sg]), dsucc_vs0=float((res[sg] - res[0]).mean()), dsucc_ci=boot_ci(res[sg] - res[0], seed=1)) for sg in SIG]
    json.dump(dict(rows=rows, reps=REPS, wall_s=time.time() - t0), open(os.path.join(OUT, 'noise.json'), 'w'), indent=1); print(rows)
