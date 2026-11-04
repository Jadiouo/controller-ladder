"""spike12a driver. python run12.py <stage> ... ; data in .scratch/s12, results in results/spike12a. Training seeds >= 500000; CAL 1000-1099 for selection; final eval 700200-700399. No TEST seed."""
import sys, os, json, time, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c12 import *
CAL = list(range(1000, 1100)); FINAL = list(range(700200, 700400))
def J(name, obj=None):
    p = os.path.join(OUT12, name)
    if obj is None: return json.load(open(p)) if os.path.exists(p) else {}
    json.dump(obj, open(p, 'w'), indent=1)
def parse(spec):
    p = spec.split(':'); return p[0], p[1], (int(p[2]) if len(p) > 2 and p[2] else None), (p[3] if len(p) > 3 else None)
def ci(x): return list(boot_ci(np.asarray(x, float)))
def pack(rs, meta):
    X = np.concatenate([r['X'] for r in rs]); R = np.concatenate([r['R'] for r in rs]); n = [len(r['X']) for r in rs]; ends = np.repeat(np.cumsum(n) - 1, n)
    return dict(X=X, R=R, ends=ends.astype(np.int64), seed=np.repeat([m[0] for m in meta], n), plan=np.repeat([m[1] for m in meta], n), succ=np.repeat([r['success'] for r in rs], n))
def evalset(pool, model, seeds, cal, **kw):
    r = pool.map(run_ep12, [dict(seed=s, rep=None, model=model, cal=cal, **kw) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r])
    return dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist(), min_dist=float(np.mean([x['min_dist'] for x in r])), diverged=float(np.mean([x['diverged'] for x in r])), unstable=float(np.mean([x['inst'] > 0 for x in r])))
if __name__ == '__main__':
    stage = sys.argv[1]; pool = mp.Pool(8); t0 = time.time()
    if stage == 'data':
        NS = int(sys.argv[2]); s0 = int(sys.argv[3]); tag = sys.argv[4]; K = 4
        for canon, nm in ((False, 'multi'), (True, 'canon')):
            f = os.path.join(SD12, f'D_{nm}_{tag}.npz')
            if os.path.exists(f): continue
            meta = [(s0 + i, j) for i in range(NS) for j in range(K)]
            rs = pool.map(run_ep12, [dict(seed=s, rep=0, plan_j=j, canon=canon, record=True) for s, j in meta], chunksize=1); d = pack(rs, meta); np.savez(f, **d)
            res = J('data.json'); res[f'{nm}_{tag}'] = dict(scenarios=NS, plans_per_scenario=K, episodes=len(rs), samples=int(len(d['X'])), teacher_succ=float(np.mean([r['success'] for r in rs])), wall_s=time.time() - t0); J('data.json', res); print(nm, res[f'{nm}_{tag}'], flush=True)
    elif stage == 'cal':
        # argv: ckpt paths (name:path[:steps]) ; evaluates on CAL (n=100) and stores to cal.json
        res = J('cal.json')
        for spec in sys.argv[2:]:
            name, path, st, nz = parse(spec); m = evalset(pool, path, CAL, True, steps=st, noise=nz); m.pop('per_seed'); res = J('cal.json'); res[name] = m; J('cal.json', res); print(name, m['succ'], f'{time.time()-t0:.0f}s', flush=True)
    elif stage == 'final':
        # argv: name:path[:steps] ... ; baselines added with 'baselines'
        res = J('final.json')
        if 'oracle_canon' not in res:
            r = pool.map(run_ep, [dict(seed=s, rep=None, cond='abc', student=None, H=6, stride=5, label='fixed', canon=True) for s in FINAL], chunksize=1); su = np.array([x['success'] for x in r]); res['oracle_canon'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist())
            r = pool.map(run10, [(spec_of('adapt3_full'), s, None, 'abc', ((700000, 800000),)) for s in FINAL], chunksize=1); su = np.array([x['success'] for x in r]); res['adapt3full'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist(), diverged=float(np.mean([x['diverged'] for x in r])))
            r = pool.map(run10, [('nom', s, None, 'abc', ((700000, 800000),)) for s in FINAL], chunksize=1); su = np.array([x['success'] for x in r]); res['nom'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist()); J('final.json', res); print({k: v['succ'] for k, v in res.items()}, flush=True)
        for spec in sys.argv[2:]:
            name, path, st, nz = parse(spec); m = evalset(pool, path, FINAL, False, steps=st, noise=nz); res = J('final.json'); res[name] = m; J('final.json', res); print(name, m['succ'], m['ci'], f'{time.time()-t0:.0f}s', flush=True)
    elif stage == 'spread':
        # closed-loop plan spread: 40 CAL scenarios x 4 noise salts; RMS over time of executed-reference differences between salts (rad), compared with teacher within-scenario spread
        res = J('spread.json')
        for spec in sys.argv[2:]:
            name, path, st, nz = parse(spec); S = CAL[:40]
            r = pool.map(run_ep12, [dict(seed=s, rep=None, model=path, cal=True, steps=st, noise=nz, salt=j, log_ref=True) for s in S for j in range(4)], chunksize=1)
            Q = [r[i * 4 + j]['QR'] for i in range(len(S)) for j in range(4)]; w = []
            for i in range(len(S)):
                for a_ in range(4):
                    for b_ in range(a_ + 1, 4):
                        qa, qb = Q[i * 4 + a_], Q[i * 4 + b_]; n_ = min(len(qa), len(qb)); w.append(np.sqrt(((qa[:n_] - qb[:n_]) ** 2).mean()))
            res = J('spread.json'); res[name] = dict(rms_between_noise_draws_rad=float(np.mean(w)), n_scen=len(S)); J('spread.json', res); print(name, res[name], flush=True)
