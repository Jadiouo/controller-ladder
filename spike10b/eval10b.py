"""LOCKED evaluation. argv[1] in {TESTB, TEST}: TESTB = seeds 20000-20499 (n=500, primary endpoint), TEST = seeds 0-49. RUN ONCE per set after freeze10b.py.
Refuses to run if results/spike10b/raw_<set>.pkl exists. Appends one line to results/.test_access.log before touching the seeds (a crash-resume from the .partial file does not log again).
All methods run on the same seeds; the true plant theta of every seed is asserted identical across methods."""
import sys, os, json, pickle, time, datetime, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
SETS = dict(TESTB=(20000, 20499), TEST=(0, 49))
def job(a):
    kind, spec, s, lo_hi = a
    if kind == 'cl': return run10((spec, s, None, 'abc', (lo_hi,)))
    base, layers = spec; return run_residual_episode10((base, layers, s, None, 'abc', (lo_hi,)))
if __name__ == '__main__':
    setn = sys.argv[1]; lo, hi = SETS[setn]; S = list(range(lo, hi + 1)); OUTP = os.path.join(OUT, f'raw_{setn}.pkl'); PART = OUTP + '.partial'
    assert not os.path.exists(OUTP), f'{setn} already evaluated once; refusing to run again'
    FIN = pickle.load(open(os.path.join(OUT, 'final_methods.pkl'), 'rb')); best = FIN['best_classical']; bspec = spec_of(best)
    out = pickle.load(open(PART, 'rb')) if os.path.exists(PART) else {'_best': best}
    if not os.path.exists(PART):
        with open(os.path.join(ROOT, 'results/.test_access.log'), 'a') as f: f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\tspike10b/eval10b.py\tH11/H12 single locked {setn} run, all methods, seeds={lo}-{hi}\n")
    pool = mp.Pool(8); t0 = time.time(); jobs = []
    def addcl(n): jobs.append((f'T-{n}', 'cl', spec_of(n)))
    addcl(best)                                                       # priority order (resilience to wall-clock cuts): primary comparison first
    for s in range(3): jobs.append((f'L-est-full#{s}', 'cl', spec_of(f'lest_full#{s}')))
    for s, d in sorted(FIN['ppo'].items()): jobs.append((f'R-PPO-hist#{s}', 'res', (bspec, d['layers'])))
    for n in ['oracle', 'adapt', 'adapt2', 'nom', 'robust']: addcl(n)
    for var in ('eff', 'ood'):
        for s in range(3): jobs.append((f'L-est-{var}#{s}', 'cl', spec_of(f'lest_{var}#{s}')))
    for n in ['adapt3_ood', 'adapt3_full', 'adapt3_grad']:
        if f'T-{n}' not in [j[0] for j in jobs]: addcl(n)
    for name, kind, spec in jobs:
        if name in out: continue
        t = time.time(); r = pool.map(job, [(kind, spec, s, (lo, hi)) for s in S], chunksize=1); out[name] = {k: np.array([x[k] for x in r]) for k in r[0] if k != 'theta'}; out[name]['theta'] = np.array([x['theta'] for x in r]); out[name]['wall'] = time.time() - t
        print(name, 'succ', out[name]['success'].mean(), 'unstable', np.mean(out[name]['inst'] > 0), f'({time.time()-t0:.0f}s)', flush=True); pickle.dump(out, open(PART, 'wb'))
    th = [v['theta'] for k, v in out.items() if k != '_best']; assert all(np.allclose(c, th[0]) for c in th), 'plant errors differ across methods'
    out['best-classical'] = out[f'T-{best}']; pickle.dump(out, open(OUTP, 'wb')); os.remove(PART); print('done; identical plant (theta) across all methods: OK')
