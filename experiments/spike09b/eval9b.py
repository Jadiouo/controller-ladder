"""LOCKED TEST evaluation (seeds 0-49) for spike09b (H9w / H10w). RUN ONCE after everything is frozen; refuses to run if results/spike09b/test_raw.pkl exists.
Appends one line to results/.test_access.log before touching TEST seeds. Chain errors identical across methods (asserted per seed).
Methods: T-nom, T-robust, T-adapt, T-oracle, best-classical (alias of the holdout-selected one), L-est#s, R-PPO-hist#s, R-oracle#s (residual on best-classical; checkpoint chosen on CAL), all frozen in results/spike09b/final_methods.json."""
import sys, os, json, pickle, time, datetime, hashlib, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c9b import *
OUTP = os.path.join(OUT, 'test_raw.pkl'); TEST = list(range(50))
def job(a):
    kind, spec, s = a
    if kind == 'cl':
        if isinstance(spec, tuple) and spec[0] == 'oracle': spec = ('oracle', tvec(true_chain(s)).tolist())
        return run_spec((spec, s, None, True))
    base, hist, oracle, state = spec; return run_residual_episode((base, hist, oracle, state, s, None, True))
if __name__ == '__main__':
    assert not os.path.exists(OUTP), 'TEST already evaluated once; refusing to run again'
    FINAL = pickle.load(open(os.path.join(OUT, 'final_methods.pkl'), 'rb')); best = FINAL['best_classical']
    with open(os.path.join(ROOT, 'results/.test_access.log'), 'a') as f: f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\tspike09b/eval9b.py\tH9w/H10w single locked TEST run, all methods, seeds=0-49\n")
    pool = mp.Pool(8); out = {'_best': best}; t0 = time.time(); jobs = []
    for n in ['nom', 'robust', 'adapt']: jobs.append((f'T-{n}', 'cl', n))
    jobs.append(('T-oracle', 'cl', ('oracle', None)))
    for s, st in sorted(FINAL['lest'].items()): jobs.append((f'L-est#{s}', 'cl', ('est', f'lest{s}_' + hashlib.md5(np.concatenate([w.ravel() for w, b in st['layers']]).tobytes()).hexdigest()[:8], MLPEst(st))))
    for m in ('R-PPO-hist', 'R-oracle'):
        for s, d in sorted(FINAL[m].items()): jobs.append((f'{m}#{s}', 'res', (best, True, m == 'R-oracle', d['layers'])))
    for name, kind, spec in jobs:
        t = time.time(); r = pool.map(job, [(kind, spec, s) for s in TEST], chunksize=1); out[name] = {k: np.array([x[k] for x in r]) for k in r[0] if k != 'chain'}; out[name]['chain'] = np.array([x['chain'] for x in r]); out[name]['wall'] = time.time() - t
        print(name, 'succ', out[name]['success'].mean(), 'md', np.nanmean(out[name]['min_dist']), f'({time.time()-t0:.0f}s)', flush=True); pickle.dump(out, open(OUTP + '.partial', 'wb'))
    ch = [v['chain'] for k, v in out.items() if k != '_best']; assert all(np.allclose(c, ch[0]) for c in ch), 'plant errors differ across methods'
    out['best-classical'] = out[f'T-{best}']; pickle.dump(out, open(OUTP, 'wb')); os.remove(OUTP + '.partial'); print('done; identical chain errors across all methods: OK')
