"""Step 4: dose curve on holdout (seeds 1100-(1100+N-1)), conditions x0.5, abc (x1), x2, a, b, c (single factors at x1). Methods listed in argv[2] (comma list; lest_<var> expands to the training seeds in argv[3]).
Resumable per (method, cond). n per level = argv[1] (>=200 required; 150 allowed only after the documented cut)."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
if __name__ == '__main__':
    N = int(sys.argv[1]); S = list(range(1100, 1100 + N)); guard(S); names0 = sys.argv[2].split(','); tseeds = sys.argv[3].split(',') if len(sys.argv) > 3 else ['0', '1', '2']
    names = []
    for n in names0: names += [f'{n}#{s}' for s in tseeds] if n.startswith('lest_') else [n]
    f = os.path.join(OUT, 'dose.json'); res = json.load(open(f)) if os.path.exists(f) else {}; pool = mp.Pool(8); t0 = time.time(); H1 = json.load(open(os.path.join(OUT, 'holdout1.json')))
    for cond in ['abc', 'x0.5', 'x2', 'a', 'b', 'c']:
        for n in names:
            k = f'{n}|{cond}'
            if k in res and len(res[k]['success']) >= N: continue
            if cond == 'abc' and n in H1 and len(H1[n]['success']) >= N:
                res[k] = dict(success=H1[n]['success'][:N], min_dist=H1[n]['min_dist'][:N], inst=H1[n]['inst'][:N], diverged=H1[n]['diverged'][:N], source='holdout1 (first N seeds)')
            else:
                r = pool.map(run10, [(spec_of(n), s, None, cond) for s in S], chunksize=1); res[k] = dict(success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], inst=[x['inst'] for x in r], diverged=[x['diverged'] for x in r], wall=time.time() - t0)
            print(k, np.mean(res[k]['success']), 'unstable', np.mean(np.array(res[k]['inst']) > 0), f'{time.time()-t0:.0f}s', flush=True); json.dump(res, open(f, 'w')); t0 = time.time()
