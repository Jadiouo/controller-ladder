"""Pipeline evaluation of estimator outputs: theta_hat per episode -> identical probe/replay/plan/execute pipeline (Probe3, as R0/R1 of spike13b).
usage: evaluate.py <cal|hold|new> <method,method,...>   theta_hat files: DAT/est/<method>_<set>.npy (rows = seeds of the set in order). Resumable. Results: results/spike14a/pipe_<set>.json"""
import sys, os, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import c14; from c14 import *
SEEDS = dict(cal=CAL_SEEDS, hold=HOLD, new=NEW)
if __name__ == '__main__':
    st, methods = sys.argv[1], sys.argv[2].split(','); D = pickle.load(open(os.path.join(DAT, f'raw_{st}.pkl'), 'rb')); seeds = SEEDS[st]; assert list(D['seed']) == seeds
    f = os.path.join(OUT14, f'pipe_{st}.json'); res = json.load(open(f)) if os.path.exists(f) else {}
    miss = [m for m in methods if m not in res and not os.path.exists(os.path.join(DAT, 'est', f'{m}_{st}.npy'))]; print('SKIPPED (no estimate file):', miss, flush=True)
    todo = [m for m in methods if m not in res and m not in miss]
    if not todo: print('nothing to do'); sys.exit()
    tab = {m: {k: np.load(os.path.join(DAT, 'est', f'{m}_{st}.npy'))[i] for i, k in enumerate(D['key'])} for m in todo}
    tp = os.path.join(DAT, f'tab_{st}_{os.getpid()}.pkl'); pickle.dump(tab, open(tp, 'wb')); pool = mp.Pool(int(os.environ.get('NW', 8)), initializer=c14._init_tab, initargs=(tp,))
    for m in todo:
        t0 = time.time(); r = pool.map(run14, [(m, s) for s in seeds], chunksize=2); res = json.load(open(f)) if os.path.exists(f) else {}
        res[m] = dict(seeds=seeds, success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], inst=[x['inst'] for x in r], diverged=[x['diverged'] for x in r], wall=time.time() - t0)
        json.dump(res, open(f, 'w')); print(st, m, f"{100*np.mean(res[m]['success']):.1f}", 'unstable', np.mean(np.array(res[m]['inst']) > 0), f'{time.time()-t0:.0f}s', flush=True)
    os.remove(tp)
