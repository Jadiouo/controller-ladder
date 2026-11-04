"""Evaluate the CAL-selected checkpoint(s) of a run on a seed set (paired with the zero-residual init) -> json with paired bootstrap CI.
usage: evalsel.py TAG SEEDSET [bias]   SEEDSET in {gate (1100-1299), holdout (1100-1599), fresh (700400-700599)}"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rl13 import *
SETS = dict(gate=list(range(1100, 1300)), holdout=list(range(1100, 1600)), fresh=list(range(700400, 700600)))
def boot(d, B=10000, seed=0):
    r = np.random.default_rng(seed); n = len(d); m = d[r.integers(0, n, (B, n))].mean(1); return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
if __name__ == '__main__':
    tag, sset = sys.argv[1], sys.argv[2]; bias = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; bl = None if bias == 0 else bias_vec(bias).tolist()
    runs = pickle.load(open(os.path.join(RUNS, tag + '.pkl'), 'rb')); seeds = SETS[sset]; pool = mp.Pool(8); outp = os.path.join(ROOT, f'results/spike13a/eval_{tag}_{sset}.json')
    cache = os.path.join(ROOT, f'results/spike13a/init_{sset}_b{bias:g}_cap12.npy')       # zero residual: independent of the cap
    if os.path.exists(cache): init = np.load(cache)
    else: init = eval_set(pool, None, 'ppo', seeds, 12.0, bl)['per']; np.save(cache, init)
    out = dict(tag=tag, set=sset, n=len(seeds), bias=bias, init=float(init.mean()), seeds={}); per = []
    for s, r in sorted(runs.items()):
        cap = r['args']['cap']; e = eval_set(pool, r['selected_layers'], r['kind'], seeds, cap, bl); d = e['per'] - init; m, lo, hi = boot(d); per.append(e['per'])
        out['seeds'][s] = dict(succ=e['succ'], diff=m, ci=[lo, hi], res_mean=e['res_mean'], res_max=e['res_max'], cal_sel=r['selected']['succ'], cal_init=r['init']['succ'], selected_it=r['selected']['it'])
        print(tag, sset, 'seed', s, 'succ', e['succ'], 'init', init.mean(), 'diff %.3f [%.3f, %.3f]' % (m, lo, hi), 'res_mean', e['res_mean'], flush=True)
    P = np.mean(per, 0); m, lo, hi = boot(P - init); out['pooled'] = dict(succ=float(P.mean()), diff=m, ci=[lo, hi]); print(tag, sset, 'pooled over', len(per), 'seeds: succ %.3f diff %.3f [%.3f, %.3f]' % (P.mean(), m, lo, hi))
    json.dump(out, open(outp, 'w'), indent=1); np.save(outp.replace('.json', '.npy'), np.array(per))
