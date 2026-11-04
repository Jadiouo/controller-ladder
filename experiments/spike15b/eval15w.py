"""Evaluate CAL-selected checkpoints of a spike15b S6 run on a seed set, paired with the zero-residual init. usage: eval15w.py TAG SET [bias]  SET in gate|holdout|fresh"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rl15w import *
def boot(d, B=10000, seed=0):
    r = np.random.default_rng(seed); n = len(d); m = d[r.integers(0, n, (B, n))].mean(1); return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
if __name__ == '__main__':
    tag, sset = sys.argv[1], sys.argv[2]; bias = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; bl = None if bias == 0 else bias_vec(bias).tolist()
    runs = pickle.load(open(os.path.join(RUNS, tag + '.pkl'), 'rb')); seeds = SETS[sset]; pool = mp.Pool(8); outp = os.path.join(ROOT, f'results/spike15b/s6_eval_{tag}_{sset}.json')
    BASE = list(runs.values())[0]['args'].get('base', 'adapt'); cache = os.path.join(ROOT, f'results/spike15b/s6_init_{sset}_b{bias:g}' + ('' if BASE == 'adapt' else '_' + BASE) + '.npy')
    if os.path.exists(cache): init = np.load(cache)
    else: init = eval_set(pool, None, seeds, bl, BASE)['per']; np.save(cache, init)
    out = dict(tag=tag, set=sset, n=len(seeds), bias=bias, init=float(init.mean()), seeds={}); per = []
    for s, r in sorted(runs.items()):
        e = eval_set(pool, r['selected_layers'], seeds, bl, BASE); d = e['per'] - init; m, lo, hi = boot(d); per.append(e['per'])
        out['seeds'][s] = dict(succ=e['succ'], diff=m, ci=[lo, hi], res_mean=e['res_mean'], res_max=e['res_max'], cal_sel=r['selected']['succ'], cal_init=r['init']['succ'], selected_it=r['selected']['it'])
        print(tag, sset, 'seed', s, 'succ', e['succ'], 'init', init.mean(), 'diff %.3f [%.3f, %.3f]' % (m, lo, hi), 'res_mean', e['res_mean'], flush=True)
    P = np.mean(per, 0); m, lo, hi = boot(P - init); out['pooled'] = dict(succ=float(P.mean()), diff=m, ci=[lo, hi]); print(tag, sset, 'pooled over', len(per), 'seeds: succ %.3f diff %.3f [%.3f, %.3f]' % (P.mean(), m, lo, hi))
    json.dump(out, open(outp, 'w'), indent=1); np.save(outp.replace('.json', '.npy'), np.array(per))
