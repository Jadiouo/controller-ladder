"""Evaluate CAL-selected checkpoints of a spike15b S5 run on a seed set (paired with the zero-residual init), 8 processes. usage: eval15j.py TAG SET [bias]  SET in gate|holdout|fresh. success = reached the 300-hit cap."""
import sys, os, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rl15j import *
PC = json.load(open(os.path.join(ROOT, 'results/spike07/best.json')))['params']
def job(t):
    torch.set_num_threads(1); state, hid, obs, seeds, bias = t
    actor = None
    if state is not None: actor = mk(obs, 3, hid); actor.load_state_dict(state)
    r = evaluate(PC, actor, np.array(seeds), None if bias == 0 else bias_vec(bias)); return (r['hits'] >= 300).astype(float), r['res_mean'], r['hits']
def run(pool, state, hid, obs, seeds, bias, nch=8):
    ch = [seeds[i::nch] for i in range(nch)]; outs = pool.map(job, [(state, hid, obs, c, bias) for c in ch], chunksize=1); succ = np.empty(len(seeds)); hits = np.empty(len(seeds))
    for i, o in enumerate(outs): succ[i::nch] = o[0]; hits[i::nch] = o[2]
    return succ, float(np.mean([o[1] for o in outs])), hits
def boot(d, B=10000, seed=0):
    r = np.random.default_rng(seed); n = len(d); m = d[r.integers(0, n, (B, n))].mean(1); return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
if __name__ == '__main__':
    tag, sset = sys.argv[1], sys.argv[2]; bias = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; runs = pickle.load(open(os.path.join(RUNS, tag + '.pkl'), 'rb')); seeds = [int(s) for s in SETS[sset]]; pool = mp.get_context('fork').Pool(8)
    outp = os.path.join(ROOT, f'results/spike15b/s5_eval_{tag}_{sset}.json'); cache = os.path.join(ROOT, f'results/spike15b/s5_init_{sset}_b{bias:g}.npy')
    if os.path.exists(cache): init = np.load(cache)
    else: init = run(pool, None, 0, 0, seeds, bias)[0]; np.save(cache, init)
    out = dict(tag=tag, set=sset, n=len(seeds), bias=bias, init=float(init.mean()), seeds={}); per = []
    for s, r in sorted(runs.items()):
        succ, rm, hits = run(pool, r['state'], r['hid'], r['obs'], seeds, bias); d = succ - init; m, lo, hi = boot(d); per.append(succ)
        out['seeds'][s] = dict(succ=float(succ.mean()), diff=m, ci=[lo, hi], res_mean=rm, mean_hits=float(hits.mean()), cal_sel=r['selected']['frac_cap'], cal_init=r['init']['frac_cap'], selected_it=r['selected']['it'])
        print(tag, sset, 'seed', s, 'succ', succ.mean(), 'init', init.mean(), 'diff %.3f [%.3f, %.3f]' % (m, lo, hi), 'res_mean', rm, flush=True)
    P = np.mean(per, 0); m, lo, hi = boot(P - init); out['pooled'] = dict(succ=float(P.mean()), diff=m, ci=[lo, hi]); print(tag, sset, 'pooled over', len(per), 'seeds: succ %.3f diff %.3f [%.3f, %.3f]' % (P.mean(), m, lo, hi))
    json.dump(out, open(outp, 'w'), indent=1); np.save(outp.replace('.json', '.npy'), np.array(per))
