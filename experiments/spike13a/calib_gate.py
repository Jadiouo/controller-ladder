"""Gate calibration: zero-residual success of the base under a post-probe torque bias b*(1,-1,1) on CAL-A (n=50); also checks b=0 equals spike10b's run_residual_episode10."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rl13 import *
if __name__ == '__main__':
    pool = mp.Pool(8); out = {}
    r0 = pool.map(run_residual_episode10, [(base_spec(), None, s, None, 'abc', ()) for s in CAL_A[:20]], chunksize=1)
    r1 = pool.map(run13, [(None, 'ppo', s, 12.0, None) for s in CAL_A[:20]], chunksize=1)
    print('equivalence b=0 (20 eps): spike10b', np.mean([x['success'] for x in r0]), 'rl13', np.mean([x['success'] for x in r1]), 'identical per-episode', all(x['success'] == y['success'] and abs(x['min_dist'] - y['min_dist']) < 1e-9 for x, y in zip(r0, r1)), flush=True)
    for b in [0, 1, 2, 3, 4, 6]:
        bv = None if b == 0 else bias_vec(b).tolist(); t = time.time(); e = eval_set(pool, None, 'ppo', CAL_A, 12.0, bv); out[b] = e['succ']; print('bias', b, 'CAL-A succ', e['succ'], 'min_dist', e['min_dist'], f'{time.time()-t:.0f}s', flush=True)
    json.dump(out, open(os.path.join(ROOT, 'results/spike13a/gate_calib.json'), 'w'))
