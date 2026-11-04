"""S6 gate calibration: zero-residual success of T-adapt under post-probe bias b on CAL-A (n=50)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rl15w import *
if __name__ == '__main__':
    pool = mp.Pool(8); out = {}
    for b in [0, 1, 2, 3, 4, 6]:
        bv = None if b == 0 else bias_vec(b).tolist(); t = time.time(); e = eval_set(pool, None, CAL_A, bv); out[b] = e['succ']; print('bias', b, 'CAL-A succ', e['succ'], 'min_dist', e['min_dist'], f'{time.time()-t:.0f}s', flush=True)
    json.dump(out, open(os.path.join(ROOT, 'results/spike15b/s6_gate_calib.json'), 'w'))
