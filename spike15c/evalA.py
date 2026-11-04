"""spike15c approach A evaluation: evalA.py <tag> <bias> <set: hold|new> -> .scratch/s15c/A_<tag>_<set>.npz (per-episode success of zero-residual init and of each seed's CAL-selected checkpoint)."""
import sys, os, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import rl15c as R; from c15c import *
tag, bias, SET = sys.argv[1], float(sys.argv[2]), sys.argv[3]; seeds = dict(hold=HOLD, new=NEW)[SET]; bl = None if bias == 0 else R.R13.bias_vec(bias).tolist()
R.TAB.update(R.load_tab()); resd = pickle.load(open(os.path.join(R.RUNS15, tag + '.pkl'), 'rb')); pool = mp.Pool(8); out = {}
t0 = time.time(); out['init'] = R.eval_set(pool, None, seeds, 12.0, bl)['per']; print('init', out['init'].mean(), flush=True)
for s, r in resd.items(): out[f'seed{s}'] = R.eval_set(pool, r['selected_layers'], seeds, 12.0, bl)['per']; print('seed', s, out[f'seed{s}'].mean(), f'{time.time()-t0:.0f}s', flush=True)
np.savez(os.path.join(S15C, f'A_{tag}_{SET}.npz'), **out); print('EVALDONE')
