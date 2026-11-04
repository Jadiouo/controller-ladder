import sys, os, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import rl15c as R; from c15c import *
R.TAB.update(R.load_tab()); e = R.ResEnv15(600001, 0, 12.0, None); print('x', e.x.shape, 'priv', e.priv().shape, 'nq', e.env.data.qpos.shape, 'n', e.nj)
for _ in range(200): e.step(np.zeros(3))
print('priv after probe', np.round(e.priv()[:6], 2), np.round(e.priv()[26:30], 2))
pool = mp.Pool(8); t0 = time.time(); a = R.eval_set(pool, None, list(range(1000, 1016)), 12.0, None); print('zero-res CAL16', a['succ'], f'{time.time()-t0:.0f}s')
b = R.eval_set(pool, None, list(range(1000, 1016)), 12.0, R.R13.bias_vec(6).tolist()); print('bias6 CAL16', b['succ'])
pl = R.eval_set(pool, None, list(range(600000, 600016)), 12.0, None, rep=0); print('pool16 (rep0)', pl['succ'])
