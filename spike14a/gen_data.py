"""raw probe data: train x1 (500000-511999, same seeds as spike10b L-est-full), train extra (540000-575999 => x4 total), CAL, HOLD, NEW. rep=0 for training (as spike10b)."""
import sys, os, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
if __name__ == '__main__':
    t0 = time.time(); pool = mp.Pool(8)
    sets = dict(cal=[(s, None, 'abc') for s in CAL_SEEDS], hold=[(s, None, 'abc') for s in HOLD], new=[(s, None, 'abc') for s in NEW],
                train1=[(500000 + i, 0, 'abc') for i in range(12000)], train_extra=[(540000 + i, 0, 'abc') for i in range(36000)])
    for k, it in sets.items():
        if len(sys.argv) > 1 and k not in sys.argv[1:]: continue
        D = gen_set(pool, it)
        if k.startswith('train'): D = {a: (v.astype(np.float32) if v.dtype == np.float64 and a not in ('TH', 'q0') else v) for a, v in D.items()}      # eval sets stay float64 (exact pipeline match)
        pickle.dump(D, open(os.path.join(DAT, f'raw_{k}.pkl'), 'wb')); print(k, D['taus'].shape, f'{time.time()-t0:.0f}s', flush=True)
