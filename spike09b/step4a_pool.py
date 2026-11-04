"""Step 4a: fill the plan caches for the M training scenarios (target seed 500000+i, chain error independent draw rep 0) of a base controller, with ZERO residual. Reports the base success on the training pool."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c9b import *
if __name__ == '__main__':
    base = sys.argv[1]; M = int(sys.argv[2]); t0 = time.time(); pool = mp.Pool(8)
    r = pool.map(run_residual_episode, [(base, True, False, None, 500000 + i, 0, False) for i in range(M)], chunksize=2)
    su = np.array([x['success'] for x in r]); print(base, 'pool', M, 'succ', su.mean(), 'mean steps', np.mean([x['steps'] for x in r]), f'{time.time()-t0:.0f}s')
    json.dump(dict(base=base, M=M, succ=float(su.mean()), mean_steps=float(np.mean([x['steps'] for x in r])), wall=time.time() - t0), open(os.path.join(OUT, f'pool_{base}.json'), 'w'))
