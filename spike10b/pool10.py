"""Fill the estimation/plan caches for the M PPO training scenarios (seed 500000+i, rep 0, condition abc) of the base controller with ZERO residual. argv: base_json M."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
if __name__ == '__main__':
    sp = json.load(open(sys.argv[1]))['spec']; base = tuple(sp) if isinstance(sp, list) else sp; M = int(sys.argv[2]); t0 = time.time(); pool = mp.Pool(8)
    r = pool.map(run_residual_episode10, [(base, None, 500000 + i, 0, 'abc', ()) for i in range(M)], chunksize=2); su = np.array([x['success'] for x in r])
    print('pool', M, 'base succ', su.mean(), 'mean steps', np.mean([x['steps'] for x in r]), 'unstable', np.mean([x['inst'] > 0 for x in r]), f'{time.time()-t0:.0f}s')
    json.dump(dict(base=str(base)[:80], M=M, succ=float(su.mean()), mean_steps=float(np.mean([x['steps'] for x in r])), unstable=float(np.mean([x['inst'] > 0 for x in r])), wall=time.time() - t0), open(os.path.join(OUT, 'pool_base.json'), 'w'))
