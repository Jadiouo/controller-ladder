"""Step 1b: T-nom, T-robust, T-adapt, T-oracle on holdout seeds 1100-1599 (chain error from the seed, spike09a dist w=0.2). best-classical = highest holdout success among T-nom/T-robust/T-adapt (ties: lower mean min_dist).
Not TEST. Saves per-seed arrays."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c9b import *
def job(a):
    name, s = a
    if name == 'oracle': return run_spec((('oracle', tvec(true_chain(s)).tolist()), s, None, False))
    return run_spec((name, s, None, False))
if __name__ == '__main__':
    S = list(range(1100, 1600)); guard(S); pool = mp.Pool(8); out = {}; t0 = time.time()
    for name in ['nom', 'robust', 'adapt', 'oracle']:
        r = pool.map(job, [(name, s) for s in S], chunksize=1); out[name] = {k: [x[k] for x in r] for k in ('success', 'min_dist', 'hit_speed', 'energy', 'diverged')}
        su = np.array(out[name]['success']); print(name, 'succ', su.mean(), 'ci', A.boot_ci(su), 'md', np.nanmean(out[name]['min_dist']), f'{time.time()-t0:.0f}s', flush=True)
    sel = max(['nom', 'robust', 'adapt'], key=lambda n: (round(float(np.mean(out[n]['success'])), 6), -float(np.mean(out[n]['min_dist']))))
    out['best_classical'] = sel; out['seeds'] = '1100-1599'; out['n'] = len(S); out['wall_s'] = time.time() - t0
    d = lambda a, b: np.array(out[a]['success']) - np.array(out[b]['success'])
    out['paired'] = {f'{a}-{b}': dict(diff=float(d(a, b).mean()), ci=A.boot_ci(d(a, b), seed=1)) for a, b in (('adapt', 'nom'), ('adapt', 'robust'), ('robust', 'nom'), ('oracle', 'adapt'))}
    json.dump(out, open(os.path.join(OUT, 'holdout.json'), 'w'), indent=1); print('best-classical:', sel, out['paired'])
