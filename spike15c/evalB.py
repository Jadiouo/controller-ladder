"""spike15c approach B driver. usage: evalB.py <set: cal|hold|new> collect [--noexec]   -> .scratch/s15c/B_<set>.pkl  (per-episode S, candidates, executed outcomes)
                                      evalB.py <set> extra                 -> adds S columns for members 3..8 (if their estimates exist)
                                      evalB.py <set> refs                  -> R0 / R1 episodes (spike13b pipeline) -> .scratch/s15c/refs_<set>.json
Run with cwd = .scratch/run15c under flock + systemd-run."""
import sys, os, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c15c import *
SET, STAGE = sys.argv[1], sys.argv[2]; NW = int(os.environ.get('NW', 8)); SEEDS = dict(cal=CAL, hold=HOLD, new=NEW)[SET]; LIM = int(os.environ.get('LIM', len(SEEDS))); SEEDS = SEEDS[:LIM]
D = pickle.load(open(os.path.join(S15C, f'raw_{SET}.pkl'), 'rb')); assert list(D['seed'][:LIM]) == SEEDS
def ests(ms): return np.stack([np.load(os.path.join(S15C, f'est_{SET}_m{m}.npy'))[:LIM] for m in ms], 1)       # (N, M, 13)
SUF = '' if LIM == len(dict(cal=CAL, hold=HOLD, new=NEW)[SET]) else f'_lim{LIM}'; OUTP = os.path.join(S15C, f'B_{SET}{SUF}.pkl')
if STAGE == 'collect':
    E3 = ests(range(3)); cen = E3.mean(1); tab = dict(mem={k: E3[i] for i, k in enumerate(D['key'][:LIM])}, cen={k: cen[i] for i, k in enumerate(D['key'][:LIM])}, npool=3); tp = os.path.join(S15C, f'tabB_{SET}{SUF}.pkl'); pickle.dump(tab, open(tp, 'wb'))
    init_mem(tp); pool = mp.Pool(NW, initializer=init_mem, initargs=(tp,)); t0 = time.time(); res = pool.map(run_B, [(s, True) for s in SEEDS], chunksize=1)
    pickle.dump(dict(seeds=SEEDS, res=res, wall=time.time() - t0, members=[0, 1, 2]), open(OUTP, 'wb')); print('collect', SET, len(res), f'{time.time()-t0:.0f}s', 'base succ', np.mean([r['succ'][r['base']] for r in res]), flush=True)
elif STAGE == 'extra':
    B = pickle.load(open(OUTP, 'rb')); have = B['members']; new = [m for m in range(3, 9) if os.path.exists(os.path.join(S15C, f'est_{SET}_m{m}.npy')) and m not in have]
    if not new: print('nothing to add'); sys.exit(0)
    E = ests(new); pool = mp.Pool(NW); t0 = time.time(); cols = pool.map(run_score, [(s, B['res'][i]['cands'], E[i]) for i, s in enumerate(SEEDS)], chunksize=1)
    for i, c in enumerate(cols): B['res'][i]['S'] = np.concatenate([B['res'][i]['S'], c], 1)
    B['members'] = have + new; pickle.dump(B, open(OUTP, 'wb')); print('extra members', new, f'{time.time()-t0:.0f}s', flush=True)
elif STAGE == 'refs':
    pool = mp.Pool(NW); out = {}
    for meth in ('adapt3_full', 'R1'):
        t0 = time.time(); r = pool.map(c13.run13, [(meth, s, 'abc') for s in SEEDS], chunksize=1); out[meth] = dict(success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], wall=time.time() - t0); print(meth, SET, np.mean(out[meth]['success']), f'{time.time()-t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(S15C, f'refs_{SET}.json'), 'w'))
