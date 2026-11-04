"""spike15d driver. usage: evalD.py <set> Bnew            (set=new only) L-dec4 B collect (centre = mean of 3 frozen nets) -> .scratch/s15d/B_new.pkl
                          evalD.py <set> refs            (new only) R0 / R1 reference episodes -> .scratch/s15d/refs_new.json
                          evalD.py <set> collect <MS|LAP>  classical-ensemble collect (centre = R0, members 3, npool 3) -> .scratch/s15d/D_<fam>_<set>.pkl
                          evalD.py <set> extra <LAP>     adds S columns of LAP members 4..9 (descriptive M9)
Run with cwd = .scratch/s15d/run under flock + systemd-run."""
import sys, os, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c15d import *
SET, STAGE = sys.argv[1], sys.argv[2]; NW = int(os.environ.get('NW', 8)); SEEDS = SETS[SET]; LIM = int(os.environ.get('LIM', len(SEEDS))); SEEDS = SEEDS[:LIM]; SUF = '' if LIM == len(SETS[SET]) else f'_lim{LIM}'
if STAGE == 'Bnew':
    assert SET == 'new'; D = pickle.load(open(os.path.join(S15D, 'raw_new.pkl'), 'rb')); assert list(D['seed'][:LIM]) == SEEDS
    E3 = np.stack([np.load(os.path.join(S15D, f'est_new_m{m}.npy'))[:LIM] for m in range(3)], 1); cen = E3.mean(1); keys = D['key'][:LIM]
    tp = os.path.join(S15D, f'tabB_new{SUF}.pkl'); pickle.dump(dict(mem={k: E3[i] for i, k in enumerate(keys)}, cen={k: cen[i] for i, k in enumerate(keys)}, npool=3), open(tp, 'wb'))
    init_mem(tp); pool = mp.Pool(NW, initializer=init_mem, initargs=(tp,)); t0 = time.time(); res = pool.map(run_B, [(s, True) for s in SEEDS], chunksize=1)
    pickle.dump(dict(seeds=SEEDS, res=res, wall=time.time() - t0, members=[0, 1, 2]), open(os.path.join(S15D, f'B_new{SUF}.pkl'), 'wb')); print('Bnew', len(res), f'{time.time()-t0:.0f}s base succ', np.mean([r['succ'][r['base']] for r in res]), flush=True)
elif STAGE == 'refs':
    assert SET == 'new'; pool = mp.Pool(NW); out = {}
    for meth in ('adapt3_full', 'R1'):
        t0 = time.time(); r = pool.map(c13.run13, [(meth, s, 'abc') for s in SEEDS], chunksize=1); out[meth] = dict(success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], wall=time.time() - t0); print(meth, SET, np.mean(out[meth]['success']), f'{time.time()-t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(S15D, 'refs_new.json'), 'w'))
else:
    fam = sys.argv[3]; C = pickle.load(open(os.path.join(S15D, f'cls_{SET}.pkl'), 'rb')); assert list(C['seeds'][:LIM]) == SEEDS; keys = C['key'][:LIM]
    mem = np.stack([(r['ms'] if fam == 'MS' else r['lap'][:3]) for r in C['res'][:LIM]]); cen = np.stack([r['center'] for r in C['res'][:LIM]]); OUTP = os.path.join(S15D, f'D_{fam}_{SET}{SUF}.pkl')
    if STAGE == 'collect':
        tp = os.path.join(S15D, f'tab_{fam}_{SET}{SUF}.pkl'); pickle.dump(dict(mem={k: mem[i] for i, k in enumerate(keys)}, cen={k: cen[i] for i, k in enumerate(keys)}, npool=3), open(tp, 'wb'))
        init_mem(tp); pool = mp.Pool(NW, initializer=init_mem, initargs=(tp,)); t0 = time.time(); res = pool.map(run_B, [(s, True) for s in SEEDS], chunksize=1)
        pickle.dump(dict(seeds=SEEDS, res=res, wall=time.time() - t0, members=[0, 1, 2]), open(OUTP, 'wb')); print('collect', fam, SET, len(res), f'{time.time()-t0:.0f}s base succ', np.mean([r['succ'][r['base']] for r in res]), flush=True)
    elif STAGE == 'extra':
        assert fam == 'LAP'; B = pickle.load(open(OUTP, 'rb'))
        if len(B['members']) > 3: print('already'); sys.exit(0)
        E = np.stack([r['lap'][3:] for r in C['res'][:LIM]]); pool = mp.Pool(NW); t0 = time.time(); cols = pool.map(run_score, [(s, B['res'][i]['cands'], E[i]) for i, s in enumerate(SEEDS)], chunksize=1)
        for i, c in enumerate(cols): B['res'][i]['S'] = np.concatenate([B['res'][i]['S'], c], 1)
        B['members'] = list(range(9)); pickle.dump(B, open(OUTP, 'wb')); print('extra', fam, SET, f'{time.time()-t0:.0f}s', flush=True)
