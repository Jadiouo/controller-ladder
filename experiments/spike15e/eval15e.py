"""spike15e: v9 locked confirmation on TEST-D (seeds 40000-40499), run ONCE. Methods frozen (spike15c/15d code reused unchanged: ProbeB/run_B/run_score, cls15d.ens_fit; weights results/spike15c/models).
usage: eval15e.py hold   -> reproduction check on holdout 1100-1599 (env LIM=n for a plumbing smoke on the first n holdout seeds, separate dir)
       eval15e.py testd  -> the ONE locked run on 40000-40499. Refuses unless .scratch/s15e/repro_ok exists (written by stats15e.py hold when the check passes).
       Writes ONE line to results/.test_access.log before the first TEST-D seed is touched (marker .scratch/s15e/testd_logged; a resume after a crash does not log again).
Everything is resumable per chunk of 50 episodes (chunk files in .scratch/s15e/<set>/). Run with cwd .scratch/s15e/run under flock + systemd-run."""
import sys, os, json, time, pickle, datetime, subprocess, multiprocessing as mp
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', 'spike15d'))
MODE = sys.argv[1]; assert MODE in ('hold', 'testd')
import c15d, cls15d, c15c                                  # c15d: guards for 15d; we override every guard below
from c15c import *
S15E = os.path.join(ROOT, '.scratch/s15e'); LIM = int(os.environ.get('LIM', 10 ** 9)); NW = int(os.environ.get('NW', 8))
FULL = list(range(1100, 1600)) if MODE == 'hold' else list(range(40000, 40500)); SEEDS = FULL[:LIM]; LO, HI = (1100, 1599) if MODE == 'hold' else (40000, 40499)
OUTD = os.path.join(S15E, MODE + ('' if LIM >= len(FULL) else f'_lim{LIM}')); os.makedirs(OUTD, exist_ok=True)
def guard15e(seed):
    s = int(seed); assert (LO <= s <= HI) or s == 1000, f'seed {s} not allowed in spike15e mode {MODE} (s==1000 only builds the nominal context)'
for m in (c15c, c15d, cls15d): m.guard15c = guard15e; m.guard15d = guard15e
c13.guard13 = guard15e; c14.guard14 = guard15e; c13.ALLOW = ((LO, HI),); c15d.guard15d = guard15e
[guard15e(s) for s in SEEDS]
LOG = os.path.join(ROOT, 'results/.test_access.log'); MARK = os.path.join(S15E, 'testd_logged')
def log_access():
    if MODE != 'testd': return
    if os.path.exists(MARK): print('TEST-D already logged; resuming without a new log line'); return
    msg = 'v9 single locked TEST-D run, seeds=40000-40499'
    with open(LOG, 'a') as f: f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\tspike15e/eval15e.py\t{msg}\n")
    open(MARK, 'a').write(datetime.datetime.now().isoformat() + ' ' + msg + '\n'); print('logged:', msg, flush=True)
# ---- CPU timing of the selection pipeline: only the first (ordinary) _swing_start of an episode = planning + scoring in all models (not the executions of the 11 candidates, which exist only for offline evaluation)
TIM = [0.0]; _orig = c15c.ProbeB._swing_start
def _timed(self, obs):
    if self.forced is None and self.score is None:
        t0 = time.process_time()
        try: return _orig(self, obs)
        finally: TIM[0] += time.process_time() - t0
    return _orig(self, obs)
c15c.ProbeB._swing_start = _timed
def run_B_t(a): TIM[0] = 0.0; r = c15c.run_B(a); r['cpu_sel'] = TIM[0]; return r
def run_score_t(a): return c15c.run_score(a)
def chunks(name, fn, items, pool, CH=50):
    out = []
    for k in range(0, len(items), CH):
        p = os.path.join(OUTD, f'{name}_{k//CH:03d}.pkl')
        if os.path.exists(p): out += pickle.load(open(p, 'rb')); continue
        t0 = time.time(); r = pool.map(fn, items[k:k + CH], chunksize=1); pickle.dump(r, open(p + '.tmp', 'wb')); os.replace(p + '.tmp', p); out += r; print(name, k + len(r), '/', len(items), f'{time.time()-t0:.0f}s', flush=True)
    return out
def stage_done(name): return os.path.exists(os.path.join(OUTD, name + '.done'))
def mark(name, **kw): json.dump(kw, open(os.path.join(OUTD, name + '.done'), 'w'))
if __name__ == '__main__':
    if MODE == 'testd':
        assert os.path.exists(os.path.join(S15E, 'repro_ok')), 'holdout reproduction check has not passed (stats15e.py hold): TEST-D refused'
    log_access(); t00 = time.time(); K = len(SEEDS)
    # ---- A raw probe data
    rawp = os.path.join(OUTD, 'raw.pkl'); pool = mp.Pool(NW)
    if not os.path.exists(rawp):
        D = c14.gen_set(pool, [(s, None, 'abc') for s in SEEDS], chunk=10); assert list(D['seed']) == SEEDS; pickle.dump(D, open(rawp, 'wb')); print('raw', K, f'{time.time()-t00:.0f}s', flush=True)
    pool.close(); pool.join(); D = pickle.load(open(rawp, 'rb')); keys = D['key']
    # ---- B inference (CPU subprocess, 9 frozen nets; the first 3 = L-ens / B members, 9 = descriptive B9)
    if not all(os.path.exists(os.path.join(OUTD, f'est_m{m}.npy')) for m in range(9)):
        t0 = time.time(); subprocess.run([sys.executable, os.path.join(H, 'infer15e.py'), rawp, os.path.join(OUTD, 'est')], check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}); mark('infer', wall=time.time() - t0)
    infer_wall = json.load(open(os.path.join(OUTD, 'infer.done')))['wall'] if stage_done('infer') else None
    E = np.stack([np.load(os.path.join(OUTD, f'est_m{m}.npy')) for m in range(9)], 1); E3 = E[:, :3]; cen = E3.mean(1)
    # ---- C B collect (centre = mean of 3 frozen nets, members 0-2, pool c11, + 6 extra members' scores later)
    tp = os.path.join(OUTD, 'tabB.pkl'); pickle.dump(dict(mem={k: E3[i] for i, k in enumerate(keys)}, cen={k: cen[i] for i, k in enumerate(keys)}, npool=3), open(tp, 'wb'))
    if not stage_done('B'):
        init_mem(tp); pool = mp.Pool(NW, initializer=init_mem, initargs=(tp,)); t0 = time.time(); res = chunks('B', run_B_t, [(s, True) for s in SEEDS], pool); pool.close(); pool.join()
        pickle.dump(dict(seeds=SEEDS, res=res, wall=time.time() - t0, members=[0, 1, 2]), open(os.path.join(OUTD, 'B.pkl'), 'wb')); mark('B', wall=time.time() - t0)
    # ---- D classical Laplace sets from R0 probe data (cls15d.ens_fit, unchanged)
    if not stage_done('cls'):
        t0 = time.time(); pool = mp.Pool(NW); idx = [list(range(i, min(i + 4, K))) for i in range(0, K, 4)]
        r = chunks('cls', cls15d.work, [(D, c) for c in idx], pool, CH=13); pool.close(); pool.join()
        res = sum([a for a, b in r], []); cpu = sum(b for a, b in r); pickle.dump(dict(seeds=SEEDS, key=keys, TH=D['TH'], res=res, cpu=cpu, wall=time.time() - t0), open(os.path.join(OUTD, 'cls.pkl'), 'wb')); mark('cls', wall=time.time() - t0, cpu=cpu)
    C = pickle.load(open(os.path.join(OUTD, 'cls.pkl'), 'rb')); assert list(C['seeds']) == SEEDS
    # ---- E LAP collect (centre = R0 estimate, members = first 3 Laplace draws, pool c11)
    if not stage_done('LAP'):
        mem = np.stack([r['lap'][:3] for r in C['res']]); cenc = np.stack([r['center'] for r in C['res']]); tpl = os.path.join(OUTD, 'tabLAP.pkl'); pickle.dump(dict(mem={k: mem[i] for i, k in enumerate(keys)}, cen={k: cenc[i] for i, k in enumerate(keys)}, npool=3), open(tpl, 'wb'))
        init_mem(tpl); pool = mp.Pool(NW, initializer=init_mem, initargs=(tpl,)); t0 = time.time(); res = chunks('LAP', run_B_t, [(s, True) for s in SEEDS], pool); pool.close(); pool.join()
        pickle.dump(dict(seeds=SEEDS, res=res, wall=time.time() - t0, members=[0, 1, 2]), open(os.path.join(OUTD, 'LAP.pkl'), 'wb')); mark('LAP', wall=time.time() - t0)
    # ---- F references R0 (T-adapt3-full) and R1 (true theta)
    if not stage_done('refs'):
        pool = mp.Pool(NW); out = {}; t0 = time.time()
        for meth in ('adapt3_full', 'R1'):
            r = chunks('ref_' + meth, c13.run13, [(meth, s, 'abc') for s in SEEDS], pool); out[meth] = dict(success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r])
        pool.close(); pool.join(); json.dump(out, open(os.path.join(OUTD, 'refs.json'), 'w')); mark('refs', wall=time.time() - t0)
    # ---- G B9 (descriptive): scores of the 6 extra frozen nets on B's stored candidates
    if not stage_done('B9'):
        B = pickle.load(open(os.path.join(OUTD, 'B.pkl'), 'rb')); pool = mp.Pool(NW); t0 = time.time(); cols = chunks('B9', run_score_t, [(s, B['res'][i]['cands'], E[i, 3:]) for i, s in enumerate(SEEDS)], pool); pool.close(); pool.join()
        for i, c in enumerate(cols): B['res'][i]['S'] = np.concatenate([B['res'][i]['S'], c], 1)
        B['members'] = list(range(9)); pickle.dump(B, open(os.path.join(OUTD, 'B9.pkl'), 'wb')); mark('B9', wall=time.time() - t0)
    print('ALL DONE', f'{time.time()-t00:.0f}s', flush=True)
