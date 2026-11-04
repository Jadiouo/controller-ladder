"""spike15f: where do MuJoCo 'Nan, Inf or huge value in QACC' warnings come from in the whip pipeline (spike15c/d/e), and how many ACTUALLY EXECUTED episodes diverge?
Instrumentation only (no existing file is modified): mujoco.set_mju_user_warning callback counts QACC warnings per phase (plan = CMA-ES stage1 / member plans, model = all other
model rollouts: replay / closed-loop scoring / estimator fits, plant = inside arena.env.mj_step of the true plant); the plant shim also counts MuJoCo auto-resets (time jumps back)
and non-finite qpos/qvel. run_episode is wrapped to record per-execution counters. Seeds: holdout 1100-1199 only. Plans/estimator caches are redirected to a fresh dir so the planner really runs.
usage: instab.py R0|B|LAP   (cwd .scratch/s15f/run)"""
import sys, os, pickle, time, multiprocessing as mp
H = os.path.dirname(os.path.abspath(__file__))
for d in ('spike15d', 'spike15c'): sys.path.insert(0, os.path.join(H, '..', d))
import c15d, c15c, c13, c14, c10b
from c15c import *
import mujoco
S15F = os.path.join(ROOT, '.scratch/s15f'); LO, HI = 1100, 1199; SEEDS = list(range(LO, HI + 1)); NW = int(os.environ.get('NW', 8)); LIM = int(os.environ.get('LIM', 10 ** 9)); SEEDS = SEEDS[:LIM]
def guard15f(seed):
    s = int(seed); assert (LO <= s <= HI) or s == 1000, f'seed {s} not allowed in spike15f'
for m in (c15c, c15d, c13, c14, c10b):
    for nm in ('guard15c', 'guard15d', 'guard13', 'guard14'):
        if hasattr(m, nm): setattr(m, nm, guard15f)
c13.ALLOW = ((LO, HI),)
FRESH = os.path.join(S15F, 'plans'); os.makedirs(FRESH, exist_ok=True)
for m in list(sys.modules.values()):
    if getattr(m, 'PL', None) == c10b.PL and m is not None: m.PL = FRESH
# ---- counters
CNT = dict(plan=0, model=0, plant=0, plant_reset=0, plant_nonfinite=0, other_warn=0); CTX = ['model']
def _cb(msg):
    if 'Nan, Inf' in msg: CNT[CTX[0]] += 1
    else: CNT['other_warn'] += 1
mujoco.set_mju_user_warning(_cb)
class Shim:
    def __getattr__(self, k): return getattr(mujoco, k)
    def mj_step(self, m, d, *a, **k):
        t0 = d.time; CTX[0], prev = 'plant', CTX[0]
        try: mujoco.mj_step(m, d, *a, **k)
        finally: CTX[0] = prev
        if d.time < t0: CNT['plant_reset'] += 1; c10b.INST[0] += 1
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()): CNT['plant_nonfinite'] += 1
import arena.env as AE; AE.mujoco = Shim()
def phase(fn, name):
    def f(*a, **k):
        prev = CTX[0]; CTX[0] = name
        try: return fn(*a, **k)
        finally: CTX[0] = prev
    return f
c15c.get_picked = phase(c15c.get_picked, 'plan'); c10b.Tuned3S.plan = phase(c10b.Tuned3S.plan, 'plan')
CALLS = []
_orig_re = c15c.run_episode
def _re(sc, ctrl, seed, *a, **k):
    s0 = dict(CNT); box = {}
    def fac(ctx): p = ctrl(ctx); box['p'] = p; return p
    tr = _orig_re(sc, fac if callable(ctrl) else ctrl, seed, *a, **k)
    p = box.get('p'); fz = getattr(p, 'forced', None)
    rec = {k_: CNT[k_] - s0[k_] for k_ in CNT}; rec.update(seed=int(seed), forced=None if fz is None else np.asarray(fz, float), diverged=bool(tr.diverged), outcome=tr.outcome,
        err_finite=bool(np.isfinite(tr.err).all()) if tr.err.size else True, q_finite=bool(np.isfinite(tr.q).all())); CALLS.append(rec); return tr
for m in list(sys.modules.values()):
    if getattr(m, 'run_episode', None) is _orig_re: m.run_episode = _re
def _fin(x): return bool(np.isfinite(x))
def work_R0(s):
    del CALLS[:]; r = c13.run13(('adapt3_full', s, 'abc')); return dict(seed=s, r={k: float(v) for k, v in r.items()}, calls=list(CALLS))
def work_B(s):
    del CALLS[:]; r = c15c.run_B((s, True)); return dict(seed=s, r=r, calls=list(CALLS))
TAB = {'B': 'tabB.pkl', 'LAP': 'tabLAP.pkl'}
if __name__ == '__main__':
    mode = sys.argv[1]; assert mode in ('R0', 'B', 'LAP'); t0 = time.time(); out = os.path.join(S15F, f'{mode}.pkl')
    if mode == 'R0': pool = mp.Pool(NW); fn = work_R0
    else:
        tp = os.path.join(ROOT, '.scratch/s15e/hold', TAB[mode]); init_mem(tp); pool = mp.Pool(NW); fn = work_B
    res = []
    for i, r in enumerate(pool.imap(fn, SEEDS, chunksize=1)):
        res.append(r)
        if (i + 1) % 10 == 0: print(mode, i + 1, '/', len(SEEDS), f'{time.time()-t0:.0f}s', flush=True)
    pool.close(); pool.join(); pickle.dump(dict(seeds=SEEDS, res=res, wall=time.time() - t0), open(out, 'wb')); print('DONE', mode, f'{time.time()-t0:.0f}s')
