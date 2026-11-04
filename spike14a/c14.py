"""spike14a shared code: estimator duel (whip, realistic conditions abc x1). Imports spike10b/spike13b code unchanged.
Run with cwd = .scratch/run14 (MuJoCo writes MUJOCO_LOG.TXT in cwd). Interface: probe data -> theta13 -> identical replay + planner pipeline (Probe3)."""
import sys, os, json, time, pickle, hashlib, numpy as np
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'spike10b'))
from r10b import *
OUT14 = os.path.join(ROOT, 'results/spike14a'); os.makedirs(OUT14, exist_ok=True)
DAT = os.path.join(ROOT, '.scratch/s14'); os.makedirs(DAT, exist_ok=True)
CAL_SEEDS = list(range(1000, 1050)) + list(range(2000, 2150)); HOLD = list(range(1100, 1600)); NEW = list(range(701000, 701200))
def guard14(seed):
    s = int(seed); ok = (1000 <= s < 1050) or (2000 <= s < 2150) or (1100 <= s <= 1599) or (701000 <= s <= 701199) or (500000 <= s < 700000)
    assert ok, f'seed {s} not allowed in spike14a (never TEST 0-49 / TEST-B 20000-20499)'
PR = dict(T=0.6, amp=0.6)
def pkey(q0, taus, Qm): return hashlib.md5(np.round(np.concatenate([q0, taus.ravel(), Qm.ravel()]), 9).tobytes()).hexdigest()
def gen_raw(args):
    """items [(seed, rep, cond)] -> raw probe data + true theta13 + nominal replay residual inputs. Same probe as the pipeline (T=0.6, amp 0.6)."""
    items = args; AMP = np.array([0.5, 0.5, 0.4]) * PR['amp']; R = dict(q0=[], taus=[], Qm=[], Vm=[], Q0n=[], V0n=[], TH=[], key=[], seed=[])
    for sd, rep, cond in items:
        guard14(sd); reset_probe_globals(); sc = scen(sd, cond, rep); env = sc.make_env(); ctx = env.ctx; obs = env.reset(sd); q0, taus, Qm, Vm = probe_run3(env, ctx, obs, PR['T'], AMP)
        Q0n, V0n, _ = replay3(ctx, np.concatenate([[1.0], np.ones(NL), [1.0, 0.0]]), q0, taus)
        for k, v in zip(('q0', 'taus', 'Qm', 'Vm', 'Q0n', 'V0n', 'TH', 'key', 'seed'), (q0, taus, Qm, Vm, Q0n, V0n, true_theta(sd, cond, rep), pkey(q0, taus, Qm), sd)): R[k].append(v)
    return R
def merge(parts):
    out = {}
    for k in parts[0]: out[k] = np.concatenate([np.array(p[k]) for p in parts]) if k != 'key' else sum([list(p[k]) for p in parts], [])
    out['key'] = np.array(out['key']); return out
def gen_set(pool, items, chunk=50):
    return merge(pool.map(gen_raw, [items[i:i + chunk] for i in range(0, len(items), chunk)], chunksize=1))
_CTX = [None]
def nominal_ctx():
    if _CTX[0] is None: _CTX[0] = scen(1000, 'abc').make_env().ctx
    return _CTX[0]
def theta_err(TH, TE):
    """parameter errors, per episode arrays: dict of arrays (N,). ls: abs; lm_*: log mass; ld: log damping; fr: 1e-3 Nm"""
    d = {}; d['ls'] = TE[:, 0] - TH[:, 0]; lm = np.log(TE[:, 1:11]) - np.log(TH[:, 1:11]); d['lm_mean'] = lm.mean(1); d['lm_rms_link'] = np.sqrt((lm ** 2).mean(1)); d['lm_shape'] = np.sqrt(((lm - lm.mean(1, keepdims=True)) ** 2).mean(1))
    d['ld'] = np.log(TE[:, 11]) - np.log(TH[:, 11]); d['fr'] = (TE[:, 12] - TH[:, 12]) * 1e3; return d
def err_summary(TH, TE):
    d = theta_err(TH, TE); s = {k: float(np.sqrt((v ** 2).mean())) for k, v in d.items()}; s['bias_ld'] = float(d['ld'].mean()); return s
# ---- pipeline with a lookup estimator
class Lookup:
    def __init__(self, tab): self.tab = tab
    def __call__(self, ctx, q0, taus, Qm, Vm): return np.asarray(self.tab[pkey(q0, taus, Qm)], float)
TAB = {}
def _init_tab(path):
    global TAB; TAB = pickle.load(open(path, 'rb'))
def run14(args):
    meth, seed = args; guard14(seed); reset_probe_globals(); INST[0] = 0; sc = scen(seed, 'abc', None); est = Lookup(TAB[meth])
    tr = run_episode(sc, lambda ctx: Probe3(ctx, est, 'lk_' + meth, PR['T'], np.array([0.5, 0.5, 0.4]) * PR['amp']), int(seed)); m = episode_metrics(sc, tr)
    return dict(success=m['success'], min_dist=m['min_dist'], inst=int(INST[0]), diverged=m['diverged'])
