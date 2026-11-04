"""spike15c shared code. Approach B: ensemble/posterior-robust plan SELECTION on the whip (realistic conditions abc x1).
Per episode: the probe is open-loop and seed-deterministic; at the end of the probe the baseline planner (spike13b/14a Probe3 + Tuned3S) produces 8 candidate swings from the CENTER theta
(mean of the 3 frozen L-dec4 nets; stage-1 CMA-ES is cached by the pipeline), optionally + the best candidate of each member's own plan. Every candidate is
  (i)  scored in the closed-loop MODEL of each ensemble member (each member with its own replay-reconstructed chain state)  -> S[c, m]  (min tip distance)
  (ii) EXECUTED on the true plant                                                                                         -> Out[c] (success, min_dist)
so that any selection rule (aggregator, member subset, candidate pool) is evaluated OFFLINE without further simulation. The baseline pick (argmin S[c, center]) is also the ordinary L-ens episode (verified).
Seeds: only CAL 1000-1049/2000-2149, holdout 1100-1599, new 703000-703199, training pool 500000-699999. Never TEST/TEST-B/TEST-C. Run with cwd = .scratch/run15c."""
import sys, os, json, time, pickle, numpy as np
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'spike13b')); sys.path.insert(0, os.path.join(_H, '..', 'spike14a'))
import c13, c14
from c14 import *          # r10b + c10b names (Probe3, Tuned3S, replay3, scen, ...), pkey, Lookup
S15C = os.path.join(ROOT, '.scratch/s15c'); os.makedirs(S15C, exist_ok=True); RES15C = os.path.join(ROOT, 'results/spike15c'); os.makedirs(RES15C, exist_ok=True)
CAL = list(range(1000, 1050)) + list(range(2000, 2150)); HOLD = list(range(1100, 1600)); NEW = list(range(703000, 703200)); POOL = list(range(600000, 600300))
def guard15c(seed):
    s = int(seed); ok = (1000 <= s < 1050) or (2000 <= s < 2150) or (1100 <= s <= 1599) or (703000 <= s <= 703199) or (500000 <= s < 700000)
    assert ok, f'seed {s} not allowed in spike15c (never TEST 0-49 / TEST-B 20000-20499 / TEST-C 30000-30499)'
c13.guard13 = guard15c; c14.guard14 = guard15c; c13.ALLOW = ((703000, 703199),)
import c14 as _c14; _c14.guard14 = guard15c
def get_picked(inner, q0, target):
    n = inner.ctx.robot.n; k1 = hashlib.md5(repr((int(inner.seed), inner.thk, inner.xh)).encode()).hexdigest(); f1 = os.path.join(PL, f's13_{k1}.npy')
    if os.path.exists(f1):
        try: return np.load(f1)
        except Exception: pass
    picked = inner._stage1(q0, target); np.save(f1 + f'.{os.getpid()}.npy', picked); os.replace(f1 + f'.{os.getpid()}.npy', f1); return picked
MEM = {}            # pkey -> (M, 13) member thetas (members 0..2 = frozen L-dec4 seeds 0..2, then extra)
CEN = {}            # pkey -> (13,) center theta
NPOOL = [0]         # number of member-own-best candidates appended (0 or 3)
def init_mem(path):
    global MEM, CEN, NPOOL
    d = pickle.load(open(path, 'rb')); MEM = d['mem']; CEN = d['cen']; NPOOL[0] = int(d.get('npool', 0))
class _Done(Exception): pass
class ProbeB(Probe3):
    """Probe3 with (a) candidate collection + scoring in all member models, (b) forced execution of a given candidate."""
    forced = None; rec = None; score = None          # score = (cands (C,4,n), thetas (K,13)) -> columns S[c,k] only, then abort the episode (class _Done)
    def _swing_start(self, obs):
        taus, Qm, Vm = self.history(); self.pq0 = self.hq0; key = pkey(self.pq0, taus, Qm); dur = self.ctx.task_info['duration'] - self.T; target = np.asarray(obs.ref['target'], float); self.q0 = np.asarray(obs.q, float).copy(); n = self.ctx.robot.n
        if self.score is not None:
            cands, ths = self.score; cols = np.full((len(cands), len(ths)), np.nan)
            for j, th in enumerate(ths):
                _, _, xj = replay3(self.ctx, th, self.pq0, taus); ij = Tuned3S(self.ctx, th, xj, dur, self.qp, self.qv); ij.reset(self.seed)
                for c, Qk in enumerate(cands): cols[c, j] = min(ij._closed_loop(self.q0, Qk, target), 1.0)
            self.cols = cols; raise _Done()
        if self.forced is not None: self.knots = np.asarray(self.forced, float).reshape(4, n); self.plan_info = {}; return
        thc = np.asarray(CEN[key], float); self.est = thc; _, _, x0c = replay3(self.ctx, thc, self.pq0, taus); inner = Tuned3S(self.ctx, thc, x0c, dur, self.qp, self.qv); inner.reset(self.seed)
        picked = get_picked(inner, self.q0, target); cands = [x.reshape(4, n) for x in picked]; ths = np.asarray(MEM[key], float)
        for m in range(NPOOL[0]):            # member-own best candidate (member m plans on its own theta and its own reconstructed chain state)
            _, _, x0m = replay3(self.ctx, ths[m], self.pq0, taus); im = Tuned3S(self.ctx, ths[m], x0m, dur, self.qp, self.qv); im.reset(self.seed); cands.append(np.asarray(im.plan(self.q0, target), float).reshape(4, n))
        allth = np.concatenate([thc[None], ths]); S = np.full((len(cands), len(allth)), np.nan)          # column 0 = center, 1.. = members
        for j, th in enumerate(allth):
            _, _, xj = replay3(self.ctx, th, self.pq0, taus) if j else (None, None, x0c)
            ij = inner if j == 0 else Tuned3S(self.ctx, th, xj, dur, self.qp, self.qv)
            if j: ij.reset(self.seed)
            for c, Qk in enumerate(cands): S[c, j] = min(ij._closed_loop(self.q0, Qk, target), 1.0)
        self.rec = dict(cands=np.array(cands), S=S, base=int(np.argmin(S[:len(picked), 0]))); self.knots = cands[self.rec['base']]; self.plan_info = {}
def _metrics(sc, tr): m = episode_metrics(sc, tr); return dict(success=float(m['success']), min_dist=float(m['min_dist']), inst=int(INST[0]), diverged=float(m['diverged']))
def run_B(args):
    """(seed, execute_all) -> dict(S, cands, base, succ[c], dist[c]); candidate `base` is the ordinary L-ens episode"""
    seed, exe_all = args; guard15c(seed); pr = probe_of_lest(); amp = np.array([0.5, 0.5, 0.4]) * pr['amp']; out = {}
    def go(forced):
        reset_probe_globals(); INST[0] = 0; sc = scen(seed, 'abc', None); box = {}
        def fac(ctx): p = ProbeB(ctx, None, 'B', pr['T'], amp); p.forced = forced; box['p'] = p; return p
        tr = run_episode(sc, fac, int(seed)); return _metrics(sc, tr), box['p']
    m0, p0 = go(None); rec = p0.rec; C = len(rec['cands']); succ = np.full(C, np.nan); dist = np.full(C, np.nan); inst = np.zeros(C, int); b = rec['base']; succ[b], dist[b], inst[b] = m0['success'], m0['min_dist'], m0['inst']
    if exe_all:
        for c in range(C):
            if c == b: continue
            same = np.where(np.abs(rec['cands'] - rec['cands'][c]).max((1, 2)) < 1e-12)[0]
            if len(same) and same[0] < c: succ[c], dist[c], inst[c] = succ[same[0]], dist[same[0]], inst[same[0]]; continue
            m, _ = go(rec['cands'][c]); succ[c], dist[c], inst[c] = m['success'], m['min_dist'], m['inst']
    return dict(seed=int(seed), S=rec['S'], cands=rec['cands'], base=int(b), succ=succ, dist=dist, inst=inst)

def run_score(args):
    """(seed, cands, thetas) -> S columns (C, K) in the closed-loop models of the given thetas (episode aborted after the probe)"""
    seed, cands, ths = args; guard15c(seed); pr = probe_of_lest(); amp = np.array([0.5, 0.5, 0.4]) * pr['amp']; reset_probe_globals(); INST[0] = 0; sc = scen(seed, 'abc', None); box = {}
    def fac(ctx): p = ProbeB(ctx, None, 'B', pr['T'], amp); p.score = (cands, ths); box['p'] = p; return p
    try: run_episode(sc, fac, int(seed))
    except _Done: pass
    return box['p'].cols
