"""spike03a: tuned classical opponents. Reuses experiments/spike/sim.py and experiments/spike02/sim2.py unchanged."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spike02'))
import numpy as np, mujoco
from sim2 import *   # W, WeakEnv, sample_*, Teacher, DT_CTRL, NSTEP, TAU_MAX, SS_START
from sim import lqr_gain

CAL_SEEDS = np.arange(1000, 1050); TEST = np.arange(50)
BOUND = 16.0
THR = 0.02            # rad, settling / recovery threshold on max-joint |error|
W['W0'] = dict(cap=80.0, delay=0, fc=0.0, fv=0.0)

class Ctrl:
    """CTC (nominal model, e=0) with scaled LQR gains; optional integral term with anti-windup, extra torque clamped to +-BOUND."""
    def __init__(self, sp=1.0, sd=1.0, ki=0.0):
        self.T = Teacher(0.0); k = lqr_gain(); self.Kp, self.Kd, self.ki = k[0]*sp, k[1]*sd, ki; self.z = None
    def reset(self, N): self.z = np.zeros((N, 2))
    def _ctc(self, q, qd, v):
        T = self.T; out = np.zeros_like(q)
        for i in range(len(q)):
            d = T.d; d.qpos[:] = q[i]; d.qvel[:] = qd[i]; d.qacc[:] = v[i]; d.qfrc_applied[:] = 0; d.ctrl[:] = 0
            mujoco.mj_inverse(T.model, d); out[i] = d.qfrc_inverse
        return out
    def __call__(self, q, qd, qr, qdr, qddr):
        if self.z is None: self.reset(len(q))
        e, ed = qr - q, qdr - qd
        vb = qddr + self.Kp*e + self.Kd*ed
        tb = self._ctc(q, qd, vb)
        if self.ki == 0.0: return np.clip(tb, -TAU_MAX, TAU_MAX)
        tf = self._ctc(q, qd, vb + self.ki*self.z)
        ex = np.clip(tf - tb, -BOUND, BOUND)
        sat = np.abs(tf - tb) > BOUND
        self.z = np.where(sat & (np.sign(self.z) == np.sign(e)), self.z, self.z + e*DT_CTRL)   # conditional integration
        return np.clip(tb + ex, -TAU_MAX, TAU_MAX)

def rollout3(policy, w, P, ie, dist=None):
    """like sim2.rollout2 but keeps the max-joint error trace for settling/recovery metrics"""
    N = len(P['c']); env = WeakEnv(N, w); env.reset(P, ie, dist)
    if hasattr(policy, 'reset'): policy.reset(N)
    e2 = np.zeros((NSTEP, N)); emax = np.zeros((NSTEP, N)); en = np.zeros(N); sat = np.zeros(N)
    for k in range(NSTEP):
        qr, qdr, qddr = env.refs(); q, qd = env.state()
        tau = policy(q, qd, qr, qdr, qddr)
        e2[k] = ((q - qr)**2).mean(1); emax[k] = np.abs(q - qr).max(1)
        ex, s = env.step(tau); en += (ex**2).sum(1) * DT_CTRL; sat += s
    e2 = np.nan_to_num(e2, nan=4.0, posinf=4.0); emax = np.nan_to_num(emax, nan=4.0, posinf=4.0)
    return dict(rmse=np.sqrt(e2.mean(0)), ss_rmse=np.sqrt(e2[SS_START:].mean(0)), energy=en, sat=sat/NSTEP, emax=emax, dist=dist)

def settle_time(emax):
    """first time after which max-joint error stays < THR until end (s); 4.0 if never (censored)"""
    N = emax.shape[1]; out = np.full(N, NSTEP * DT_CTRL)
    for i in range(N):
        bad = np.where(emax[:, i] >= THR)[0]
        out[i] = 0.0 if len(bad) == 0 else (NSTEP if bad[-1] + 1 >= NSTEP else bad[-1] + 1) * DT_CTRL
    return out

def recovery_time(emax, dist):
    """mean over pulses (merged nonzero runs of the disturbance) of: time from pulse end until error stays < THR until next pulse onset / 1 s window; capped at the window length; 0 if error never exceeded THR after pulse onset"""
    N = emax.shape[1]; out = np.zeros(N)
    for i in range(N):
        on = np.abs(dist[i]).sum(1) > 0
        edges = np.diff(np.concatenate([[0], on.astype(int), [0]])); st = np.where(edges == 1)[0]; en = np.where(edges == -1)[0]
        vals = []
        for j, (s, e) in enumerate(zip(st, en)):
            wend = min(st[j+1] if j+1 < len(st) else NSTEP, e + int(1.0/DT_CTRL))
            seg = emax[s:wend, i]
            if (seg < THR).all(): vals.append(0.0); continue
            bad = np.where(seg >= THR)[0]; last = s + bad[-1] + 1
            vals.append(0.0 if last <= e else min((last - e) * DT_CTRL, (wend - e) * DT_CTRL))
        out[i] = np.mean(vals) if vals else 0.0
    return out

def evaluate(policy_factory, w, seeds):
    P = sample_refs(seeds); ie = sample_init_err(seeds); ds = sample_dist(seeds)
    c = rollout3(policy_factory(), w, P, ie); d = rollout3(policy_factory(), w, P, ie, dist=ds)
    return dict(clean=c, dist=d)

def objective(params, w, kind, pool=None):
    """calibration-set objective = 0.5*(mean RMSE clean) + 0.5*(mean RMSE dist), transient included"""
    sp, sd = np.exp(params[0]), np.exp(params[1]); ki = np.exp(params[2]) if kind == 'C2' else 0.0
    r = evaluate(lambda: Ctrl(sp, sd, ki), w, CAL_SEEDS)
    return 0.5*r['clean']['rmse'].mean() + 0.5*r['dist']['rmse'].mean(), r
