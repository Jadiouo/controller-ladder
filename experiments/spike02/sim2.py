"""spike02 plant with un-modelled weaknesses. Reuses experiments/spike/sim.py (unchanged). Teacher is the spike01 e=0 teacher (knows nothing of the weaknesses)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spike'))
import numpy as np, mujoco
import sim
from sim import (MJCF, DT_SIM, SUBSTEPS, DT_CTRL, NSTEP, M_PAYLOAD_TRUE, TAU_MAX, Teacher, ref_at,
                 sample_refs, sample_init_err, sample_dist)

SS_START = int(round(0.5 / DT_CTRL))   # steady-state window starts at 0.5 s
TEST_SEEDS = np.arange(50)

# weakness = dict(cap=torque limit Nm, delay=control steps (5 ms each), fc=Coulomb Nm, fv=viscous Nm s/rad)
W = {}   # filled by calibrated values below (see calib.py / README)
W['W0'] = dict(cap=80.0, delay=0, fc=0.0, fv=0.0)
# calibrated with calib.py on teacher only (calibration seeds 1000-1049). cap first chosen 27 (x2.0 on calib seeds) but on the 50 TEST seeds the teacher had 5/50 diverging trajectories (cliff) -> cap raised to 32 (teacher-only look at test seeds, no student involved). The nominal 10 ms delay did NOT hurt the teacher (x0.94) -> raised to 30 ms
W['W1'] = dict(cap=32.0, delay=0, fc=0.0, fv=0.0)
W['W2'] = dict(cap=80.0, delay=6, fc=0.0, fv=0.0)
W['W3'] = dict(cap=80.0, delay=0, fc=1.5, fv=0.3)
W['W4'] = dict(cap=32.0, delay=6, fc=1.5, fv=0.3)

def make_model_cap(cap):
    return mujoco.MjModel.from_xml_string(MJCF.format(dt=DT_SIM, m=M_PAYLOAD_TRUE, tm=cap))

class WeakEnv:
    """N parallel true plants (payload 1.5 kg) with weakness w. Step 0 identical in structure to sim.rollout."""
    def __init__(self, N, w):
        self.w, self.N = w, N
        self.model = make_model_cap(w['cap']); self.datas = [mujoco.MjData(self.model) for _ in range(N)]
        self.fc = np.full(2, w['fc']); self.fv = np.full(2, w['fv'])
    def reset(self, P, ie, dist=None):
        self.P, self.dist, self.k, self.hist = P, dist, 0, []
        q0, qd0, _ = ref_at(P, 0.0)
        for i, d in enumerate(self.datas):
            mujoco.mj_resetData(self.model, d)
            d.qpos[:] = q0[i] + ie[i, :2]; d.qvel[:] = qd0[i] + ie[i, 2:]
    def state(self):
        return np.array([d.qpos for d in self.datas]), np.array([d.qvel for d in self.datas])
    def refs(self):
        return ref_at(self.P, self.k * DT_CTRL)
    def step(self, cmd):
        """cmd (N,2) = commanded torque. Returns executed (post-delay, post-saturation) torque and saturation flag (N,)."""
        D, cap = self.w['delay'], self.w['cap']
        self.hist.append(np.array(cmd, dtype=float))
        applied = self.hist[max(0, len(self.hist) - 1 - D)]
        if len(self.hist) > D + 2: self.hist.pop(0)
        sat = (np.abs(applied) >= cap - 1e-9).any(1)
        ex = np.clip(applied, -cap, cap)
        dk = None if self.dist is None else self.dist[:, self.k]
        for i, d in enumerate(self.datas):
            d.ctrl[:] = ex[i]
            for _ in range(SUBSTEPS):
                f = -(self.fc * np.tanh(d.qvel / 0.05) + self.fv * d.qvel)
                if dk is not None: f = f + dk[i]
                d.qfrc_applied[:] = f
                mujoco.mj_step(self.model, d)
        self.k += 1
        return ex, sat

def rollout2(policy, w, P, ie, dist=None, exec_noise=None, record=False):
    """policy(q,qd,qr,qdr,qddr)->commanded tau (N,2). exec_noise added to the command (DART). Returns per-traj metrics."""
    N = len(P['c']); env = WeakEnv(N, w); env.reset(P, ie, dist)
    e2 = np.zeros((NSTEP, N)); en = np.zeros(N); sat = np.zeros(N)
    obs_l, tau_l = [], []
    for k in range(NSTEP):
        qr, qdr, qddr = env.refs(); q, qd = env.state()
        tau = policy(q, qd, qr, qdr, qddr)
        if record:
            obs_l.append(np.concatenate([q, qd, qr, qdr, qddr], 1)); tau_l.append(tau.copy())
        cmd = tau if exec_noise is None else np.clip(tau + exec_noise[:, k], -TAU_MAX, TAU_MAX)
        e2[k] = ((q - qr)**2).mean(1)
        ex, s = env.step(cmd)
        en += (ex**2).sum(1) * DT_CTRL; sat += s
    res = dict(rmse=np.sqrt(e2.mean(0)), ss_rmse=np.sqrt(e2[SS_START:].mean(0)), energy=en, sat=sat / NSTEP)
    if record:
        res['obs'] = np.concatenate(obs_l); res['tau'] = np.concatenate(tau_l)
    return res

def eval_policy(policy, w):
    """both test conditions on the fixed 50 test trajectories"""
    P = sample_refs(TEST_SEEDS); ie = sample_init_err(TEST_SEEDS); ds = sample_dist(TEST_SEEDS)
    return dict(clean=rollout2(policy, w, P, ie), dist=rollout2(policy, w, P, ie, dist=ds))
