"""spike04: sensor noise + delay. Imports experiments/spike03a/lib.py (and through it experiments/spike/sim.py, experiments/spike02/sim2.py) unchanged."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spike03a'))
from lib import *     # np, mujoco, W, WeakEnv, Teacher, Ctrl, sample_*, settle_time, recovery_time, CAL_SEEDS, TEST, BOUND, DT_CTRL, NSTEP, TAU_MAX, SS_START

COND = {   # plant weakness part of each condition (sigma is set separately, from results/spike04/calib_sigma.json)
    'N1': dict(cap=80.0, delay=6, fc=0.0, fv=0.0),   # noise + 30 ms delay
    'N2': dict(cap=80.0, delay=0, fc=0.0, fv=0.0),   # noise only
}
NOISE_OFF = 11_000_000

def make_noise(seeds, sigma):
    """(N, NSTEP+2, 2) angle noise [rad]; index k = noise on the measurement at step k; last two entries = previous-sample noise for k=0 differencing. Seeded per trajectory seed -> identical for every method."""
    out = np.zeros((len(seeds), NSTEP + 2, 2))
    for i, s in enumerate(seeds): out[i] = np.random.default_rng(int(s) + NOISE_OFF).standard_normal((NSTEP + 2, 2)) * sigma
    return out

class NoisyEnv(WeakEnv):
    """state() = MEASURED (q, qd): q+noise, qd = finite difference of the noisy measured angles. state_true() = real state (metrics, reward)."""
    def reset(self, P, ie, dist=None, noise=None):
        super().reset(P, ie, dist); self.nz = noise; self._c = None; self._qm = None
    def state_true(self): return WeakEnv.state(self)
    def state(self):
        qt, qdt = WeakEnv.state(self)
        if self.nz is None: return qt, qdt
        if self._c is not None and self._c[0] == self.k: return self._c[1], self._c[2]
        qm = qt + self.nz[:, self.k]
        prev = self._qm if (self._qm is not None and self.k > 0) else qt - qdt*DT_CTRL + self.nz[:, NSTEP + 1]
        qdm = (qm - prev) / DT_CTRL
        self._qm = qm; self._c = (self.k, qm, qdm)
        return qm, qdm

class Ctrl4(Ctrl):
    """Ctrl (CTC + scaled LQR gains) acting on the measured state, optional first-order low-pass (cutoff fc Hz) on measured q and qd."""
    def __init__(self, sp=1.0, sd=1.0, fc=None):
        super().__init__(sp, sd, 0.0); self.fc = fc; self.f = None
        self.alpha = None if fc is None else 1 - np.exp(-2*np.pi*fc*DT_CTRL)
    def reset(self, N): self.z = np.zeros((N, 2)); self.f = None
    def __call__(self, q, qd, qr, qdr, qddr):
        if self.alpha is not None:
            if self.f is None: self.f = (q.copy(), qd.copy())
            else: self.f = (self.f[0] + self.alpha*(q - self.f[0]), self.f[1] + self.alpha*(qd - self.f[1]))
            q, qd = self.f
        return super().__call__(q, qd, qr, qdr, qddr)
    act = __call__

def rollout4(policy, w, P, ie, dist, noise):
    N = len(P['c']); env = NoisyEnv(N, w); env.reset(P, ie, dist, noise)
    if hasattr(policy, 'reset'): policy.reset(N)
    e2 = np.zeros((NSTEP, N)); emax = np.zeros((NSTEP, N)); en = np.zeros(N); sat = np.zeros(N)
    for k in range(NSTEP):
        qr, qdr, qddr = env.refs(); qm, qdm = env.state(); qt, _ = env.state_true()
        tau = policy(qm, qdm, qr, qdr, qddr)
        e2[k] = ((qt - qr)**2).mean(1); emax[k] = np.abs(qt - qr).max(1)
        ex, s = env.step(tau); en += (ex**2).sum(1) * DT_CTRL; sat += s
    e2 = np.nan_to_num(e2, nan=4.0, posinf=4.0); emax = np.nan_to_num(emax, nan=4.0, posinf=4.0)
    return dict(rmse=np.sqrt(e2.mean(0)), ss_rmse=np.sqrt(e2[SS_START:].mean(0)), energy=en, sat=sat/NSTEP, emax=emax, dist=dist)

def evaluate4(policy_factory, w, seeds, sigma):
    P = sample_refs(seeds); ie = sample_init_err(seeds); ds = sample_dist(seeds); nz = make_noise(seeds, sigma) if sigma > 0 else None
    return dict(clean=rollout4(policy_factory(), w, P, ie, None, nz), dist=rollout4(policy_factory(), w, P, ie, ds, nz))

def objective4(params, w, kind, sigma):
    """calibration objective (same as spike03a) = 0.5*mean RMSE clean + 0.5*mean RMSE dist. kind: C1 (Kp,Kd scale) or C1F (+ LPF cutoff)"""
    sp, sd = np.exp(params[0]), np.exp(params[1]); fc = np.exp(params[2]) if kind == 'C1F' else None
    r = evaluate4(lambda: Ctrl4(sp, sd, fc), w, CAL_SEEDS, sigma)
    return 0.5*r['clean']['rmse'].mean() + 0.5*r['dist']['rmse'].mean(), r
