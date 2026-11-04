"""2-DoF vertical-plane arm in MuJoCo, reference generation, computed-torque+LQR teacher, batched rollout."""
import numpy as np, mujoco
from scipy.linalg import solve_continuous_are

# ---------------- fixed experiment constants ----------------
DT_SIM = 0.0025          # MuJoCo timestep
SUBSTEPS = 2             # -> control period 5 ms = 200 Hz
DT_CTRL = DT_SIM * SUBSTEPS
T_EP = 4.0               # episode length (s)
NSTEP = int(round(T_EP / DT_CTRL))   # 800
M_PAYLOAD_TRUE = 1.5     # kg (true payload at tip)
TAU_MAX = 80.0           # Nm, actuator ctrlrange (both teacher and student clipped)
JR = np.array([[-1.2, 1.2], [0.3, 2.3]])   # reference range per joint (rad)

MJCF = """
<mujoco model="arm2">
  <option timestep="{dt}" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <body name="l1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.5" armature="0.01"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03" mass="1.0" contype="0" conaffinity="0"/>
      <body name="l2" pos="0.5 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.3" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.025" mass="0.8" contype="0" conaffinity="0"/>
        <body name="payload" pos="0.4 0 0">
          <geom type="sphere" size="0.04" mass="{m}" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1" ctrllimited="true" ctrlrange="-{tm} {tm}"/>
    <motor joint="j2" gear="1" ctrllimited="true" ctrlrange="-{tm} {tm}"/>
  </actuator>
</mujoco>
"""

def make_model(m_payload):
    return mujoco.MjModel.from_xml_string(MJCF.format(dt=DT_SIM, m=m_payload, tm=TAU_MAX))

# ---------------- reference trajectories ----------------
def sample_refs(seeds):
    """Each ref = per joint: center + 2 sines. Returns params dict of arrays (N,2,...)."""
    N = len(seeds); P = dict(c=np.zeros((N,2)), a=np.zeros((N,2,2)), w=np.zeros((N,2,2)), ph=np.zeros((N,2,2)))
    for i, s in enumerate(seeds):
        r = np.random.default_rng(int(s))
        for j in range(2):
            A = r.uniform(0.2, 0.6)
            frac = r.uniform(0.3, 0.7)
            a = np.array([A*frac, A*(1-frac)])
            lo, hi = JR[j]
            P['c'][i,j] = r.uniform(lo + A, hi - A)
            P['a'][i,j] = a
            P['w'][i,j] = 2*np.pi*r.uniform(0.2, 1.0, size=2)
            P['ph'][i,j] = r.uniform(0, 2*np.pi, size=2)
    return P

def ref_at(P, t):
    """returns qref, qdref, qddref each (N,2) at time t"""
    arg = P['w']*t + P['ph']
    q = P['c'] + (P['a']*np.sin(arg)).sum(-1)
    qd = (P['a']*P['w']*np.cos(arg)).sum(-1)
    qdd = -(P['a']*P['w']**2*np.sin(arg)).sum(-1)
    return q, qd, qdd

def sample_init_err(seeds):
    out = np.zeros((len(seeds), 4))
    for i, s in enumerate(seeds):
        r = np.random.default_rng(int(s) + 7_000_000)
        out[i] = np.concatenate([r.uniform(-0.15, 0.15, 2), r.uniform(-0.3, 0.3, 2)])
    return out

def sample_dist(seeds):
    """random torque pulses: 3 per episode, 0.1 s long, magnitude U(4,8) Nm random sign, random joint. -> (N,NSTEP,2)"""
    out = np.zeros((len(seeds), NSTEP, 2))
    for i, s in enumerate(seeds):
        r = np.random.default_rng(int(s) + 9_000_000)
        for _ in range(3):
            k0 = r.integers(int(0.3/DT_CTRL), NSTEP - int(0.3/DT_CTRL)); j = r.integers(0, 2)
            out[i, k0:k0+int(0.1/DT_CTRL), j] += r.choice([-1, 1]) * r.uniform(4, 8)
    return out

# ---------------- teacher ----------------
def lqr_gain(qp=1.6e5, qv=800.0):
    A = np.array([[0,1],[0,0.]]); B = np.array([[0],[1.]])
    Pm = solve_continuous_are(A, B, np.diag([qp, qv]), np.array([[1.]]))
    return (B.T @ Pm).ravel()   # [Kp, Kd]

class Teacher:
    """computed torque with nominal model (payload mass = true*(1+e)); LQR gain on per-joint double-integrator error dynamics."""
    def __init__(self, e):
        self.model = make_model(M_PAYLOAD_TRUE * (1 + e)); self.d = mujoco.MjData(self.model)
        self.K = lqr_gain()
    def act(self, q, qd, qr, qdr, qddr):
        out = np.zeros_like(q)
        for i in range(len(q)):
            v = qddr[i] + self.K[0]*(qr[i]-q[i]) + self.K[1]*(qdr[i]-qd[i])
            d = self.d; d.qpos[:] = q[i]; d.qvel[:] = qd[i]; d.qacc[:] = v
            d.qfrc_applied[:] = 0; d.ctrl[:] = 0
            mujoco.mj_inverse(self.model, d)
            out[i] = d.qfrc_inverse   # torque that produces qacc=v under nominal model
        return np.clip(out, -TAU_MAX, TAU_MAX)

# ---------------- batched rollout in the TRUE model ----------------
def rollout(policy, P, init_err, dist=None, exec_noise=None, record=False):
    """policy(q,qd,qr,qdr,qddr)->tau (N,2). exec_noise: (N,NSTEP,2) added to executed torque (DART); labels are clean.
    Returns per-traj rmse (N,), energy (N,), and optionally (obs, clean_tau) arrays."""
    model = make_model(M_PAYLOAD_TRUE); N = len(P['c'])
    datas = [mujoco.MjData(model) for _ in range(N)]
    q0, qd0, _ = ref_at(P, 0.0)
    for i, d in enumerate(datas):
        d.qpos[:] = q0[i] + init_err[i, :2]; d.qvel[:] = qd0[i] + init_err[i, 2:]
    se = np.zeros(N); en = np.zeros(N)
    obs_l, tau_l = [], []
    for k in range(NSTEP):
        t = k * DT_CTRL
        qr, qdr, qddr = ref_at(P, t)
        q = np.array([d.qpos for d in datas]); qd = np.array([d.qvel for d in datas])
        tau = policy(q, qd, qr, qdr, qddr)
        if record:
            obs_l.append(np.concatenate([q, qd, qr, qdr, qddr], 1)); tau_l.append(tau.copy())
        ex = tau if exec_noise is None else np.clip(tau + exec_noise[:, k], -TAU_MAX, TAU_MAX)
        se += ((q - qr)**2).mean(1); en += (ex**2).sum(1) * DT_CTRL
        for i, d in enumerate(datas):
            d.ctrl[:] = ex[i]
            d.qfrc_applied[:] = 0 if dist is None else dist[i, k]
            mujoco.mj_step(model, d, nstep=SUBSTEPS)
    res = dict(rmse=np.sqrt(se / NSTEP), energy=en)
    if record:
        res['obs'] = np.concatenate(obs_l); res['tau'] = np.concatenate(tau_l)   # (NSTEP*N, .)
    return res
