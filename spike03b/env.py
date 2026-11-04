"""spike03b: planar 3-DoF arm + flat racket + 1 ball (MuJoCo, x-z plane). Mirror-law teacher (Buhler/Koditschek style) + CTC joint control.
Conventions: link direction angle phi_i = cumulative joint angle, direction (cos phi, sin phi) in (x,z); hinge axis "0 -1 0".
Wrist (end of link 2) = racket plate centre; plate normal = (-sin phi3, cos phi3); phi3 = q1+q2+q3.
zb' = ball centre height above the plate-surface-contact level (z_c_wrist + R_BALL + HALF_T): ball centre at contact has zb'=0."""
import numpy as np, mujoco, json, os
DT_SIM = 0.001; SUB = 5; DT = DT_SIM * SUB          # control 200 Hz
G = 9.81; L1 = L2 = 0.5
R_BALL = 0.03; HALF_T = 0.008; M_BALL = 0.05; OFF = R_BALL + HALF_T
X_STAR = 0.6; H_STAR = 0.5; WRIST_Z0 = 0.0          # hit point (x*, wrist z), apex target above contact level
TAU_MAX = 60.0; RES_BOUND = 0.2 * TAU_MAX            # residual cap = 20% of torque limit = 12 Nm
HITS_CAP = 100; T_MAX = 100.0; NMAX = int(T_MAX / DT)
KAPPA_MAX = 0.40
# contact parameters (documented in README): solref = (TC, zeta), condim 1 (frictionless normal contact), default solimp
TC = 0.02
E_NOM = 0.8
M1, M2, M3 = 1.0, 0.8, 0.3                          # nominal link / racket masses (kg)
DAMP = (0.5, 0.3, 0.1)

XML = """<mujoco model="juggle"><option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast"/>
<default><geom contype="0" conaffinity="0"/></default>
<worldbody>
 <body name="l1" pos="0 0 0"><joint name="j1" type="hinge" axis="0 -1 0" damping="{d1}" armature="0.01"/>
  <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03" mass="{m1}"/>
  <body name="l2" pos="0.5 0 0"><joint name="j2" type="hinge" axis="0 -1 0" damping="{d2}" armature="0.01"/>
   <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.025" mass="{m2}"/>
   <body name="wrist" pos="0.5 0 0"><joint name="j3" type="hinge" axis="0 -1 0" damping="{d3}" armature="0.002"/>
    <geom name="racket" type="box" size="0.12 0.02 {ht}" mass="{m3}" contype="1" conaffinity="1" condim="1" solref="{tc} {zeta}"/>
    <site name="tip" pos="0 0 0"/></body></body></body>
{ball}</worldbody>
<actuator><motor joint="j1" ctrllimited="true" ctrlrange="-{tm} {tm}"/><motor joint="j2" ctrllimited="true" ctrlrange="-{tm} {tm}"/><motor joint="j3" ctrllimited="true" ctrlrange="-{tm} {tm}"/></actuator></mujoco>"""
BALL = """<body name="ball" pos="0 0 0"><joint name="bx" type="slide" axis="1 0 0"/><joint name="bz" type="slide" axis="0 0 1"/>
 <geom name="ballg" type="sphere" size="{r}" mass="{mb}" contype="1" conaffinity="1" condim="1" solref="{tc} {zeta}"/></body>"""

def build(zeta, mscale=1.0, ball=True):
    b = BALL.format(r=R_BALL, mb=M_BALL, tc=TC, zeta=zeta) if ball else ''
    return mujoco.MjModel.from_xml_string(XML.format(dt=DT_SIM, d1=DAMP[0], d2=DAMP[1], d3=DAMP[2], m1=M1*mscale, m2=M2*mscale, m3=M3*mscale,
                                                      ht=HALF_T, tc=TC, zeta=zeta, ball=b, tm=TAU_MAX))
# ---- restitution -> damping-ratio map (from sanity.py drop test, stored in zeta_map.json) ----
_ZP = os.path.join(os.path.dirname(__file__), 'zeta_map.json')
def zeta_for(e):
    m = json.load(open(_ZP)); es = np.array(m['e']); zs = np.array(m['zeta']); o = np.argsort(es)
    return float(np.interp(e, es[o], zs[o]))

# ---- kinematics ----
def fk(q):
    q = np.asarray(q); p1 = q[..., 0]; p12 = p1 + q[..., 1]
    return np.stack([L1*np.cos(p1) + L2*np.cos(p12), L1*np.sin(p1) + L2*np.sin(p12), p12 + q[..., 2]], -1)
def ik(x, z, phi):
    r2 = x*x + z*z; c2 = np.clip((r2 - L1*L1 - L2*L2) / (2*L1*L2), -1, 1); q2 = -np.arccos(c2)
    q1 = np.arctan2(z, x) - np.arctan2(L2*np.sin(q2), L1 + L2*np.cos(q2))
    return np.stack([q1, q2, phi - q1 - q2], -1)
def jac_inv_apply(q, pd):
    """qd = J^-1 pd for pd=(xd,zd,phid), batched"""
    p1 = q[:, 0]; p12 = p1 + q[:, 1]; s1, c1, s12, c12 = np.sin(p1), np.cos(p1), np.sin(p12), np.cos(p12)
    J = np.zeros((len(q), 3, 3)); J[:, 0, 0] = -L1*s1 - L2*s12; J[:, 0, 1] = -L2*s12; J[:, 1, 0] = L1*c1 + L2*c12; J[:, 1, 1] = L2*c12; J[:, 2, :] = 1
    return np.linalg.solve(J, pd[..., None])[..., 0]

Q0 = ik(np.array([X_STAR]), np.array([WRIST_Z0]), np.array([0.0]))[0]

class Cond:
    """true-plant condition. teacher model is always nominal (e=0.8, masses nominal, perfect sensing)."""
    def __init__(self, e=E_NOM, mscale=1.0, sigma=0.0002, delay=0, name='nominal'):
        self.e, self.mscale, self.sigma, self.delay, self.name = e, mscale, sigma, delay, name
    def __repr__(self): return f'Cond({self.name}: e={self.e}, m x{self.mscale}, sigma={self.sigma*1000:.1f}mm, delay={self.delay*DT*1000:.0f}ms)'

# teacher parameters (tuned once on nominal; see tune_teacher.py / README)
MODEL_MS = 1.0   # teacher's internal (CTC) model mass scale; 1.0 = nominal (only changed in the ctrl_teacher.py control)
TP = dict(pred=0, kE=1.0, Kx=1.1, vxmax=1.2, kp=900.0, kd=60.0, a=0.5, b=0.1, phimax=0.35)

def init_cond(seeds):
    """test / calib / train initial conditions from seed: ball x offset, drop height, vx, racket start offset (zero)"""
    out = []
    for s in seeds:
        r = np.random.default_rng(int(s))
        out.append(dict(x=X_STAR + r.uniform(-0.08, 0.08), h=r.uniform(0.2, 0.5), vx=r.uniform(-0.3, 0.3)))
    return out

class JugEnv:
    """N parallel plants. step(tau) -> arrays; holds controller-side estimator state (alpha-beta filter on delayed noisy ball position)."""
    def __init__(self, N, cond):
        self.N, self.c = N, cond
        self.model = build(zeta_for(cond.e), cond.mscale)
        self.nom = build(zeta_for(E_NOM), MODEL_MS, ball=False); self.nd = mujoco.MjData(self.nom)
        self.d = [mujoco.MjData(self.model) for _ in range(N)]
        self.active = np.zeros(N, bool); self.rngs = [None]*N
    def reset_one(self, i, ic, seed):
        d = self.d[i]; mujoco.mj_resetData(self.model, d)
        d.qpos[:3] = Q0; d.qpos[3] = ic['x']; d.qpos[4] = WRIST_Z0 + OFF + ic['h']; d.qvel[3] = ic['vx']
        mujoco.mj_forward(self.model, d)
        if not hasattr(self, 'est'): self._alloc()
        r = np.random.default_rng(int(seed) + 7_000_000); self.noise = getattr(self, 'noise', [None]*self.N); self.noise[i] = r.normal(0, 1, (NMAX + 5, 2)) * self.c.sigma
        self.buf[i] = [(ic['x'], d.qpos[4])] * (self.c.delay + 1)
        self.est[i] = [ic['x'], d.qpos[4] - OFF - WRIST_Z0, ic['vx'], 0.0]
        self.k[i] = 0; self.hits[i] = 0; self.active[i] = True; self.vzprev[i] = 0.0; self.fail[i] = ''
        self.ts[i] = 0; self.stats[i] = dict(apex_err=[], hit_x=[], tau2=0.0, nsteps=0, hit_t=[])
    def _alloc(self):
        N = self.N; self.est = np.zeros((N, 4)); self.k = np.zeros(N, int); self.hits = np.zeros(N, int); self.vzprev = np.zeros(N); self.dead = np.zeros(N, int)
        self.buf = [None]*N; self.fail = ['']*N; self.ts = np.zeros(N); self.stats = [None]*N
    def reset(self, ics, seeds):
        self._alloc()
        for i in range(self.N): self.reset_one(i, ics[i], seeds[i])
    # ---- observations / estimation ----
    def arm(self):
        return np.array([d.qpos[:3] for d in self.d]), np.array([d.qvel[:3] for d in self.d])
    def ball_true(self):
        return np.array([[d.qpos[3], d.qpos[4] - OFF - WRIST_Z0, d.qvel[3], d.qvel[4]] for d in self.d])
    def measure_update(self):
        """sensor: noisy + delayed ball position -> alpha-beta filter (velocity estimated). Called once per control step before control."""
        a, b = TP['a'], TP['b']
        for i in range(self.N):
            if not self.active[i]: continue
            d = self.d[i]; self.buf[i].append((d.qpos[3], d.qpos[4])); m = self.buf[i].pop(0)
            zx = m[0] + self.noise[i][self.k[i], 0]; zz = m[1] - OFF - WRIST_Z0 + self.noise[i][self.k[i], 1]
            e = self.est[i]; px = e[0] + e[2]*DT; pz = e[1] + e[3]*DT; pvz = e[3] - G*DT   # predict with gravity
            rx, rz = zx - px, zz - pz
            e[0] = px + a*rx; e[2] = e[2] + b/DT*rx; e[1] = pz + a*rz; e[3] = pvz + b/DT*rz
    # ---- teacher ----
    def teacher(self):
        """returns tau_T (N,3), refs (qr,qdr), kappa. Mirror law on estimated ball state, nominal model CTC."""
        e = self.est; q, qd = self.arm(); xh, zh, vx, vz = e.T
        if TP['pred']:
            tp = TP['pred']*DT; xh = xh + vx*tp; zh = zh + vz*tp - 0.5*G*tp*tp; vz = vz - G*tp
        E = G*np.maximum(zh, 0) + 0.5*vz**2
        kap = np.clip((1 - E_NOM)/(1 + E_NOM) + TP['kE']*(G*H_STAR - E)/G, 0, KAPPA_MAX)
        zr = WRIST_Z0 - kap*zh; zrd = -kap*vz
        xr = np.clip(xh, X_STAR - 0.35, X_STAR + 0.35); xrd = np.where(xr == xh, vx, 0.0)
        vimp = np.sqrt(vz**2 + 2*G*np.maximum(zh, 0)); vimp = np.maximum(vimp, 0.5)
        vxd = np.clip(TP['Kx']*(X_STAR - xh), -TP['vxmax'], TP['vxmax']); dvx = vxd - vx
        s2 = np.clip(2*dvx/((1 + E_NOM)*(1 + kap)*vimp), -0.95, 0.95); phi = np.clip(-0.5*np.arcsin(s2), -TP['phimax'], TP['phimax'])
        qr = ik(xr, zr, phi); qdr = jac_inv_apply(qr, np.stack([xrd, zrd, np.zeros(self.N)], 1))
        acc = TP['kp']*(qr - q) + TP['kd']*(qdr - qd)
        tau = np.zeros((self.N, 3)); nd = self.nd
        for i in range(self.N):
            nd.qpos[:] = q[i]; nd.qvel[:] = qd[i]; nd.qacc[:] = acc[i]; mujoco.mj_inverse(self.nom, nd); tau[i] = nd.qfrc_inverse
        self.last = dict(qr=qr, qdr=qdr, kap=kap)
        return np.clip(tau, -TAU_MAX, TAU_MAX)
    def obs(self, tT):
        q, qd = self.arm(); L = self.last; e = self.est
        return np.concatenate([q, qd/3, L['qr'], L['qdr']/3, (e[:, :1] - X_STAR)*5, e[:, 1:2]*2, e[:, 2:3], e[:, 3:4]/3, tT/30, L['kap'][:, None]*3, (q - L['qr'])*10], 1).astype(np.float32)
    # ---- dynamics ----
    def step(self, tau):
        """apply torques; returns dict of events (hit, apex, fail, done, apex_err) arrays"""
        N = self.N; hit = np.zeros(N, bool); apexe = np.full(N, np.nan); done = np.zeros(N, bool)
        for i in range(N):
            if not self.active[i]: continue
            d = self.d[i]; d.ctrl[:3] = tau[i]; vz0 = d.qvel[4]
            mujoco.mj_step(self.model, d, nstep=SUB)
            self.k[i] += 1; st = self.stats[i]; st['tau2'] += float((np.clip(tau[i], -TAU_MAX, TAU_MAX)**2).sum()) * DT; st['nsteps'] += 1
            zb = d.qpos[4] - OFF - WRIST_Z0; vz = d.qvel[4]
            if vz0 < -0.1 and vz > 0 and zb < 0.12:
                self.hits[i] += 1; hit[i] = True; st['hit_x'].append(abs(d.qpos[3] - X_STAR)); st['hit_t'].append(self.k[i]*DT)
            if vz0 > 0 and vz <= 0 and self.hits[i] >= 3:
                apexe[i] = zb - H_STAR; st['apex_err'].append(abs(zb - H_STAR))
            self.dead[i] = self.dead[i] + 1 if (zb < 0.04 and abs(vz) < 0.5) else 0
            f = ''
            if self.dead[i] > 60: f = 'dead'
            elif zb < -0.30: f = 'dropped'
            elif abs(d.qpos[3] - X_STAR) > 0.5: f = 'lateral'
            elif not np.isfinite(d.qpos).all(): f = 'nan'
            elif self.k[i] >= NMAX: f = 'timeout'
            if f: self.fail[i] = f; self.active[i] = False; done[i] = True
            elif self.hits[i] >= HITS_CAP: self.fail[i] = 'cap'; self.active[i] = False; done[i] = True
        return dict(hit=hit, apex_err=apexe, done=done)

def run_policy(env, policy, ics, seeds, record=None, maxk=None):
    """policy(env, tT, obs) -> tau (N,3); policy=None -> pure teacher. Returns per-episode dict arrays."""
    env.reset(ics, seeds); N = env.N; maxres = 0.0; frames = []
    while env.active.any():
        env.measure_update(); tT = env.teacher()
        tau = tT if policy is None else policy(env, tT, env.obs(tT))
        maxres = max(maxres, float(np.abs(np.clip(tau, -TAU_MAX, TAU_MAX) - tT)[env.active].max())) if env.active.any() else maxres
        if record is not None: record(env)
        env.step(tau)
        if maxk is not None:
            for i in np.where(env.active & (env.k >= maxk))[0]: env.active[i] = False; env.fail[i] = 'trunc'
    S = env.stats
    return dict(hits=env.hits.copy(), fail=list(env.fail), apex=np.array([np.mean(s['apex_err']) if s['apex_err'] else np.nan for s in S]),
                drift=np.array([np.mean(s['hit_x']) if s['hit_x'] else np.nan for s in S]),
                tau2=np.array([s['tau2']/max(s['nsteps']*DT, 1e-9) for s in S]), dur=np.array([s['nsteps']*DT for s in S]), maxres=maxres)
