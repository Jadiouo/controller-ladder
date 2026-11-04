"""spike05: H8 environment. Reuses spike03b/env.py (plant XML, kinematics, mirror-law teacher, alpha-beta filter) and adds
 (1) per-episode randomised plant (restitution e, ball mass scale, sensor delay) drawn from the episode seed,
 (2) a parametrised teacher (all tunable teacher parameters in dict P),
 (3) the classical online estimator for T-adapt (restitution from bounce velocity ratio, delay from impact time),
 (4) observation history for R-DR-hist.
Seed pools: test 0-49 (locked), calibration 1000-1049, RL training >= 500000. Plant parameters and initial condition both derive from the seed."""
import sys, os, json, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spike03b'))
import env as E
from env import DT, DT_SIM, SUB, G, L1, L2, OFF, X_STAR, H_STAR, WRIST_Z0, TAU_MAX, RES_BOUND, Q0, fk, ik, jac_inv_apply, TC

HITS_CAP = 300; T_MAX = 300.0; NMAX = int(T_MAX / DT)
SIGMA = 0.004                      # 4 mm ball position noise
E_RANGE = (0.40, 0.85); MB_RANGE = (1.0, 5.0); DELAY_MS_RANGE = (0.0, 150.0)
HERE = os.path.dirname(os.path.abspath(__file__))
_ZM = json.load(open(os.path.join(HERE, 'zeta_map5.json')))

def zeta_for(e):
    es = np.array(_ZM['e']); zs = np.array(_ZM['zeta']); o = np.argsort(es)
    return float(np.interp(e, es[o], zs[o]))

def draw_params(seed):
    """plant error for one episode, determined by seed only (identical for every method)"""
    r = np.random.default_rng(int(seed) + 13_000_000)
    e = r.uniform(*E_RANGE); mb = r.uniform(*MB_RANGE); dms = r.uniform(*DELAY_MS_RANGE)
    return dict(e=float(e), mb=float(mb), delay_ms=float(dms), delay=int(round(dms / (DT * 1000))))

def build5(zeta, mb_scale):
    ball = E.BALL.format(r=E.R_BALL, mb=E.M_BALL * mb_scale, tc=TC, zeta=zeta)
    return mujoco.MjModel.from_xml_string(E.XML.format(dt=DT_SIM, d1=E.DAMP[0], d2=E.DAMP[1], d3=E.DAMP[2], m1=E.M1, m2=E.M2, m3=E.M3,
                                                       ht=E.HALF_T, tc=TC, zeta=zeta, ball=ball, tm=TAU_MAX))

# ---- teacher parameters. T-nom = spike03b teacher (TP + KAPPA_MAX + E_NOM) ----
P_NOM = dict(kappa_max=0.40, e_m=0.80, kE=1.0, Kx=1.1, vxmax=1.2, kp=900.0, kd=60.0, a=0.5, b=0.1, phimax=0.35, pred=0.0,
             adapt=0, e0=0.7, n0e=2.0, bmin_e=0.15, d0=10.0, n0d=2.0, bmin_d=0.15, cd=1.0, c0=0.0, thr=0.06)
HF = [12, 13, 14, 15, 19, 20, 21, 22]          # obs columns kept in the history (ball state, kappa, tracking error)
HIST_K, HIST_STRIDE = 16, 16                   # 16 lags x 16 control steps (80 ms) = 1.28 s of history
HDIM = len(HF) + 3

class JugEnv5:
    """N parallel plants, each with its own (e, ball mass, delay) from its episode seed. API compatible with spike03b JugEnv / run_policy."""
    def __init__(self, N, P=None, hist=False, cap=None):
        self.N = N; self.P = dict(P_NOM if P is None else P); self.use_hist = hist; self.cap = HITS_CAP if cap is None else cap
        self.nom = E.build(zeta_for(0.8), 1.0, ball=False); self.nd = mujoco.MjData(self.nom)
        self.models = [None] * N; self.d = [None] * N; self.active = np.zeros(N, bool); self.pp = [None] * N
        self._alloc()
    def _alloc(self):
        N = self.N
        self.est = np.zeros((N, 4)); self.k = np.zeros(N, int); self.hits = np.zeros(N, int); self.vzprev = np.zeros(N); self.dead = np.zeros(N, int)
        self.buf = [None] * N; self.fail = [''] * N; self.ts = np.zeros(N); self.stats = [None] * N; self.noise = [None] * N
        self.ehat = np.full(N, self.P['e0']); self.dhat = np.full(N, self.P['d0']); self.ne = np.zeros(N); self.nde = np.zeros(N)
        self.raw = np.zeros((N, 256, 2)); self.qdprev = np.zeros((N, 3)); self.accprev = np.zeros((N, 3)); self.last_imp = np.full(N, -999)
        self.pend = [None] * N; self.log_e = [[] for _ in range(N)]; self.log_d = [[] for _ in range(N)]
        self.hbuf = np.zeros((N, HIST_K * HIST_STRIDE + 1, HDIM), np.float32); self.prev_act = np.zeros((N, 3), np.float32); self.hfresh = np.ones(N, bool)
    def reset(self, ics, seeds):
        self._alloc()
        for i in range(self.N): self.reset_one(i, ics[i], seeds[i])
    def reset_one(self, i, ic, seed):
        pp = draw_params(seed); self.pp[i] = pp
        self.models[i] = build5(zeta_for(pp['e']), pp['mb']); m = self.models[i]
        self.d[i] = mujoco.MjData(m); d = self.d[i]
        d.qpos[:3] = Q0; d.qpos[3] = ic['x']; d.qpos[4] = WRIST_Z0 + OFF + ic['h']; d.qvel[3] = ic['vx']
        mujoco.mj_forward(m, d)
        r = np.random.default_rng(int(seed) + 7_000_000); self.noise[i] = r.normal(0, 1, (NMAX + 5, 2)) * SIGMA
        self.buf[i] = [(ic['x'], d.qpos[4])] * (pp['delay'] + 1)
        self.est[i] = [ic['x'], d.qpos[4] - OFF - WRIST_Z0, ic['vx'], 0.0]
        self.k[i] = 0; self.hits[i] = 0; self.active[i] = True; self.vzprev[i] = 0.0; self.fail[i] = ''; self.dead[i] = 0
        self.ts[i] = 0; self.stats[i] = dict(apex_err=[], hit_x=[], tau2=0.0, nsteps=0, hit_t=[])
        self.ehat[i] = self.P['e0']; self.dhat[i] = self.P['d0']; self.ne[i] = 0; self.nde[i] = 0; self.last_imp[i] = -999; self.pend[i] = None
        self.qdprev[i] = 0; self.accprev[i] = 0; self.log_e[i] = []; self.log_d[i] = []
        self.hbuf[i] = 0; self.prev_act[i] = 0; self.hfresh[i] = True
    # ---- sensing / estimation ----
    def arm(self):
        return np.array([d.qpos[:3] for d in self.d]), np.array([d.qvel[:3] for d in self.d])
    def measure_update(self):
        a, b = self.P['a'], self.P['b']; ad = self.P['adapt']
        for i in range(self.N):
            if not self.active[i]: continue
            d = self.d[i]; k = self.k[i]; self.buf[i].append((d.qpos[3], d.qpos[4])); mm = self.buf[i].pop(0)
            nx, nz = self.noise[i][k]; zx = mm[0] + nx; zz = mm[1] - OFF - WRIST_Z0 + nz
            self.raw[i, k % 256] = (zx, mm[1] + nz)
            e = self.est[i]; px = e[0] + e[2] * DT; pz = e[1] + e[3] * DT; pvz = e[3] - G * DT
            rx, rz = zx - px, zz - pz
            e[0] = px + a * rx; e[2] = e[2] + b / DT * rx; e[1] = pz + a * rz; e[3] = pvz + b / DT * rz
            if ad: self._adapt(i, k, d)
    def _fit(self, i, j0, j1, t0):
        """least squares ballistic fit on raw samples idx j0..j1 (receive steps). x = ax + bx*tau ; z + g tau^2/2 = az + bz*tau; tau = (j - t0)*DT"""
        js = np.arange(j0, j1 + 1); tau = (js - t0) * DT; S = self.raw[i, js % 256]
        A = np.stack([np.ones_like(tau), tau], 1)
        cx = np.linalg.lstsq(A, S[:, 0], rcond=None)[0]; cz = np.linalg.lstsq(A, S[:, 1] + 0.5 * G * tau**2, rcond=None)[0]
        return cx, cz
    def _adapt(self, i, k, d):
        P = self.P; qd = d.qvel[:3].copy()
        # impact detection from encoders: unexplained jump of joint velocity vs the commanded acceleration
        if k >= 30 and (k - self.last_imp[i]) > 40 and self.pend[i] is None:
            s = np.abs(qd - self.qdprev[i] - self.accprev[i] * DT).max()
            if s > P['thr']:
                self.last_imp[i] = k; q = d.qpos[:3]; W = 20; j1 = k - 3; j0 = j1 - W
                if k - W - 3 >= 0:
                    cx, cz = self._fit(i, j0, j1, j1)
                    pw = fk(self._qprev[i]); ph = pw[2]; n = np.array([-np.sin(ph), np.cos(ph)]); zc = pw[1] + OFF * n[1]
                    disc = cz[1]**2 - 2 * G * (zc - cz[0])
                    if disc > 0 and cz[1] - G * 0 > -50:
                        tauc = (cz[1] + np.sqrt(disc)) / G; tc_idx = j1 + tauc / DT
                        d_steps = float(np.clip(tc_idx - (k - 1.5), 0, 40)); tm = tc_idx + 2.0
                        taum = (tm - j1) * DT; vin = np.array([cx[1], cz[1] - G * taum])
                        p1, p12 = self._qprev[i][0], self._qprev[i][0] + self._qprev[i][1]; qdp = self.qdprev[i]
                        vr = np.array([-L1 * np.sin(p1) * qdp[0] - L2 * np.sin(p12) * (qdp[0] + qdp[1]), L1 * np.cos(p1) * qdp[0] + L2 * np.cos(p12) * (qdp[0] + qdp[1])])
                        # delay update
                        nb = max(P['bmin_d'], 1.0 / (self.nde[i] + P['n0d'])); self.dhat[i] += nb * (d_steps - self.dhat[i]); self.nde[i] += 1; self.log_d[i].append(d_steps)
                        self.pend[i] = dict(j0=int(np.ceil(tc_idx)) + 6, tm=tm, vin=vin, vr=vr, n=n)
        if self.pend[i] is not None:
            pd = self.pend[i]; PN = 15
            if k >= pd['j0'] + PN - 1:
                j0 = pd['j0']; cx, cz = self._fit(i, j0, j0 + PN - 1, j0); taum = (pd['tm'] - j0) * DT
                vout = np.array([cx[1], cz[1] - G * taum]); n = pd['n']; den = float((pd['vin'] - pd['vr']) @ n); num = float((vout - pd['vr']) @ n)
                if den < -0.3:
                    eo = float(np.clip(-num / den, 0.2, 1.1)); nb = max(P['bmin_e'], 1.0 / (self.ne[i] + P['n0e']))
                    self.ehat[i] += nb * (eo - self.ehat[i]); self.ne[i] += 1; self.log_e[i].append(eo)
                self.pend[i] = None
        self.qdprev[i] = qd
    # ---- teacher (parametrised) ----
    def teacher(self):
        P = self.P; e = self.est; q, qd = self.arm(); xh, zh, vx, vz = e.T; N = self.N
        if P['adapt']:
            em = np.clip(self.ehat, 0.2, 0.95); pr = np.maximum(P['c0'] + P['cd'] * self.dhat, 0.0)
        else:
            em = np.full(N, P['e_m']); pr = np.full(N, P['pred'])
        tp = pr * DT; xh = xh + vx * tp; zh = zh + vz * tp - 0.5 * G * tp * tp; vz = vz - G * tp
        En = G * np.maximum(zh, 0) + 0.5 * vz**2
        kap = np.clip((1 - em) / (1 + em) + P['kE'] * (G * H_STAR - En) / G, 0, P['kappa_max'])
        zr = WRIST_Z0 - kap * zh; zrd = -kap * vz
        xr = np.clip(xh, X_STAR - 0.35, X_STAR + 0.35); xrd = np.where(xr == xh, vx, 0.0)
        vimp = np.sqrt(vz**2 + 2 * G * np.maximum(zh, 0)); vimp = np.maximum(vimp, 0.5)
        vxd = np.clip(P['Kx'] * (X_STAR - xh), -P['vxmax'], P['vxmax']); dvx = vxd - vx
        s2 = np.clip(2 * dvx / ((1 + em) * (1 + kap) * vimp), -0.95, 0.95); phi = np.clip(-0.5 * np.arcsin(s2), -P['phimax'], P['phimax'])
        qr = ik(xr, zr, phi); qdr = jac_inv_apply(qr, np.stack([xrd, zrd, np.zeros(N)], 1))
        acc = P['kp'] * (qr - q) + P['kd'] * (qdr - qd)
        tau = np.zeros((N, 3)); nd = self.nd
        for i in range(N):
            nd.qpos[:] = q[i]; nd.qvel[:] = qd[i]; nd.qacc[:] = acc[i]; mujoco.mj_inverse(self.nom, nd); tau[i] = nd.qfrc_inverse
        self.last = dict(qr=qr, qdr=qdr, kap=kap); self.accprev = acc; self._qprev = q.copy()
        return np.clip(tau, -TAU_MAX, TAU_MAX)
    def obs(self, tT):
        q, qd = self.arm(); L = self.last; e = self.est
        o = np.concatenate([q, qd / 3, L['qr'], L['qdr'] / 3, (e[:, :1] - X_STAR) * 5, e[:, 1:2] * 2, e[:, 2:3], e[:, 3:4] / 3, tT / 30, L['kap'][:, None] * 3, (q - L['qr']) * 10], 1).astype(np.float32)
        if not self.use_hist: return o
        f = np.concatenate([o[:, HF], self.prev_act], 1)
        for i in np.where(self.hfresh)[0]: self.hbuf[i] = f[i]; self.hfresh[i] = False
        self.hbuf = np.roll(self.hbuf, 1, axis=1); self.hbuf[:, 0] = f           # lag 0 at index 0
        lags = self.hbuf[:, HIST_STRIDE::HIST_STRIDE][:, :HIST_K].reshape(self.N, -1)   # lags 16,32,...,256 (wraps are fresh-filled)
        return np.concatenate([o, lags], 1)
    # ---- dynamics ----
    def step(self, tau):
        N = self.N; hit = np.zeros(N, bool); apexe = np.full(N, np.nan); done = np.zeros(N, bool)
        for i in range(N):
            if not self.active[i]: continue
            d = self.d[i]; d.ctrl[:3] = tau[i]; vz0 = d.qvel[4]
            mujoco.mj_step(self.models[i], d, nstep=SUB)
            self.k[i] += 1; st = self.stats[i]; st['tau2'] += float((np.clip(tau[i], -TAU_MAX, TAU_MAX)**2).sum()) * DT; st['nsteps'] += 1
            zb = d.qpos[4] - OFF - WRIST_Z0; vz = d.qvel[4]
            if vz0 < -0.1 and vz > 0 and zb < 0.12:
                self.hits[i] += 1; hit[i] = True; st['hit_x'].append(abs(d.qpos[3] - X_STAR)); st['hit_t'].append(self.k[i] * DT)
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
            elif self.hits[i] >= self.cap: self.fail[i] = 'cap'; self.active[i] = False; done[i] = True
        return dict(hit=hit, apex_err=apexe, done=done)

def run_policy(env, policy, ics, seeds, record=None, maxk=None):
    """policy(env, tT, obs) -> tau (N,3); None = pure (classical) teacher"""
    env.reset(ics, seeds); N = env.N; maxres = 0.0
    while env.active.any():
        env.measure_update(); tT = env.teacher()
        tau = tT if policy is None else policy(env, tT, env.obs(tT))
        if policy is not None: maxres = max(maxres, float(np.abs(np.clip(tau, -TAU_MAX, TAU_MAX) - tT)[env.active].max()))
        if record is not None: record(env)
        env.step(tau)
        if maxk is not None:
            for i in np.where(env.active & (env.k >= maxk))[0]: env.active[i] = False; env.fail[i] = 'trunc'
    S = env.stats
    out = dict(hits=env.hits.copy(), fail=list(env.fail), apex=np.array([np.mean(s['apex_err']) if s['apex_err'] else np.nan for s in S]),
               drift=np.array([np.mean(s['hit_x']) if s['hit_x'] else np.nan for s in S]),
               tau2=np.array([s['tau2'] / max(s['nsteps'] * DT, 1e-9) for s in S]), dur=np.array([s['nsteps'] * DT for s in S]), maxres=maxres,
               steps=int(sum(s['nsteps'] for s in S)), params=[dict(p) for p in env.pp])
    if env.P['adapt']: out['e_est'] = [list(x) for x in env.log_e]; out['d_est'] = [list(x) for x in env.log_d]; out['ehat_final'] = env.ehat.copy(); out['dhat_final'] = env.dhat.copy()
    return out

def init_cond(seeds): return E.init_cond(seeds)
def evaluate(P, seeds, policy=None, hist=False, cap=None, maxk=None):
    env = JugEnv5(len(seeds), P, hist=hist, cap=cap); return run_policy(env, policy, init_cond(seeds), seeds, maxk=maxk)
