"""spike11a student: obs = noisy arm (q, qd) history + time + previous torque + task target (reference) -> tracking reference (qr - q, qdr, qddr) -> same CTC inner loop as the teacher.
Rollout with optional teacher labelling (fixed plan = T-oracle policy, or replanning every k steps), BC / DAgger / DART data collection, numpy MLP forward, torch training."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c11 import *
TAU_SC = 80.0
def feat_dim(H): return 6 + 6 * H + 1 + 3 + 2 + 3
class Feat:
    def __init__(self, H, stride): self.lags = [stride * (i + 1) for i in range(H)]; self.H = H; self.qh, self.vh = [], []; self.q1 = None
    def push(self, obs):
        if self.q1 is None: self.q1 = np.array(obs.q, float)
        self.qh.append(np.array(obs.q, float)); self.vh.append(np.array(obs.qd, float))
    def __call__(self, obs, prev_tau, target):
        q, v = self.qh[-1], self.vh[-1]; f = [q, v]
        for l in self.lags:
            j = max(len(self.qh) - 1 - l, 0); f.append(q - self.qh[j])
        for l in self.lags:
            j = max(len(self.qh) - 1 - l, 0); f.append(self.vh[j])
        f += [[obs.k / NSTEP], prev_tau / TAU_SC, target, self.q1]
        return np.concatenate([np.asarray(x, float).ravel() for x in f]).astype(np.float32)
def mlp_fwd(st, x):
    z = (x - st['mu']) / st['sd']
    for i, (W_, b) in enumerate(st['layers']):
        z = W_ @ z + b
        if i < len(st['layers']) - 1: z = np.tanh(z)
    return z * st['ysd'] + st['ymu']
_FLOWS = {}
def flow_model(st):
    import torch; torch.set_num_threads(1)
    if st['key'] not in _FLOWS:
        from arena.learn.models import Flow
        m = Flow(st['obs_dim'], 9, 1, st['width'], st['depth']); m.load_state_dict({k: torch.from_numpy(v) for k, v in st['sd'].items()}); m.eval(); _FLOWS[st['key']] = m
    return _FLOWS[st['key']]
def flow_y(st, x, z):
    import torch
    from arena.learn.models import time_embed
    m = flow_model(st)
    with torch.inference_mode():
        o = (torch.from_numpy(x.astype(np.float32))[None] - m.obs_mean) / m.obs_std; zz = torch.from_numpy(z.astype(np.float32))[None]; t = torch.zeros(1)
        y = zz + m._f(o, zz, t)                                  # ONE Euler step from z at t=0 (flow, steps=1)
        return (y * m.act_std + m.act_mean)[0].numpy().astype(np.float64)
def run_ep(args):
    """args = dict(seed, rep, cond, student(state or None), H, stride, label ('fixed'|int k), dart (sigma, rho) or None, record, cal, beta (prob. that the TEACHER acts each 25-step block))
    -> metrics (+ X, Y if record)"""
    a = args; seed, rep, cond = a['seed'], a['rep'], a['cond']; guard11([seed], a.get('cal', False)); INST[0] = 0; sc, env, obs = new_env(seed, cond, rep); ctx = env.ctx
    arm = ArmModel(ctx.robot); Kp, Kd = lqr_double_integrator(DEF['qp'], DEF['qv']); st = a.get('student'); fz = Feat(a['H'], a['stride']); target = np.asarray(obs.ref['target'], float)
    label = a.get('label'); T = Teacher(env, seed, cond, rep, None if (label in (None, 'fixed')) else int(label), cfg=a.get('cfg'), warm=a.get('warm', False), canon=a.get('canon', False)) if label is not None else None
    rng = np.random.default_rng([int(seed), 7711, int(a.get('salt', 0))]); dart = a.get('dart'); nz = np.zeros(3); beta = a.get('beta', 0.0); teach_blocks = {}
    zfix = (np.zeros(9) if a.get('zero_noise') else np.random.default_rng([int(seed), 4242]).standard_normal(9))
    X, Y = [], []; tips = [env.task.chain_tip(env).copy()]; prev = np.zeros(3); done = False; info = {}; fz.push(obs)
    prior = None
    if a.get('prior'): Pk = canon_x0(target)
    priv = None
    if a.get('priv'): th = true_theta(seed, cond, rep); priv = np.concatenate([[(th[0] - 1) / 0.2], np.log(th[1:11] / 0.015), [np.log(th[11])], [th[12] * 1e3 - 2.0]]).astype(np.float32)   # DIAGNOSTIC ONLY: true plant fed to the student
    while not done:
        k = obs.k; x = fz(obs, prev, target)
        if a.get('prior'):
            pq, pv, pa = (z[0] for z in spline_g(fz.q1, np.zeros(3), Pk, 1.6, np.array([k * DT]))); prior = np.concatenate([pq - obs.q, pv, pa]); x = np.concatenate([x, prior.astype(np.float32)])
        if priv is not None: x = np.concatenate([x, priv])
        if T is not None: T.maybe_plan(k, target)
        if T is not None and a.get('record'):
            qr, qdr, qddr = T.ref(k); X.append(x); Y.append((np.concatenate([qr - obs.q, qdr, qddr]) - (prior if prior is not None else 0.0)).astype(np.float32))
        blk = k // 25
        if blk not in teach_blocks: teach_blocks[blk] = bool(rng.random() < beta) if st is not None else True
        if st is None or teach_blocks[blk]:
            if T is None: raise ValueError('teacher control needs a teacher')
            qr, qdr, qddr = T.ref(k)
        else:
            y = (flow_y(st, x, zfix) if 'sd' in st else mlp_fwd(st, x.astype(np.float64))) + (prior if prior is not None else 0.0); qr, qdr, qddr = obs.q + y[:3], y[3:6], y[6:9]
        tau = ctc(arm, Kp, Kd, ctx.tau_limit, obs.q, obs.qd, qr, qdr, qddr)
        if dart is not None: nz = dart[1] * nz + np.sqrt(1 - dart[1] ** 2) * dart[0] * rng.standard_normal(3); tau = np.clip(tau + nz, -ctx.tau_limit, ctx.tau_limit)
        obs, _, term, trunc, info = env.step(tau); prev = tau; tips.append(env.task.chain_tip(env).copy()); done = term or trunc
        if not done: fz.push(obs)
    dist = np.linalg.norm(np.array(tips) - env.task.target, axis=1)
    out = dict(success=float(info['outcome'] == 'success'), min_dist=float(dist.min()), diverged=float(info.get('diverged', False)), inst=int(INST[0]), steps=env.k, plan_cost=float(np.sum(T.costs)) if T is not None else 0.0, nplan=len(T.costs) if T is not None else 0)
    if a.get('record'): out['X'] = np.array(X, np.float32); out['Y'] = np.array(Y, np.float32)
    return out
def train_mlp(X, Y, hid=256, depth=2, epochs=40, lr=1e-3, wd=1e-5, bs=512, seed=0, init=None, max_steps=40000, val_frac=0.05):
    import torch, torch.nn as nn; torch.set_num_threads(2); torch.manual_seed(seed); rs = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6; ymu, ysd = Y.mean(0), Y.std(0) + 1e-6; Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)); Yn = torch.from_numpy(((Y - ymu) / ysd).astype(np.float32))
    perm = rs.permutation(len(X)); nv = int(len(X) * val_frac); vi, ti = perm[:nv], perm[nv:]
    layers = []; d = X.shape[1]
    for _ in range(depth): layers += [nn.Linear(d, hid), nn.Tanh()]; d = hid
    net = nn.Sequential(*layers, nn.Linear(d, Y.shape[1]))
    steps_per = max(1, len(ti) // bs); epochs = int(min(epochs, max(5, max_steps // steps_per))); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ti = torch.from_numpy(ti)
    for ep in range(epochs):
        p = ti[torch.randperm(len(ti))]
        for i in range(0, len(p) - bs + 1, bs):
            idx = p[i:i + bs]; loss = ((net(Xn[idx]) - Yn[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad(): val = float(((net(Xn[torch.from_numpy(vi)]) - Yn[torch.from_numpy(vi)]) ** 2).mean()); per = ((net(Xn[torch.from_numpy(vi)]) - Yn[torch.from_numpy(vi)]) ** 2).mean(0).numpy().tolist()
    lay = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]
    return dict(mu=mu.astype(np.float64), sd=sd.astype(np.float64), ymu=ymu.astype(np.float64), ysd=ysd.astype(np.float64), layers=lay, val_mse_z=val, val_per=per, epochs=epochs, n=len(X))
