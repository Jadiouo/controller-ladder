"""spike04: spike02 residual PPO (hyperparameters unchanged) under sensor noise + delay, base = C0 or C1* (fixed, acts on the same measured obs). Usage: python spike04/run_rl4.py --cond N1 --base C1STAR --seeds 0,1,2,3,4 --steps 2000000
Changes vs spike02/run_rl.py: env = NoisyEnv (obs = measured q,qd); reward uses TRUE state; base policy = Ctrl4 (stateful if LPF) called exactly once per observation; eval via evaluate4."""
import argparse, time, json, pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__)); import json
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
from lib4 import *
ap = argparse.ArgumentParser()
ap.add_argument('--cond', required=True); ap.add_argument('--base', required=True, choices=['C0', 'C1STAR']); ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--steps', type=int, default=1_500_000)
ap.add_argument('--bound', type=float, default=16.0, help='residual bound Nm (=20%% of the 80 Nm nominal torque limit, same for all W)')
ap.add_argument('--nenv', type=int, default=64); ap.add_argument('--horizon', type=int, default=200)
ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--mb', type=int, default=2048)
ap.add_argument('--tracking_w', type=float, default=200.0); ap.add_argument('--res_pen', type=float, default=1e-4)
ap.add_argument('--out', default=None); ap.add_argument('--no_pulses', action='store_true', help='ablation: train without torque pulses'); ap.add_argument('--fresh_eval', action='store_true', help='ablation: also evaluate on fresh seeds 2000-2049')
a = ap.parse_args(); w = COND[a.cond]; out = a.out or f'results/spike04/raw_rl_{a.cond}_{a.base}.pkl'
SIG = json.load(open('results/spike04/calib_sigma.json'))['sigma']; TUN = json.load(open(f'results/spike04/tuned_{a.cond}.json')) if a.base != 'C0' else None
def make_base():
    if a.base == 'C0': return Ctrl4(1.0, 1.0)
    c = TUN[TUN['Cstar']]; return Ctrl4(c['sp'], c['sd'], c.get('fc'))
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
T = make_base(); B = a.bound; N = a.nenv; H = a.horizon; GAM, LAM, CLIP = 0.99, 0.95, 0.2
SC = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/30, 1/30, 20, 20, 3, 3], dtype=np.float32)
def feats(q, qd, qr, qdr, qddr, tT):
    x = np.concatenate([q, qd, qr, qdr, qddr / 3, tT / 30, (q - qr) * 20, (qd - qdr) * 3], 1)   # spike01 obs + teacher action (+ error features)
    return x.astype(np.float32)
def mlp(i, o, h=128, out_scale=1.0):
    n = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, out_scale); return n

class ResidualPolicy:
    def __init__(self, actor): self.actor = actor; self.maxres = 0.0; self.T = make_base()
    def reset(self, N): self.T.reset(N)
    def __call__(self, q, qd, qr, qdr, qddr):
        tT = self.T.act(q, qd, qr, qdr, qddr)
        with torch.no_grad(): m = self.actor(torch.from_numpy(feats(q, qd, qr, qdr, qddr, tT))).numpy()
        out = np.clip(tT + B * np.clip(m, -1, 1), -TAU_MAX, TAU_MAX); self.maxres = max(self.maxres, float(np.abs(out - tT).max()))
        return out

def train(seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    actor, critic = mlp(16, 2, out_scale=0.01), mlp(16, 1, out_scale=1.0)
    logstd = nn.Parameter(torch.full((2,), -1.5))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [logstd], lr=a.lr)
    env = NoisyEnv(N, w); iters = a.steps // (N * H); ep_idx = 0; log = []
    t0 = time.time(); tsim = 0.
    for it in range(iters):
        if (it * H) % NSTEP == 0:   # new batch of episodes: fresh training scenarios (seeds disjoint from test 0..49 and calib)
            sd = 500_000 + seed * 100_000 + ep_idx * N + np.arange(N); ep_idx += 1
            P = sample_refs(sd); ie = sample_init_err(sd); ds = sample_dist(sd); ds[::2] = 0; ds[:] = 0 if a.no_pulses else ds   # half the episodes get torque pulses
            env.reset(P, ie, ds, make_noise(sd, SIG)); T.reset(N)
            q, qd = env.state(); qr, qdr, qddr = env.refs(); tT = T.act(q, qd, qr, qdr, qddr)
        x = feats(q, qd, qr, qdr, qddr, tT)
        bx, ba, blp, bv, br, bd = [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x)
            with torch.no_grad():
                mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu)
                lp = (-0.5 * ((act - mu) / sd_)**2 - logstd - 0.9189385).sum(1); v = critic(xt).squeeze(1)
            an = act.numpy(); cmd = np.clip(tT + B * np.clip(an, -1, 1), -TAU_MAX, TAU_MAX)
            ts = time.time(); env.step(cmd); q, qd = env.state(); qr, qdr, qddr = env.refs() if env.k < NSTEP else (qr, qdr, qddr); tsim += time.time() - ts
            r = -a.tracking_w * ((env.state_true()[0] - qr)**2).sum(1) - a.res_pen * ((B * np.clip(an, -1, 1))**2).sum(1)   # tracking error^2 + small residual-torque penalty
            done = env.k >= NSTEP
            bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r.astype(np.float32)); bd.append(np.full(N, float(done), dtype=np.float32))
            if not done: tT = T.act(q, qd, qr, qdr, qddr); x = feats(q, qd, qr, qdr, qddr, tT)
        with torch.no_grad(): lastv = critic(torch.from_numpy(x)).squeeze(1).numpy() if not done else np.zeros(N, np.float32)
        R_, V_, D_ = np.array(br), np.array(bv), np.array(bd); adv = np.zeros_like(R_); g = 0
        for h in reversed(range(H)):
            nv = lastv if h == H - 1 else V_[h + 1]; nd = D_[h]
            delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * LAM * (1 - nd) * g; adv[h] = g
        ret = adv + V_
        X = torch.from_numpy(np.concatenate(bx)); A = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp))
        ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1)); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(X)
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, a.mb):
                idx = perm[i:i + a.mb]; mu = actor(X[idx]); lp = (-0.5 * ((A[idx] - mu) / logstd.exp())**2 - logstd - 0.9189385).sum(1)
                ratio = (lp - LP[idx]).exp(); l1 = ratio * ADV[idx]; l2 = ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[idx]
                loss = -torch.min(l1, l2).mean() + 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx])**2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()) + [logstd], 0.5); opt.step()
        if it % 20 == 0 or it == iters - 1:
            trk = float(np.sqrt(-R_.mean() / a.tracking_w / 2)) if False else float(-R_.mean())
            log.append((it, (it + 1) * N * H, float(R_.mean()), float(logstd.exp().mean().detach()), time.time() - t0))
            print(f"  [{a.cond}/{a.base} s{seed}] it {it}/{iters} steps {(it+1)*N*H} meanR {R_.mean():.4f} std {float(logstd.exp().mean()):.3f} t {time.time()-t0:.0f}s (sim {tsim:.0f}s)", flush=True)
    return actor, log, time.time() - t0

t_all = time.time()
for seed in [int(s) for s in a.seeds.split(',')]:
    if ('rl', seed) in res: continue
    actor, log, tt = train(seed); t = time.time()
    pol = ResidualPolicy(actor); r = evaluate4(lambda: pol, w, TEST, SIG); r['maxres'] = pol.maxres; assert pol.maxres <= B + 1e-6
    te = time.time() - t
    # check residual bound is respected: |executed command - teacher command| <= B (by construction in ResidualPolicy; verify numerically)
    res[('rl', seed)] = dict(**r, log=log, train_time=tt, steps=a.steps, state={k: v.clone() for k, v in actor.state_dict().items()})
    dd = r['clean']
    print(f"{a.cond}/{a.base} rl seed={seed} rmse={dd['rmse'].mean():.4f} dist={r['dist']['rmse'].mean():.4f} train {tt:.0f}s eval {te:.0f}s", flush=True)
    pickle.dump(res, open(out, 'wb'))
json.dump(dict(args=vars(a), weak=w, total=time.time() - t_all), open(out.replace('.pkl', f'_timing_{a.seeds.replace(",","")}.json'), 'w'))
