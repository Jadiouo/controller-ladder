"""Residual PPO + domain randomisation on top of the spike03b mirror-law teacher (T-nom). Same PPO/reward/residual cap (12 Nm) as experiments/spike03b/run_rl.py.
variant dr: 23-d spike03b observation. variant hist: same 23-d + 16 lags (stride 16 control steps = 80 ms, 1.28 s) of 11 features (ball est x,z,vx,vz, kappa, tracking error q-qr (3), previous residual action (3)).
Training episodes: seeds >= 500000 (initial condition AND plant error drawn from the seed). Test set is NOT touched here (evaluation: eval_all.py)."""
import argparse, pickle, time, sys, os, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env5 as V
torch.set_num_threads(2)
ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True, choices=['dr', 'hist']); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--steps', type=int, default=3_000_000)
ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--mb', type=int, default=2048); ap.add_argument('--hid', type=int, default=256); ap.add_argument('--out', default=None)
a = ap.parse_args(); out = a.out or f'results/spike05/raw_rl_{a.variant}.pkl'; hist = a.variant == 'hist'
OBS = 23 + (V.HIST_K * V.HDIM if hist else 0)
GAM, LAM, CLIP, B, N, H = 0.995, 0.95, 0.2, V.RES_BOUND, a.nenv, a.horizon
R_HIT, C_APEX, R_FAIL, RES_PEN = 1.0, 3.0, 3.0, 2e-6
def mk(i, o, sc=1.0):
    n = nn.Sequential(nn.Linear(i, a.hid), nn.Tanh(), nn.Linear(a.hid, a.hid), nn.Tanh(), nn.Linear(a.hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
def train(seed):
    torch.manual_seed(seed); np.random.seed(seed); actor, critic = mk(OBS, 3, 0.01), mk(OBS, 1); logstd = nn.Parameter(torch.full((3,), -2.0))
    params = list(actor.parameters()) + list(critic.parameters()) + [logstd]; opt = torch.optim.Adam(params, lr=a.lr)
    env = V.JugEnv5(N, V.P_NOM, hist=hist); ctr = [0]; base = 500_000 + seed * 1_000_000
    def newep(i):
        sd = base + ctr[0]; ctr[0] += 1; assert sd >= 500000; env.reset_one(i, V.init_cond([sd])[0], sd)
    for i in range(N): newep(i)
    iters = a.steps // (N * H); t0 = time.time(); log = []; epret = np.zeros(N); hitsl = []
    env.measure_update(); tT = env.teacher(); x = env.obs(tT); simsteps = 0
    for it in range(iters):
        bx, ba, blp, bv, br, bd = [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x)
            with torch.no_grad():
                mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu)
                lp = (-0.5 * ((act - mu) / sd_)**2 - logstd - 0.9189385).sum(1); v = critic(xt).squeeze(1)
            an = act.numpy(); res_t = B * np.clip(an, -1, 1); cmd = np.clip(tT + res_t, -V.TAU_MAX, V.TAU_MAX); env.prev_act = np.clip(an, -1, 1).astype(np.float32)
            simsteps += int(env.active.sum()); ev = env.step(cmd)
            r = R_HIT * ev['hit'] - C_APEX * np.nan_to_num(np.abs(ev['apex_err'])) - RES_PEN * (res_t**2).sum(1)
            fl = np.array(env.fail); fail = ev['done'] & (fl != 'cap') & (fl != 'trunc'); r = r - R_FAIL * fail
            epret += r; bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r.astype(np.float32)); bd.append(ev['done'].astype(np.float32))
            for i in np.where(ev['done'])[0]: hitsl.append(env.hits[i]); epret[i] = 0; newep(i)
            env.measure_update(); tT = env.teacher(); x = env.obs(tT)
        with torch.no_grad(): lastv = critic(torch.from_numpy(x)).squeeze(1).numpy()
        R_, V_, D_ = np.array(br), np.array(bv), np.array(bd); adv = np.zeros_like(R_); g = 0
        for h in reversed(range(H)):
            nv = lastv if h == H - 1 else V_[h + 1]; nd = D_[h]
            delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * LAM * (1 - nd) * g; adv[h] = g
        ret = adv + V_
        X = torch.from_numpy(np.concatenate(bx)); A = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp))
        ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1)); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); n = len(X)
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, a.mb):
                idx = perm[i:i + a.mb]; mu = actor(X[idx]); lp = (-0.5 * ((A[idx] - mu) / logstd.exp())**2 - logstd - 0.9189385).sum(1)
                ratio = (lp - LP[idx]).exp(); l1 = ratio * ADV[idx]; l2 = ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[idx]
                loss = -torch.min(l1, l2).mean() + 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx])**2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 0.5); opt.step()
        if it % 10 == 0 or it == iters - 1:
            mh = float(np.median(hitsl[-30:])) if hitsl else float('nan'); log.append((it, (it + 1) * N * H, mh, time.time() - t0))
            print(f'  [{a.variant} s{seed}] it {it}/{iters} steps {(it+1)*N*H} recent-train-episodes median hits {mh} (n={len(hitsl)}) std {float(logstd.exp().mean().detach()):.3f} t {time.time()-t0:.0f}s', flush=True)
    return actor, log, time.time() - t0, simsteps
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
for seed in [int(s) for s in a.seeds.split(',')]:
    if seed in res: continue
    actor, log, tt, ss = train(seed)
    res[seed] = dict(log=log, train_time=tt, steps=a.steps, env_steps_exec=ss, state={k: v.clone() for k, v in actor.state_dict().items()}, hid=a.hid, obs=OBS, variant=a.variant)
    pickle.dump(res, open(out, 'wb')); print(f'{a.variant} seed {seed} trained in {tt:.0f}s', flush=True)
