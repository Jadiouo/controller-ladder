"""Residual PPO on top of the mirror-law teacher. action = clip(tau_T + 12 Nm * clip(a,-1,1), +-60). Reward: +1 per hit, -3*|apex err| at each apex, -3 on failure, tiny residual penalty. No test info."""
import argparse, pickle, time, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from stu import *
ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--steps', type=int, default=2_000_000)
ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--mb', type=int, default=2048); ap.add_argument('--out', default=None)
a = ap.parse_args(); cond = load_conds()[a.cond]; out = a.out or f'results/spike03b/raw_rl_{a.cond}.pkl'
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
GAM, LAM, CLIP, B, N, H = 0.995, 0.95, 0.2, E.RES_BOUND, a.nenv, a.horizon
R_HIT, C_APEX, R_FAIL, RES_PEN = 1.0, 3.0, 3.0, 2e-6
def mk(i, o, sc=1.0):
    n = nn.Sequential(nn.Linear(i, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
def train(seed):
    torch.manual_seed(seed); actor, critic = mk(OBS, 3, 0.01), mk(OBS, 1); logstd = nn.Parameter(torch.full((3,), -2.0))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [logstd], lr=a.lr)
    env = E.JugEnv(N, cond); ctr = [0]; base = 500_000 + seed * 100_000
    def newep(i):
        sd = base + ctr[0]; ctr[0] += 1; env.reset_one(i, E.init_cond([sd])[0], sd)
    env._alloc(); env.noise = [None]*N
    for i in range(N): newep(i)
    iters = a.steps // (N * H); t0 = time.time(); log = []; epret = np.zeros(N); rets = []; hitsl = []
    env.measure_update(); tT = env.teacher(); x = env.obs(tT)
    for it in range(iters):
        bx, ba, blp, bv, br, bd = [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x)
            with torch.no_grad():
                mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu)
                lp = (-0.5 * ((act - mu) / sd_)**2 - logstd - 0.9189385).sum(1); v = critic(xt).squeeze(1)
            an = act.numpy(); res_t = B * np.clip(an, -1, 1); cmd = np.clip(tT + res_t, -E.TAU_MAX, E.TAU_MAX)
            ev = env.step(cmd)
            r = R_HIT * ev['hit'] - C_APEX * np.nan_to_num(np.abs(ev['apex_err'])) - RES_PEN * (res_t**2).sum(1)
            fail = ev['done'] & (np.array(env.fail) != 'cap') & (np.array(env.fail) != 'trunc'); r = r - R_FAIL * fail
            epret += r; bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r.astype(np.float32)); bd.append(ev['done'].astype(np.float32))
            for i in np.where(ev['done'])[0]: rets.append(epret[i]); hitsl.append(env.hits[i]); epret[i] = 0; newep(i)
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
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()) + [logstd], 0.5); opt.step()
        if it % 10 == 0 or it == iters - 1:
            mh = float(np.median(hitsl[-30:])) if hitsl else float('nan'); log.append((it, (it + 1) * N * H, mh, time.time() - t0))
            print(f'  [{a.cond} s{seed}] it {it}/{iters} steps {(it+1)*N*H} recent-train-episodes median hits {mh} (n={len(hitsl)}) std {float(logstd.exp().mean().detach()):.3f} t {time.time()-t0:.0f}s', flush=True)
    return actor, log, time.time() - t0
t_all = time.time()
if 'teacher' not in res: res['teacher'] = evaluate(cond, None)
for seed in [int(s) for s in a.seeds.split(',')]:
    if ('rl', seed) in res: continue
    actor, log, tt = train(seed); r = evaluate(cond, ResPolicy(actor)); assert r['maxres'] <= E.RES_BOUND + 1e-6, r['maxres']
    res[('rl', seed)] = dict(**r, log=log, train_time=tt, steps=a.steps, state={k: v.clone() for k, v in actor.state_dict().items()})
    print(f'{a.cond} RL seed {seed}: test median hits {np.median(r["hits"])} (teacher {np.median(res["teacher"]["hits"])}) maxres {r["maxres"]:.2f} train {tt:.0f}s', flush=True)
    pickle.dump(res, open(out, 'wb'))
