"""spike07 residual RL stacked on best-classical. algo ppo|sac, variant hist|dr|oracle. Derived from spike06/rl6.py (same env, reward, features, residual cap 12 Nm); NEW options:
  --cap_bootstrap 1 : episodes ending at the 300-hit cap are TRUNCATIONS (value bootstrapped with gamma*V(s_t)), not terminals (spike06 treated them as terminals)
  --target_kl x     : stop the PPO epochs of an iteration when the approx KL exceeds 1.5*x
  --bc c            : loss += c*mean(mu^2)  (behaviour cloning toward the teacher = zero residual)
  --nenv / --horizon: batch size;  --gamma
  7 equally spaced checkpoints (every iters/7) evaluated on CAL (seeds 1000-1049, mean action, cap 300); selected = best (frac_cap, mean); the untrained init is not a candidate.
  Residual norm (L2 over 3 joints, Nm) is logged for every CAL evaluation (mean over active steps, max).
Training seeds >= 500000 (initial condition and plant error from the seed). TEST seeds are never touched here (asserted)."""
import argparse, pickle, time, sys, os, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env7 as W; V = W.V; W.use_file()
torch.set_num_threads(2)
CAL = np.arange(1000, 1050)
def mk(i, o, hid, sc=1.0):
    n = nn.Sequential(nn.Linear(i, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
class SACActorMean(nn.Module):
    """evaluation wrapper of a SAC actor: deterministic action tanh(mean)"""
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return torch.tanh(self.net(x)[..., :3])
def oracle_extra(env):
    """true plant error, normalised: restitution, ball mass multiplier, sensor delay (ms). 3 columns."""
    return np.array([[(p['e'] - 0.585) * 4.0, (p['mb'] - 8.5) / 7.5, (p['delay_ms'] - 200.0) / 200.0] for p in env.pp], np.float32)
class Res:
    """evaluation policy: mean action (logstd None) or sampled. Tracks residual L2 norm over active steps."""
    def __init__(self, actor, logstd=None, seed=0, oracle=False): self.actor = actor; self.ls = logstd; self.rng = np.random.default_rng(seed); self.oracle = oracle; self.sum = 0.0; self.cnt = 0; self.mx = 0.0
    def __call__(self, env, tT, obs):
        if self.oracle: obs = np.concatenate([obs, oracle_extra(env)], 1)
        with torch.no_grad(): m = self.actor(torch.from_numpy(obs)).numpy()
        if self.ls is not None: m = m + np.exp(self.ls) * self.rng.standard_normal(m.shape)
        m = np.clip(m, -1, 1); env.prev_act = m.astype(np.float32); r = V.RES_BOUND * m
        n = np.linalg.norm(r, axis=1)[env.active]; self.sum += float(n.sum()); self.cnt += int(len(n)); self.mx = max(self.mx, float(n.max()) if len(n) else 0.0)
        return np.clip(tT + r, -V.TAU_MAX, V.TAU_MAX)
def evaluate(P, actor, seeds, hist, oracle=False):
    pol = None if actor is None else Res(actor, None, 0, oracle); r = V.evaluate(P, seeds, policy=pol, hist=hist)
    assert actor is None or r['maxres'] <= V.RES_BOUND * 1.7321 + 1e-6
    if pol is not None: r['res_mean'] = pol.sum / max(pol.cnt, 1); r['res_max'] = pol.mx
    return r
def summ(r):
    h = r['hits']; d = dict(frac_cap=float((h >= 300).mean()), mean=float(h.mean()), median=float(np.median(h)), steps=r['steps'])
    if 'res_mean' in r: d['res_mean'] = r['res_mean']; d['res_max'] = r['res_max']
    return d
def build_args(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--algo', default='ppo', choices=['ppo', 'sac']); ap.add_argument('--variant', required=True, choices=['dr', 'hist', 'oracle']); ap.add_argument('--classical', required=True)
    ap.add_argument('--seeds', default='0'); ap.add_argument('--steps', type=int, default=3_000_000); ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--mb', type=int, default=2048); ap.add_argument('--hid', type=int, default=256)
    ap.add_argument('--logstd0', type=float, default=-3.5); ap.add_argument('--logstd_min', type=float, default=-5.0); ap.add_argument('--out', default=None); ap.add_argument('--check', type=int, default=0)
    ap.add_argument('--nck', type=int, default=7); ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--cap_bootstrap', type=int, default=0)
    ap.add_argument('--target_kl', type=float, default=0.0); ap.add_argument('--bc', type=float, default=0.0)
    ap.add_argument('--utd', type=float, default=0.125); ap.add_argument('--sac_bs', type=int, default=256); ap.add_argument('--buf', type=int, default=300_000); ap.add_argument('--warm', type=int, default=30_000); ap.add_argument('--actor_delay', type=int, default=60_000)
    ap.add_argument('--tag', default=''); ap.add_argument('--alpha_fixed', type=float, default=0.0); ap.add_argument('--sac_ls_max', type=float, default=-3.0); ap.add_argument('--tgt_ent', type=float, default=-6.0); ap.add_argument('--c_apex', type=float, default=3.0); ap.add_argument('--r_fail', type=float, default=3.0); ap.add_argument('--ncal', type=int, default=50)  # ncal<50 only for crash smoke tests
    return ap.parse_args(argv)
GAM_DEFAULT = 0.995
R_HIT, RES_PEN = 1.0, 2e-6
def train(a, seed, PC):
    hist = a.variant in ('hist', 'oracle'); oracle = a.variant == 'oracle'; OBS = 23 + (V.HIST_K * V.HDIM if hist else 0) + (3 if oracle else 0)
    B, N = V.RES_BOUND, a.nenv; GAM = a.gamma; C_APEX, R_FAIL = a.c_apex, a.r_fail
    torch.manual_seed(seed); np.random.seed(seed)
    env = V.JugEnv5(N, PC, hist=hist); ctr = [0]; base = 500_000 + seed * 1_000_000
    def newep(i):
        sd = base + ctr[0]; ctr[0] += 1; assert sd >= 500000; env.reset_one(i, V.init_cond([sd])[0], sd)
    def getobs(tT):
        o = env.obs(tT); return np.concatenate([o, oracle_extra(env)], 1) if oracle else o
    for i in range(N): newep(i)
    env.measure_update(); tT = env.teacher(); x = getobs(tT); simsteps = 0; hitsl = []; log = []; ck = []; best_ck = None; ck_steps = 0; t0 = time.time()
    if a.algo == 'ppo':
        actor, critic = mk(OBS, 3, a.hid, 0.01), mk(OBS, 1, a.hid); logstd = nn.Parameter(torch.full((3,), a.logstd0)); evalactor = actor
    else:
        actor = mk(OBS, 6, a.hid, 0.01); evalactor = SACActor = SACActorMean(actor)
        with torch.no_grad(): actor[-1].weight[:3].zero_(); actor[-1].bias[:3].zero_(); actor[-1].bias[3:] = a.logstd0     # zero initial residual (the gain-0.01 init gave ~1 Nm and CAL 0-4% in the first SAC runs)
        q1, q2, q1t, q2t = [mk(OBS + 3, 1, a.hid) for _ in range(4)]; q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        log_alpha = torch.tensor(np.log(a.alpha_fixed if a.alpha_fixed > 0 else 0.02), dtype=torch.float32, requires_grad=True); TGT_ENT = a.tgt_ent
    ck_at = set(int(round((k + 1) * (a.steps // (N * (a.horizon if a.algo == 'ppo' else 1))) / a.nck)) for k in range(a.nck))
    def do_ckpt(it, extra):
        nonlocal best_ck, ck_steps
        rr = evaluate(PC, evalactor, CAL[:a.ncal], hist, oracle); sm = summ(rr); ck_steps += rr['steps']; sm['it'] = it; sm['train_steps'] = simsteps; sm.update(extra); ck.append(sm); print('  ckpt', sm, flush=True)
        if best_ck is None or (sm['frac_cap'], sm['mean']) > (best_ck[0]['frac_cap'], best_ck[0]['mean']): best_ck = (sm, {k: v.clone() for k, v in evalactor.state_dict().items()}) if a.algo == 'ppo' else (sm, {k: v.clone() for k, v in actor.state_dict().items()})
    def trainlog(it, nit, extra=''):
        rec = hitsl[-60:]; fc = float(np.mean(np.array(rec) >= 300)) if rec else float('nan'); log.append((it, simsteps, fc, time.time() - t0))
        print(f'  [{a.algo} {a.variant} s{seed}] it {it}/{nit} steps {simsteps} train-frac_cap(last60) {fc:.2f} meanhits {np.mean(rec) if rec else float("nan"):.1f} (n={len(hitsl)}) {extra} rss {int(open("/proc/self/statm").read().split()[1])*4096/1e9:.2f}GB t {time.time()-t0:.0f}s', flush=True)
    if a.algo == 'ppo':
        H = a.horizon; params = list(actor.parameters()) + list(critic.parameters()) + [logstd]; opt = torch.optim.Adam(params, lr=a.lr); iters = a.steps // (N * H); kls = []; cfs = []
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
                if a.cap_bootstrap: capd = ev['done'] & (fl == 'cap'); r = r + GAM * v.numpy() * capd
                bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r.astype(np.float32)); bd.append(ev['done'].astype(np.float32))
                for i in np.where(ev['done'])[0]: hitsl.append(env.hits[i]); newep(i)
                env.measure_update(); tT = env.teacher(); x = getobs(tT)
            with torch.no_grad(): lastv = critic(torch.from_numpy(x)).squeeze(1).numpy()
            R_, V_, D_ = np.array(br), np.array(bv), np.array(bd); adv = np.zeros_like(R_); g = 0
            for h in reversed(range(H)):
                nv = lastv if h == H - 1 else V_[h + 1]; nd = D_[h]
                delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * 0.95 * (1 - nd) * g; adv[h] = g
            ret = adv + V_
            X = torch.from_numpy(np.concatenate(bx)); A = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp))
            ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1)); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); n = len(X); stop = False; kl_it = []; cf_it = []
            for ep in range(a.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, a.mb):
                    idx = perm[i:i + a.mb]; mu = actor(X[idx]); lp = (-0.5 * ((A[idx] - mu) / logstd.exp())**2 - logstd - 0.9189385).sum(1)
                    ratio = (lp - LP[idx]).exp(); l1 = ratio * ADV[idx]; l2 = ratio.clamp(0.8, 1.2) * ADV[idx]
                    loss = -torch.min(l1, l2).mean() + 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx])**2).mean() + a.bc * (mu**2).mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                    with torch.no_grad():
                        logstd.clamp_(min=a.logstd_min); kl = float(((ratio - 1) - (ratio.log())).mean()); kl_it.append(kl); cf_it.append(float(((ratio - 1).abs() > 0.2).float().mean()))
                    if a.target_kl > 0 and kl > 1.5 * a.target_kl: stop = True; break
                if stop: break
            kls.append(float(np.mean(kl_it))); cfs.append(float(np.mean(cf_it)))
            if (it + 1) in ck_at: do_ckpt(it + 1, dict(kl=float(np.mean(kls[-20:])), clipfrac=float(np.mean(cfs[-20:]))))
            if it % 10 == 0 or it == iters - 1: trainlog(it, iters, f'kl {np.mean(kls[-10:]):.4f} clipfrac {np.mean(cfs[-10:]):.2f} std {float(logstd.exp().mean().detach()):.3f}')
        extra = dict(logstd=logstd.detach().numpy().copy()); nit = iters
    else:
        import torch.nn.functional as F
        n_buf = a.buf; ob = np.zeros((n_buf, OBS), np.float32); ac = np.zeros((n_buf, 3), np.float32); rw = np.zeros(n_buf, np.float32); nob = np.zeros((n_buf, OBS), np.float32); dn = np.zeros(n_buf, np.float32); ptr = 0; full = False
        aopt = torch.optim.Adam(actor.parameters(), lr=a.lr); copt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4); alopt = torch.optim.Adam([log_alpha], lr=3e-4)
        nit = a.steps // N; upd = 0; updacc = 0.0; TAU = 0.005
        def pol(xt, sample=True):
            o = actor(xt); mu, ls = o[:, :3], o[:, 3:].clamp(-5.0, -3.0); u = mu + ls.exp() * torch.randn_like(mu) if sample else mu; t = torch.tanh(u)
            lp = (-0.5 * ((u - mu) / ls.exp())**2 - ls - 0.9189385).sum(1) - torch.log(1 - t**2 + 1e-6).sum(1); return t, lp
        qlog = []
        for it in range(nit):
            xt = torch.from_numpy(x)
            with torch.no_grad(): t, _ = pol(xt)
            an = t.numpy(); res_t = B * an; cmd = np.clip(tT + res_t, -V.TAU_MAX, V.TAU_MAX); env.prev_act = an.astype(np.float32)
            simsteps += int(env.active.sum()); ev = env.step(cmd)
            r = R_HIT * ev['hit'] - C_APEX * np.nan_to_num(np.abs(ev['apex_err'])) - RES_PEN * (res_t**2).sum(1)
            fl = np.array(env.fail); fail = ev['done'] & (fl != 'cap') & (fl != 'trunc'); r = r - R_FAIL * fail
            x_prev = x
            for i in np.where(ev['done'])[0]: hitsl.append(env.hits[i]); newep(i)
            env.measure_update(); tT = env.teacher(); x = getobs(tT)
            nxt = np.where((ev['done'] & ~fail)[:, None], x_prev, x)      # cap truncation: next obs approximated by the current obs (bootstrapped); real failures are masked terminals
            sl = (ptr + np.arange(N)) % n_buf; ob[sl] = x_prev; ac[sl] = an; rw[sl] = r; nob[sl] = nxt; dn[sl] = fail.astype(np.float32); ptr = (ptr + N) % n_buf; full = full or ptr < N
            size = n_buf if full else ptr
            if simsteps >= a.warm and size >= 5000:
                updacc += N * a.utd
                while updacc >= 1:
                    updacc -= 1; j = np.random.randint(0, size, a.sac_bs); o_, a_, r_, n_, d_ = [torch.from_numpy(z[j]) for z in (ob, ac, rw, nob, dn)]
                    with torch.no_grad():
                        t2, lp2 = pol(n_); sa = torch.cat([n_, t2], 1); qt = torch.min(q1t(sa), q2t(sa)).squeeze(1) - log_alpha.exp() * lp2; y = r_ + GAM * (1 - d_) * qt
                    sa = torch.cat([o_, a_], 1); cl = F.mse_loss(q1(sa).squeeze(1), y) + F.mse_loss(q2(sa).squeeze(1), y)
                    copt.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0); copt.step()
                    if simsteps >= a.actor_delay:
                        t3, lp3 = pol(o_); sa3 = torch.cat([o_, t3], 1); al = (log_alpha.exp().detach() * lp3 - torch.min(q1(sa3), q2(sa3)).squeeze(1)).mean() + a.bc * (t3**2).mean()
                        aopt.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); aopt.step()
                        if a.alpha_fixed <= 0: alo = -(log_alpha * (lp3.detach() + TGT_ENT)).mean(); alopt.zero_grad(); alo.backward(); alopt.step()
                    with torch.no_grad():
                        for p, pt in zip(list(q1.parameters()) + list(q2.parameters()), list(q1t.parameters()) + list(q2t.parameters())): pt.mul_(1 - TAU).add_(TAU * p)
                    upd += 1
            if (it + 1) in ck_at: do_ckpt(it + 1, dict(updates=upd, alpha=float(log_alpha.exp())))
            if it % 2000 == 0 or it == nit - 1: trainlog(it, nit, f'upd {upd} alpha {float(log_alpha.exp()):.4f}')
        extra = dict(logstd=None, updates=upd)
    last_state = {k: v.clone() for k, v in (actor.state_dict()).items()}
    res = dict(log=log, train_time=time.time() - t0, steps=a.steps, env_steps_exec=simsteps, ckpt_eval_steps=ck_steps, ckpts=ck, last_state=last_state, state=(best_ck[1] if best_ck else last_state), selected=(best_ck[0] if best_ck else None),
               hid=a.hid, obs=OBS, variant=a.variant, algo=a.algo, args=vars(a), classical=a.classical, hitsl=hitsl, **extra)
    res['cal'] = best_ck[0]; print('CAL selected ckpt (mean action):', res['cal'], flush=True); return res
def load_actor(v):
    if v.get('algo', 'ppo') == 'ppo': n = mk(v['obs'], 3, v['hid']); n.load_state_dict(v['state']); return n
    n = mk(v['obs'], 6, v['hid']); n.load_state_dict(v['state']); return SACActorMean(n)
if __name__ == '__main__':
    a = build_args(); PC = json.load(open(a.classical))['params']
    if a.check:
        hist = a.variant in ('hist', 'oracle'); out = dict(classical=summ(evaluate(PC, None, CAL, False))); print('classical CAL', out['classical'], flush=True); json.dump(out, open(a.out, 'w'), indent=1); sys.exit()
    out = a.out; resd = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
    for s in [int(x) for x in a.seeds.split(',')]:
        if s in resd: continue
        resd[s] = train(a, s, PC); pickle.dump(resd, open(out, 'wb')); print(f'{a.algo} {a.variant} seed {s} done in {resd[s]["train_time"]:.0f}s', flush=True)
