"""spike06 residual PPO stacked on a CLASSICAL controller (R-on-best). Corrected recipe vs spike05: log-std init <= -3.5 with floor -5, lr <= 1e-4, 4 epochs.
Everything else = spike05/diag/run_rl5_fix.py (PPO gamma .995 lambda .95 clip .2, 32 envs x 256, minibatch 2048, tanh 256x256, reward +1/hit -3|apex err| -3 fail -2e-6|res|^2, residual cap 12 Nm).
Residual is added to the classical torque: tau = clip(tau_classical + 12*clip(a,-1,1)). Observation/history features = spike05 (23-d, + 16 lags x 11 features for hist).
Training seeds >= 500000 (initial condition AND plant error from the seed). The error distribution is results/spike06/dist.json. TEST seeds are never touched here.
modes:  train   : --variant dr|hist --classical results/spike06/<best>.json --seeds 0 --steps N --lr --logstd0 --out X.pkl   (also evaluates the trained policy on CAL, mean action)
        check   : same args + --check 1 : no training; evaluates the INITIAL policy on CAL with mean action and with sample action, and the classical alone."""
import argparse, pickle, time, sys, os, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env6 as W; V = W.V; W.use_file()
torch.set_num_threads(2)
CAL = np.arange(1000, 1050)
def mk(i, o, hid, sc=1.0):
    n = nn.Sequential(nn.Linear(i, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
class Res:
    """evaluation policy. logstd=None -> mean action; else sample action (the exact distribution used in training)."""
    def __init__(self, actor, logstd=None, seed=0): self.actor = actor; self.ls = logstd; self.rng = np.random.default_rng(seed)
    def __call__(self, env, tT, obs):
        with torch.no_grad(): m = self.actor(torch.from_numpy(obs)).numpy()
        if self.ls is not None: m = m + np.exp(self.ls) * self.rng.standard_normal(m.shape)
        m = np.clip(m, -1, 1); env.prev_act = m.astype(np.float32)
        return np.clip(tT + V.RES_BOUND * m, -V.TAU_MAX, V.TAU_MAX)
def evaluate(P, actor, seeds, hist, logstd=None):
    r = V.evaluate(P, seeds, policy=None if actor is None else Res(actor, logstd), hist=hist); assert r['maxres'] <= V.RES_BOUND + 1e-6 or actor is None
    return r
def summ(r): h = r['hits']; return dict(frac_cap=float((h >= 300).mean()), mean=float(h.mean()), median=float(np.median(h)), steps=r['steps'])
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True, choices=['dr', 'hist']); ap.add_argument('--classical', required=True); ap.add_argument('--seeds', default='0')
    ap.add_argument('--steps', type=int, default=3_000_000); ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--mb', type=int, default=2048); ap.add_argument('--hid', type=int, default=256)
    ap.add_argument('--logstd0', type=float, default=-3.5); ap.add_argument('--logstd_min', type=float, default=-5.0); ap.add_argument('--out', default=None); ap.add_argument('--check', type=int, default=0)
    ap.add_argument('--cal_eval', type=int, default=1); ap.add_argument('--ckpt_every', type=int, default=60)
    a = ap.parse_args(); hist = a.variant == 'hist'; OBS = 23 + (V.HIST_K * V.HDIM if hist else 0)
    PC = json.load(open(a.classical))['params']
    GAM, LAM, CLIP, B, N, H = 0.995, 0.95, 0.2, V.RES_BOUND, a.nenv, a.horizon
    R_HIT, C_APEX, R_FAIL, RES_PEN = 1.0, 3.0, 3.0, 2e-6
    def train(seed):
        torch.manual_seed(seed); np.random.seed(seed); actor, critic = mk(OBS, 3, a.hid, 0.01), mk(OBS, 1, a.hid); logstd = nn.Parameter(torch.full((3,), a.logstd0))
        if a.check:
            out = {}
            for nm, ls in (('mean', None), ('sample', logstd.detach().numpy().copy())): out[nm] = summ(evaluate(PC, actor, CAL, hist, ls)); print('init', a.variant, nm, out[nm], flush=True)
            out['classical'] = summ(evaluate(PC, None, CAL, False)); print('classical', out['classical'], flush=True); return out
        params = list(actor.parameters()) + list(critic.parameters()) + [logstd]; opt = torch.optim.Adam(params, lr=a.lr)
        env = V.JugEnv5(N, PC, hist=hist); ctr = [0]; base = 500_000 + seed * 1_000_000
        def newep(i):
            sd = base + ctr[0]; ctr[0] += 1; assert sd >= 500000; env.reset_one(i, V.init_cond([sd])[0], sd)
        for i in range(N): newep(i)
        iters = a.steps // (N * H); t0 = time.time(); log = []; hitsl = []; ck = []; best_ck = None; ck_steps = 0
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
                bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r.astype(np.float32)); bd.append(ev['done'].astype(np.float32))
                for i in np.where(ev['done'])[0]: hitsl.append(env.hits[i]); newep(i)
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
                    with torch.no_grad(): logstd.clamp_(min=a.logstd_min)
            if a.ckpt_every and ((it + 1) % a.ckpt_every == 0 or it == iters - 1):
                # checkpoint selection on CAL (mean action, cap 300). The untrained initial policy (= classical) is NOT a candidate; it is reported separately by --check.
                rr = evaluate(PC, actor, CAL, hist); sm = summ(rr); ck_steps += rr['steps']; sm['it'] = it + 1; ck.append(sm); print('  ckpt', sm, flush=True)
                if best_ck is None or (sm['frac_cap'], sm['mean']) > (best_ck[0]['frac_cap'], best_ck[0]['mean']): best_ck = (sm, {k: v.clone() for k, v in actor.state_dict().items()})
            if it % 10 == 0 or it == iters - 1:
                rec = hitsl[-60:]; fc = float(np.mean(np.array(rec) >= 300)) if rec else float('nan'); log.append((it, (it + 1) * N * H, fc, time.time() - t0))
                print(f'  [{a.variant} s{seed}] it {it}/{iters} steps {(it+1)*N*H} recent-train-episodes frac_cap {fc:.2f} mean hits {np.mean(rec) if rec else float("nan"):.1f} (n={len(hitsl)}) std {float(logstd.exp().mean().detach()):.3f} t {time.time()-t0:.0f}s', flush=True)
        last_state = {k: v.clone() for k, v in actor.state_dict().items()}
        res = dict(log=log, train_time=time.time() - t0, steps=a.steps, env_steps_exec=simsteps, ckpt_eval_steps=ck_steps, ckpts=ck, last_state=last_state, state=(best_ck[1] if best_ck else last_state), selected=(best_ck[0] if best_ck else None), hid=a.hid, obs=OBS, variant=a.variant,
                   logstd=logstd.detach().numpy().copy(), args=vars(a), classical=a.classical)
        res['cal'] = best_ck[0] if best_ck else summ(evaluate(PC, actor, CAL, hist)); print('CAL selected ckpt (mean action):', res['cal'], flush=True)
        return res
    if a.check: json.dump({a.variant: train(0)}, open(a.out, 'w'), indent=1); sys.exit()
    out = a.out; resd = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
    for s in [int(x) for x in a.seeds.split(',')]:
        if s in resd: continue
        resd[s] = train(s); pickle.dump(resd, open(out, 'wb')); print(f'{a.variant} seed {s} trained in {resd[s]["train_time"]:.0f}s', flush=True)
