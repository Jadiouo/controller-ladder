"""spike15b S5: spike07 residual PPO (variant hist, algo ppo) on the juggling task, derived from experiments/spike07/rl7.py; ONLY changes: (1) --logstd0 default -2.0 (the exploration scale under test), (2) optional G1 gate bias
(constant torque b*(1,-1,1) added to the command after teacher + residual), (3) CAL = 1000-1049 + 2000-2049, (4) PPO/hist only, (5) the maxres assertion of rl7 is dropped (bias is not a residual). TEST seeds blocked."""
import argparse, pickle, time, sys, os, json, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'spike07')); import env7 as W; V = W.V; W.use_file(os.path.join(HERE, '..', '..', 'results/spike06/dist.json'))
torch.set_num_threads(1)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CAL = np.concatenate([np.arange(1000, 1050), np.arange(2000, 2050)]); CAL_A = np.arange(1000, 1050)
SETS = dict(gate=np.arange(702000, 702500), holdout=np.arange(1100, 1600), fresh=np.arange(702500, 702700))
RUNS = os.path.join(ROOT, 'results/spike15b/runs_s5'); os.makedirs(RUNS, exist_ok=True)
def bias_vec(b): return np.array([1.0, -1.0, 1.0]) * float(b)
def mk(i, o, hid, sc=1.0):
    n = nn.Sequential(nn.Linear(i, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
class Res:
    """evaluation policy: mean action (+ optional bias). actor None = zero residual."""
    def __init__(self, actor, bias): self.actor = actor; self.bias = None if bias is None else np.asarray(bias, float); self.sum = 0.0; self.cnt = 0
    def __call__(self, env, tT, obs):
        if self.actor is None: m = np.zeros((len(tT), 3))
        else:
            with torch.no_grad(): m = self.actor(torch.from_numpy(obs)).numpy()
        m = np.clip(m, -1, 1); env.prev_act = m.astype(np.float32); r = V.RES_BOUND * m
        n = np.linalg.norm(r, axis=1)[env.active]; self.sum += float(n.sum()); self.cnt += int(len(n))
        cmd = tT + r
        if self.bias is not None: cmd = cmd + self.bias
        return np.clip(cmd, -V.TAU_MAX, V.TAU_MAX)
def evaluate(P, actor, seeds, bias=None):
    assert np.min(seeds) >= 1000, 'TEST seeds forbidden'
    pol = Res(actor, bias); r = V.evaluate(P, seeds, policy=pol, hist=True); r['res_mean'] = pol.sum / max(pol.cnt, 1); return r
def summ(r):
    h = r['hits']; return dict(frac_cap=float((h >= 300).mean()), mean=float(h.mean()), median=float(np.median(h)), steps=r['steps'], res_mean=r['res_mean'])
def build_args(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--classical', default=os.path.join(ROOT, 'results/spike07/best.json')); ap.add_argument('--seeds', default='0'); ap.add_argument('--steps', type=int, default=2_500_000)
    ap.add_argument('--nenv', type=int, default=128); ap.add_argument('--horizon', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--mb', type=int, default=8192); ap.add_argument('--hid', type=int, default=256)
    ap.add_argument('--logstd0', type=float, default=-2.0); ap.add_argument('--logstd_min', type=float, default=-5.0); ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--nck', type=int, default=7)
    ap.add_argument('--c_apex', type=float, default=0.3); ap.add_argument('--r_fail', type=float, default=3.0); ap.add_argument('--bias', type=float, default=0.0); return ap.parse_args(argv)
R_HIT, RES_PEN = 1.0, 2e-6
def train(a, seed, PC):
    OBS = 23 + V.HIST_K * V.HDIM; B, N = V.RES_BOUND, a.nenv; GAM = a.gamma; C_APEX, R_FAIL = a.c_apex, a.r_fail; bias = None if a.bias == 0 else bias_vec(a.bias)
    torch.manual_seed(seed); np.random.seed(seed)
    env = V.JugEnv5(N, PC, hist=True); ctr = [0]; base = 500_000 + seed * 1_000_000
    def newep(i):
        sd = base + ctr[0]; ctr[0] += 1; assert sd >= 500000; env.reset_one(i, V.init_cond([sd])[0], sd)
    for i in range(N): newep(i)
    env.measure_update(); tT = env.teacher(); x = env.obs(tT); simsteps = 0; hitsl = []; ck = []; best_ck = None; ck_steps = 0; t0 = time.time(); log = []
    actor, critic = mk(OBS, 3, a.hid, 0.01), mk(OBS, 1, a.hid); logstd = nn.Parameter(torch.full((3,), a.logstd0))
    iters = a.steps // (N * a.horizon); ck_at = set(int(round((k + 1) * iters / a.nck)) for k in range(a.nck))
    init = summ(evaluate(PC, None, CAL, bias)); ck_steps += init['steps']; print('  init CAL', init, flush=True)
    def do_ckpt(it, extra):
        nonlocal best_ck, ck_steps
        rr = evaluate(PC, actor, CAL, bias); sm = summ(rr); ck_steps += rr['steps']; sm['it'] = it; sm['train_steps'] = simsteps; sm.update(extra); ck.append(sm); print('  ckpt', sm, flush=True)
        if best_ck is None or (sm['frac_cap'], sm['mean']) > (best_ck[0]['frac_cap'], best_ck[0]['mean']): best_ck = (sm, {k: v.clone() for k, v in actor.state_dict().items()})
    H = a.horizon; params = list(actor.parameters()) + list(critic.parameters()) + [logstd]; opt = torch.optim.Adam(params, lr=a.lr); kls = []
    for it in range(iters):
        bx, ba, blp, bv, br, bd = [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x)
            with torch.no_grad():
                mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu)
                lp = (-0.5 * ((act - mu) / sd_)**2 - logstd - 0.9189385).sum(1); v = critic(xt).squeeze(1)
            an = act.numpy(); res_t = B * np.clip(an, -1, 1); cmd = tT + res_t
            if bias is not None: cmd = cmd + bias
            cmd = np.clip(cmd, -V.TAU_MAX, V.TAU_MAX); env.prev_act = np.clip(an, -1, 1).astype(np.float32)
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
            delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * 0.95 * (1 - nd) * g; adv[h] = g
        ret = adv + V_
        X = torch.from_numpy(np.concatenate(bx)); A = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp))
        ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1)); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); n = len(X); kl_it = []
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, a.mb):
                idx = perm[i:i + a.mb]; mu = actor(X[idx]); lp = (-0.5 * ((A[idx] - mu) / logstd.exp())**2 - logstd - 0.9189385).sum(1)
                ratio = (lp - LP[idx]).exp(); l1 = ratio * ADV[idx]; l2 = ratio.clamp(0.8, 1.2) * ADV[idx]
                loss = -torch.min(l1, l2).mean() + 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx])**2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                with torch.no_grad(): logstd.clamp_(min=a.logstd_min); kl_it.append(float(((ratio - 1) - (ratio.log())).mean()))
        kls.append(float(np.mean(kl_it)))
        if (it + 1) in ck_at: do_ckpt(it + 1, dict(kl=float(np.mean(kls[-20:]))))
        if it % 10 == 0 or it == iters - 1:
            rec = hitsl[-60:]; fc = float(np.mean(np.array(rec) >= 300)) if rec else float('nan'); log.append((it, simsteps, fc, time.time() - t0))
            print(f'  [{a.tag} s{seed}] it {it}/{iters} steps {simsteps} train-frac_cap(last60) {fc:.2f} meanhits {np.mean(rec) if rec else float("nan"):.1f} (n={len(hitsl)}) kl {np.mean(kls[-10:]):.4f} std {float(logstd.exp().mean().detach()):.3f} t {time.time()-t0:.0f}s', flush=True)
    return dict(seed=seed, init=init, ckpts=ck, selected=best_ck[0], state=best_ck[1], hid=a.hid, obs=OBS, train_episodes=len(hitsl), ckpt_eval_steps=ck_steps, env_steps_exec=simsteps, train_time=time.time() - t0, steps=a.steps, args=vars(a), log=log, logstd=logstd.detach().numpy().copy(), hitsl=hitsl)
def load_actor(v): n = mk(v['obs'], 3, v['hid']); n.load_state_dict(v['state']); return n
if __name__ == '__main__':
    a = build_args(); PC = json.load(open(a.classical))['params']; out = os.path.join(RUNS, a.tag + '.pkl'); resd = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
    for s in [int(x) for x in a.seeds.split(',')]:
        if s in resd: continue
        resd[s] = train(a, s, PC); pickle.dump(resd, open(out, 'wb')); r = resd[s]; print(f'{a.tag} seed {s} done in {r["train_time"]:.0f}s init {r["init"]["frac_cap"]:.3f} selected {r["selected"]["frac_cap"]:.3f}', flush=True)
