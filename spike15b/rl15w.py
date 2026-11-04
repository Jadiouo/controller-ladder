"""spike15b S6: residual PPO on the idealised whip (spike09b setting, base T-adapt) with the exploration scale as the ONLY change vs spike09b/train9.py,
plus the G1 gate bias (constant post-probe torque b*(1,-1,1)). Seeds: TEST (<1000) is blocked by guard(); only CAL / holdout / fresh / gate seeds are used."""
import sys, os, time, json, pickle, argparse, multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spike09b'))
from c9b import *
import torch, torch.nn as nn; torch.set_num_threads(2)
CAL_A = list(range(1000, 1050)); CAL = CAL_A + list(range(2000, 2150))
SETS = dict(gate=list(range(702000, 702500)), holdout=list(range(1100, 1600)), fresh=list(range(702500, 702700)))
RUNS = os.path.join(ROOT, 'results/spike15b/runs_s6'); os.makedirs(RUNS, exist_ok=True)
def bias_vec(b): return np.array([1.0, -1.0, 1.0]) * float(b)
class ResEnvB(ResEnv):
    def __init__(self, base, fz, seed, rep, bias=None): self.bias = np.zeros(3) if bias is None else np.asarray(bias, float); super().__init__(base, fz, seed, rep)
    def step(self, a):
        a = np.clip(a, -1, 1); on = self.obs.k >= getattr(self.base, 'n_pr', 0); res = BOUND * a * on
        tau = np.clip(self.tb + res + self.bias * on, -self.ctx.tau_limit, self.ctx.tau_limit)
        if on: self.rn.append(float(np.linalg.norm(res)))
        obs, _, term, trunc, info = self.env.step(tau); self.en.append(float((info['applied'] ** 2).sum())); self.tips.append(self.task.chain_tip(self.env).copy()); d = float(np.linalg.norm(self.task.chain_tip(self.env) - self.task.target)); r = 10.0 * max(self.dmin - d, 0.0); self.dmin = min(self.dmin, d)
        succ = info['outcome'] == 'success'; r += 5.0 * succ - 5.0 * (info['outcome'] == 'failure') - 2e-6 * float((res ** 2).sum()); done = term or trunc
        if not done: self.obs = obs; self.fz.push(obs); self.tb = self.base.act(obs); self.x = self.fz(obs, self.tb, self.base, self.yt)
        return self.x, r, done, dict(success=succ, min_dist=self.dmin)
def run15(args):
    state, seed, bias = args[:3]; base = args[3] if len(args) > 3 else 'adapt'; guard([seed]); e = ResEnvB(base, Feat(True, False), seed, None, bias); done = False; x = e.x
    while not done:
        a = np.zeros(3) if state is None else np.clip(np_mlp(state, x.astype(np.float64)), -1, 1); x, r, done, info = e.step(a)
    return dict(success=float(info['success']), min_dist=float(e.dmin), res_mean=float(np.mean(e.rn)) if e.rn else 0.0, res_max=float(np.max(e.rn)) if e.rn else 0.0)
def eval_set(pool, state, seeds, bias, base='adapt'):
    r = pool.map(run15, [(state, s, bias, base) for s in seeds], chunksize=1)
    return dict(succ=float(np.mean([x['success'] for x in r])), min_dist=float(np.mean([x['min_dist'] for x in r])), res_mean=float(np.mean([x['res_mean'] for x in r])), res_max=float(np.max([x['res_max'] for x in r])), episodes=len(seeds), per=np.array([x['success'] for x in r]))
def worker(conn, wid, E, bias, M, seed, bname='adapt'):
    rng = np.random.default_rng([seed, wid, 77]); envs = [None] * E
    def new(i): envs[i] = ResEnvB(bname, Feat(True, False), 500000 + int(rng.integers(0, M)), 0, bias)
    while True:
        cmd, a = conn.recv()
        if cmd == 'reset':
            for i in range(E): new(i)
            conn.send(np.stack([e.x for e in envs]))
        elif cmd == 'step':
            xs, rs, ds, eps = [], [], [], []
            for i in range(E):
                x, r, d, info = envs[i].step(a[i])
                if d: eps.append(dict(success=float(info['success']))); new(i); x = envs[i].x
                xs.append(x); rs.append(r); ds.append(d)
            conn.send((np.stack(xs), np.array(rs, np.float32), np.array(ds, np.float32), eps))
        elif cmd == 'close': break
def mk(i, o, hid, sc=1.0):
    n = nn.Sequential(nn.Linear(i, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
def layers_of(net): return [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]
def train(a, seed, evalpool):
    torch.manual_seed(seed); np.random.seed(seed); fz = Feat(True, False); OBS = fz.dim; N, H = a.nenv, a.horizon; GAM = a.gamma; bias = None if a.bias == 0 else bias_vec(a.bias).tolist()
    nw = a.workers; E = N // nw; conns, procs = [], []
    for w in range(nw):
        pc, cc = mp.Pipe(); p = mp.Process(target=worker, args=(cc, w, E, bias, a.pool, seed * 1000 + 7, a.base), daemon=True); p.start(); conns.append(pc); procs.append(p)
    for c in conns: c.send(('reset', None))
    x = np.concatenate([c.recv() for c in conns]); actor, critic = mk(OBS, 3, a.hid, 0.01), mk(OBS, 1, a.hid); logstd = nn.Parameter(torch.full((3,), a.logstd0))
    with torch.no_grad(): actor[-1].weight.zero_(); actor[-1].bias.zero_()
    params = list(actor.parameters()) + list(critic.parameters()) + [logstd]; opt = torch.optim.Adam(params, lr=a.lr); iters = a.steps // (N * H); ck_at = set(int(round((k + 1) * iters / 7)) for k in range(7))
    init = eval_set(evalpool, layers_of(actor), CAL, bias, a.base); print('  init CAL', {k: v for k, v in init.items() if k != 'per'}, flush=True); cks = []; best = None; eps_done = 0; succ_hist = []; t0 = time.time(); log = []; eval_eps = init['episodes']
    for it in range(iters):
        bx, ba, blp, bv, br, bd = [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x)
            with torch.no_grad(): mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu); lp = (-0.5 * ((act - mu) / sd_) ** 2 - logstd - 0.9189385).sum(1); v = critic(xt).squeeze(1)
            an = act.numpy(); an_c = np.clip(an, -1, 1)
            for w, c in enumerate(conns): c.send(('step', an_c[w * E:(w + 1) * E]))
            outs = [c.recv() for c in conns]; xn = np.concatenate([o[0] for o in outs]); r = np.concatenate([o[1] for o in outs]); d = np.concatenate([o[2] for o in outs])
            for o in outs:
                for e in o[3]: eps_done += 1; succ_hist.append(e['success'])
            bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r); bd.append(d); x = xn
        with torch.no_grad(): lastv = critic(torch.from_numpy(x)).squeeze(1).numpy()
        R_, V_, D_ = np.array(br), np.array(bv), np.array(bd); adv = np.zeros_like(R_); g = 0
        for h in reversed(range(H)):
            nv = lastv if h == H - 1 else V_[h + 1]; nd = D_[h]; delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * 0.95 * (1 - nd) * g; adv[h] = g
        ret = adv + V_; X = torch.from_numpy(np.concatenate(bx)); A_ = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp)); ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); n = len(X); kl_it = []
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, a.mb):
                idx = perm[i:i + a.mb]; mu = actor(X[idx]); lp = (-0.5 * ((A_[idx] - mu) / logstd.exp()) ** 2 - logstd - 0.9189385).sum(1); ratio = (lp - LP[idx]).exp()
                loss = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]).mean() + 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                with torch.no_grad(): logstd.clamp_(min=a.logstd_min); kl_it.append(float(((ratio - 1) - ratio.log()).mean()))
        if (it + 1) in ck_at:
            c = eval_set(evalpool, layers_of(actor), CAL, bias, a.base); eval_eps += c['episodes']; c.update(it=it + 1, train_steps=(it + 1) * N * H, train_episodes=eps_done); cks.append(c); print('  ckpt', {k: v for k, v in c.items() if k != 'per'}, flush=True)
            if best is None or (c['succ'], -c['min_dist']) > (best[0]['succ'], -best[0]['min_dist']): best = (c, layers_of(actor))
        if it % 10 == 0 or it == iters - 1:
            rec = succ_hist[-200:]; log.append((it, eps_done, float(np.mean(rec)) if rec else float('nan'), time.time() - t0)); print(f'  [{a.tag} s{seed}] it {it}/{iters} eps {eps_done} train-succ(last200) {np.mean(rec) if rec else float("nan"):.3f} kl {np.mean(kl_it):.4f} std {float(logstd.exp().mean().detach()):.3f} t {time.time()-t0:.0f}s', flush=True)
    for c in conns: c.send(('close', None))
    for p in procs: p.join(timeout=5)
    sel = {k: v for k, v in best[0].items() if k != 'per'}
    return dict(seed=seed, init={k: v for k, v in init.items()}, ckpts=[{k: v for k, v in c.items() if k != 'per'} for c in cks], selected=sel, selected_layers=best[1], train_episodes=eps_done, cal_eval_episodes=eval_eps, train_steps=iters * N * H, wall=time.time() - t0, log=log, args=vars(a), logstd=logstd.detach().numpy().copy())
def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--bias', type=float, default=0.0); ap.add_argument('--seeds', default='0'); ap.add_argument('--steps', type=int, default=3_000_000)
    ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--mb', type=int, default=2048)
    ap.add_argument('--hid', type=int, default=256); ap.add_argument('--logstd0', type=float, default=-2.0); ap.add_argument('--logstd_min', type=float, default=-5.0); ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--pool', type=int, default=1200); ap.add_argument('--base', default='adapt')
    return ap
if __name__ == '__main__':
    a = parser().parse_args(); out = os.path.join(RUNS, a.tag + '.pkl'); resd = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}; evalpool = mp.Pool(8)
    for s in [int(v) for v in a.seeds.split(',')]:
        if s in resd: continue
        resd[s] = train(a, s, evalpool); pickle.dump(resd, open(out, 'wb')); r = resd[s]; print(f'{a.tag} seed {s} done {r["wall"]:.0f}s init {r["init"]["succ"]:.3f} selected {r["selected"]["succ"]:.3f} (res_mean {r["selected"]["res_mean"]:.3f})', flush=True)
