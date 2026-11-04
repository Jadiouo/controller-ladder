"""spike15c approach A: asymmetric actor-critic residual PPO on top of the L-ens (3-model mean L-dec4) whip pipeline.
Actor input = spike13a features (arm history only). Critic input = same features + privileged vector (true theta13 z-scored, theta_hat z-scored, simulator chain qpos/qvel).
Recipe = spike13a "fixed" (separate actor/critic grad clip, probe steps masked from the policy loss), log-std0 and steps as arguments. Gate: --bias B adds a constant torque bias after the probe.
Training scenarios: seeds 600000-600299 rep 0 (not in the estimator training sets). Eval seeds: guard15c. Run with cwd = .scratch/run15c."""
import sys, os, time, json, pickle, argparse, multiprocessing as mp
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H); sys.path.insert(0, os.path.join(_H, '..', 'spike13a'))
from c15c import *
import rl13 as R13, c9b
import torch, torch.nn as nn
NP = 13 + 13 + 10 + 10        # privileged dims
POOLSEEDS = 600000
RUNS15 = os.path.join(RES15C, 'runs'); os.makedirs(RUNS15, exist_ok=True)
TAB = {}
def load_tab():
    """pkey -> L-ens theta (mean of frozen members 0..2) for cal/hold/new/pool"""
    t = {}
    for k in ('cal', 'hold', 'new', 'pool'):
        D = pickle.load(open(os.path.join(S15C, f'raw_{k}.pkl'), 'rb')); E = np.mean([np.load(os.path.join(S15C, f'est_{k}_m{m}.npy')) for m in range(3)], 0)
        for i, key in enumerate(D['key']): t[key] = E[i]
    return t
def base_spec15():
    pr = probe_of_lest(); return ('est', 'lk_ens_dec4', Lookup(TAB), pr['T'], pr['amp'])
def zth(th): th = np.asarray(th, float); return np.concatenate([[(th[0] - 1) / 0.12], np.log(np.maximum(th[1:11], 1e-6)), [np.log(max(th[11], 1e-6)) / 1.2], [(th[12] * 1e3 - 2.0) / 0.6]]).astype(np.float32)
class ResEnv15(R13.ResEnv13):
    def __init__(self, seed, rep, cap, bias):
        guard15c(seed); super().__init__(base_spec15(), c9b.Feat(True, False), seed, rep, 'abc', cap, bias); self.tz = zth(true_theta(seed, 'abc', rep)); self.nj = self.ctx.robot.n
    def priv(self):
        est = getattr(self.base, 'est', None); d = self.env.data; n = self.nj
        return np.concatenate([self.tz, zth(est) if est is not None else np.zeros(13, np.float32), np.clip(d.qpos[n:n + 10], -4, 4), np.clip(d.qvel[n:n + 10], -30, 30) * 0.2]).astype(np.float32)
def run15(args):
    state, seed, rep, cap, bias = args; guard15c(seed); INST[0] = 0; e = ResEnv15(seed, rep, cap, bias); done = False; x = e.x
    while not done:
        a = np.zeros(3) if state is None else np.clip(c9b.np_mlp(state, x.astype(np.float64))[:3], -1, 1); x, r, done, info = e.step(a)
    return dict(success=float(info['success']), min_dist=float(e.dmin), res_mean=float(np.mean(e.rn)) if e.rn else 0.0, res_max=float(np.max(e.rn)) if e.rn else 0.0, inst=int(INST[0]))
def eval_set(pool, state, seeds, cap, bias, rep=None):
    r = pool.map(run15, [(state, s, rep, cap, bias) for s in seeds], chunksize=1)
    return dict(succ=float(np.mean([x['success'] for x in r])), min_dist=float(np.mean([x['min_dist'] for x in r])), res_mean=float(np.mean([x['res_mean'] for x in r])), res_max=float(np.max([x['res_max'] for x in r])), episodes=len(seeds), per=np.array([x['success'] for x in r]), inst=float(np.mean([x['inst'] > 0 for x in r])))
def worker(conn, wid, E, cap, bias, M, seed):
    reset_probe_globals(); rng = np.random.default_rng([seed, wid, 77]); envs = [None] * E
    def new(i): envs[i] = ResEnv15(POOLSEEDS + int(rng.integers(0, M)), 0, cap, bias)
    while True:
        cmd, a = conn.recv()
        if cmd == 'reset':
            for i in range(E): new(i)
            conn.send((np.stack([e.x for e in envs]), np.stack([e.priv() for e in envs])))
        elif cmd == 'step':
            xs, ps, rs, ds, ons, eps = [], [], [], [], [], []
            for i in range(E):
                x, r, d, info = envs[i].step(a[i]); ons.append(info['on'])
                if d: eps.append(dict(success=float(info['success']), steps=len(envs[i].en), res=float(np.mean(envs[i].rn)) if envs[i].rn else 0.0)); new(i); x = envs[i].x
                xs.append(x); ps.append(envs[i].priv()); rs.append(r); ds.append(d)
            conn.send((np.stack(xs), np.stack(ps), np.array(rs, np.float32), np.array(ds, np.float32), np.array(ons, np.float32), eps))
        elif cmd == 'close': break
def start_workers(a, E, nw, seed):
    conns, procs = [], []
    for w in range(nw):
        pc, cc = mp.Pipe(); p = mp.Process(target=worker, args=(cc, w, E, a.cap, a.bias_vec, a.pool, seed * 1000 + 7), daemon=True); p.start(); conns.append(pc); procs.append(p)
    for c in conns: c.send(('reset', None))
    o = [c.recv() for c in conns]; return conns, procs, np.concatenate([q[0] for q in o]), np.concatenate([q[1] for q in o])
def vstep(conns, E, an_c):
    for w, c in enumerate(conns): c.send(('step', an_c[w * E:(w + 1) * E]))
    o = [c.recv() for c in conns]
    return (np.concatenate([q[0] for q in o]), np.concatenate([q[1] for q in o]), np.concatenate([q[2] for q in o]), np.concatenate([q[3] for q in o]), np.concatenate([q[4] for q in o]), [e for q in o for e in q[5]])
def train_ppo(a, seed, evalpool, CALS):
    reset_probe_globals(); torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2); OBS = c9b.Feat(True, False).dim; N, H, GAM = a.nenv, a.horizon, a.gamma; E = N // a.workers
    conns, procs, x, pv = start_workers(a, E, a.workers, seed)
    actor, critic = R13.mk(OBS, 3, a.hid, 0.01), R13.mk(OBS + NP, 1, a.hid); logstd = nn.Parameter(torch.full((3,), a.logstd0))
    with torch.no_grad(): actor[-1].weight.zero_(); actor[-1].bias.zero_()
    pa, pc = list(actor.parameters()) + [logstd], list(critic.parameters()); opt_a = torch.optim.Adam(pa, lr=a.lr); opt_c = torch.optim.Adam(pc, lr=a.lr_c or a.lr)
    iters = a.steps // (N * H); ck_at = sorted(set(int(round((k + 1) * iters / 7)) for k in range(7)))
    init = eval_set(evalpool, R13.layers_of(actor), CALS, a.cap, a.bias_list); print('  init CAL', {k: v for k, v in init.items() if k != 'per'}, flush=True); cks = []; eps_done = 0; succ_hist = []; t0 = time.time(); log = []; eval_eps = init['episodes']
    for it in range(iters):
        bx, bp, ba, blp, bv, br, bd, bon = [], [], [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x); pt = torch.from_numpy(pv)
            with torch.no_grad(): mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu); lp = (-0.5 * ((act - mu) / sd_) ** 2 - logstd - 0.9189385).sum(1); v = critic(torch.cat([xt, pt], 1)).squeeze(1)
            an = act.numpy(); xn, pn, r, d, on, eps = vstep(conns, E, np.clip(an, -1, 1))
            for e in eps: eps_done += 1; succ_hist.append(e['success'])
            bx.append(x); bp.append(pv); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r * a.rscale); bd.append(d); bon.append(on); x, pv = xn, pn
        with torch.no_grad(): lastv = critic(torch.cat([torch.from_numpy(x), torch.from_numpy(pv)], 1)).squeeze(1).numpy()
        R_, V_, D_ = np.array(br), np.array(bv), np.array(bd); adv = np.zeros_like(R_); g = 0
        for h in reversed(range(H)):
            nv = lastv if h == H - 1 else V_[h + 1]; nd = D_[h]; delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * 0.95 * (1 - nd) * g; adv[h] = g
        ret = adv + V_; X = torch.from_numpy(np.concatenate(bx)); P_ = torch.from_numpy(np.concatenate(bp)); A_ = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp)); ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1)); ON = torch.from_numpy(np.concatenate(bon))
        m_ = ON > 0.5; ADV = (ADV - ADV[m_].mean()) / (ADV[m_].std() + 1e-8); n = len(X); kl_it = []; g_act = []
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, a.mb):
                idx = perm[i:i + a.mb]; mu = actor(X[idx]); ls = logstd; lp = (-0.5 * ((A_[idx] - mu) / ls.exp()) ** 2 - ls - 0.9189385).sum(1); ratio = (lp - LP[idx]).exp()
                pl = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]); w_ = ON[idx]; pol = (pl * w_).sum() / w_.sum().clamp(min=1.0)
                vl = 0.5 * ((critic(torch.cat([X[idx], P_[idx]], 1)).squeeze(1) - RET[idx]) ** 2).mean(); loss_a = pol - a.ent * (ls + 1.4189385).sum() + a.bc * (mu ** 2).mean()
                opt_a.zero_grad(); loss_a.backward(); ga = nn.utils.clip_grad_norm_(pa, 0.5); opt_a.step(); opt_c.zero_grad(); vl.backward(); nn.utils.clip_grad_norm_(pc, 0.5); opt_c.step(); g_act.append(float(ga))
                with torch.no_grad(): logstd.clamp_(min=a.logstd_min); kl_it.append(float(((ratio - 1) - ratio.log()).mean()))
        if (it + 1) in ck_at:
            c = eval_set(evalpool, R13.layers_of(actor), CALS, a.cap, a.bias_list); eval_eps += c['episodes']; c.update(it=it + 1, train_steps=(it + 1) * N * H, train_episodes=eps_done, layers=R13.layers_of(actor), logstd=logstd.detach().numpy().copy()); cks.append(c); print('  ckpt', {k: v for k, v in c.items() if k not in ('per', 'layers', 'logstd')}, flush=True)
        if it % 10 == 0 or it == iters - 1:
            rec = succ_hist[-200:]; log.append((it, eps_done, float(np.mean(rec)) if rec else float('nan'), time.time() - t0)); print(f'  [{a.tag} s{seed}] it {it}/{iters} eps {eps_done} train-succ(last200) {np.mean(rec) if rec else float("nan"):.3f} kl {np.mean(kl_it):.4f} std {float(logstd.exp().mean().detach()):.3f} gn {np.mean(g_act):.2f} t {time.time()-t0:.0f}s', flush=True)
    for c in conns: c.send(('close', None))
    for p in procs: p.join(timeout=5)
    best = max(cks, key=lambda c: (c['succ'], -c['min_dist'])); sel = {k: v for k, v in best.items() if k not in ('layers', 'logstd')}
    return dict(seed=seed, init=init, ckpts=cks, selected=sel, selected_layers=best['layers'], learned_cal=bool(best['succ'] >= init['succ'] + 0.05), train_episodes=eps_done, cal_eval_episodes=eval_eps, wall=time.time() - t0, log=log, args=vars(a))
def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--bias', type=float, default=0.0); ap.add_argument('--seeds', default='0'); ap.add_argument('--steps', type=int, default=3_000_000)
    ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--lr_c', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--mb', type=int, default=2048); ap.add_argument('--hid', type=int, default=256); ap.add_argument('--logstd0', type=float, default=-2.0); ap.add_argument('--logstd_min', type=float, default=-5.0)
    ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--bc', type=float, default=0.0); ap.add_argument('--cap', type=float, default=12.0); ap.add_argument('--pool', type=int, default=300); ap.add_argument('--ent', type=float, default=0.0); ap.add_argument('--rscale', type=float, default=1.0)
    return ap
if __name__ == '__main__':
    a = parser().parse_args(); a.bias_list = None if a.bias == 0 else R13.bias_vec(a.bias).tolist(); a.bias_vec = None if a.bias == 0 else R13.bias_vec(a.bias); TAB.update(load_tab())
    out = os.path.join(RUNS15, a.tag + '.pkl'); resd = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
    evalpool = mp.Pool(8)
    for s in [int(v) for v in a.seeds.split(',')]:
        if s in resd: continue
        resd[s] = train_ppo(a, s, evalpool, CAL); pickle.dump(resd, open(out, 'wb')); r = resd[s]
        print(f'{a.tag} seed {s} done {r["wall"]:.0f}s init {r["init"]["succ"]:.3f} selected {r["selected"]["succ"]:.3f} (res_mean {r["selected"]["res_mean"]:.3f}) learned_cal {r["learned_cal"]}', flush=True)
