"""spike13a: residual PPO / SAC on the whip (spike10b realistic conditions, base T-adapt3-full) with switches for every candidate pipeline bug.
Recipe R0 == experiments/spike10b/train10.py (joint grad clip over actor+critic, all steps in the policy loss, no entropy term). Fixes are opt-in flags so the gate can attribute them.
Env 'gate': a constant torque bias is added after the probe (known learnable space: residual = -bias restores the base).
TEST (0-49) / TEST-B (20000-20499) are blocked by guard(). Run with cwd = .scratch/run13 (MuJoCo writes its log into cwd)."""
import sys, os, time, json, pickle, argparse, multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spike10b'))
from r10b import *
import torch, torch.nn as nn
CAL_A = list(range(1000, 1050)); CAL_B = list(range(2000, 2150)); CAL = (CAL_A + CAL_B)[:int(os.environ.get("CAL_N", 200))]
RUNS = os.path.join(ROOT, 'results/spike13a/runs'); os.makedirs(RUNS, exist_ok=True)
BASE_SPEC = None
def base_spec():
    sp = json.load(open(os.path.join(ROOT, 'results/spike10b/best_classical.json')))['spec']; return (sp[0], sp[1])
def bias_vec(b): return np.array([1.0, -1.0, 1.0]) * float(b)
class ResEnv13(ResEnv10):
    """ResEnv10 + residual cap + optional post-probe constant torque bias; info carries 'on' (residual/bias active) so the trainer can mask the probe steps."""
    def __init__(self, base, fz, seed, rep, cond, cap, bias):
        self.cap = float(cap); self.bias = np.zeros(3) if bias is None else np.asarray(bias, float); super().__init__(base, fz, seed, rep, cond)
    def step(self, a):
        a = np.clip(a, -1, 1); on = bool(self.obs.k >= getattr(self.base, 'n_pr', 0)); res = self.cap * a * on
        tau = np.clip(self.tb + res + self.bias * on, -self.ctx.tau_limit, self.ctx.tau_limit)
        if on: self.rn.append(float(np.linalg.norm(res)))
        obs, _, term, trunc, info = self.env.step(tau); self.en.append(float((info['applied'] ** 2).sum())); self.tips.append(self.task.chain_tip(self.env).copy())
        d = float(np.linalg.norm(self.task.chain_tip(self.env) - self.task.target)); r = 10.0 * max(self.dmin - d, 0.0); self.dmin = min(self.dmin, d)
        succ = info['outcome'] == 'success'; r += 5.0 * succ - 5.0 * (info['outcome'] == 'failure') - 2e-6 * float((res ** 2).sum()); done = term or trunc
        if not done: self.obs = obs; self.fz.push(obs); self.tb = self.base.act(obs); self.x = self.fz(obs, self.tb, self.base, self.yt)
        return self.x, r, done, dict(success=succ, min_dist=self.dmin, on=on, term=bool(term))
def act_of(state, x, kind):
    o = c9b.np_mlp(state, x.astype(np.float64))[:3]; return np.tanh(o) if kind == 'sac' else np.clip(o, -1, 1)
def run13(args):
    """one evaluation episode: (state_layers|None, kind, seed, cap, bias(list|None)) -> metrics"""
    state, kind, seed, cap, bias = args; guard([seed]); INST[0] = 0; fz = c9b.Feat(True, False); e = ResEnv13(base_spec(), fz, seed, None, 'abc', cap, bias); done = False; x = e.x
    while not done:
        a = np.zeros(3) if state is None else act_of(state, x, kind); x, r, done, info = e.step(a)
    return dict(success=float(info['success']), min_dist=float(e.dmin), res_mean=float(np.mean(e.rn)) if e.rn else 0.0, res_max=float(np.max(e.rn)) if e.rn else 0.0, steps=len(e.en), inst=int(INST[0]))
def eval_set(pool, state, kind, seeds, cap, bias):
    r = pool.map(run13, [(state, kind, s, cap, bias) for s in seeds], chunksize=1)
    return dict(succ=float(np.mean([x['success'] for x in r])), min_dist=float(np.mean([x['min_dist'] for x in r])), res_mean=float(np.mean([x['res_mean'] for x in r])), res_max=float(np.max([x['res_max'] for x in r])), episodes=len(seeds),
                per=np.array([x['success'] for x in r]))
def worker(conn, wid, E, cap, bias, M, seed):
    reset_probe_globals(); rng = np.random.default_rng([seed, wid, 77]); envs = [None] * E; base = base_spec()
    def new(i): envs[i] = ResEnv13(base, c9b.Feat(True, False), 500000 + int(rng.integers(0, M)), 0, 'abc', cap, bias)
    while True:
        cmd, a = conn.recv()
        if cmd == 'reset':
            for i in range(E): new(i)
            conn.send(np.stack([e.x for e in envs]))
        elif cmd == 'step':
            xs, rs, ds, ons, eps = [], [], [], [], []
            for i in range(E):
                x, r, d, info = envs[i].step(a[i]); ons.append(info['on'])
                if d: eps.append(dict(success=float(info['success']), steps=len(envs[i].en), res=float(np.mean(envs[i].rn)) if envs[i].rn else 0.0)); new(i); x = envs[i].x
                xs.append(x); rs.append(r); ds.append(d)
            conn.send((np.stack(xs), np.array(rs, np.float32), np.array(ds, np.float32), np.array(ons, np.float32), eps))
        elif cmd == 'close': break
def mk(i, o, hid, sc=1.0):
    n = nn.Sequential(nn.Linear(i, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
def layers_of(net): return [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]
def start_workers(a, E, nw, seed):
    conns, procs = [], []
    for w in range(nw):
        pc, cc = mp.Pipe(); p = mp.Process(target=worker, args=(cc, w, E, a.cap, a.bias_vec, a.pool, seed * 1000 + 7), daemon=True); p.start(); conns.append(pc); procs.append(p)
    for c in conns: c.send(('reset', None))
    return conns, procs, np.concatenate([c.recv() for c in conns])
def vstep(conns, E, an_c):
    for w, c in enumerate(conns): c.send(('step', an_c[w * E:(w + 1) * E]))
    outs = [c.recv() for c in conns]
    return (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]), np.concatenate([o[2] for o in outs]), np.concatenate([o[3] for o in outs]), [e for o in outs for e in o[4]])
# ------------------------------------------------------------------------------------------------ PPO
def train_ppo(a, seed, evalpool):
    reset_probe_globals(); torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2); OBS = c9b.Feat(True, False).dim; N, H, GAM = a.nenv, a.horizon, a.gamma; E = N // a.workers
    conns, procs, x = start_workers(a, E, a.workers, seed)
    actor, critic = mk(OBS, 3, a.hid, 0.01), mk(OBS, 1, a.hid); logstd = nn.Parameter(torch.full((3,), a.logstd0))
    with torch.no_grad(): actor[-1].weight.zero_(); actor[-1].bias.zero_()
    pa, pc = list(actor.parameters()) + [logstd], list(critic.parameters()); params = pa + pc
    if a.sepclip: opt_a = torch.optim.Adam(pa, lr=a.lr); opt_c = torch.optim.Adam(pc, lr=a.lr_c or a.lr)
    else: opt = torch.optim.Adam(params, lr=a.lr)
    iters = a.steps // (N * H); ck_at = sorted(set(int(round((k + 1) * iters / 7)) for k in range(7)))
    init = eval_set(evalpool, layers_of(actor), 'ppo', CAL, a.cap, a.bias_list); print('  init CAL', {k: v for k, v in init.items() if k != 'per'}, flush=True); cks = []; eps_done = 0; succ_hist = []; t0 = time.time(); log = []; eval_eps = init['episodes']; gstat = []
    for it in range(iters):
        bx, ba, blp, bv, br, bd, bon = [], [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x)
            with torch.no_grad(): mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu); lp = (-0.5 * ((act - mu) / sd_) ** 2 - logstd - 0.9189385).sum(1); v = critic(xt).squeeze(1)
            an = act.numpy(); xn, r, d, on, eps = vstep(conns, E, np.clip(an, -1, 1))
            for e in eps: eps_done += 1; succ_hist.append(e['success'])
            bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r * a.rscale); bd.append(d); bon.append(on); x = xn
        with torch.no_grad(): lastv = critic(torch.from_numpy(x)).squeeze(1).numpy()
        R_, V_, D_ = np.array(br), np.array(bv), np.array(bd); adv = np.zeros_like(R_); g = 0
        for h in reversed(range(H)):
            nv = lastv if h == H - 1 else V_[h + 1]; nd = D_[h]; delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * 0.95 * (1 - nd) * g; adv[h] = g
        ret = adv + V_; X = torch.from_numpy(np.concatenate(bx)); A_ = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp)); ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1)); ON = torch.from_numpy(np.concatenate(bon))
        if a.maskprobe: m_ = ON > 0.5; ADV = (ADV - ADV[m_].mean()) / (ADV[m_].std() + 1e-8)
        else: ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(X); kl_it = []; g_act = []; g_tot = []; scl = []
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, a.mb):
                idx = perm[i:i + a.mb]; mu = actor(X[idx]); ls = logstd; lp = (-0.5 * ((A_[idx] - mu) / ls.exp()) ** 2 - ls - 0.9189385).sum(1); ratio = (lp - LP[idx]).exp()
                pl = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx])
                if a.maskprobe: w_ = ON[idx]; pol = (pl * w_).sum() / w_.sum().clamp(min=1.0)
                else: pol = pl.mean()
                ent = (ls + 1.4189385).sum(); vl = 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx]) ** 2).mean()
                loss_a = pol - a.ent * ent + a.bc * (mu ** 2).mean()
                if a.sepclip:
                    opt_a.zero_grad(); loss_a.backward(); ga = nn.utils.clip_grad_norm_(pa, 0.5); opt_a.step()
                    opt_c.zero_grad(); vl.backward(); gc = nn.utils.clip_grad_norm_(pc, 0.5); opt_c.step(); g_act.append(float(ga)); g_tot.append(float(gc)); scl.append(1.0)
                else:
                    loss = loss_a + vl; opt.zero_grad(); loss.backward()
                    ga = float(torch.sqrt(sum((p.grad ** 2).sum() for p in pa))); gt = nn.utils.clip_grad_norm_(params, 0.5); g_act.append(ga); g_tot.append(float(gt)); scl.append(min(1.0, 0.5 / (float(gt) + 1e-12))); opt.step()
                with torch.no_grad(): logstd.clamp_(min=a.logstd_min); kl_it.append(float(((ratio - 1) - ratio.log()).mean()))
        gstat.append((float(np.mean(g_act)), float(np.mean(g_tot)), float(np.mean(scl))))
        if (it + 1) in ck_at:
            c = eval_set(evalpool, layers_of(actor), 'ppo', CAL, a.cap, a.bias_list); eval_eps += c['episodes']; c.update(it=it + 1, train_steps=(it + 1) * N * H, train_episodes=eps_done, layers=layers_of(actor), logstd=logstd.detach().numpy().copy()); cks.append(c); print('  ckpt', {k: v for k, v in c.items() if k not in ('per', 'layers', 'logstd')}, flush=True)
        if it % 10 == 0 or it == iters - 1:
            rec = succ_hist[-200:]; log.append((it, eps_done, float(np.mean(rec)) if rec else float('nan'), time.time() - t0, gstat[-1])); print(f'  [{a.tag} s{seed}] it {it}/{iters} eps {eps_done} train-succ(last200) {np.mean(rec) if rec else float("nan"):.3f} kl {np.mean(kl_it):.4f} std {float(logstd.exp().mean().detach()):.3f} gn(actor,tot,scale) {gstat[-1][0]:.3f},{gstat[-1][1]:.3f},{gstat[-1][2]:.2f} t {time.time()-t0:.0f}s', flush=True)
    for c in conns: c.send(('close', None))
    for p in procs: p.join(timeout=5)
    return finish(a, seed, init, cks, eps_done, eval_eps, time.time() - t0, log, 'ppo')
# ------------------------------------------------------------------------------------------------ SAC
def train_sac(a, seed, evalpool):
    reset_probe_globals(); torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4); OBS = c9b.Feat(True, False).dim; N = a.nenv; E = N // a.workers; GAM = a.gamma
    conns, procs, x = start_workers(a, E, a.workers, seed)
    actor = mk(OBS, 6, a.hid, 0.01); q1, q2, t1, t2 = (mk(OBS + 3, 1, a.hid) for _ in range(4))
    with torch.no_grad(): actor[-1].weight.zero_(); actor[-1].bias.zero_(); actor[-1].bias[3:] = a.logstd0
    t1.load_state_dict(q1.state_dict()); t2.load_state_dict(q2.state_dict()); oa = torch.optim.Adam(actor.parameters(), lr=a.lr); oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=a.lr_c or 3e-4)
    CAPN = 500_000; bx = np.zeros((CAPN, OBS), np.float32); bx2 = np.zeros((CAPN, OBS), np.float32); ba = np.zeros((CAPN, 3), np.float32); br = np.zeros(CAPN, np.float32); bd = np.zeros(CAPN, np.float32); bn = 0; ptr = 0
    def pol(xt, det=False):
        o = actor(xt); mu, ls = o[:, :3], o[:, 3:].clamp(-5.0, -1.0)
        if det: return torch.tanh(mu), None
        u = mu + ls.exp() * torch.randn_like(mu); act = torch.tanh(u); lp = (-0.5 * ((u - mu) / ls.exp()) ** 2 - ls - 0.9189385).sum(1) - torch.log(1 - act ** 2 + 1e-6).sum(1); return act, lp
    K = 8; iters = a.steps // (N * K); ck_at = sorted(set(int(round((k + 1) * iters / 7)) for k in range(7))); upd_per = int(round(N * K * a.utd))
    init = eval_set(evalpool, layers_of(actor), 'sac', CAL, a.cap, a.bias_list); print('  init CAL', {k: v for k, v in init.items() if k != 'per'}, flush=True); cks = []; eps_done = 0; succ_hist = []; t0 = time.time(); log = []; eval_eps = init['episodes']
    for it in range(iters):
        for h in range(K):
            with torch.no_grad(): act, _ = pol(torch.from_numpy(x))
            an = act.numpy(); xn, r, d, on, eps = vstep(conns, E, an)
            for e in eps: eps_done += 1; succ_hist.append(e['success'])
            # xn at a done step is the NEXT episode's first obs; terminal transitions are masked by (1-d), so the stored next obs is irrelevant there
            for j in range(N): bx[ptr] = x[j]; bx2[ptr] = xn[j]; ba[ptr] = an[j]; br[ptr] = r[j] * a.rscale; bd[ptr] = d[j]; ptr = (ptr + 1) % CAPN; bn = min(bn + 1, CAPN)
            x = xn
        if bn >= a.warmup:
            for _ in range(upd_per):
                idx = np.random.randint(0, bn, 256); X, X2, A, R, D = (torch.from_numpy(z[idx]) for z in (bx, bx2, ba, br, bd))
                with torch.no_grad(): a2, lp2 = pol(X2); xa2 = torch.cat([X2, a2], 1); tq = torch.min(t1(xa2), t2(xa2)).squeeze(1) - a.alpha * lp2; y = R + GAM * (1 - D) * tq
                xa = torch.cat([X, A], 1); lq = ((q1(xa).squeeze(1) - y) ** 2).mean() + ((q2(xa).squeeze(1) - y) ** 2).mean(); oq.zero_grad(); lq.backward(); nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0); oq.step()
                ap, lp = pol(X); xap = torch.cat([X, ap], 1); la = (a.alpha * lp - torch.min(q1(xap), q2(xap)).squeeze(1)).mean() + a.bc * (ap ** 2).mean(); oa.zero_grad(); la.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 5.0); oa.step()
                with torch.no_grad():
                    for t, q in ((t1, q1), (t2, q2)):
                        for pt, pq in zip(t.parameters(), q.parameters()): pt.mul_(0.995).add_(0.005 * pq)
        if (it + 1) in ck_at:
            c = eval_set(evalpool, layers_of(actor), 'sac', CAL, a.cap, a.bias_list); eval_eps += c['episodes']; c.update(it=it + 1, train_steps=(it + 1) * N * K, train_episodes=eps_done, layers=layers_of(actor)); cks.append(c); print('  ckpt', {k: v for k, v in c.items() if k not in ('per', 'layers')}, flush=True)
        if it % 20 == 0 or it == iters - 1:
            rec = succ_hist[-200:]; log.append((it, eps_done, float(np.mean(rec)) if rec else float('nan'), time.time() - t0)); print(f'  [{a.tag} s{seed}] it {it}/{iters} eps {eps_done} train-succ(last200) {np.mean(rec) if rec else float("nan"):.3f} t {time.time()-t0:.0f}s', flush=True)
    for c in conns: c.send(('close', None))
    for p in procs: p.join(timeout=5)
    return finish(a, seed, init, cks, eps_done, eval_eps, time.time() - t0, log, 'sac')
def finish(a, seed, init, cks, eps_done, eval_eps, wall, log, kind):
    best = max(cks, key=lambda c: (c['succ'], -c['min_dist'])); sel = {k: v for k, v in best.items() if k not in ('layers', 'logstd')}
    return dict(seed=seed, kind=kind, init={k: v for k, v in init.items()}, ckpts=cks, selected=sel, selected_layers=best['layers'], learned_cal=bool(best['succ'] >= init['succ'] + 0.05), train_episodes=eps_done, cal_eval_episodes=eval_eps, wall=wall, log=log, args=vars(a))
def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--mode', default='ppo', choices=['ppo', 'sac']); ap.add_argument('--bias', type=float, default=0.0, help='gate: torque bias magnitude (Nm), 0 = real condition')
    ap.add_argument('--seeds', default='0'); ap.add_argument('--steps', type=int, default=1_500_000); ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--lr_c', type=float, default=0.0); ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--mb', type=int, default=2048); ap.add_argument('--hid', type=int, default=256)
    ap.add_argument('--logstd0', type=float, default=-3.5); ap.add_argument('--logstd_min', type=float, default=-5.0); ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--bc', type=float, default=0.0)
    ap.add_argument('--cap', type=float, default=12.0); ap.add_argument('--pool', type=int, default=300); ap.add_argument('--ent', type=float, default=0.0); ap.add_argument('--rscale', type=float, default=1.0)
    ap.add_argument('--sepclip', type=int, default=0); ap.add_argument('--maskprobe', type=int, default=0)
    ap.add_argument('--alpha', type=float, default=0.01); ap.add_argument('--utd', type=float, default=0.2); ap.add_argument('--warmup', type=int, default=20000)
    return ap
if __name__ == '__main__':
    a = parser().parse_args(); a.bias_list = None if a.bias == 0 else bias_vec(a.bias).tolist(); a.bias_vec = None if a.bias == 0 else bias_vec(a.bias)
    out = os.path.join(RUNS, a.tag + '.pkl'); resd = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}; evalpool = mp.Pool(8)
    for s in [int(v) for v in a.seeds.split(',')]:
        if s in resd: continue
        resd[s] = (train_ppo if a.mode == 'ppo' else train_sac)(a, s, evalpool); pickle.dump(resd, open(out, 'wb'))
        r = resd[s]; print(f'{a.tag} seed {s} done {r["wall"]:.0f}s init {r["init"]["succ"]:.3f} selected {r["selected"]["succ"]:.3f} (res_mean {r["selected"]["res_mean"]:.3f}) learned_cal {r["learned_cal"]}', flush=True)
