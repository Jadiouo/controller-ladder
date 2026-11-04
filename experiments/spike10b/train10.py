"""spike09b residual PPO on best-classical (whip). Recipe = spike07 PPO (log-std init -3.5, floor -5, lr 1e-4, 4 epochs, minibatch 2048, 256x256 tanh, gamma .995, GAE .95, clip .2, grad-clip .5),
large batch 32 envs x 256 steps = 8192/iter, residual cap 12 Nm. Reward = 10*(decrease of the running minimum chain-tip distance) + 5*success - 5*failure - 2e-6*|res|^2 (privileged tip position is used ONLY in the reward).
Training scenarios: pool of M fixed target seeds 500000..500000+M-1 with chain error drawn independently from the seed (rep 0); CAL (seeds 1000-1049, chain from the seed) selects among 7 equally spaced checkpoints (the init is a reference, not a candidate).
TEST seeds are never touched (guard)."""
import sys, os, time, json, pickle, argparse, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
import torch, torch.nn as nn; torch.set_num_threads(2)
CAL = list(range(1000, 1050))
def worker(conn, wid, E, base, hist, oracle, M, seed):
    reset_probe_globals()
    rng = np.random.default_rng([seed, wid, 77]); envs = [None] * E
    def new(i): envs[i] = ResEnv10(base, c9b.Feat(hist, oracle), 500000 + int(rng.integers(0, M)), 0, 'abc')
    while True:
        cmd, a = conn.recv()
        if cmd == 'reset':
            for i in range(E): new(i)
            conn.send(np.stack([e.x for e in envs]))
        elif cmd == 'step':
            xs, rs, ds, eps = [], [], [], []
            for i in range(E):
                x, r, d, info = envs[i].step(a[i])
                if d: eps.append(dict(success=float(info['success']), steps=len(envs[i].en), res=float(np.mean(envs[i].rn)) if envs[i].rn else 0.0)); new(i); x = envs[i].x
                xs.append(x); rs.append(r); ds.append(d)
            conn.send((np.stack(xs), np.array(rs, np.float32), np.array(ds, np.float32), eps))
        elif cmd == 'close': break
def mk(i, o, hid, sc=1.0):
    n = nn.Sequential(nn.Linear(i, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
def layers_of(net): return [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]
def cal_eval(pool, base, hist, oracle, layers):
    r = pool.map(run_residual_episode10, [(base, layers, s, None, 'abc', ()) for s in CAL], chunksize=1)
    return dict(succ=float(np.mean([x['success'] for x in r])), min_dist=float(np.mean([x['min_dist'] for x in r])), res_mean=float(np.mean([x['res_mean'] for x in r])), res_max=float(np.max([x['res_max'] for x in r])), episodes=len(CAL))
def train(a, seed, evalpool):
    reset_probe_globals()
    torch.manual_seed(seed); np.random.seed(seed); hist = a.variant in ('hist', 'oracle'); oracle = a.variant == 'oracle'; fz = c9b.Feat(hist, oracle); OBS = fz.dim; N, H = a.nenv, a.horizon; GAM = a.gamma
    nw = a.workers; E = N // nw; conns, procs = [], []
    for w in range(nw):
        pc, cc = mp.Pipe(); p = mp.Process(target=worker, args=(cc, w, E, a.base, hist, oracle, a.pool, seed * 1000 + 7), daemon=True); p.start(); conns.append(pc); procs.append(p)
    for c in conns: c.send(('reset', None))
    x = np.concatenate([c.recv() for c in conns]); actor, critic = mk(OBS, 3, a.hid, 0.01), mk(OBS, 1, a.hid); logstd = nn.Parameter(torch.full((3,), a.logstd0))
    with torch.no_grad(): actor[-1].weight.zero_(); actor[-1].bias.zero_()      # init = exactly zero residual (the whip is hypersensitive: a 0.1 Nm random residual already cost 38 pp on CAL)
    params = list(actor.parameters()) + list(critic.parameters()) + [logstd]; opt = torch.optim.Adam(params, lr=a.lr); iters = a.steps // (N * H); ck_at = set(int(round((k + 1) * iters / 7)) for k in range(7))
    init = cal_eval(evalpool, a.base, hist, oracle, layers_of(actor)); print('  init CAL', init, flush=True); cks = []; best = None; eps_done = 0; succ_hist = []; t0 = time.time(); log = []; eval_eps = init['episodes']
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
                loss = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]).mean() + 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx]) ** 2).mean() + a.bc * (mu ** 2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                with torch.no_grad(): logstd.clamp_(min=a.logstd_min); kl_it.append(float(((ratio - 1) - ratio.log()).mean()))
        if (it + 1) in ck_at:
            c = cal_eval(evalpool, a.base, hist, oracle, layers_of(actor)); eval_eps += c['episodes']; c.update(it=it + 1, train_steps=(it + 1) * N * H, train_episodes=eps_done); cks.append(c); print('  ckpt', c, flush=True)
            if best is None or (c['succ'], -c['min_dist']) > (best[0]['succ'], -best[0]['min_dist']): best = (c, layers_of(actor), {k: v.clone() for k, v in actor.state_dict().items()})
        if it % 10 == 0 or it == iters - 1:
            rec = succ_hist[-200:]; log.append((it, eps_done, float(np.mean(rec)) if rec else float('nan'), time.time() - t0)); print(f'  [{a.base_name} {a.variant} s{seed}] it {it}/{iters} eps {eps_done} train-succ(last200) {np.mean(rec) if rec else float("nan"):.3f} kl {np.mean(kl_it):.4f} std {float(logstd.exp().mean().detach()):.3f} t {time.time()-t0:.0f}s', flush=True)
    for c in conns: c.send(('close', None))
    for p in procs: p.join(timeout=5)
    sel = best[0]; learned = sel['succ'] >= init['succ'] + 0.05
    return dict(seed=seed, init=init, ckpts=cks, selected=sel, selected_layers=best[1], learned=bool(learned), train_episodes=eps_done, cal_eval_episodes=eval_eps, train_steps=iters * N * H, wall=time.time() - t0, log=log, args=vars(a), obs=OBS,
                last_layers=layers_of(actor), logstd=logstd.detach().numpy().copy())
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True, help='path to json {"spec": ...} (list -> tuple)'); ap.add_argument('--variant', required=True, choices=['hist', 'oracle']); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--steps', type=int, default=4_000_000)
    ap.add_argument('--nenv', type=int, default=32); ap.add_argument('--horizon', type=int, default=256); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--mb', type=int, default=2048)
    ap.add_argument('--hid', type=int, default=256); ap.add_argument('--logstd0', type=float, default=-3.5); ap.add_argument('--logstd_min', type=float, default=-5.0); ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--bc', type=float, default=0.0)
    ap.add_argument('--pool', type=int, default=600); ap.add_argument('--out', required=True); a = ap.parse_args()
    a.base = (lambda sp: tuple(sp) if isinstance(sp, list) else sp)(json.load(open(a.base))['spec']); a.base_name = a.base if isinstance(a.base, str) else a.base[0]
    resd = pickle.load(open(a.out, 'rb')) if os.path.exists(a.out) else {}; evalpool = mp.Pool(8)
    for s in [int(x) for x in a.seeds.split(',')]:
        if s in resd: continue
        resd[s] = train(a, s, evalpool); pickle.dump(resd, open(a.out, 'wb')); print(f'{a.base_name} {a.variant} seed {s} done {resd[s]["wall"]:.0f}s selected {resd[s]["selected"]} learned {resd[s]["learned"]}', flush=True)
