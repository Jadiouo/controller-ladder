"""BC + DART (torque-noise injected teacher rollouts in the TRUE (error) plant, labels = clean teacher torque). 10 seeds."""
import argparse, pickle, time, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from stu import *
ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--ntraj', type=int, default=60); ap.add_argument('--maxk', type=int, default=1500); ap.add_argument('--sigma', type=float, default=2.0); ap.add_argument('--epochs', type=int, default=30)
a = ap.parse_args(); cond = load_conds()[a.cond]; res = {}; t0 = time.time()
res['teacher'] = evaluate(cond, None); print(a.cond, cond, 'teacher median', np.median(res['teacher']['hits']), flush=True)
for s in range(a.seeds):
    t1 = time.time(); torch.manual_seed(s); rng = np.random.default_rng(s); X, Y = [], []
    seeds = 100_000 + s * 1000 + np.arange(a.ntraj); env = E.JugEnv(a.ntraj, cond)
    def pol(env_, tT, obs):
        act = env_.active; X.append(obs[act]); Y.append(tT[act])
        return np.clip(tT + rng.normal(0, a.sigma, tT.shape), -E.TAU_MAX, E.TAU_MAX)        # DART: noisy execution, clean label
    E.run_policy(env, pol, E.init_cond(seeds), seeds, maxk=a.maxk)
    X = np.concatenate(X); Y = np.concatenate(Y); xm, xs, ym, ys = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    net = mlp(OBS, 3); Xt = torch.tensor((X - xm) / xs).float(); Yt = torch.tensor((Y - ym) / ys).float()
    opt = torch.optim.Adam(net.parameters(), 1e-3); n = len(Xt); BS = 1024; sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * ((n + BS - 1) // BS))
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]; loss = ((net(Xt[idx]) - Yt[idx])**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    r = evaluate(cond, BCPolicy(net, xm, xs, ym, ys)); res[('bc', s)] = r
    print(f'{a.cond} BC+DART seed {s}: n={n} loss {float(loss):.4f} median hits {np.median(r["hits"])} (teacher {np.median(res["teacher"]["hits"])}) {time.time()-t1:.0f}s', flush=True)
res['wall'] = time.time() - t0; pickle.dump(res, open(f'results/spike03b/raw_bc_{a.cond}.pkl', 'wb'))
