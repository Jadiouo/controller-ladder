"""spike11a driver. python run11.py <stage> [args]. Stages: base (CAL baselines) | bc (collect BC data, train BC for 3 history windows, CAL select) | dart | dagger <label> <iters> | final
Data in .scratch/s11/, results in results/spike11a/. Training seeds >= 500000 only; CAL 1000-1049 for selection, 1050-1099 fresh."""
import sys, os, json, time, pickle, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from s11 import *
SD = os.path.join(ROOT, '.scratch/s11'); os.makedirs(SD, exist_ok=True)
CAL = list(range(1000, 1050)); CAL2 = list(range(1050, 1100))
def J(name, obj=None):
    p = os.path.join(OUT11, name)
    if obj is None: return json.load(open(p)) if os.path.exists(p) else {}
    json.dump(obj, open(p, 'w'), indent=1)
def ci(x): return list(boot_ci(np.asarray(x, float)))
PRIV = False; PRIOR = False
def collect(pool, seeds, **kw):
    r = pool.map(run_ep, [dict(seed=s, rep=0, cond='abc', record=True, priv=PRIV, prior=PRIOR, **kw) for s in seeds], chunksize=1)
    return np.concatenate([x['X'] for x in r]), np.concatenate([x['Y'] for x in r]), r
def evaluate(pool, st, H, stride, seeds=CAL):
    global PRIV, PRIOR
    r = pool.map(run_ep, [dict(seed=s, rep=None, cond='abc', student=st, H=H, stride=stride, cal=True, priv=PRIV, prior=PRIOR) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r])
    return dict(succ=float(su.mean()), ci=ci(su), min_dist=float(np.mean([x['min_dist'] for x in r])), diverged=float(np.mean([x['diverged'] for x in r])), unstable=float(np.mean([x['inst'] > 0 for x in r])), n=len(seeds)), su
if __name__ == '__main__':
    stage = sys.argv[1]; pool = mp.Pool(8); t0 = time.time()
    if stage == 'base':
        res = J('base.json')
        for name, seeds in (('cal', CAL), ('cal2', CAL2)):
            r = pool.map(run_teacher, [(s, None, 'abc', None, False, None, True) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r]); res[f'oracle_{name}'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist(), unstable=float(np.mean([x['inst'] > 0 for x in r])), diverged=float(np.mean([x['diverged'] for x in r])))
            r = pool.map(run10, [(spec_of('adapt3_full'), s, None, 'abc', ()) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r]); res[f'adapt3full_{name}'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist(), unstable=float(np.mean([x['inst'] > 0 for x in r])), diverged=float(np.mean([x['diverged'] for x in r])))
            # prior-only baseline: the oracle plan structure with nominal plant? (not run) ; zero information student = T-nom
            r = pool.map(run10, [('nom', s, None, 'abc', ()) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r]); res[f'nom_{name}'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist())
            print(name, {k: v['succ'] for k, v in res.items()}, f'{time.time()-t0:.0f}s', flush=True); J('base.json', res)
    elif stage == 'canon_teacher':
        res = J('canon_teacher.json'); r = pool.map(run_ep, [dict(seed=s, rep=None, cond='abc', student=None, H=6, stride=5, label='fixed', canon=True, cal=True) for s in CAL + CAL2], chunksize=1); su = np.array([x['success'] for x in r])
        res['canon_teacher'] = dict(cal=float(su[:50].mean()), cal2=float(su[50:].mean()), all=float(su.mean()), ci=ci(su), plan_s=float(np.mean([x['plan_cost'] for x in r])), unstable=float(np.mean([x['inst'] > 0 for x in r])), diverged=float(np.mean([x['diverged'] for x in r]))); print(res, flush=True); J('canon_teacher.json', res)
    elif stage == 'canon_dagger':
        H, stride, iters, NE = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); name = 'canon_dagger'; PRIV = len(sys.argv) > 6 and sys.argv[6] == 'priv'
        if PRIV: name += '_priv'
        res = J('canon_dagger.json'); res[name] = {}
        X, Y, r = collect(pool, range(500000, 500400), student=None, H=H, stride=stride, label='fixed', canon=True); res[name]['teacher_data'] = dict(episodes=400, samples=len(X), teacher_succ=float(np.mean([x['success'] for x in r]))); st = train_mlp(X, Y, seed=0); m, su = evaluate(pool, st, H, stride)
        m.update(episodes=400, samples=len(X), val_mse_z=st['val_mse_z']); res[name]['iter0_bc'] = m; print('bc', m, flush=True); J('canon_dagger.json', res); eps = 400
        for it in range(1, iters + 1):
            seeds = range(580000 + NE * (it - 1), 580000 + NE * it); Xn, Yn, r = collect(pool, seeds, student=st, H=H, stride=stride, label='fixed', canon=True, salt=it); eps += NE
            X, Y = np.concatenate([X, Xn]), np.concatenate([Y, Yn]); st = train_mlp(X, Y, seed=it); m, su = evaluate(pool, st, H, stride)
            m.update(episodes=eps, samples=len(X), student_train_succ=float(np.mean([x['success'] for x in r])), val_mse_z=st['val_mse_z'], wall=time.time() - t0); res = J('canon_dagger.json'); res[name][f'iter{it}'] = m; print(name, it, m, flush=True); J('canon_dagger.json', res); pickle.dump(st, open(os.path.join(SD, f'{name}_it{it}.pkl'), 'wb'))
    elif stage == 'final':
        res = J('final.json'); H, stride = 6, 5
        for name in sys.argv[2:]:
            st = pickle.load(open(os.path.join(SD, name + '.pkl'), 'rb')); m, su = evaluate(pool, st, H, stride, CAL2); m['per_seed'] = su.tolist(); res[name] = m; print(name, m['succ'], m['ci'], flush=True); J('final.json', res)
    elif stage == 'final1':
        res = J('final1.json'); H, stride = 6, 5
        for name in sys.argv[2:]:
            st = pickle.load(open(os.path.join(SD, name + '.pkl'), 'rb')); m, su = evaluate(pool, st, H, stride, CAL); m['per_seed'] = su.tolist(); res[name] = m; print(name, m['succ'], m['ci'], flush=True); J('final1.json', res)
    elif stage == 'canon_dagger2':
        H, stride, iters, NE = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); PRIOR = len(sys.argv) > 6 and sys.argv[6] == 'prior'; beta0 = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0; name = ('canon_dagger3' if PRIOR else 'canon_dagger2') + (f'_b{beta0}' if beta0 > 0 else ''); res = J('canon_dagger2.json'); res[name] = {}
        X, Y, r = collect(pool, range(500000, 500400), student=None, H=H, stride=stride, label='fixed', canon=True); st = train_mlp(X, Y, seed=0, epochs=60, max_steps=100000); eps = 400
        for it in range(0, iters + 1):
            if it > 0:
                seeds = range(600000 + NE * (it - 1), 600000 + NE * it); Xn, Yn, r = collect(pool, seeds, student=st, H=H, stride=stride, label='fixed', canon=True, salt=it, beta=beta0 * 0.5 ** (it - 1)); eps += NE; X, Y = np.concatenate([X, Xn]), np.concatenate([Y, Yn]); st = train_mlp(X, Y, seed=it, epochs=60, max_steps=100000)
            m1, s1 = evaluate(pool, st, H, stride, CAL); m2, s2 = evaluate(pool, st, H, stride, CAL2)
            m = dict(episodes=eps, samples=len(X), cal1=m1['succ'], cal1_ci=m1['ci'], cal2=m2['succ'], cal2_ci=m2['ci'], per_seed_cal1=s1.tolist(), per_seed_cal2=s2.tolist(), diverged=(m1['diverged'] + m2['diverged']) / 2, unstable=(m1['unstable'] + m2['unstable']) / 2, train_succ=float(np.mean([x['success'] for x in r])), val_mse_z=st['val_mse_z'], wall=time.time() - t0)
            res = J('canon_dagger2.json'); res.setdefault(name, {})[f'iter{it}'] = m; print(name, it, {k: v for k, v in m.items() if not k.startswith('per_seed')}, flush=True); J('canon_dagger2.json', res); pickle.dump(st, open(os.path.join(SD, f'{name}_it{it}.pkl'), 'wb'))
        np.savez(os.path.join(SD, f'{name}_data.npz'), X=X, Y=Y)
    elif stage == 'fresh':
        # supplementary robustness set: unseen seeds >= 700000 (never used for training or selection), seed-derived plant (rep None) like CAL. argv: n then student names (cd3 style: prior flag via name prefix 'prior:')
        n = int(sys.argv[2]); seeds = list(range(700000, 700000 + n)); res = J('fresh.json')
        if 'oracle' not in res:
            r = pool.map(run_teacher, [(s, None, 'abc', None, False, None, False) for s in seeds], chunksize=1) if False else pool.map(run_ep, [dict(seed=s, rep=None, cond='abc', student=None, H=6, stride=5, label='fixed', canon=True) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r]); res['oracle_canon'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist())
            r = pool.map(run10, [(spec_of('adapt3_full'), s, None, 'abc', ((700000, 800000),)) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r]); res['adapt3full'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist(), diverged=float(np.mean([x['diverged'] for x in r])), unstable=float(np.mean([x['inst'] > 0 for x in r])))
            r = pool.map(run10, [('nom', s, None, 'abc', ((700000, 800000),)) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r]); res['nom'] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist()); res['oracle'] = 1; J('fresh.json', res); print({k: v['succ'] for k, v in res.items() if k != 'oracle'}, flush=True)
        for name in sys.argv[3:]:
            pr = name.startswith('prior:'); fn = name.split(':')[-1]; st = pickle.load(open(os.path.join(SD, fn + '.pkl'), 'rb'))
            r = pool.map(run_ep, [dict(seed=s, rep=None, cond='abc', student=st, H=6, stride=5, cal=False, prior=pr) for s in seeds], chunksize=1); su = np.array([x['success'] for x in r])
            res = J('fresh.json'); res[name] = dict(succ=float(su.mean()), ci=ci(su), per_seed=su.tolist(), diverged=float(np.mean([x['diverged'] for x in r])), unstable=float(np.mean([x['inst'] > 0 for x in r]))); print(name, res[name]['succ'], res[name]['ci'], flush=True); J('fresh.json', res)
    elif stage == 'flow':
        # offline: train the arena/learn Flow policy (one Euler step, noise fixed per episode) on the aggregated canon+prior DAgger data of canon_dagger3 (same data as the MLP of its last iteration)
        import torch; from arena.learn.models import Flow; torch.set_num_threads(2); torch.manual_seed(0); d = np.load(os.path.join(SD, 'canon_dagger3_data.npz')); X, Y = d['X'], d['Y']; steps = int(sys.argv[2]); width, depth = 256, 4
        m = Flow(X.shape[1], 9, 1, width, depth); m.set_norm(X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6); Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)[:, None, :]; Xn = (Xt - m.obs_mean) / m.obs_std; Yn = (Yt - m.act_mean) / m.act_std
        opt = torch.optim.Adam(m.parameters(), lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); g = torch.Generator().manual_seed(0); tl = time.time()
        for i in range(steps):
            idx = torch.randint(0, len(Xn), (512,), generator=g); loss = m.loss(Xn[idx], Yn[idx], g); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if i % 5000 == 0: print('flow step', i, float(loss), f'{time.time()-tl:.0f}s', flush=True)
        st = dict(key='flow%d' % steps, obs_dim=X.shape[1], width=width, depth=depth, sd={k: v.detach().numpy() for k, v in m.state_dict().items()}); res = J('flow.json'); PRIOR = True
        for zn in (False, True):
            m1, s1 = None, None
            r1 = pool.map(run_ep, [dict(seed=s, rep=None, cond='abc', student=st, H=6, stride=5, cal=True, prior=True, zero_noise=zn) for s in CAL], chunksize=1); r2 = pool.map(run_ep, [dict(seed=s, rep=None, cond='abc', student=st, H=6, stride=5, cal=True, prior=True, zero_noise=zn) for s in CAL2], chunksize=1)
            a1, a2 = np.mean([x['success'] for x in r1]), np.mean([x['success'] for x in r2]); res['flow1step_' + ('z0' if zn else 'zfixed')] = dict(cal1=float(a1), cal2=float(a2), both=float((a1 + a2) / 2), train_steps=steps, samples=len(X), width=width, depth=depth, diverged=float(np.mean([x['diverged'] for x in r1 + r2])), per_seed=[x['success'] for x in r1 + r2]); print(res, flush=True); J('flow.json', res)
    elif stage == 'bc':
        NE = int(sys.argv[2]) if len(sys.argv) > 2 else 400; res = J('bc.json'); data = {}
        for H, stride in ((6, 2), (6, 5), (6, 10)):
            f = os.path.join(SD, f'bc_data_{H}_{stride}.npz')
            if os.path.exists(f): d = np.load(f); X, Y = d['X'], d['Y']
            else:
                X, Y, r = collect(pool, range(500000, 500000 + NE), student=None, H=H, stride=stride, label='fixed'); np.savez(f, X=X, Y=Y); res[f'teacher_bc_data_{H}_{stride}'] = dict(episodes=NE, samples=len(X), succ=float(np.mean([x['success'] for x in r])), wall=time.time() - t0); J('bc.json', res)
            st = train_mlp(X, Y, seed=0); pickle.dump(st, open(os.path.join(SD, f'bc_{H}_{stride}.pkl'), 'wb')); m, su = evaluate(pool, st, H, stride); m.update(val_mse_z=st['val_mse_z'], val_per=st['val_per'], samples=len(X), episodes=NE); res[f'bc_{H}_{stride}'] = m; print(H, stride, m, f'{time.time()-t0:.0f}s', flush=True); J('bc.json', res)
    elif stage == 'dart':
        H, stride, sigma, rho, NE = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]); res = J('dart.json')
        d = np.load(os.path.join(SD, f'bc_data_{H}_{stride}.npz')); X0, Y0 = d['X'], d['Y']
        X1, Y1, r = collect(pool, range(510000, 510000 + NE), student=None, H=H, stride=stride, label='fixed', dart=(sigma, rho), salt=1); rt = np.mean([x['success'] for x in r])
        st = train_mlp(np.concatenate([X0, X1]), np.concatenate([Y0, Y1]), seed=0); m, su = evaluate(pool, st, H, stride); m.update(teacher_noisy_succ=float(rt), sigma=sigma, rho=rho, episodes=len(d['X']) and (NE + 400), samples=len(X0) + len(X1), val_mse_z=st['val_mse_z']); res[f'dart_s{sigma}_r{rho}'] = m; print(m, f'{time.time()-t0:.0f}s', flush=True); J('dart.json', res); pickle.dump(st, open(os.path.join(SD, f'dart_{sigma}.pkl'), 'wb'))
    elif stage == 'dagger':
        H, stride, label, iters, NE = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]); lab = 'fixed' if label == 'fixed' else int(label); name = f'dagger_{label}'; res = J('dagger.json'); beta0 = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
        if beta0 > 0: name += f'_b{beta0}'
        if len(sys.argv) > 7 and sys.argv[7] == 'priv':
            PRIV = True; name += '_priv'; X, Y, r = collect(pool, range(500000, 500400), student=None, H=H, stride=stride, label='fixed'); st = train_mlp(X, Y, seed=0); m, su = evaluate(pool, st, H, stride); res = J('dagger.json'); res[name] = {'iter0': {**m, 'episodes': 400}}; print(name, 0, m, flush=True); J('dagger.json', res); base_seed = 560000; eps = 400
        else:
            d = np.load(os.path.join(SD, f'bc_data_{H}_{stride}.npz')); X, Y = d['X'], d['Y']; st = pickle.load(open(os.path.join(SD, f'bc_{H}_{stride}.pkl'), 'rb')); base_seed = (520000 if label == 'fixed' else 540000) + (100000 if beta0 > 0 else 0); eps = 400
        res = J('dagger.json'); res[name] = res.get(name, {}); res[name].setdefault('iter0', dict(res_of='bc', episodes=eps)); J('dagger.json', res)
        for it in range(1, iters + 1):
            seeds = range(base_seed + NE * (it - 1), base_seed + NE * it); Xn, Yn, r = collect(pool, seeds, student=st, H=H, stride=stride, label=lab, warm=(lab != 'fixed'), beta=beta0 * 0.5 ** (it - 1), salt=it); eps += NE
            X, Y = np.concatenate([X, Xn]), np.concatenate([Y, Yn]); st = train_mlp(X, Y, seed=it); m, su = evaluate(pool, st, H, stride)
            m.update(episodes=eps, samples=len(X), student_train_succ=float(np.mean([x['success'] for x in r])), label_plan_s=float(np.mean([x['plan_cost'] for x in r])), val_mse_z=st['val_mse_z'], wall=time.time() - t0); res = J('dagger.json'); res[name][f'iter{it}'] = m; print(name, it, m, flush=True); J('dagger.json', res)
            pickle.dump(st, open(os.path.join(SD, f'{name}_it{it}.pkl'), 'wb'))
        np.savez(os.path.join(SD, f'{name}_data.npz'), X=X, Y=Y)
