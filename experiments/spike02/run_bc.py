"""BC+DART and DAgger students for one weakness condition. Usage: python experiments/spike02/run_bc.py --w W1 [--seeds 10]"""
import argparse, time, json, pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
torch.set_num_threads(2)
from sim2 import *
from bc_lib import *
ap = argparse.ArgumentParser()
ap.add_argument('--w', required=True); ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--ntrain', type=int, default=200); ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--dart_sigma', type=float, default=3.0); ap.add_argument('--dagger_rounds', type=int, default=5)
ap.add_argument('--dagger_epochs', type=int, default=10); ap.add_argument('--methods', default='dart,dagger')
ap.add_argument('--out', default=None)
a = ap.parse_args(); w = W[a.w]; out = a.out or f'results/spike02/raw_bc_{a.w}.pkl'
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
timing = dict(collect=0., train=0., eval=0.); t_all = time.time()
T = Teacher(0.0)

if ('teacher', 0) not in res:
    t = time.time(); res[('teacher', 0)] = eval_policy(T.act, w); timing['eval'] += time.time() - t
print(a.w, w, 'teacher clean rmse %.4f dist %.4f' % (res[('teacher', 0)]['clean']['rmse'].mean(), res[('teacher', 0)]['dist']['rmse'].mean()), flush=True)

class DAggerMix:
    """executes beta*teacher + (1-beta) student per-step (random switch per env); logs obs and CLEAN teacher label"""
    def __init__(self, student, beta, rng):
        self.s, self.beta, self.rng, self.X, self.Y = student, beta, rng, [], []
    def __call__(self, q, qd, qr, qdr, qddr):
        tt = T.act(q, qd, qr, qdr, qddr)
        self.X.append(np.concatenate([q, qd, qr, qdr, qddr], 1)); self.Y.append(tt.copy())
        if self.s is None: return tt
        use_t = self.rng.random(len(q)) < self.beta
        return np.where(use_t[:, None], tt, self.s(q, qd, qr, qdr, qddr))

for seed in range(a.seeds):
    tr = 100_000 + seed * 1000 + np.arange(a.ntrain)
    for method in a.methods.split(','):
        if (method, seed) in res: continue
        torch.manual_seed(seed); t = time.time()
        if method == 'dart':
            P = sample_refs(tr); ie = sample_init_err(tr)
            noise = np.random.default_rng(555 + seed).normal(0, a.dart_sigma, (a.ntrain, NSTEP, 2))
            d = rollout2(T.act, w, P, ie, exec_noise=noise, record=True)
            X, Y = d['obs'], d['tau']; timing['collect'] += time.time() - t; t = time.time()
            net = new_net(); stats = norm_stats(X, Y); loss = fit(net, X, Y, stats, a.epochs)
            pol = NPPolicy(net, *stats); ntraj = a.ntrain
        else:  # dagger: R rounds, equal total trajectories (ntrain/R each)
            R = a.dagger_rounds; per = a.ntrain // R; rng = np.random.default_rng(777 + seed)
            net = new_net(); pol = None; Xs, Ys = [], []; tc = tt_ = 0.
            for r in range(R):
                sl = tr[r*per:(r+1)*per]; P = sample_refs(sl); ie = sample_init_err(sl)
                beta = 1.0 if r == 0 else (0.5 if r == 1 else 0.0)
                mix = DAggerMix(pol, beta, rng); t0 = time.time()
                rollout2(mix, w, P, ie); tc += time.time() - t0
                Xs.append(np.concatenate(mix.X)); Ys.append(np.concatenate(mix.Y))
                X, Y = np.concatenate(Xs), np.concatenate(Ys); t0 = time.time()
                if r == 0: stats = norm_stats(X, Y)   # fixed after round 0 (warm start)
                loss = fit(net, X, Y, stats, a.epochs if r == 0 else a.dagger_epochs)
                pol = NPPolicy(net, *stats); tt_ += time.time() - t0
            timing['collect'] += tc; t = time.time() - tt_  # so that train += tt_ below
            timing['train'] += tt_; ntraj = per * R
        if method == 'dart': timing['train'] += time.time() - t
        t = time.time()
        res[(method, seed)] = dict(**eval_policy(pol, w), final_loss=loss, ntraj=ntraj)
        timing['eval'] += time.time() - t
        c, dd = res[(method, seed)]['clean'], res[(method, seed)]['dist']
        print(f"{a.w} {method:6s} seed={seed} loss={loss:.4f} rmse={c['rmse'].mean():.4f} dist_rmse={dd['rmse'].mean():.4f} "
              f"(teacher {res[('teacher',0)]['clean']['rmse'].mean():.4f}/{res[('teacher',0)]['dist']['rmse'].mean():.4f}) elapsed={time.time()-t_all:.0f}s", flush=True)
        pickle.dump(res, open(out, 'wb'))
timing['total'] = time.time() - t_all
print('TIMING', json.dumps(timing)); json.dump(dict(timing=timing, args=vars(a), weak=w), open(out.replace('.pkl', '_timing.json'), 'w'))
