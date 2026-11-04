"""Distillation feasibility spike. Usage: python experiments/spike/run_spike.py [--seeds 10] [--ntrain 200] [--epochs 40]"""
import argparse, time, json, pickle, os, numpy as np, torch, torch.nn as nn
from sim import *
torch.set_num_threads(2)  # 4 threads was 10x slower under machine load (other jobs); 2 is within the cap
ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--ntrain', type=int, default=200)
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--dart_sigma', type=float, default=3.0)
ap.add_argument('--errs', type=str, default='-0.3,0,0.1,0.3,1.0'); ap.add_argument('--out', default='results/spike01/raw.pkl')
a = ap.parse_args()
ERRS = [float(x) for x in a.errs.split(',')]
HID, BS, LR = 128, 1024, 1e-3

# fixed test set: SAME for teacher and all students
TEST_SEEDS = np.arange(50)                      # reference/init-err/disturbance seeds 0..49
PT = sample_refs(TEST_SEEDS); IET = sample_init_err(TEST_SEEDS); DT_ = sample_dist(TEST_SEEDS)

class NPPolicy:
    def __init__(self, net, xm, xs, ym, ys):
        self.W = [(l.weight.detach().numpy().T.astype(np.float64), l.bias.detach().numpy().astype(np.float64)) for l in net if isinstance(l, nn.Linear)]
        self.xm, self.xs, self.ym, self.ys = xm, xs, ym, ys
    def __call__(self, q, qd, qr, qdr, qddr):
        h = (np.concatenate([q, qd, qr, qdr, qddr], 1) - self.xm) / self.xs
        for i, (W, b) in enumerate(self.W):
            h = h @ W + b
            if i < len(self.W) - 1: h = np.maximum(h, 0)
        return np.clip(h * self.ys + self.ym, -TAU_MAX, TAU_MAX)

def train(X, Y, seed):
    torch.manual_seed(seed)
    xm, xs, ym, ys = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32); Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(10, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 2))
    opt = torch.optim.Adam(net.parameters(), lr=LR); n = len(Xt); steps = a.epochs * ((n + BS - 1) // BS)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]; loss = ((net(Xt[idx]) - Yt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return NPPolicy(net, xm, xs, ym, ys), float(loss.detach())

res = pickle.load(open(a.out, 'rb')) if os.path.exists(a.out) else {}
t_all = time.time(); timing = {'teacher_eval': 0., 'collect': 0., 'train': 0., 'eval': 0.}
for e in ERRS:
    T = Teacher(e)
    if (e, 'teacher', 0) not in res:
        t = time.time()
        res[(e, 'teacher', 0)] = dict(clean=rollout(T.act, PT, IET), dist=rollout(T.act, PT, IET, dist=DT_))
        timing['teacher_eval'] += time.time() - t
    for seed in range(a.seeds):
        tr_seeds = 100_000 + seed * 1000 + np.arange(a.ntrain)   # disjoint from test seeds 0..49
        P = sample_refs(tr_seeds); ie = sample_init_err(tr_seeds)
        for method in ['bc', 'dart']:
            if (e, method, seed) in res: continue
            t = time.time()
            noise = None
            if method == 'dart':
                noise = np.random.default_rng(555 + seed).normal(0, a.dart_sigma, (a.ntrain, NSTEP, 2))
            d = rollout(T.act, P, ie, exec_noise=noise, record=True)    # labels = teacher's CLEAN action at visited states
            timing['collect'] += time.time() - t; t = time.time()
            pol, loss = train(d['obs'], d['tau'], seed)
            timing['train'] += time.time() - t; t = time.time()
            res[(e, method, seed)] = dict(clean=rollout(pol, PT, IET), dist=rollout(pol, PT, IET, dist=DT_), final_loss=loss,
                                          train_rmse=float(d['rmse'].mean()))
            timing['eval'] += time.time() - t
            c, dd = res[(e, method, seed)]['clean'], res[(e, method, seed)]['dist']
            print(f"e={e:+.1f} {method:4s} seed={seed} loss={loss:.4f} rmse={c['rmse'].mean():.4f} dist_rmse={dd['rmse'].mean():.4f} "
                  f"(teacher {res[(e,'teacher',0)]['clean']['rmse'].mean():.4f}) elapsed={time.time()-t_all:.0f}s", flush=True)
            pickle.dump(res, open(a.out, 'wb'))
timing['total_this_invocation'] = time.time() - t_all
print('TIMING', json.dumps(timing)); json.dump(dict(timing=timing, args=vars(a)), open(a.out.replace('.pkl', '_timing.json'), 'w'))
