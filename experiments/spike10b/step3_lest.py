"""Step 3: learned estimators (probe features -> plant theta). Variants:
  full: 13 outputs [ls, log m_1..m_10, log ds, fric]  trained on cond abc (x1)             -> L-est-full   (train dist == eval dist; sim labels)
  eff : 4 outputs  [ls, log total mass, log ds, fric] trained on cond abc (x1)              -> L-est-eff    (no per-link labels)
  ood : 13 outputs, trained on x0.5 and x2 (half each), evaluated on x1                     -> L-est-OOD
Training data: probe-only episodes on TRAINING seeds >= 500000 (full variant & eff: seeds 500000.., rep 0; ood: seeds 520000.., rep 0). The probe (T, amp) is the one of the chosen T-adapt3.
CAL (seeds 1000-1049, reps None,1,2,3 = 200 draws; ood: x0.5/x2 alternating) selects hyper-parameters by label MSE (z-space). Then 3 training seeds each.
argv[1] = path of json with the probe {T, amp}."""
import sys, os, time, json, pickle, itertools, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c10b import *
import torch, torch.nn as nn; torch.set_num_threads(2)
NTRAIN = 12000
def train_mlp(X, Y, hid, wd, epochs, seed, lr=1e-3, bs=256):
    torch.manual_seed(seed); mu, sd = X.mean(0), X.std(0) + 1e-9; ymu, ysd = Y.mean(0), Y.std(0) + 1e-9; Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)); Yt = torch.from_numpy(((Y - ymu) / ysd).astype(np.float32))
    net = nn.Sequential(nn.Linear(X.shape[1], hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, Y.shape[1])); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), bs):
            idx = perm[i:i + bs]; loss = ((net(Xn[idx]) - Yt[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    layers = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]
    return dict(mu=mu.astype(np.float64), sd=sd.astype(np.float64), ymu=ymu.astype(np.float64), ysd=ysd.astype(np.float64), layers=layers, hid=hid, wd=wd, epochs=epochs, seed=seed)
def predict(st, X):
    Z = (X - st['mu']) / st['sd']
    for i, (Wt, b) in enumerate(st['layers']): Z = Z @ Wt.T + b; Z = np.tanh(Z) if i < len(st['layers']) - 1 else Z
    return Z * st['ysd'] + st['ymu']
def gen_set(pool, items, T, amp):
    chunks = [(items[i:i + 100], T, amp) for i in range(0, len(items), 100)]; r = pool.map(gen_probe10, chunks, chunksize=1); return np.concatenate([a for a, b in r]), np.concatenate([b for a, b in r])
if __name__ == '__main__':
    pr = json.load(open(sys.argv[1])); T, amp = pr['T'], pr['amp']; t0 = time.time(); pool = mp.Pool(8); stats = {}; D = {}
    D['abc'] = gen_set(pool, [(500000 + i, 0, 'abc') for i in range(NTRAIN)], T, amp); print('train abc', D['abc'][0].shape, f'{time.time()-t0:.0f}s', flush=True)
    D['ood'] = gen_set(pool, [(520000 + i, 0, 'x0.5' if i % 2 == 0 else 'x2') for i in range(NTRAIN)], T, amp); print('train ood', f'{time.time()-t0:.0f}s', flush=True)
    reps = [None, 1, 2, 3]
    C = {'abc': gen_set(pool, [(s, r, 'abc') for r in reps for s in range(1000, 1050)], T, amp), 'ood': gen_set(pool, [(s, r, 'x0.5' if (j % 2 == 0) else 'x2') for j, (r, s) in enumerate((r, s) for r in reps for s in range(1000, 1050))], T, amp)}
    print('cal sets', f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(D=D, C=C, T=T, amp=amp), open(os.path.join(ROOT, '.scratch/lest_data.pkl'), 'wb'))
    out = dict(T=T, amp=amp, ntrain=NTRAIN, variants={}); models = {}
    for var, (dset, kind) in dict(full=('abc', 'full'), eff=('abc', 'eff'), ood=('ood', 'full')).items():
        X, TH = D[dset]; Y = theta_to_y(TH, kind); Xc, THc = C[dset]; Yc = theta_to_y(THc, kind); sweep = []
        for hid, wd, ep in itertools.product([128, 256], [0.0, 1e-4], [100, 300]):
            st = train_mlp(X, Y, hid, wd, ep, seed=0); P = predict(st, Xc); mse = float((((P - Yc) / st['ysd']) ** 2).mean()); sweep.append(dict(hid=hid, wd=wd, epochs=ep, cal_mse_z=mse, per_output_z=(((P - Yc) / st['ysd']) ** 2).mean(0).tolist(), wall=time.time() - t0)); print(var, sweep[-1]['hid'], sweep[-1]['wd'], sweep[-1]['epochs'], round(mse, 4), flush=True)
        best = min(sweep, key=lambda s: s['cal_mse_z']); out['variants'][var] = dict(kind=kind, train_set=dset, sweep=sweep, chosen=best, label_sd=Y.std(0).tolist(), cal_prior_mse_z=float((((Yc - Y.mean(0)) / Y.std(0)) ** 2).mean()))
        for s in range(3):
            st = train_mlp(X, Y, best['hid'], best['wd'], best['epochs'], seed=s); st['kind'] = kind; pickle.dump(st, open(os.path.join(OUT, f'lest_{var}_s{s}.pkl'), 'wb'))
        # CAL errors in physical units for seed-0 model of the chosen config
        P = predict(pickle.load(open(os.path.join(OUT, f'lest_{var}_s0.pkl'), 'rb')), Xc); out['variants'][var]['cal_rmse_phys'] = np.sqrt(((P - Yc) ** 2).mean(0)).tolist()
        json.dump(out, open(os.path.join(OUT, 'lest_train.json'), 'w'), indent=1); print(var, 'chosen', best, flush=True)
    out['wall'] = time.time() - t0; out['probe_episodes_total'] = 2 * NTRAIN + 2 * 200 + 3 * 0; json.dump(out, open(os.path.join(OUT, 'lest_train.json'), 'w'), indent=1)
