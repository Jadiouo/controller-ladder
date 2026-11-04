"""Step 3: L-est. Data: probe-only episodes on TRAINING seeds (>=500000; target seed 500000+i, chain error independent draw rep=0): features = probe history minus nominal-chain replay, label = true (ls,ms,ds) scaled to [-1,1].
Hyper-parameters (hidden size, weight decay, epochs; lr 1e-3, Adam, MSE) are chosen ONLY on CAL (seeds 1000-1049, chain from the seed): lowest mean squared error in label space. Then 3 training seeds (init + data order)."""
import sys, os, time, json, pickle, itertools, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c9b import *
import torch, torch.nn as nn; torch.set_num_threads(2)
NTRAIN = 12000
def gen(chunk): return gen_probe_data(chunk)
def train_mlp(X, Y, hid, wd, epochs, seed, lr=1e-3, bs=256):
    torch.manual_seed(seed); mu, sd = X.mean(0), X.std(0) + 1e-9; Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)); Yt = torch.from_numpy(Y)
    net = nn.Sequential(nn.Linear(X.shape[1], hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 3)); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), bs):
            idx = perm[i:i + bs]; loss = ((net(Xn[idx]) - Yt[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    layers = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]
    return dict(mu=mu.astype(np.float64), sd=sd.astype(np.float64), layers=layers, hid=hid, wd=wd, epochs=epochs, seed=seed)
if __name__ == '__main__':
    t0 = time.time(); pairs = [(500000 + i, 0) for i in range(NTRAIN)]; chunks = [pairs[i:i + 100] for i in range(0, NTRAIN, 100)]
    pool = mp.Pool(8); r = pool.map(gen, chunks, chunksize=1); pool.close(); X = np.concatenate([a for a, b in r]); Y = np.concatenate([b for a, b in r]); print('data', X.shape, f'{time.time()-t0:.0f}s', flush=True)
    # CAL set (chain from the seed): features need CAL seeds -> same generator but rep=None path
    Xc, Yc = [], []
    for sd in range(1000, 1050):
        t = true_chain(sd); env = scenario_of(t).make_env(); ctx = env.ctx; obs = env.reset(sd); _, q0, taus, Qm, Vm = run_probe(env, ctx, obs); Xc.append(probe_features(ctx, q0, taus, Qm, Vm)); Yc.append(y_of(tvec(t)))
    Xc, Yc = np.array(Xc), np.array(Yc); sweep = []
    for hid, wd, ep in itertools.product([128, 256], [0.0, 1e-4], [100, 300]):
        st = train_mlp(X, Y, hid, wd, ep, seed=0)
        Z = (Xc - st['mu']) / st['sd']
        for i, (Wt, b) in enumerate(st['layers']): Z = Z @ Wt.T + b; Z = np.tanh(Z) if i < len(st['layers']) - 1 else Z
        mse = float(((np.clip(Z, -1, 1) - Yc) ** 2).mean()); mse_c = ((np.clip(Z, -1, 1) - Yc) ** 2).mean(0).tolist(); sweep.append(dict(hid=hid, wd=wd, epochs=ep, cal_mse=mse, per_param=mse_c, wall=time.time() - t0)); print(sweep[-1], flush=True)
    best = min(sweep, key=lambda s: s['cal_mse']); models = {}
    for s in range(3):
        models[s] = train_mlp(X, Y, best['hid'], best['wd'], best['epochs'], seed=s); pickle.dump(models[s], open(os.path.join(OUT, f'lest_s{s}.pkl'), 'wb'))
    json.dump(dict(ntrain=NTRAIN, sweep=sweep, chosen=best, feature_dim=int(X.shape[1]), prior_mse=float((Yc ** 2).mean()), wall=time.time() - t0, probe_episodes_total=NTRAIN + 50 * len(sweep),
                   label_var_train=Y.var(0).tolist()), open(os.path.join(OUT, 'lest_train.json'), 'w'), indent=1); print('chosen', best)
