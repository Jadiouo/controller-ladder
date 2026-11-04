"""Learned estimators for spike14a (torch, run with .venv-gpu through a GPU job queue). Raw probe time series (120 steps x 18 channels) -> theta13.
variants: gru1 / gru4 (MSE in z-space; x1 = 12000 train episodes, x4 = 48000), dec1 / dec4 (same net, loss = (1-a) z-MSE + a * Gauss-Newton tip-trajectory error with per-episode sensitivity J_i), mlp4 (spike10b MLP recipe on stride-2 residual features, x4 data).
usage: nn_train.py <variant> <hid> <lr> <alpha> <seeds comma> <tag> [epochs]      Every config: checkpoint = epoch with the lowest CAL loss (CAL = 1000-1049 + 2000-2149 labels). Writes predictions (theta13) for cal/hold/new and the CAL curves."""
import sys, os, time, pickle, json, numpy as np, torch, torch.nn as nn
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); DAT = os.path.join(ROOT, '.scratch/s14'); OUT = os.path.join(ROOT, 'results/spike14a'); NL = 10
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
def theta_to_y(TH): return np.column_stack([TH[:, 0], np.log(TH[:, 1:11]), np.log(TH[:, 11]), TH[:, 12] * 1e3])
BOX = dict(ls=(0.8, 1.2), lm=(-1.5, 1.5), ld=(-2.2, 2.2), fr=(0.0, 8.0))
def y_to_theta(Y):
    TH = np.column_stack([np.clip(Y[:, 0], *BOX['ls']), np.exp(np.clip(Y[:, 1:11], *BOX['lm'])), np.exp(np.clip(Y[:, 11], *BOX['ld'])), np.clip(Y[:, 12], 0, BOX['fr'][1]) * 1e-3]); return TH
def seq_feats(D):
    q0 = D['q0'][:, None, :].astype(np.float32); Qm, Vm, tau = D['Qm'], D['Vm'], D['taus']; dQ = np.clip(np.nan_to_num(Qm - D['Q0n']), -2, 2); dV = np.clip(np.nan_to_num(Vm - D['V0n']), -2, 2)
    return np.concatenate([Qm - q0, Vm, tau, dQ, dV, np.repeat(q0, Qm.shape[1], 1)], 2).astype(np.float32)
def mlp_feats(D): return np.clip(np.nan_to_num(np.concatenate([(D['Qm'] - D['Q0n'])[:, 1::2].reshape(len(D['Qm']), -1), (D['Vm'] - D['V0n'])[:, 1::2].reshape(len(D['Vm']), -1)], 1)), -2, 2).astype(np.float32)
class GRUNet(nn.Module):
    def __init__(s, din, H, dout=13):
        super().__init__(); s.g = nn.GRU(din, H, num_layers=2, batch_first=True, dropout=0.0); s.h = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, dout))
    def forward(s, x): o, _ = s.g(x); return s.h(torch.cat([o[:, -1], o.mean(1)], 1))
def mlp(din, H, dout=13): return nn.Sequential(nn.Linear(din, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, dout))
def load(name): return pickle.load(open(os.path.join(DAT, f'raw_{name}.pkl'), 'rb'))
def run(var, H, lr, alpha, seeds, tag, EP):
    t0 = time.time(); big = var.endswith('4'); feats = mlp_feats if var.startswith('mlp') else seq_feats
    parts = [load('train1')] + ([load('train_extra')] if big else []); TH = np.concatenate([p['TH'] for p in parts]); X = np.concatenate([feats(p) for p in parts]); Y = theta_to_y(TH)
    ev = {k: load(k) for k in ('cal', 'hold', 'new')}; EX = {k: feats(v) for k, v in ev.items()}; Yc = theta_to_y(ev['cal']['TH'])
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-9; mu = X.reshape(-1, X.shape[-1]).mean(0) if X.ndim == 3 else X.mean(0); sd = (X.reshape(-1, X.shape[-1]).std(0) if X.ndim == 3 else X.std(0)) + 1e-6
    tX = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(dev); Xt = tX(X); Zt = torch.from_numpy(((Y - ymu) / ysd).astype(np.float32)).to(dev); Xc = tX(EX['cal']); Zc = ((Yc - ymu) / ysd)
    W = None
    if var.startswith('dec'):
        S = pickle.load(open(os.path.join(DAT, 'sens_train.pkl'), 'rb')); Jall = np.stack([s['J'] for s in S]).astype(np.float64); nJ = len(S); Wi = np.einsum('nti,ntj->nij', Jall, Jall); good = ~np.array([x['bad'] for x in S]); Wbar = Wi[good].mean(0)
        Wfull = np.broadcast_to(Wbar, (len(Y), 13, 13)).copy(); Wfull[:nJ][good] = Wi[good]; d0 = (Y - ymu); s = np.einsum('ni,nij,nj->n', d0, Wfull, d0).mean(); Wfull = Wfull / s; Wt = torch.from_numpy(Wfull.astype(np.float32)).to(dev); ysd_t = torch.from_numpy(ysd.astype(np.float32)).to(dev); W = (Wt, ysd_t, Wfull[0] * s / s)
        Wc_bar = torch.from_numpy((Wbar / s).astype(np.float32)).to(dev)
    res = dict(var=var, H=H, lr=lr, alpha=alpha, EP=EP, ntrain=len(Y), runs={})
    for seed in seeds:
        torch.manual_seed(seed); net = (mlp(Xt.shape[1], H) if var.startswith('mlp') else GRUNet(Xt.shape[2], H)).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); bs = 256
        steps = EP * ((len(Xt) + bs - 1) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1); best = (1e9, None, -1); curve = []
        for ep in range(EP):
            net.train(); perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), bs):
                idx = perm[i:i + bs]; out = net(Xt[idx]); loss = ((out - Zt[idx]) ** 2).mean()
                if W is not None:
                    dlt = (out - Zt[idx]) * W[1]; dec = torch.einsum('bi,bij,bj->b', dlt, W[0][idx], dlt).mean(); loss = (1 - alpha) * loss + alpha * dec
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            net.eval()
            with torch.no_grad(): pc = net(Xc).cpu().numpy()
            zm = float(((pc - Zc) ** 2).mean())
            if W is not None: dd = torch.from_numpy((pc - Zc) * ysd).to(dev).float(); zm = float((1 - alpha) * zm + alpha * torch.einsum('bi,ij,bj->b', dd, Wc_bar, dd).mean())
            curve.append(zm)
            if zm < best[0]: best = (zm, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
        net.load_state_dict(best[1]); net.eval(); preds = {}
        with torch.no_grad():
            for k in ev: preds[k] = y_to_theta(net(tX(EX[k])).cpu().numpy().astype(np.float64) * ysd + ymu)
        pickle.dump(dict(preds=preds, best_ep=best[2], cal_z=best[0], curve=curve, wall=time.time() - t0), open(os.path.join(DAT, f'pred_{tag}_s{seed}.pkl'), 'wb'))
        res['runs'][seed] = dict(best_ep=best[2], cal_z=best[0], wall=time.time() - t0); print(tag, 'seed', seed, 'cal z-mse', round(best[0], 4), 'ep', best[2], f'{time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(os.path.join(DAT, f'nnlog_{tag}.json'), 'w'), indent=1); return res
if __name__ == '__main__':
    var = sys.argv[1]
    if sys.argv[2] == 'family':      # family <Hs> <lrs> <alphas> <EP> : CAL sweep over (H, lr) at the first alpha with seed 0, then 3 training seeds for each alpha at the CAL-best (H, lr)
        Hs, lrs, alphas, EP = [int(x) for x in sys.argv[3].split(',')], [float(x) for x in sys.argv[4].split(',')], [float(x) for x in sys.argv[5].split(',')], int(sys.argv[6]); sw = []
        for H in Hs:
            for lr in lrs: r = run(var, H, lr, alphas[0], [0], f'sw_{var}_H{H}_lr{lr}', EP); sw.append((r['runs'][0]['cal_z'], H, lr, r['runs'][0]['wall']))
        sw.sort(); json.dump(sw, open(os.path.join(OUT, f'nn_sweep_{var}.json'), 'w'), indent=1); _, H, lr, _ = sw[0]; print('chosen', H, lr, flush=True)
        for a in alphas: run(var, H, lr, a, [0, 1, 2], f'{var}' + (f'_a{a}' if var.startswith('dec') else ''), EP)
        cf = os.path.join(DAT, f'chain_{var}.txt')          # follow-up families to run in the same queue slot (read at run time); line: var family Hs lrs alphas EP [tag_suffix]
        if os.path.exists(cf):
            for line in open(cf).read().strip().splitlines():
                a = line.split(); v2 = a[0]; Hs, lrs, alphas, EP2 = [int(x) for x in a[2].split(',')], [float(x) for x in a[3].split(',')], [float(x) for x in a[4].split(',')], int(a[5]); suf = a[6] if len(a) > 6 else ''; sw = []
                for H in Hs:
                    for lr in lrs: r = run(v2, H, lr, alphas[0], [0], f'sw_{v2}{suf}_H{H}_lr{lr}', EP2); sw.append((r['runs'][0]['cal_z'], H, lr, r['runs'][0]['wall']))
                sw.sort(); json.dump(sw, open(os.path.join(OUT, f'nn_sweep_{v2}{suf}.json'), 'w'), indent=1); _, H, lr, _ = sw[0]
                for a_ in alphas: run(v2, H, lr, a_, [0, 1, 2], f'{v2}' + (f'_a{a_}' if v2.startswith('dec') else '') + suf, EP2)
    else: run(var, int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), [int(x) for x in sys.argv[5].split(',')], sys.argv[6], int(sys.argv[7]) if len(sys.argv) > 7 else 60)
