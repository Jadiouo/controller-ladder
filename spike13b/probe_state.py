"""Exploratory: does the arm's probe history contain information about the chain state at swing start? Ridge regression (closed form) from probe features
(measured minus nominal replay, as L-est) to (i) true chain qpos/qvel at the end of the probe, (ii) the 13-parameter theta. Train: seeds 500000+ (rep 0). Test: holdout 1100-1299 (rep None).
R2 = 1 - SSE/SST over test seeds, per target group. 'ridge+theta' adds the TRUE theta as extra input (upper bound of what parameter knowledge adds). Exploratory, not part of the ladder."""
import sys, os, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c13 import *
def work(items):
    X, TH, CS = [], [], []; pr = probe_of_lest(); AMP = np.array([0.5, 0.5, 0.4]) * pr['amp']
    for sd, rep in items:
        guard([sd]) if sd >= 500000 else guard13(sd); sc = scen(sd, 'abc', rep); env = sc.make_env(); ctx = env.ctx; obs = env.reset(sd); q0, taus, Qm, Vm = probe_run3(env, ctx, obs, pr['T'], AMP); n = ctx.robot.n
        X.append(probe_features10(ctx, q0, taus, Qm, Vm)); TH.append(true_theta(sd, 'abc', rep)); CS.append(np.concatenate([env.data.qpos[n:], env.data.qvel[n:]]))
    return np.array(X), np.array(TH), np.array(CS)
def gen(pool, items): ch = [items[i:i + 50] for i in range(0, len(items), 50)]; r = pool.map(work, ch, chunksize=1); return [np.concatenate([a[i] for a in r]) for i in range(3)]
def ridge(Xtr, Ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; A = (Xtr - mu) / sd; B = (Xte - mu) / sd; ym = Ytr.mean(0); W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - ym)); return B @ W + ym
def r2(Y, P): return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
if __name__ == '__main__':
    pool = mp.Pool(8); N = int(sys.argv[1]); tr = gen(pool, [(500000 + i, 0) for i in range(N)]); te = gen(pool, [(s, None) for s in range(1100, 1300)]); out = {}
    Ytr = {'chain_state': tr[2], 'theta': np.column_stack([tr[1][:, 0], np.log(tr[1][:, 1:12]), tr[1][:, 12] * 1e3])}; Yte = {'chain_state': te[2], 'theta': np.column_stack([te[1][:, 0], np.log(te[1][:, 1:12]), te[1][:, 12] * 1e3])}
    nch = tr[2].shape[1] // 2
    for tgt in Ytr:
        for lam in (1e-3, 1e-2, 1e-1, 1.0):                  # picked on a split of train, not test
            k = int(0.8 * N); P = ridge(tr[0][:k], Ytr[tgt][:k], tr[0][k:], lam); out[f'{tgt}|lam{lam}|val'] = r2(Ytr[tgt][k:], P)
        lam = max((1e-3, 1e-2, 1e-1, 1.0), key=lambda l: out[f'{tgt}|lam{l}|val']); P = ridge(tr[0], Ytr[tgt], te[0], lam); out[f'{tgt}|ridge_probe_features'] = dict(lam=lam, r2=r2(Yte[tgt], P))
        if tgt == 'chain_state':
            out['chain_state|r2_qpos_only'] = r2(Yte[tgt][:, :nch], P[:, :nch]); out['chain_state|r2_qvel_only'] = r2(Yte[tgt][:, nch:], P[:, nch:])
            Xa = np.column_stack([tr[0], np.log(tr[1][:, 1:12]), tr[1][:, 0], tr[1][:, 12] * 1e3]); Xb = np.column_stack([te[0], np.log(te[1][:, 1:12]), te[1][:, 0], te[1][:, 12] * 1e3])
            P2 = ridge(Xa, Ytr[tgt], Xb, lam); out['chain_state|ridge_probe_features+true_theta'] = r2(Yte[tgt], P2)
            Z = np.column_stack([np.log(tr[1][:, 1:12]), tr[1][:, 0], tr[1][:, 12] * 1e3]); Zt = np.column_stack([np.log(te[1][:, 1:12]), te[1][:, 0], te[1][:, 12] * 1e3]); out['chain_state|ridge_true_theta_only'] = r2(Yte[tgt], ridge(Z, Ytr[tgt], Zt, 1e-3))
            out['chain_state|var_total_qpos'] = float(Yte[tgt][:, :nch].var(0).sum()); out['chain_state|var_total_qvel'] = float(Yte[tgt][:, nch:].var(0).sum())
    out['N_train'] = N; out['N_test'] = 200; json.dump(out, open(os.path.join(OUT13, 'probe_state.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
