"""Decision-oriented sensitivity for training episodes: oracle plan (true theta) of training episode i, then J_i = d tip_xz(t) / d y (y = [ls, log m_1..10, log ds, fric_mm]) of the planner's own swing model (WT.SwingSim, PD servo on the plan) by central differences.
Gauss-Newton: tip-trajectory error of the oracle plan executed on a plant with theta_hat ~ (y_hat - y)^T J^T J (y_hat - y). Training seeds only (500000+)."""
import sys, os, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
DY = np.array([0.03] + [0.08] * 10 + [0.15, 0.6])      # half-step in y units
def tip_traj(ctx, th, knots, q0):
    c2 = dataclasses.replace(ctx, task_info={**ctx.task_info, **theta_arrays(ctx, th)}, make_model=lambda: model_theta(ctx, th)); sim = WT.SwingSim(c2)
    tip, bad = sim.run(q0, knots[None], float(ctx.task_info['duration']), nthread=1); return tip[0][19::20], bool(bad[0])       # every 0.05 s
def y2th(y): return np.concatenate([[y[0]], np.exp(y[1:11]), [np.exp(y[11]), max(y[12], 0.0) * 1e-3]])
def job(sd):
    guard14(sd); assert sd >= 500000; reset_probe_globals(); sc = scen(sd, 'abc', 0); env = sc.make_env(); ctx = env.ctx; obs = env.reset(sd); th = true_theta(sd, 'abc', 0)
    t0 = time.time(); _, arr = (None, theta_arrays(type('C', (), dict(task_info=dict(chain_len=np.full(NL, .1), chain_mass=np.full(NL, .015), chain_damp=np.full(NL, .0004))))(), th))
    ctrl = OracleFull(ctx, arr, float(th[12])); ctrl.reset(sd); q0 = np.asarray(obs.q, float).copy(); knots = ctrl.plan(q0, np.asarray(obs.ref['target'], float)); tp = time.time() - t0
    y = np.concatenate([[th[0]], np.log(th[1:11]), [np.log(th[11]), th[12] * 1e3]]); base, b0 = tip_traj(ctx, th, knots, q0); J = np.zeros((base.size, 13)); badany = b0
    for i in range(13):
        e = np.zeros(13); e[i] = DY[i]; yp, ym = y + e, y - e
        if i == 12: yp[12] = max(yp[12], 0) ; ym[12] = max(ym[12], 0)
        tp_, b1 = tip_traj(ctx, y2th(yp), knots, q0); tm_, b2 = tip_traj(ctx, y2th(ym), knots, q0); badany |= b1 | b2; J[:, i] = ((tp_ - tm_).ravel()) / (yp[i] - ym[i] if yp[i] != ym[i] else 1.0)
    return dict(seed=sd, J=J.astype(np.float32), bad=badany, t_plan=tp, dmin=ctrl.plan_info.get('planned_min_dist', np.nan))
if __name__ == '__main__':
    N = int(sys.argv[1]); t0 = time.time(); pool = mp.Pool(8); out = pool.map(job, [500000 + i for i in range(N)], chunksize=2)
    pickle.dump(out, open(os.path.join(DAT, 'sens_train.pkl'), 'wb')); print(N, f'{time.time()-t0:.0f}s', 'bad', np.mean([o['bad'] for o in out]), 'plan s', np.mean([o['t_plan'] for o in out]), flush=True)
