"""Build the canonical-teacher grid: nominal-plant plans over a (distance, angle) grid of targets, warm-started along a serpentine path (continuation), so that neighbouring targets get neighbouring plans."""
import sys, os, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c11 import *
if __name__ == '__main__':
    sc, env, obs = new_env(1000, 'none', None); th = np.concatenate([[1.0], np.ones(NL), [1.0, 2e-3]]); arr = theta_arrays(type('C', (), dict(task_info=dict(chain_len=np.full(NL, .1), chain_mass=np.full(NL, .015), chain_damp=np.full(NL, .0004))))(), th)
    ctx = env.ctx; info = {**ctx.task_info, **{k: np.asarray(v, float).copy() for k, v in arr.items()}}; n = ctx.robot.n; c2 = dataclasses.replace(ctx, task_info=info)
    def mk():
        t = make_task('whip', c2.robot, n_links=int(info['n_links'])); t.chain_len, t.chain_mass, t.chain_damp = (np.array(info[k], float) for k in ('chain_len', 'chain_mass', 'chain_damp'))
        m = mujoco.MjModel.from_xml_string(build_mjcf(c2.robot, t.tip_xml(c2.robot))); m.dof_frictionloss[n:] = 2e-3; return m
    c2 = dataclasses.replace(c2, make_model=mk); d = env.data; x0 = (d.qpos[:13].copy(), d.qvel[:13].copy()); q0 = x0[0][:3]
    K = np.zeros((len(GRID_D), len(GRID_A), 4, 3)); sc_ = np.zeros(K.shape[:2]); prev = None; order = [(i, j if i % 2 == 0 else len(GRID_A) - 1 - j) for i in range(len(GRID_D)) for j in range(len(GRID_A))]; log = []
    for (i, j) in order:
        tg = GRID_D[i] * np.array([np.cos(GRID_A[j]), np.sin(GRID_A[j])]); rp = Replanner(c2, dict(sigma0=0.25, restarts=1) if prev is not None else None)
        p = rp.plan(x0, 0.0, tg, 7, x0k=None if prev is None else prev); 
        if prev is not None and p['score'] > 0.04: rp2 = Replanner(c2); p2 = rp2.plan(x0, 0.0, tg, 7); log.append(('cold-fallback', i, j, p['score'], p2['score'])); p = p2 if p2['score'] < p['score'] else p
        K[i, j] = p['knots']; sc_[i, j] = p['score']; prev = p['knots']; print(i, j, round(p['score'], 3), flush=True)
    dist = [np.linalg.norm(K[i, j] - K[i2, j2]) for i in range(len(GRID_D)) for j in range(len(GRID_A)) for (i2, j2) in ((i + 1, j), (i, j + 1)) if i2 < len(GRID_D) and j2 < len(GRID_A)]
    np.savez(GRID_F, K=K, score=sc_); out = dict(score_max=float(sc_.max()), score_mean=float(sc_.mean()), n_fallback=len(log), adjacent_knot_dist_mean=float(np.mean(dist)), adjacent_knot_dist_max=float(np.max(dist)), fallbacks=[list(map(float, l[1:])) for l in log]); print(out); json.dump(out, open(os.path.join(OUT11, 'canon_grid.json'), 'w'), indent=1)
