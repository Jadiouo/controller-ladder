"""Diagnostic: is the teacher plan a smooth function of (plant, target)? For 12 CAL scenarios plan 4x with different CMA-ES streams (seed offset) and report the spread of the arm-reference trajectories
(RMS over t of qr differences between independent plans) vs the spread between different scenarios. All plans reach the target (score < r_hit) or are reported."""
import sys, os, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c11 import *
def job(a):
    seed, salt = a; sc, env, obs = new_env(seed, 'abc', None); c2 = oracle_ctx(env.ctx, seed, 'abc', None); CAN = len(sys.argv) > 1 and sys.argv[1] == 'canon'; rp = Replanner(c2, CANON_CFG if CAN else None); d = env.data
    x0 = (d.qpos[:rp.model.nq].copy(), d.qvel[:rp.model.nv].copy()); p = rp.plan(x0, 0.0, np.asarray(obs.ref['target'], float), seed * 10 + salt, x0k=canon_x0(np.asarray(obs.ref['target'], float)) if CAN else None); t = np.arange(0, 1.8, 0.005); qr, _, _ = spline_g(p['q0'], p['v0'], p['knots'], p['S'], t); return qr, p['score']
if __name__ == '__main__':
    pool = mp.Pool(8); S = list(range(1000, 1012)); r = pool.map(job, [(s, j) for s in S for j in range(4)], chunksize=1); Q = np.array([x[0] for x in r]).reshape(len(S), 4, -1, 3); sc = np.array([x[1] for x in r]).reshape(len(S), 4)
    within = np.mean([np.sqrt(((Q[i, a] - Q[i, b]) ** 2).mean()) for i in range(len(S)) for a in range(4) for b in range(a + 1, 4)]); between = np.mean([np.sqrt(((Q[i, 0] - Q[j, 0]) ** 2).mean()) for i in range(len(S)) for j in range(i + 1, len(S))])
    out = dict(rms_within_scenario_rad=float(within), rms_between_scenarios_rad=float(between), frac_plans_hit=float((sc < 0.08).mean()), scores_mean=float(sc.mean()))
    print(out); json.dump(out, open(os.path.join(OUT11, 'multimodal_canon.json' if len(sys.argv) > 1 and sys.argv[1] == 'canon' else 'multimodal.json'), 'w'), indent=1)
