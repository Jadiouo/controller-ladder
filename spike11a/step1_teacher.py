"""Step 1: teacher query cost + quality on CAL 1000-1049 (cond abc): fixed plan (= T-oracle structure), replan every 50 / 25 steps (cold start), replan every 25 (warm start)."""
import sys, os, json, time, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c11 import *
if __name__ == '__main__':
    S = list(range(1000, 1050)); pool = mp.Pool(8); out = {}
    for name, every, warm in [('fixed', None, False), ('replan50', 50, False), ('replan25', 25, False), ('replan25warm', 25, True)]:
        t0 = time.time(); r = pool.map(run_teacher, [(s, None, 'abc', every, warm, None, True) for s in S], chunksize=1); su = np.array([x['success'] for x in r]); c = np.concatenate([x['costs'] for x in r])
        out[name] = dict(succ=float(su.mean()), ci=boot_ci(su), min_dist=float(np.mean([x['min_dist'] for x in r])), n_queries=int(len(c)), q_cost_mean=float(c.mean()), q_cost_med=float(np.median(c)), q_cost_p90=float(np.quantile(c, .9)), per_episode_plan_s=float(np.mean([x['plan_cost'] for x in r])), unstable=float(np.mean([x['inst'] > 0 for x in r])), diverged=float(np.mean([x['diverged'] for x in r])), wall=time.time() - t0)
        print(name, out[name], flush=True); json.dump(out, open(os.path.join(OUT11, 'step1_teacher.json'), 'w'), indent=1)
