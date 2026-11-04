"""Experiment B: teacher sensitivity to additive iid Gaussian torque noise, injected exactly like the RL residual in experiments/spike07/rl7.py (PPO path):
   res = RES_BOUND * clip(a, -1, 1) with a = mean + std*N(0,1);  tau = clip(tT + res, -TAU_MAX, TAU_MAX)  (every control step, same place).
   Here mean=0 and std_action = sigma/RES_BOUND, so sigma is in Nm per joint. Noise is per-episode seeded (independent of chunking). CAL seeds 1000-1049 only.
   Decision rule (pre-registered): success(sigma=0.35) <= success(sigma=0) - 15 pp -> 'exploration noise itself breaks the loop'."""
import sys, os, json, time, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from common import *
SIGMAS = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]; REPS = 10; SEEDS = np.arange(1000, 1050); B = V.RES_BOUND
class Noise:
    def __init__(self, seeds, sigma, rep): self.rng = [np.random.default_rng([int(s), rep, 8080]) for s in seeds]; self.s = sigma / B
    def __call__(self, env, tT, obs):
        z = np.stack([g.standard_normal(3) for g in self.rng]); a = np.clip(self.s * z, -1, 1); return np.clip(tT + B * a, -V.TAU_MAX, V.TAU_MAX)
import functools
def mk(sigma, rep): return functools.partial(Noise, sigma=sigma, rep=rep)
if __name__ == '__main__':
    nw = 8; pool = mp.Pool(nw); t0 = time.time(); res = {}; steps = 0
    for sg in SIGMAS:
        reps = [0] if sg == 0 else list(range(REPS)); H = []
        for rp in reps:
            h, st = peval(pool, PROB, SEEDS, nw, polf=None if sg == 0 else mk(sg, rp)); H.append(h); steps += st
        H = np.array(H); succ = (H >= 300).astype(float).mean(0); hits = H.mean(0)       # per-episode, averaged over noise reps
        res[sg] = dict(succ_ep=succ.tolist(), hits_ep=hits.tolist(), per_rep_succ=[float((h >= 300).mean()) for h in H]); print(sg, 'succ', succ.mean(), 'hits', hits.mean(), f'{time.time()-t0:.0f}s', flush=True)
    s0 = np.array(res[0]['succ_ep']); h0 = np.array(res[0]['hits_ep']); rows = []
    for sg in SIGMAS:
        s = np.array(res[sg]['succ_ep']); h = np.array(res[sg]['hits_ep']); r = dict(sigma=sg, succ=float(s.mean()), succ_ci=boot_ci(s), hits=float(h.mean()), hits_ci=boot_ci(h), dsucc_vs0=float((s - s0).mean()), dsucc_ci=boot_ci(s - s0, seed=1), dhits_vs0=float((h - h0).mean()), dhits_ci=boot_ci(h - h0, seed=2), per_rep_succ=res[sg]['per_rep_succ']); rows.append(r); print(r, flush=True)
    # actual log-std noise of the spike07 RL (read from code defaults + learned values)
    import pickle; ls = {}
    for k in ('ppo_s0', 'ppo_s1', 'ppo_s2', 'oracle_s0', 'oracle_s1', 'oracle_s2'):
        v = pickle.load(open(os.path.join(ROOT, f'results/spike07/final_{k}.pkl'), 'rb')); v = list(v.values())[0]; ls[k] = dict(logstd0_arg=v['args']['logstd0'], final_logstd=[float(x) for x in v['logstd']], final_noise_Nm=[float(B * np.exp(x)) for x in v['logstd']])
    json.dump(dict(rows=rows, reps=REPS, rule_drop_pp=15, sim_steps=steps, wall_s=time.time() - t0, rl_noise=ls, B=B, noise_at_default_logstd0=B * float(np.exp(-3.5))), open(os.path.join(OUT, 'expB.json'), 'w'), indent=1)
