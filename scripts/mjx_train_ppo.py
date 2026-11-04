"""Residual PPO on MJX (GPU). Usage (through a GPU job queue!):
  python scripts/mjx_train_ppo.py --w W3 --seeds 0,1,2,3,4 --steps 2000000 --tag b2m [--nenv 64] [--pool train|fresh]
W0 = nominal plant; W3 = Coulomb 1.5 Nm (tanh, eps .05) + 0.3 Nm s/rad viscous (spike02 definitions).
Training scenarios (refs, init errors, torque pulses on every second env) are drawn from `--pool`:
  train = arena TRAIN_SEEDS 10000-10199 (protocol rule, default); fresh = spike02's 500000+seed*100000+... (outside every split).
Progress is logged on CAL seeds 1000-1049 only (never TEST seeds). Policies -> results/mjx/policies/<tag>_<W>_s<seed>.npz"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--w", required=True); ap.add_argument("--seeds", default="0"); ap.add_argument("--steps", type=int, default=2_000_000)
ap.add_argument("--nenv", type=int, default=64); ap.add_argument("--mb", type=int, default=2048); ap.add_argument("--tag", default="b2m")
ap.add_argument("--pool", default="train", choices=["train", "fresh"]); ap.add_argument("--no_pulses", action="store_true")
ap.add_argument("--x64", type=int, default=0); ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "mjx", "policies"))
a = ap.parse_args()
import arena_mjx
jax = arena_mjx.setup(x64=bool(a.x64))
import jax.numpy as jnp, numpy as np
from arena.eval.protocol import TRAIN_SEEDS, CAL_SEEDS
from arena.robots import RobotSpec
from arena_mjx.sampling import episode_batch
from arena_mjx.env import MjxTracking, make_params
from arena_mjx.ctc import CtcLqr
from arena_mjx.rollout import rollout
from arena_mjx.ppo import HP, init_params, make_iteration, actor_state_dict, mlp, feats

WK = dict(W0=dict(), W3=dict(fc=1.5, fv=0.3))[a.w]
hp = HP(nenv=a.nenv, mb=a.mb)
dtype = jnp.float64 if a.x64 else jnp.float32
spec = RobotSpec(); env = MjxTracking(spec, dtype); ctc = CtcLqr(env)
iteration, opt = make_iteration(env, ctc, hp)
N, H = hp.nenv, hp.horizon
iters = a.steps // (N * H); ipe = env.n_steps // H           # iterations per episode batch
os.makedirs(a.outdir, exist_ok=True)
cal = make_params(episode_batch(CAL_SEEDS, spec, pulses=False), dtype, **WK)
cal_p = make_params(episode_batch(CAL_SEEDS, spec, pulses=True), dtype, **WK)
pol_cal = lambda prm: (lambda s, q, qd, qr, qdr, qddr: ctc.act(q, qd, qr, qdr, qddr) + hp.bound * jnp.clip(
    mlp(prm["actor"], feats(q, qd, qr, qdr, qddr, ctc.act(q, qd, qr, qdr, qddr))), -1, 1))
cal_eval = jax.jit(lambda prm, pp: rollout(env, lambda *z: jnp.clip(pol_cal(prm)(*z), -80, 80), pp)["rmse"].mean())
summary = {}
for seed in [int(s) for s in a.seeds.split(",")]:
    t0, c0 = time.time(), time.process_time()
    params = init_params(jax.random.PRNGKey(seed), hp); opt_state = opt.init(params); key = jax.random.PRNGKey(10_000 + seed)
    rs = np.random.default_rng(seed)
    ep_idx = 0; log = []
    for it in range(iters):
        if it % ipe == 0:
            if a.pool == "train":
                sd = rs.choice(TRAIN_SEEDS, N, replace=N > len(TRAIN_SEEDS))
            else:
                sd = 500_000 + seed * 100_000 + ep_idx * N + np.arange(N)
            ep_idx += 1
            pulses = np.zeros(N, bool) if a.no_pulses else (np.arange(N) % 2 == 1)   # spike02: ds[::2] = 0
            p = make_params(episode_batch(sd, spec, pulses=pulses), dtype, **WK)
            state = env.reset(p)
        params, opt_state, state, key, m = iteration(params, opt_state, state, p, key)
        if it % 20 == 0 or it == iters - 1:
            m = {k: float(v) for k, v in m.items()}
            log.append((it, (it + 1) * N * H, m["meanR"], m["std"], time.time() - t0))
            print(f"  [{a.w} s{seed}] it {it}/{iters} steps {(it+1)*N*H} meanR {m['meanR']:.4f} std {m['std']:.3f} |res| {m['absres']:.2f} t {time.time()-t0:.0f}s", flush=True)
    jax.block_until_ready(params)
    wall, cpu = time.time() - t0, time.process_time() - c0
    c_clean, c_dist = float(cal_eval(params, cal)), float(cal_eval(params, cal_p))
    out = os.path.join(a.outdir, f"{a.tag}_{a.w}_s{seed}.npz")
    np.savez(out, **actor_state_dict(params), log=np.array(log), wall=wall, cpu=cpu, steps=iters * N * H, cal_clean=c_clean, cal_dist=c_dist,
             args=json.dumps(vars(a)))
    summary[seed] = dict(wall=wall, cpu=cpu, cal_clean=c_clean, cal_dist=c_dist)
    print(f"{a.w} seed={seed} steps={iters*N*H} wall {wall:.0f}s cpu {cpu:.0f}s CAL rmse clean {c_clean:.4f} dist {c_dist:.4f} -> {out}", flush=True)
try:
    ms = jax.devices()[0].memory_stats(); print("peak_bytes_in_use", ms.get("peak_bytes_in_use"))
except Exception as e:
    print("memstats n/a", e)
json.dump(summary, open(os.path.join(a.outdir, f"{a.tag}_{a.w}_summary_{a.seeds.replace(',', '')}.json"), "w"))
