"""Throughput / VRAM / CPU benchmark (use a GPU job queue): MJX CTC+LQR rollouts, N envs x 800 steps, vs CPU MuJoCo single env.
  python scripts/mjx_bench.py --n 1024,4096 [--x64 0] [--cpu]  (--cpu: time arena single-env CPU instead)"""
import argparse, os, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--n", default="1024,4096"); ap.add_argument("--x64", type=int, default=0)
ap.add_argument("--cpu", action="store_true"); ap.add_argument("--out", default=None)
a = ap.parse_args()
res = {}
if a.cpu:     # CPU MuJoCo through the arena env, 1 env, CTC+LQR, W3 + pulses (per-step cost incl. python controller)
    from arena import Scenario
    from arena.perturb import Friction, TorquePulses
    from arena.eval.rollout import run_episode
    sc = Scenario("tracking", {"ref": "joint_sines"}).with_perturb(Friction(1.5, 0.3), TorquePulses())
    env = sc.make_env(); run_episode(sc, "ctc_lqr", 5000, env=env)
    t0, c0 = time.time(), time.process_time(); n = 10
    for s in range(n): run_episode(sc, "ctc_lqr", 5000 + s, env=env)
    w, c = time.time() - t0, time.process_time() - c0
    res = dict(kind="cpu_arena_1env", control_steps_per_s=n * 800 / w, wall=w, cpu_s=c, cpu_util=c / w, load1=os.getloadavg()[0])
    # raw mj_step only (no controller): lower bound for pure physics
    import mujoco, numpy as np
    from arena.robots import build_mjcf, RobotSpec
    m = mujoco.MjModel.from_xml_string(build_mjcf(RobotSpec())); d = mujoco.MjData(m)
    t0 = time.time(); [mujoco.mj_step(m, d) for _ in range(20000)]; w2 = time.time() - t0
    res["raw_mj_step_per_s"] = 20000 / w2
else:
    import arena_mjx
    jax = arena_mjx.setup(x64=bool(a.x64))
    import jax.numpy as jnp, numpy as np
    from arena.robots import RobotSpec
    from arena_mjx.sampling import episode_batch
    from arena_mjx.env import MjxTracking, make_params
    from arena_mjx.ctc import CtcLqr
    from arena_mjx.rollout import rollout
    dtype = jnp.float64 if a.x64 else jnp.float32
    env = MjxTracking(RobotSpec(), dtype); ctc = CtcLqr(env)
    pol = lambda s, q, qd, qr, qdr, qddr: ctc.act(q, qd, qr, qdr, qddr)
    fn = jax.jit(lambda p: rollout(env, pol, p)["rmse"])
    res = dict(kind="mjx", device=str(jax.devices()[0]), x64=a.x64, runs={})
    for n in [int(x) for x in a.n.split(",")]:
        b = episode_batch(np.arange(10000, 10000 + n) % 200 + 10000, RobotSpec(), pulses=True)
        p = make_params(b, dtype, fc=1.5, fv=0.3)
        t0 = time.time(); jax.block_until_ready(fn(p)); tc = time.time() - t0
        t0, c0 = time.time(), time.process_time(); R = 3
        for _ in range(R): jax.block_until_ready(fn(p))
        w, c = (time.time() - t0) / R, (time.process_time() - c0) / R
        res["runs"][n] = dict(first_call_s=tc, wall_per_800steps=w, control_steps_per_s=n * 800 / w, physics_steps_per_s=n * 1600 / w, cpu_util=c / w)
        print(n, res["runs"][n], flush=True)
    try: res["peak_bytes_in_use"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception: pass
print(json.dumps(res, indent=1))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
