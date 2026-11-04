"""Export the push_t task for the web arena.
Writes ../../assets/push_t.xml (MJCF template, @@PUSHER@@ / @@WORLD@@ filled by js/tasks/push_t.js), push_t_nominal.xml (arm only, gravity 0),
push_t.json (constants) and ../python_push_t.json (initial states of seeds 1000-1019, Python outcomes for nominal and sensor delay 30 ms,
closed-loop trajectory + plan log of seed 1000).  Run from the repo root:
  flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 .venv/bin/python web/arena/validation/tools/export_push_t.py
"""
import json, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
import numpy as np
from arena import Scenario
from arena.controllers.base import as_factory
from arena.perturb import Delay, TFriction, TMass, TComOffset, TPoseNoise
from arena.robots import build_mjcf
from arena.tasks import _pusht_geom as G
from arena.tasks.push_t import PushT, X_RANGE, Z_RANGE, G_ACC

ASSETS = os.path.abspath(os.path.join(HERE, "..", "..", "assets"))
r12 = lambda a: np.round(np.asarray(a, float), 12).tolist()
task = PushT()
spec = task.robot
nog = lambda x: x.replace('gravity="0 0 -9.81"', 'gravity="0 0 0"')
open(os.path.join(ASSETS, "push_t.xml"), "w").write(nog(build_mjcf(spec, "@@PUSHER@@<!--@world-->@@WORLD@@")))
open(os.path.join(ASSETS, "push_t_nominal.xml"), "w").write(nog(build_mjcf(spec)))
cfg = dict(n=spec.n, link_lengths=list(spec.link_lengths), dt=spec.dt, substeps=spec.substeps, tau_max=spec.tau_max, duration=task.duration,
           mu=task.mu, t_mass=task.t_mass, table_b=task.table_b, coulomb_frac=task.coulomb_frac, v_ref=task.v_ref, pusher_radius=task.pusher_radius,
           cov_thr=task.cov_thr, max_rot=task.max_rot, goal=r12(task.goal), x_range=list(X_RANGE), z_range=list(Z_RANGE), g=G_ACC,
           geom=dict(W=G.W, B=G.B, S=G.S, L=G.L, RHO2=G.RHO2, V_BAR=G.V_BAR, V_STEM=G.V_STEM, AREA=G.AREA),
           teacher=dict(kp=900.0, kd=60.0, v_approach=0.40, v_push=0.08, a_max=2.0, depth=4, beam=10, n_try=6, travel_w=0.3, hold_s=0.2))
json.dump(cfg, open(os.path.join(ASSETS, "push_t.json"), "w"), indent=1)

SEEDS = list(range(1000, 1020))
CASES = {"nominal": [], "obs_delay_30ms": [Delay(6, where="obs")], "friction+50%": [TFriction(0.5)], "mass+100%": [TMass(1.0)],
         "com(20,0)mm": [TComOffset(0.02, 0.0)], "noise 5mm": [TPoseNoise(0.005, 0.0125)]}


def run(args):
    cname, seed, record = args
    sc = Scenario("push_t").with_perturb(*CASES[cname])
    env = sc.make_env(); obs = env.reset(seed)
    n = env.n
    init = dict(q=r12(env.data.qpos[:n + 3]), pose=r12(env.task.t0_pose), pusher=r12(env.task.p0))
    c = as_factory("push_t_teacher")(env.ctx); c.reset(seed)
    Q, TAU = [env.data.qpos[:n + 3].copy()], []
    cov_max = 0.0
    t0 = time.time()
    for k in range(env.n_steps):
        obs, rew, te, tr, inf = env.step(c.act(obs))
        cov_max = max(cov_max, env.task.coverage_now(env))
        if record:
            Q.append(env.data.qpos[:n + 3].copy()); TAU.append(env.data.ctrl[:n].copy())
        if te or tr: break
    plog = [dict(k=p["k"], prim=p["prim"], cost=float(p["cost"])) for p in c.plan_log]
    return dict(case=cname, seed=seed, init=init, outcome=inf["outcome"], steps=k + 1, coverage=float(env.task.coverage_now(env)), max_cov=float(cov_max),
                plans=plog, secs=time.time() - t0, q=np.round(Q, 10).tolist() if record else None, tau=np.round(TAU, 8).tolist() if record else None)


if __name__ == "__main__":
    jobs = [(c, s, c == "nominal" and s == 1000) for c in CASES for s in SEEDS]
    out = dict(seeds=SEEDS, cases={c: [] for c in CASES}, traj=None)
    with Pool(3) as p:
        for res in p.imap(run, jobs):
            print(res["case"], res["seed"], res["outcome"], res["steps"], f"cov {res['coverage']:.3f}", f"{res['secs']:.0f}s", flush=True)
            q, tau = res.pop("q"), res.pop("tau")
            if q is not None:
                out["traj"] = dict(q=q, tau=tau, seed=res["seed"], case=res["case"])
            out["cases"][res["case"]].append(res)
    for c, rows in out["cases"].items():
        print(c, sum(r["outcome"] == "success" for r in rows), "/", len(rows))
    json.dump(out, open(os.path.join(HERE, "..", "python_push_t.json"), "w"), separators=(",", ":"))
