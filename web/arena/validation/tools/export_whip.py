"""Export the whip task for the web arena.
Writes ../../assets/whip.xml (MJCF template: @@CHAIN@@ / @@WORLD@@ are filled in by js/tasks/whip.js), whip_nominal.xml (arm only),
whip.json (task constants + the OFFLINE PLAN TABLE: Python WhipTeacher keyframes per (target grid cell, assumed chain length)) and
../python_whip.json (Python closed-loop references for validate_whip.mjs).
Run from the repo root with the repo env:  flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 .venv/bin/python web/arena/validation/tools/export_whip.py
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
import mujoco
import numpy as np
from arena import Scenario
from arena.controllers.base import as_factory
from arena.controllers.whip_teacher import WhipTeacher, Q_LO, Q_HI, KNOT_T
from arena.perturb import ChainLength
from arena.robots import build_mjcf
from arena.tasks.whip import Whip, Q_START

ASSETS = os.path.abspath(os.path.join(HERE, "..", "..", "assets"))
r12 = lambda a: np.round(np.asarray(a, float), 12).tolist()
task = Whip()
spec = task.robot
open(os.path.join(ASSETS, "whip.xml"), "w").write(build_mjcf(spec, "@@CHAIN@@<!--@world-->@@WORLD@@"))
open(os.path.join(ASSETS, "whip_nominal.xml"), "w").write(build_mjcf(spec))

D_GRID, A_GRID = (1.20, 1.35, 1.50, 1.65), (-0.35, 0.0, 0.35)
SCALES = (0.9, 1.0, 1.1)
TRUE_SCALES = (0.9, 1.0, 1.1)
TARGETS = [dict(id=f"d{d:.2f}_a{a:+.2f}", d=d, a=a, x=float(d * np.cos(a)), z=float(d * np.sin(a))) for d in D_GRID for a in A_GRID]


def make_env(true_scale=1.0, assumed_scale=1.0):
    sc = Scenario("whip", task_kw=dict(link_len=0.1 * assumed_scale))
    if true_scale != assumed_scale:        # TRUE chain = nominal(assumed) * (1 + rel)
        sc = sc.with_perturb(ChainLength(rel=true_scale / assumed_scale - 1.0))
    return sc.make_env()


def set_target(env, tgt):
    env.task.target = np.array(tgt, float)
    env.model.geom_pos[env.model.geom("target").id] = [tgt[0], -0.05, tgt[1]]
    mujoco.mj_forward(env.model, env.data)
    return env._observe(push=True)


def run(env, ctrl, obs, record=False):
    n = env.n
    mind = np.inf
    Q, TAU = [env.data.qpos.copy()], []
    sid = env.model.site("whip_tip").id
    mind = float(np.linalg.norm(env.data.site_xpos[sid][[0, 2]] - env.task.target))
    for k in range(env.n_steps):
        obs, rew, te, tr, inf = env.step(ctrl.act(obs))
        mind = min(mind, float(np.linalg.norm(env.data.site_xpos[sid][[0, 2]] - env.task.target)))
        if record:
            Q.append(env.data.qpos.copy()); TAU.append(env.data.ctrl[:n].copy())
        if te or tr: break
    return dict(outcome=inf["outcome"], steps=k + 1, min_dist=mind, q=np.array(Q).tolist() if record else None,
                tau=np.array(TAU).tolist() if record else None)


# ---- 1. offline plan table --------------------------------------------------------------------------------------------
plans = {}
t0 = time.time()
for s in SCALES:
    env = make_env(s, s)
    env.reset(1000)
    teacher = WhipTeacher(env.ctx); teacher.reset(0)
    for T in TARGETS:
        tt = time.time()
        knots = teacher.plan(Q_START.copy(), np.array([T["x"], T["z"]]))
        plans[f"{T['id']}|{s:.1f}"] = dict(knots=r12(knots), planned_min_dist=float(teacher.plan_info["planned_min_dist"]))
        print(f"plan {T['id']} assumed {s}: planned min dist {teacher.plan_info['planned_min_dist']:.3f}  ({time.time() - tt:.1f}s)", flush=True)
print("plans", time.time() - t0, "s")

cfg = dict(n=spec.n, link_lengths=list(spec.link_lengths), dt=spec.dt, substeps=spec.substeps, tau_max=spec.tau_max,
           payload_mass=spec.payload_mass, duration=task.duration, r_hit=task.r_hit, n_links=task.n_links,
           chain_len=0.1, chain_mass=0.015, chain_damp=0.0004, q_start=r12(Q_START), knot_t=KNOT_T.tolist(), q_lo=Q_LO, q_hi=Q_HI,
           qp=1.6e5, qv=800.0, targets=TARGETS, scales=list(SCALES), plans=plans,
           note="plans precomputed offline by the Python WhipTeacher (CMA-ES keyframe search on the nominal model), seed 0, start posture Q_START")
json.dump(cfg, open(os.path.join(ASSETS, "whip.json"), "w"), indent=0, separators=(",", ":"))

# ---- 2. Python references -----------------------------------------------------------------------------------------------
ref = dict(grid=[], traj={}, cal={})
GRID_SEEDS = {T["id"]: 1000 + i for i, T in enumerate(TARGETS)}
def grid_run(T, seed, s, ts, record):
    env = make_env(ts, 1.0)
    obs = env.reset(seed); obs = set_target(env, [T["x"], T["z"]])
    teacher = as_factory("whip_teacher")(env.ctx); teacher.reset(seed)
    teacher.knots = np.array(plans[f"{T['id']}|{s:.1f}"]["knots"]); teacher.q0 = obs.q.copy()
    q0 = obs.q.copy()
    res = run(env, teacher, obs, record=record)
    return q0, res


for T in TARGETS:                                                     # execution of table plans on true chains (start posture noise from the seed)
    seed = GRID_SEEDS[T["id"]]
    for s in SCALES:
        for ts in TRUE_SCALES:
            q0, res = grid_run(T, seed, s, ts, False)
            ref["grid"].append(dict(target=T["id"], seed=seed, q0=r12(q0), assumed=s, true=ts, outcome=res["outcome"], steps=res["steps"], min_dist=res["min_dist"]))
# full-length trajectories (13 DoF, 500 steps) of the first two MISSES + the first two hits, to compare chain motion after the swing
picks = [g for g in ref["grid"] if g["outcome"] != "success"][:2] + [g for g in ref["grid"] if g["outcome"] == "success"][:2]
for g in picks:
    T = next(t for t in TARGETS if t["id"] == g["target"])
    q0, res = grid_run(T, g["seed"], g["assumed"], g["true"], True)
    ref["traj"][f"{g['target']}|{g['assumed']}|{g['true']}"] = dict(q0=r12(q0), q=np.round(res["q"], 11).tolist(), tau=np.round(res["tau"], 9).tolist(), assumed=g["assumed"], true=g["true"], target=[T["x"], T["z"]], outcome=res["outcome"])
print("grid done", sum(r["outcome"] == "success" for r in ref["grid"]), "/", len(ref["grid"]), "success")

SEEDS = list(range(1000, 1020))                                       # calibration seeds; the FULL Python teacher plans each seed itself
for cname, ts in (("nominal", 1.0), ("chain_len-10%", 0.9)):
    rows = []
    for seed in SEEDS:
        env = make_env(ts, 1.0)
        obs = env.reset(seed)
        q0, tgt = obs.q.copy(), env.task.target.copy()
        teacher = as_factory("whip_teacher")(env.ctx); teacher.reset(seed)
        res = run(env, teacher, obs)
        rows.append(dict(seed=seed, q0=r12(q0), target=r12(tgt), knots=r12(teacher.knots), outcome=res["outcome"], steps=res["steps"], min_dist=res["min_dist"]))
    ref["cal"][cname] = rows
    print(cname, sum(r["outcome"] == "success" for r in rows), "/", len(rows), flush=True)
json.dump(ref, open(os.path.join(HERE, "..", "python_whip.json"), "w"), separators=(",", ":"))
