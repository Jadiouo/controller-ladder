"""Export the pole_balance task for the web arena: MJCF (true plant + nominal arm model), task constants, LQR gain,
Python closed-loop trajectories (seed 1000, clean) and clean/hard outcomes of 20 seeds with their initial states.
Writes ../../assets/pole_balance*.{xml,json} and ../python_pole.json. Run from anywhere with the repo env."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))   # repo root
import numpy as np
from arena import Scenario
from arena.controllers.base import as_factory
from arena.controllers.ctc_lqr import PoleLQR
from arena.perturb import PayloadError, Delay
from arena.robots import build_mjcf
from arena.tasks.pole_balance import PoleBalance

ASSETS = os.path.abspath(os.path.join(HERE, "..", "..", "assets"))
r9 = lambda a: np.round(np.asarray(a, float), 12).tolist()
task = PoleBalance()
spec = task.robot
open(os.path.join(ASSETS, "pole_balance.xml"), "w").write(build_mjcf(spec, task.tip_xml(spec)))
open(os.path.join(ASSETS, "pole_balance_nominal.xml"), "w").write(build_mjcf(spec))   # arm only: what ArmModel uses
env = Scenario("pole_balance").make_env()
env.reset(1000)
pl = PoleLQR(env.ctx)
info = task.info()
cfg = dict(link_lengths=list(spec.link_lengths), dt=spec.dt, substeps=spec.substeps, tau_max=spec.tau_max, payload_mass=spec.payload_mass, duration=task.duration,
           pole_length=task.L, bob_mass=task.bob_mass, hold=r9(task.hold), q_hold=r9(task.q_hold), fall_angle=task.fall_angle,
           theta0=task.theta0, omega0=task.omega0, gravity=9.81, lqr_K=r9(pl.K), lqr_q=[20.0, 2.0, 200.0, 20.0], lqr_r=0.01, kz=[400.0, 40.0],
           ctc_null=[100.0, 20.0], pid=dict(wn=30.0, c_i=70.0, c_p=5.0, c_d=0.2))
json.dump(cfg, open(os.path.join(ASSETS, "pole_balance.json"), "w"), indent=1)

CASES = {"clean": [], "hard": [PayloadError(rel=0.3), Delay(2)]}
SEEDS = list(range(1000, 1020))
out = {"seeds": SEEDS, "cases": {}, "init": {}, "traj": {}}


def run(cn, cname, seed, record=False):
    sc = Scenario("pole_balance").with_perturb(*CASES[cname])
    env = sc.make_env(); obs = env.reset(seed)
    n = env.n
    init = dict(q=r9(env.data.qpos[:n + 1]), qd=r9(env.data.qvel[:n + 1]))
    c = as_factory(cn)(env.ctx); c.reset(seed)
    Q = [env.data.qpos[:n + 1].copy()]; TAU = []
    for k in range(env.n_steps):
        obs, rew, te, tr, inf = env.step(c.act(obs))
        Q.append(env.data.qpos[:n + 1].copy()); TAU.append(env.data.ctrl[:n].copy())
        if te or tr: break
    return init, dict(outcome=inf["outcome"], steps=k + 1, q=np.array(Q).tolist() if record else None,
                      tau=np.array(TAU).tolist() if record else None)

for cname in CASES:
    for cn in ("pid", "ctc_lqr"):
        rows = []
        for s in SEEDS:
            init, res = run(cn, cname, s, record=(cname == "clean" and s == 1000))
            if cname == "clean" and cn == "pid": out["init"][str(s)] = init
            rows.append(dict(seed=s, outcome=res["outcome"], steps=res["steps"]))
            if res["q"] is not None: out["traj"][f"{cname}/{cn}"] = dict(q=res["q"], tau=res["tau"])
        out["cases"][f"{cname}/{cn}"] = rows
        print(cname, cn, sum(r["outcome"] == "timeout" for r in rows), "/", len(rows), "balanced")
json.dump(out, open(os.path.join(HERE, "..", "python_pole.json"), "w"), separators=(",", ":"))
