"""Export the catch and toss tasks for the web arena: MJCF (true plant + nominal arm models), task constants, Python teacher
trajectories (seed 1000) and outcomes of 20 calibration seeds (1000-1019, never test seeds) with their initial states.
Writes ../../assets/{catch,toss}*.{xml,json} and ../python_{catch,toss}.json. Run from anywhere with the repo env."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))   # repo root
import numpy as np
from arena import Scenario
from arena.controllers.base import as_factory
from arena.perturb import BallObsDelay, ObjectMass, ReleaseDelay
from arena.robots import build_mjcf

ASSETS = os.path.abspath(os.path.join(HERE, "..", "..", "assets"))
r12 = lambda a: np.round(np.asarray(a, float), 12).tolist()
SEEDS = list(range(1000, 1020))          # calibration seeds only
W = lambda name, s: open(os.path.join(ASSETS, name), "w").write(s)


def export_assets():
    env = Scenario("catch").make_env()
    t, spec = env.task, env.task.robot
    W("catch.xml", build_mjcf(spec, t.tip_xml(spec)))
    W("catch_nominal.xml", build_mjcf(spec))
    cfg = dict(link_lengths=list(spec.link_lengths), dt=spec.dt, substeps=spec.substeps, tau_max=spec.tau_max, duration=t.duration,
               drag=t.drag, ball_mass=t.ball_mass, ball_radius=t.ball_radius, dwell=t.dwell, gravity=t.g, q_home=r12(t.q_home),
               drag_rate=t.drag / t.ball_mass, floor_z=-0.85)
    json.dump(cfg, open(os.path.join(ASSETS, "catch.json"), "w"), indent=1)
    env = Scenario("toss").make_env()
    t, spec = env.task, env.task.robot
    W("toss.xml", build_mjcf(spec, t.tip_xml(spec)))
    W("toss_nominal_free.xml", build_mjcf(spec))
    W("toss_nominal_hold.xml", build_mjcf(spec.with_(payload_mass=spec.payload_mass + t.ball_mass)))
    cfg = dict(link_lengths=list(spec.link_lengths), dt=spec.dt, substeps=spec.substeps, tau_max=spec.tau_max, duration=t.duration,
               drag=t.drag, ball_mass=t.ball_mass, ball_radius=t.ball_radius, gravity=t.g, q_start=r12(t.q_start), arm_reach=t.L,
               release_time=t.release_time, basket_hw=t.basket_hw, basket_h=t.basket_h, floor_z=-0.9, drag_rate=t.drag / t.ball_mass)
    json.dump(cfg, open(os.path.join(ASSETS, "toss.json"), "w"), indent=1)


def run(task, cname, pert, ctl, seed, record):
    sc = Scenario(task).with_perturb(*pert)
    env = sc.make_env(); obs = env.reset(seed)
    n = env.n
    init = dict(q=r12(env.data.qpos[:n + 2]), qd=r12(env.data.qvel[:n + 2]), basket=r12(env.task.basket) if task == "toss" else None)
    c = as_factory(ctl)(env.ctx); c.reset(seed)
    Q, TAU = [env.data.qpos[:n + 2].copy()], []
    inf = {}
    for k in range(env.n_steps):
        obs, rew, te, tr, inf = env.step(c.act(obs))
        Q.append(env.data.qpos[:n + 2].copy()); TAU.append(env.data.ctrl[:n].copy())
        if te or tr: break
    return init, dict(outcome=inf["outcome"], steps=k + 1, q=np.array(Q).tolist() if record else None,
                      tau=np.array(TAU).tolist() if record else None)


def main():
    export_assets()
    spec = {"catch": ("catch_teacher", {"clean": [], "obs_delay_2steps_10ms": [BallObsDelay(2)], "ball_mass_p30": [ObjectMass(0.3)]}),
            "toss": ("toss_teacher", {"clean": [], "release_delay_40ms": [ReleaseDelay(40)], "ball_mass_p30": [ObjectMass(0.3)]})}
    for task, (ctl, cases) in spec.items():
        out = {"seeds": SEEDS, "cases": {}, "init": {}, "traj": {}}
        for cname, pert in cases.items():
            rows = []
            for s in SEEDS:
                init, res = run(task, cname, pert, ctl, s, record=(s == 1000 and cname != "ball_mass_p30"))
                if cname == "clean": out["init"][str(s)] = init
                rows.append(dict(seed=s, outcome=res["outcome"], steps=res["steps"]))
                if res["q"] is not None: out["traj"][cname] = dict(q=res["q"], tau=res["tau"])
            out["cases"][cname] = rows
            print(task, cname, sum(r["outcome"] == "success" for r in rows), "/", len(rows), "success", flush=True)
        json.dump(out, open(os.path.join(HERE, "..", f"python_{task}.json"), "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
