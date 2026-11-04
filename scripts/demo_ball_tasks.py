"""Demo: the classical teachers of the two ball tasks (catch, toss) without model error and under one pilot perturbation each.

Run (heavy-job wrapper required on this machine):
  flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
      .venv/bin/python scripts/demo_ball_tasks.py
Writes results/ball_tasks/table.md and four clips (catch / toss, one success, one failure under a perturbation).
Seeds are DEMO_SEEDS (5000-5019): disjoint from the calibration and test sets. The perturbation strengths below are PILOT values
(chosen by eye on the demo seeds to show an effect); the formal strengths are to be decided on the calibration seeds.
"""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arena import Scenario
from arena.controllers.base import as_factory
from arena.eval import DEMO_SEEDS, evaluate, run_episode
from arena.eval.stats import bootstrap_ci, summarize
from arena.perturb import AirDrag, BallObsDelay, BallObsNoise, Friction, ObjectMass, ReleaseDelay
from arena.viz.record import _marker, _Writer

OUT = "results/ball_tasks"
TEACHER = {"catch": "catch_teacher", "toss": "toss_teacher"}
CONDITIONS = {
    "catch": [("no model error", []), ("air drag x2", [AirDrag(1.0)]), ("ball mass x2", [ObjectMass(1.0)]),
              ("ball position noise 10 mm", [BallObsNoise(0.01)]), ("ball position delay 10 ms", [BallObsDelay(2)])],
    "toss": [("no model error", []), ("air drag x2", [AirDrag(1.0)]), ("object mass x2", [ObjectMass(1.0)]),
             ("release 40 ms late", [ReleaseDelay(40)]), ("joint friction (Coulomb 5 Nm, viscous 1.0)", [Friction(coulomb=5.0, viscous=1.0)])],
}
VIDEO_PERT = {"catch": "ball position delay 10 ms", "toss": "release 40 ms late"}     # the perturbation of the failure clip
CAMERA = {"catch": dict(lookat=(0.75, 0.0, 0.1), distance=3.2), "toss": dict(lookat=(0.85, 0.0, -0.25), distance=3.3)}
TITLE = {"catch": "CATCH", "toss": "TOSS"}


def sc_for(task, pert):
    return Scenario(task).with_perturb(*pert)


def run_table():
    rows, per_seed, outcome = [], {}, {}
    for task in ("catch", "toss"):
        for name, pert in CONDITIONS[task]:
            t0 = time.time()
            r = evaluate(sc_for(task, pert), TEACHER[task], DEMO_SEEDS, label=name)
            succ = r.metrics["success"]
            m, lo, hi = bootstrap_ci(succ)
            sec = r.metrics["impact_speed"] if task == "catch" else r.metrics["landing_err"]
            s = summarize(sec)
            en = summarize(r.metrics["energy"])
            per_seed[(task, name)] = succ
            rows.append((task, name, m, lo, hi, int(succ.sum()), len(succ), s["mean"], s["half"], s["n_valid"], en["mean"], time.time() - t0))
            print(task, name, f"{m:.2f} [{lo:.2f},{hi:.2f}] sec={s['mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return rows, per_seed


def table_md(rows):
    L = ["# Ball tasks: classical teachers under model error (pilot)", "",
         f"Seeds {DEMO_SEEDS[0]}-{DEMO_SEEDS[-1]} (n={len(DEMO_SEEDS)}, DEMO set; not calibration, not test). One episode per seed, "
         "same seeds in every row (same balls / baskets, only the true task parameters or the sensing differ). "
         "success = primary metric (0/1); the interval is a percentile bootstrap 95% CI of the success rate (2000 resamples). "
         "Perturbation strengths are PILOT values picked on the demo seeds to show an effect; the formal strengths are to be "
         "calibrated on CAL seeds. Secondary: catch = relative speed ball vs basket just before the first contact "
         "(m/s, lower = softer; mean +- 95% t half-width over the episodes that touched the basket, n in the last column); "
         "toss = horizontal miss of the basket centre where the object crosses the rim plane (m). energy = mean sum(tau^2) per executed step.", ""]
    for task, title in (("catch", "Catch (3-DoF arm with a basket, ball thrown in from outside the workspace; success = ball stays >= 0.5 s)"),
                        ("toss", "Toss (object held at the tip, released at t = 0.8 s, basket 0.7-1.8 m away, arm reach 1.0 m)")):
        sec = "impact speed (m/s)" if task == "catch" else "landing error (m)"
        L += [f"## {title}", "", f"| condition | success | 95% CI | change vs no error | {sec} | n | energy |", "|---|---|---|---|---|---|---|"]
        base = [r for r in rows if r[0] == task][0][2]
        for t, name, m, lo, hi, k, n, sm, sh, nv, en, _ in rows:
            if t != task:
                continue
            d = "-" if name == "no model error" else f"{100 * (m - base):+.0f} pts"
            L.append(f"| {name} | {k}/{n} = {m:.0%} | [{lo:.0%}, {hi:.0%}] | {d} | {sm:.2f} +- {sh:.2f} | {nv} | {en:.0f} |")
        L.append("")
    return "\n".join(L)


# ---- video ------------------------------------------------------------------------------------------------------
def _font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def record(task, pert, seed, path, caption, size=(480, 640), fps=25, hold_s=1.2):
    sc = sc_for(task, pert)
    env = sc.make_env()
    h, w = size
    writer = _Writer(path, size, fps)
    env.model.vis.headlight.ambient[:] = 0.55
    env.model.vis.headlight.diffuse[:] = 0.8
    renderer = mujoco.Renderer(env.model, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAMERA[task]["lookat"]
    cam.distance, cam.azimuth, cam.elevation = CAMERA[task]["distance"], 90.0, 0.0
    every = max(1, int(round(1.0 / (fps * env.dt))))
    trail, f_big, f_small = [], _font(22), _font(16)
    state = {"frames": 0}

    def frame(e, k, banner=None):
        renderer.update_scene(e.data, camera=cam)
        for p in trail[::3]:
            _marker(renderer.scene, p, (1.0, 0.75, 0.2, 0.55), 0.012)
        if task == "toss":
            _marker(renderer.scene, (e.task.basket[0], e.task.basket[1] + e.task.basket_h + 0.12), (0.15, 0.55, 0.8, 1.0), 0.03)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        d.text((10, 8), f"{TITLE[task]}  |  {caption}  |  seed {seed}", fill=(255, 255, 255), font=f_small)
        d.text((10, 30), f"t = {k * e.dt:4.2f} s", fill=(200, 200, 200), font=f_small)
        if task == "toss":
            note = "released" if e.task._released else "holding"
            d.text((10, 52), f"{note}   basket {e.task.basket[0]:.2f} m away (arm reach {e.task.L:.1f} m)", fill=(200, 200, 200), font=f_small)
        if banner:
            col = (90, 230, 120) if banner[0] else (255, 110, 90)
            d.rectangle((0, h - 44, w, h), fill=(20, 20, 20))
            d.text((12, h - 38), banner[1], fill=col, font=f_big)
        writer.add(np.asarray(img))
        state["frames"] += 1

    def cb(e, k):
        trail.append(e.task.measure(e)["ball_pos"].copy())
        if k % every == 0:
            frame(e, k)

    try:
        tr = run_episode(sc, as_factory(TEACHER[task]), seed, env=env, frame_cb=cb)
        ok = tr.outcome == "success"
        trail.append(env.task.measure(env)["ball_pos"].copy())
        what = {"catch": ("CAUGHT: ball stayed in the basket 0.5 s", "MISSED: the ball got away"),
                "toss": ("IN THE BASKET", "MISSED THE BASKET")}[task]
        for _ in range(int(hold_s * fps)):
            frame(env, env.k, banner=(ok, what[0] if ok else what[1]))
    finally:
        renderer.close()
        writer.close()
    return tr.outcome


def pick_failure_seed(task, pert, per_seed):
    base, bad = per_seed[(task, "no model error")], per_seed[(task, VIDEO_PERT[task])]
    cand = [i for i in range(len(base)) if base[i] > 0.5 and bad[i] < 0.5]
    return int(DEMO_SEEDS[cand[0]]) if cand else int(DEMO_SEEDS[int(np.argmin(bad))])


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, per_seed = run_table()
    md = table_md(rows)
    open(f"{OUT}/table.md", "w").write(md + "\n")
    print(md)
    for task in ("catch", "toss"):
        base = per_seed[(task, "no model error")]
        s_ok = int(DEMO_SEEDS[int(np.argmax(base))])
        out = record(task, [], s_ok, f"{OUT}/{task}_success.mp4", "classical teacher, no model error")
        print(task, "success clip", s_ok, out)
        pert = dict(CONDITIONS[task])[VIDEO_PERT[task]]
        s_bad = pick_failure_seed(task, pert, per_seed)
        out = record(task, pert, s_bad, f"{OUT}/{task}_failure.mp4", f"classical teacher, {VIDEO_PERT[task]}")
        print(task, "failure clip", s_bad, out)


if __name__ == "__main__":
    main()
