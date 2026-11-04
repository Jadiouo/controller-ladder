"""Demo: the classical teachers of the two "hard for a classical teacher" tasks, whip and push_t, without model error and under
one perturbation at a time (pilot strengths) -> results/whip_pusht/{table.md, whip_success.mp4, whip_failure.mp4, push_t_success.mp4,
push_t_failure.mp4}.

Run (heavy-job wrapper required on this machine):
  flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
      .venv/bin/python scripts/demo_whip_pusht.py
Seeds are DEMO_SEEDS (5000-5019), disjoint from calibration and test sets: whip uses all 20, push_t the first N_PUSHT (default 20; episodes are 30 s
long and the planner re-plans many times; set N_PUSHT=10 for a quicker run). Perturbation strengths are PILOT values (chosen on the CAL seeds, not
calibrated for the final protocol). Also reports a no-search baseline per task (random keyframes / random push primitive).
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
from arena.perturb import ChainDamping, ChainLength, ChainMass, Delay, TComOffset, TFriction, TMass, TPoseNoise
from arena.viz.record import _marker, _Writer

OUT = "results/whip_pusht"
N_PUSHT = int(os.environ.get("N_PUSHT", 20))
TEACHER = {"whip": "whip_teacher", "push_t": "push_t_teacher"}
BASELINE = {"whip": ("whip_random", "random keyframes (no search)"), "push_t": ("push_t_random", "random push primitive (no search)")}
SEEDS = {"whip": DEMO_SEEDS, "push_t": DEMO_SEEDS[:N_PUSHT]}
CONDITIONS = {
    "whip": [("no model error", []), ("chain mass +50%", [ChainMass(0.5)]), ("chain mass -50%", [ChainMass(-0.5)]),
             ("chain damping x3", [ChainDamping(2.0)]), ("chain damping x0.25", [ChainDamping(-0.75)]),
             ("chain length +10%", [ChainLength(0.10)]), ("chain length -10%", [ChainLength(-0.10)]),
             ("uneven chain (mass, log-sd 0.3 per link)", [ChainMass(0.0, spread=0.3, pattern=1)])],
    "push_t": [("no model error", []), ("block friction x0.5", [TFriction(-0.5)]), ("block friction x2", [TFriction(1.0)]),
               ("block mass x2", [TMass(1.0)]), ("block mass x4", [TMass(3.0)]),
               ("COM offset 20 mm", [TComOffset(0.02, 0.0)]), ("COM offset 40 mm", [TComOffset(0.04, 0.0)]),
               ("block pose noise 8 mm / 0.08 rad", [TPoseNoise(0.008, 0.08)]),
               ("sensor delay 20 ms (q, qd, block pose)", [Delay(4, where="obs")]), ("sensor delay 30 ms", [Delay(6, where="obs")])],
}
VIDEO_PERT = {"whip": "chain length -10%", "push_t": "sensor delay 30 ms"}
CAMERA = {"whip": dict(lookat=(0.8, 0.0, -0.3), distance=3.6), "push_t": dict(lookat=(0.58, 0.0, 0.0), distance=1.7)}
TITLE = {"whip": "WHIP", "push_t": "PUSH-T"}


def sc_for(task, pert):
    return Scenario(task).with_perturb(*pert)


def run_table():
    rows, per_seed = [], {}
    for task in ("whip", "push_t"):
        for ctrl, name, pert in [(TEACHER[task], n, p) for n, p in CONDITIONS[task]] + [(BASELINE[task][0], BASELINE[task][1], [])]:
            t0 = time.time()
            r = evaluate(sc_for(task, pert), ctrl, SEEDS[task], label=name)
            succ = r.metrics["success"]
            m, lo, hi = bootstrap_ci(succ)
            if task == "whip":
                sec = [summarize(r.metrics["min_dist"]), summarize(r.metrics["hit_speed"])]
            else:
                sec = [summarize(r.metrics["coverage"]), summarize(r.metrics["time_to_success"]), summarize(r.metrics["path_len"])]
            per_seed[(task, name)] = succ
            rows.append((task, name, m, lo, hi, int(succ.sum()), len(succ), sec, time.time() - t0))
            print(task, name, f"{m:.2f} [{lo:.2f},{hi:.2f}] ({time.time() - t0:.0f}s)", flush=True)
    return rows, per_seed


def table_md(rows, per_seed):
    L = ["# Whip and Push-T: classical teachers under model error (pilot)", "",
         f"Seeds: DEMO set (5000-5019; not calibration, not test); whip n={len(SEEDS['whip'])}, push_t n={len(SEEDS['push_t'])}. One episode per seed, "
         "the same seeds in every row (same targets / block poses, only the TRUE task parameters differ; the planner never sees them). "
         "success = primary metric (0/1) with a percentile-bootstrap 95% CI (2000 resamples); `change` = paired difference to the "
         "no-error row in percentage points with the paired bootstrap 95% CI. Perturbation strengths are PILOT values (chosen on the CAL "
         "seeds, not calibrated; for the three block parameters of push_t no reasonable strength changed the result, see the README). whip is an under-actuated 13-DoF fling "
         "planned open loop for the chain; Push-T is contact-rich with many solutions (see the README for why its teacher turned out strong). "
         "The last row of each block is a baseline with the same low-level control and no search, to show what the planning contributes.", ""]
    for task, title, cols in (
            ("whip", "Whip (3-DoF arm + 10-link free chain, target 1.15-1.70 m from the shoulder, arm reach 1.0 m, hit radius 8 cm, 2.5 s)",
             ("min dist (m)", "tip speed at hit (m/s)")),
            ("push_t", "Push-T (2-DoF arm, T block 0.20 x 0.20 m, success = coverage >= 0.9, 30 s)",
             ("final coverage", "time to success (s)", "pusher path (m)"))):
        L += [f"## {title}", "", "| condition | success | 95% CI | change vs no error (paired) | " + " | ".join(cols) + " |",
              "|---|---|---|---|" + "---|" * len(cols)]
        base = per_seed[(task, "no model error")]
        for t, name, m, lo, hi, k, n, sec, _ in rows:
            if t != task:
                continue
            if name == "no model error":
                d = "-"
            else:
                diff = per_seed[(task, name)] - base
                dm, dlo, dhi = bootstrap_ci(diff)
                d = f"{100 * dm:+.0f} pts [{100 * dlo:+.0f}, {100 * dhi:+.0f}]"
            cells = []
            for s in sec:
                cells.append("n/a" if s["n_valid"] == 0 else f"{s['mean']:.2f} +- {0 if np.isnan(s['half']) else s['half']:.2f} (n={s['n_valid']})")
            L.append(f"| {name} | {k}/{n} = {m:.0%} | [{lo:.0%}, {hi:.0%}] | {d} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Secondary metrics are mean +- 95% t half-width over the episodes where the quantity exists (n in brackets): whip `min dist` is the closest "
          "approach of the chain tip BEFORE the episode ended (a hit ends the episode, so a successful episode shows its first-hit distance, "
          "< 8 cm); `tip speed` is the chain-tip speed at the first hit; Push-T `final coverage` is the overlap with the goal at the end of the episode "
          "(for a success: the value that triggered it), `time to success` only over successes; `pusher path` of the 30 ms sensor-delay row (tens of metres) is an unstable arm "
          "flailing, not a plan.", ""]
    return "\n".join(L)


# ---- video ------------------------------------------------------------------------------------------------------
def _font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def record(task, pert, seed, path, caption, size=(480, 640), fps=25, hold_s=1.5):
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

    def frame(e, k, banner=None):
        renderer.update_scene(e.data, camera=cam)
        for p in trail[::2]:
            _marker(renderer.scene, p, (1.0, 0.75, 0.2, 0.6), 0.008 if task == "whip" else 0.006)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, w, 74 if task == "push_t" else 50), fill=(20, 20, 20))
        d.text((10, 8), f"{TITLE[task]}  |  {caption}  |  seed {seed}", fill=(255, 255, 255), font=f_small)
        d.text((10, 30), f"t = {k * e.dt:4.2f} s", fill=(200, 200, 200), font=f_small)
        if task == "push_t":
            d.text((10, 52), f"coverage {e.task.coverage_now(e):.2f}  (success at {e.task.cov_thr:.2f})", fill=(200, 200, 200), font=f_small)
        if banner:
            col = (90, 230, 120) if banner[0] else (255, 110, 90)
            d.rectangle((0, h - 44, w, h), fill=(20, 20, 20))
            d.text((12, h - 38), banner[1], fill=col, font=f_big)
        writer.add(np.asarray(img))

    def cb(e, k):
        trail.append(e.task.chain_tip(e).copy() if task == "whip" else e.task.tip_pos(e).copy())
        if k % every == 0:
            frame(e, k)

    try:
        tr = run_episode(sc, as_factory(TEACHER[task]), seed, env=env, frame_cb=cb)
        ok = tr.outcome == "success"
        trail.append(env.task.chain_tip(env).copy() if task == "whip" else env.task.tip_pos(env).copy())
        what = {"whip": ("HIT: chain tip reached the target", "MISSED: the chain tip never got within 8 cm"),
                "push_t": ("SUCCESS: coverage >= 0.9", "FAILED: not on the goal in 30 s")}[task]
        for _ in range(int(hold_s * fps)):
            frame(env, env.k, banner=(ok, what[0] if ok else what[1]))
    finally:
        renderer.close()
        writer.close()
    return tr.outcome


def pick_seeds(task, per_seed):
    base = per_seed[(task, "no model error")]
    bad = per_seed[(task, VIDEO_PERT[task])]
    seeds = SEEDS[task]
    s_ok = int(seeds[int(np.argmax(base))])
    cand = [i for i in range(len(base)) if base[i] > 0.5 and bad[i] < 0.5]
    if cand:
        return s_ok, int(seeds[cand[0]]), VIDEO_PERT[task]
    fails = [i for i in range(len(base)) if base[i] < 0.5]
    return s_ok, int(seeds[fails[0] if fails else int(np.argmin(bad))]), "no model error"


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("CLIPS_ONLY"):          # re-record the four clips only (seeds = the ones the full run picked: 5000 for success and failure)
        for task in ("whip", "push_t"):
            why = VIDEO_PERT[task]
            print(task, record(task, [], 5000, f"{OUT}/{task}_success.mp4", "classical teacher, no model error"), flush=True)
            print(task, record(task, dict(CONDITIONS[task])[why], 5000, f"{OUT}/{task}_failure.mp4", f"classical teacher, {why}"), flush=True)
        return
    rows, per_seed = run_table()
    md = table_md(rows, per_seed)
    open(f"{OUT}/table.md", "w").write(md + "\n")
    print(md)
    for task in ("whip", "push_t"):
        s_ok, s_bad, why = pick_seeds(task, per_seed)
        out = record(task, [], s_ok, f"{OUT}/{task}_success.mp4", "classical teacher, no model error")
        print(task, "success clip", s_ok, out, flush=True)
        pert = dict(CONDITIONS[task])[why]
        out = record(task, pert, s_bad, f"{OUT}/{task}_failure.mp4", f"classical teacher, {why}")
        print(task, "failure clip", s_bad, out, flush=True)


if __name__ == "__main__":
    main()
