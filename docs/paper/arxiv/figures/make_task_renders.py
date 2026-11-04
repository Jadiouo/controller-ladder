"""Render the seven benchmark tasks (nominal parameters, CAL seed 1000) offscreen with MuJoCo and tile them into fig_tasks.pdf.
Run from the repo root:  MUJOCO_GL=egl .venv/bin/python docs/paper/arxiv/figures/make_task_renders.py
Writes figures/renders/<task>.png (1000 px wide) and figures/fig_tasks.pdf. Read-only w.r.t. arena/ and spike03b/."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spike03b"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

from arena import Scenario
from arena.eval import run_episode
from arena.viz.record import _marker

SEED = 1000                      # calibration seed (not a TEST seed)
W, H = 1000, 750                 # render size (px), 4:3 (whip: 1600x1200)
BG = np.array([0.93, 0.95, 0.98, 1.0])
OUT = os.path.join(HERE, "renders")
os.makedirs(OUT, exist_ok=True)


class _Stop(Exception):
    pass


def _light(model, w=W, h=H):
    model.vis.rgba.fog[:] = BG
    model.vis.rgba.haze[:] = BG
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.8
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)


def _render(r, data, cam, marks=None):
    """RGB render with the (black) empty background replaced by BG, found with a segmentation pass.
    marks(scene): optional callback adding marker spheres (reference dot, path) to the RGB pass only."""
    r.update_scene(data, camera=cam)
    if marks is not None:
        marks(r.scene)
    img = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=cam)
    seg = r.render()[..., 0]
    r.disable_segmentation_rendering()
    img[(seg < 0) & (img.sum(-1) < 24)] = (BG[:3] * 255).astype(np.uint8)
    return img


def _cam(lookat, dist, az=90.0, el=0.0):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def _save(name, img):
    Image.fromarray(img).save(os.path.join(OUT, name + ".png"))


def arena_snapshot(name, task, ctrl, t_snap, cam, task_kw=None, t_before_end=None, size=(H, W)):
    """Run `ctrl` on `task` (seed SEED); render the scene at time t_snap, or t_before_end seconds before the episode ends."""
    sc = Scenario(task, task_kw=task_kw or {})
    if t_before_end is not None:
        tr = run_episode(sc, ctrl, SEED)
        t_snap = max(0.0, tr.t_end - t_before_end)
        print(f"  {name}: outcome {tr.outcome}, t_end {tr.t_end:.2f}s -> snapshot at {t_snap:.2f}s")
    env = sc.make_env()
    _light(env.model, size[1], size[0])
    r = mujoco.Renderer(env.model, *size)
    k_snap = int(round(t_snap / env.dt))
    got = {}

    def cb(e, k):
        if k >= k_snap:
            def marks(scene, e=e, k=k):
                path = getattr(e.task, "EE", None)
                if path is not None:                      # tracking: reference path (blue dots) and current reference point (red)
                    for p in path[::6]:
                        _marker(scene, p, (0.2, 0.45, 0.95, 0.9), 0.008)
                    _marker(scene, e.task.ee_ref(k), (0.9, 0.1, 0.1, 1.0), 0.02)
            got["img"] = _render(r, e.data, cam, marks)
            raise _Stop

    try:
        run_episode(sc, ctrl, SEED, env=env, frame_cb=cb)
    except _Stop:
        pass
    finally:
        r.close()
    assert "img" in got, f"{name}: episode ended before snapshot time"
    _save(name, got["img"])
    print(f"  {name}: saved (t={k_snap * env.dt:.2f}s)")


def juggling():
    import env as E
    cond = E.Cond()
    je = E.JugEnv(1, cond)
    m = je.model
    _light(m)
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        m.geom_rgba[g] = {"racket": [0.12, 0.35, 0.75, 1], "ballg": [0.95, 0.35, 0.05, 1]}.get(n, [0.55, 0.55, 0.6, 1])
    ics = E.init_cond([SEED])
    je.reset(ics, [SEED])
    r = mujoco.Renderer(m, H, W)
    img = None
    try:
        while je.active.any() and je.k[0] < 800:
            je.measure_update()
            tau = je.teacher()
            je.step(tau)
            d = je.d[0]
            zb = d.qpos[4] - E.OFF - E.WRIST_Z0
            if je.hits[0] >= 3 and d.qvel[4] < 0.3 and d.qvel[4] > -0.3 and zb > 0.35:     # near the apex, after 3 hits
                img = _render(r, d, _cam((0.4, 0.0, 0.2), 1.5))
                print(f"  juggling: hits {je.hits[0]}, t={je.k[0] * E.DT:.2f}s, ball height {zb:.2f}")
                break
    finally:
        r.close()
    assert img is not None, "juggling: no apex snapshot found"
    _save("juggling", img)


def render_all():
    print("rendering")
    arena_snapshot("tracking", "tracking", "ctc_lqr", 1.6, _cam((0.35, 0, 0.0), 1.5))
    arena_snapshot("pole_balance", "pole_balance", "ctc_lqr", 1.5, _cam((0.3, 0, 0.15), 1.3))
    arena_snapshot("catch", "catch", "catch_teacher", 0.55, _cam((0.8, 0, 0.12), 2.6))
    arena_snapshot("toss", "toss", "toss_teacher", 1.1, _cam((0.85, 0, -0.3), 2.2))
    arena_snapshot("whip", "whip", "whip_teacher", None, _cam((0.85, 0, -0.3), 2.3), t_before_end=0.12, size=(1295, 1500))
    arena_snapshot("push_t", "push_t", "push_t_teacher", 6.0, _cam((0.58, 0, 0.0), 1.3))
    juggling()


PANELS = [  # (file, two-line caption)
    ("tracking", "Tracking\nfollow a moving path"),
    ("pole_balance", "Pole balance\nkeep a pole upright"),
    ("catch", "Catch\nintercept a thrown ball"),
    ("toss", "Toss\nthrow into a basket"),
    ("whip", "Whip\nflick the chain tip onto the target"),
    ("push_t", "Push-T\npush the T onto its goal"),
    ("juggling", "Juggling\nbounce a ball on a racket"),
]


def compose():
    fw = 7.1
    # 5 equal columns: whip = cols 0-1 (rows 0-1); the other six tasks = cols 2-4, two rows. Images 4:3, caption strip under each.
    cw = fw / 5
    ih = cw * 0.75                      # small-image height (in)
    cap = 0.27                          # caption height (in)
    rowh = ih + cap
    fh = 2 * rowh + 0.03
    fig = plt.figure(figsize=(fw, fh))
    order = ["whip", "tracking", "pole_balance", "catch", "toss", "push_t", "juggling"]
    titles = dict(PANELS)
    pos = {"whip": (0, 0, 2), "tracking": (2, 0, 1), "pole_balance": (3, 0, 1), "catch": (4, 0, 1),
           "toss": (2, 1, 1), "push_t": (3, 1, 1), "juggling": (4, 1, 1)}
    top = fh - 0.02
    for i, key in enumerate(order):
        c, rrow, span = pos[key]
        img = np.asarray(Image.open(os.path.join(OUT, key + ".png")))
        w_in = cw * span - 0.06
        h_in = w_in * img.shape[0] / img.shape[1]
        if key == "whip":
            h_in = 2 * rowh - cap        # fill both rows (render aspect chosen to fit)
            w_in = 2 * cw - 0.06
        x0 = c * cw + 0.03
        y_top = top - rrow * rowh
        ax = fig.add_axes([x0 / fw, (y_top - h_in) / fh, w_in / fw, h_in / fh])
        ax.imshow(img, aspect="auto"); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec="0.55", lw=0.5))
        fig.text((x0 + w_in / 2) / fw, (y_top - h_in - 0.02) / fh, f"({'abcdefg'[i]}) " + titles[key], fontsize=6.5, ha="center", va="top", linespacing=1.15)
    fig.savefig(os.path.join(HERE, "fig_tasks.pdf"), dpi=300)
    print("wrote fig_tasks.pdf")


if __name__ == "__main__":
    if "--compose-only" not in sys.argv:
        render_all()
    compose()
