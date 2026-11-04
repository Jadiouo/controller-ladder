"""Record a rollout to mp4 (ffmpeg, H.264) or gif (Pillow fallback / .gif path) with MuJoCo offscreen rendering.

Backend: MUJOCO_GL=egl (headless GPU/Mesa EGL) is selected by default; verified on this machine. osmesa is NOT
installed here; glfw works only with a display. Always call Renderer.close() (done here) or EGL prints an error at exit.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..controllers.base import as_factory
from ..eval.metrics import compute_metrics
from ..eval.rollout import run_episode
from ..scenario import Scenario


def _marker(scene, pos, rgba, radius=0.012):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.array([pos[0], -0.08, pos[1]]),
                        np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


class _Writer:
    def __init__(self, path, size, fps):
        self.path, self.size, self.fps, self.frames = path, size, fps, []
        self.gif = path.lower().endswith(".gif") or shutil.which("ffmpeg") is None
        if self.gif and not path.lower().endswith(".gif"):
            self.path = os.path.splitext(path)[0] + ".gif"
        if not self.gif:
            h, w = size
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                 "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", self.path],
                stdin=subprocess.PIPE)

    def add(self, frame):
        if self.gif:
            self.frames.append(Image.fromarray(frame))
        else:
            self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.gif:
            self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:], duration=int(1000 / self.fps), loop=0)
        else:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")


def record_rollout(scenario: Scenario, controller, seed: int, path: str, fps: int = 25, size=(360, 480),
                   max_seconds: float | None = None, label: str | None = None, camera=None) -> dict:
    """Run one episode and write the video. Returns dict(frames, path, metrics). `size` = (height, width), even numbers."""
    h, w = size
    env = scenario.make_env()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = _Writer(path, (h, w), fps)
    env.model.vis.headlight.ambient[:] = 0.5
    env.model.vis.headlight.diffuse[:] = 0.8
    renderer = mujoco.Renderer(env.model, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = camera or (0.35, 0.0, 0.0)
    cam.distance, cam.azimuth, cam.elevation = 2.2, 90.0, 0.0
    every = max(1, int(round(1.0 / (fps * env.dt))))
    last_k = int((max_seconds or 1e9) / env.dt)
    title = label or f"{getattr(controller, 'registry_name', controller if isinstance(controller, str) else '')} | {scenario.label}"
    state = {}

    def cb(e, k):
        if k % every or k > last_k:
            return
        renderer.update_scene(e.data, camera=cam)
        ref = e.task.ee_ref(k)
        if k == 0 or "path" not in state:
            state["path"] = getattr(e.task, "EE", None)
        if state["path"] is not None:
            for p in state["path"][::8]:
                _marker(renderer.scene, p, (0.2, 0.5, 1.0, 0.35), 0.006)
        if ref is not None:
            _marker(renderer.scene, ref, (1.0, 0.2, 0.2, 1.0), 0.015)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        d.text((8, 6), f"{title}   t={k * e.dt:4.2f}s", fill=(255, 255, 255))
        writer.add(np.asarray(img))
        state["frames"] = state.get("frames", 0) + 1

    try:
        tr = run_episode(scenario, as_factory(controller), seed, env=env, frame_cb=cb)
    finally:
        renderer.close()
        if state.get("frames"):
            writer.close()
    return dict(frames=state.get("frames", 0), path=writer.path, metrics=compute_metrics(tr, env.task))
