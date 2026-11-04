"""Tracking reference sanity over every seed split: ref.q starts in (-pi, pi], stays in a sane range, is continuous (no 2 pi
jumps) and its forward kinematics reproduces the commanded figure (the IK must not leap to another branch)."""
import numpy as np
import pytest

from arena.eval.protocol import CAL_SEEDS, DEMO_SEEDS, TEST_SEEDS, TRAIN_SEEDS
from arena.robots import RobotSpec
from arena.tasks import make_task

SPLITS = {"train": TRAIN_SEEDS, "cal": CAL_SEEDS, "test": TEST_SEEDS, "demo": DEMO_SEEDS}


def _refs(kind, seeds, robot):
    task = make_task("tracking", robot=robot, ref=kind)
    for s in seeds:
        task._build_reference(int(s))
        tip = task._tip_path(int(s), np.arange(task.n_steps) * robot.dt)
        yield int(s), task.Q.copy(), float(np.linalg.norm(task.EE - tip, axis=1).max())


@pytest.mark.parametrize("kind", ["lissajous", "glyph"])
@pytest.mark.parametrize("split", list(SPLITS))
def test_reference_q_in_range_and_continuous(kind, split):
    robot = RobotSpec.two_link()
    bad = []
    for s, Q, ee_err in _refs(kind, SPLITS[split], robot):
        step = np.abs(np.diff(Q, axis=0)).max()
        if np.abs(Q).max() > np.pi + 1.0 or step > 0.1 or np.abs(Q[0]).max() > np.pi or ee_err > 1e-6:
            bad.append((s, float(np.abs(Q).max()), float(step), float(ee_err)))
    assert not bad, f"{kind}/{split}: (seed, max|q|, max step, max tip error) {bad[:5]} ({len(bad)} bad)"
