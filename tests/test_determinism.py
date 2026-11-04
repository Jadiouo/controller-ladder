import numpy as np
import pytest

from arena import Scenario
from arena.eval import evaluate, run_episode
from arena.perturb import Delay, Friction, SensorNoise, TorquePulses

SC = Scenario("tracking").with_perturb(SensorNoise(q=0.002, qd=0.02), Delay(2), Friction(1.0, 0.1), TorquePulses())
SC_POLE = Scenario("pole_balance").with_perturb(SensorNoise(pole_theta=0.01, pole_omega=0.05))


@pytest.mark.parametrize("sc", [SC, SC_POLE])
@pytest.mark.parametrize("ctrl", ["pid", "ctc_lqr"])
def test_same_seed_identical_trace(sc, ctrl):
    a, b = run_episode(sc, ctrl, 11), run_episode(sc, ctrl, 11)
    assert np.array_equal(a.err, b.err) and np.array_equal(a.tau_applied, b.tau_applied)


def test_different_seed_differs():
    a, b = run_episode(SC, "ctc_lqr", 11), run_episode(SC, "ctc_lqr", 12)
    assert not np.array_equal(a.err, b.err)


def test_reused_env_equals_fresh_env():
    """evaluate() reuses one env across seeds; stale state (delay buffer, noise rng, data) must not leak."""
    seeds = [5005, 5006, 5007]
    reused = evaluate(SC, "ctc_lqr", seeds)
    fresh = [evaluate(SC, "ctc_lqr", [s]) for s in seeds]
    for m in ("rmse", "energy"):
        assert np.array_equal(reused.metrics[m], np.concatenate([f.metrics[m] for f in fresh]))


def test_order_of_seeds_does_not_matter():
    a, b = evaluate(SC, "pid", [5003, 5004]), evaluate(SC, "pid", [5004, 5003])
    assert np.array_equal(a.metrics["rmse"], b.metrics["rmse"][::-1])
