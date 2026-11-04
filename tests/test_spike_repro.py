"""Reproduction check: arena's CTC+LQR / perturbations must match the spike code (read-only imports) numerically.

spike01 : sample_refs / sample_init_err / sample_dist / Teacher(e) / rollout   (spike/sim.py)
spike02 : WeakEnv weaknesses (cap, delay, Coulomb+viscous friction)            (spike02/sim2.py)
Tolerance 1e-6 (observed differences are ~1e-12: same MuJoCo calls in the same order).
Seeds are calibration seeds (1000-1007): the test set is not touched.
"""
import os
import sys

import numpy as np
import pytest

from arena import Scenario
from arena.controllers import controller_factory
from arena.eval import compute_metrics, run_episode
from arena.perturb import Delay, Friction, PayloadError, TorqueLimit, TorquePulses
from arena.robots import RobotSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("spike", "spike02"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sim = pytest.importorskip("sim")
sim2 = pytest.importorskip("sim2")

SEEDS = np.arange(1000, 1008)
TOL = 1e-6


def arena_metrics(e, perts, seeds=SEEDS):
    sc = Scenario("tracking", {"ref": "joint_sines"}, robot=RobotSpec.two_link(payload_mass=1.5 * (1 + e)),
                  perturbations=(PayloadError(mass=1.5),) + tuple(perts))
    out = [compute_metrics(run_episode(sc, "ctc_lqr", int(s)), None) for s in seeds]
    got = {k: np.array([o[k] for o in out]) for k in ("rmse", "ss_rmse", "energy", "sat")}
    got["energy"] = got["energy"] * sc.make_env().task.duration     # arena energy is a per-step mean; spike's is the integral
    return got


@pytest.mark.parametrize("e", [0.0, 0.3, -0.3])
@pytest.mark.parametrize("pulses", [False, True])
def test_spike01_teacher(e, pulses):
    P, ie = sim.sample_refs(SEEDS), sim.sample_init_err(SEEDS)
    ds = sim.sample_dist(SEEDS) if pulses else None
    ref = sim.rollout(sim.Teacher(e).act, P, ie, dist=ds)
    got = arena_metrics(e, [TorquePulses()] if pulses else [])
    assert np.abs(got["rmse"] - ref["rmse"]).max() < TOL
    assert np.abs(got["energy"] - ref["energy"]).max() < TOL * ref["energy"].max()


def test_spike01_reference_and_init_error_identical():
    sc = Scenario("tracking", {"ref": "joint_sines"})
    env = sc.make_env()
    P, ie = sim.sample_refs(SEEDS), sim.sample_init_err(SEEDS)
    for i, s in enumerate(SEEDS[:3]):
        env.reset(int(s))
        q, qd, qdd = sim.ref_at(P, 123 * sim.DT_CTRL)
        r = env.task.reference(123)
        assert np.abs(r["q"] - q[i]).max() < 1e-12 and np.abs(r["qdd"] - qdd[i]).max() < 1e-12
        q0, qd0, _ = sim.ref_at(P, 0.0)
        assert np.abs(env.data.qpos[:2] - (q0[i] + ie[i, :2])).max() < 1e-12


@pytest.mark.parametrize("w", ["W1", "W2", "W3", "W4"])
def test_spike02_weaknesses(w):
    wk = sim2.W[w]
    P, ie, ds = sim.sample_refs(SEEDS), sim.sample_init_err(SEEDS), sim.sample_dist(SEEDS)
    T = sim.Teacher(0.0)
    ref = sim2.rollout2(T.act, wk, P, ie, dist=ds)
    perts = []
    if wk["delay"]:
        perts.append(Delay(wk["delay"]))
    if wk["cap"] < 80:
        perts.append(TorqueLimit(wk["cap"]))
    if wk["fc"] or wk["fv"]:
        perts.append(Friction(wk["fc"], wk["fv"]))
    got = arena_metrics(0.0, perts + [TorquePulses()])
    for k in ("rmse", "ss_rmse", "sat"):
        assert np.abs(got[k] - ref[k]).max() < TOL, k
    assert np.abs(got["energy"] - ref["energy"]).max() < TOL * ref["energy"].max()


def test_reported_rmse_matches_spike01_table():
    """spike01 table.md: teacher e=+0% no disturbance, 50 test trajectories: 0.0097 (4 d.p.). Same code path, test seeds."""
    got = arena_metrics(0.0, [], seeds=np.arange(50))["rmse"].mean()
    assert round(float(got), 4) == 0.0097
