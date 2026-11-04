"""iLQR / iLQR-MPC / MPPI: interface, reproducibility, accuracy vs CTC+LQR, pole balance, nominal-model-only."""
import dataclasses

import numpy as np
import pytest

from arena import Scenario
from arena.controllers import CONTROLLERS, controller_factory
from arena.eval import DEMO_SEEDS, evaluate, run_episode
from arena.perturb import PayloadError

NAMES = ("ilqr", "ilqr_mpc", "mppi")
SMALL = {"ilqr": {}, "ilqr_mpc": {}, "mppi": dict(samples=32, horizon=15)}      # light settings for the contract tests


def _tracking(duration=1.0):
    return Scenario("tracking", {"ref": "glyph", "duration": duration})


@pytest.mark.parametrize("task,kw", [("tracking", {"ref": "glyph", "duration": 0.3}), ("pole_balance", {"duration": 0.3})])
@pytest.mark.parametrize("name", NAMES)
def test_interface_contract(name, task, kw):
    assert name in CONTROLLERS
    env = Scenario(task, kw).make_env()
    ctrl = controller_factory(name, **SMALL[name])(env.ctx)
    obs = env.reset(DEMO_SEEDS[0])
    ctrl.reset(DEMO_SEEDS[0])
    for _ in range(env.n_steps):
        tau = ctrl.act(obs)
        assert tau.shape == (env.n,) and np.isfinite(tau).all() and np.abs(tau).max() <= env.ctx.tau_limit + 1e-9
        obs, _, term, trunc, _ = env.step(tau)
        if term or trunc:
            break
    assert len(ctrl.compute_times) == len(ctrl.cpu_times) == len(ctrl.info_log) == env.n_steps
    tm = ctrl.timing()
    assert tm["n"] == env.n_steps and 0 < tm["mean_ms"] <= tm["max_ms"] and tm["p95_ms"] >= 0
    assert "step_time" in ctrl.last_info


def test_unsupported_task_is_declared_in_init():
    ctx = dataclasses.replace(Scenario("tracking").make_env().ctx, task="juggling")      # a task outside tracking / pole_balance
    for name in NAMES:
        with pytest.raises(NotImplementedError):
            controller_factory(name)(ctx)


def test_mppi_same_seed_reproducible_and_seed_dependent():
    sc = _tracking(0.4)
    kw = dict(samples=32, horizon=15)
    a = run_episode(sc, controller_factory("mppi", **kw), 5000)
    b = run_episode(sc, controller_factory("mppi", **kw), 5000)
    c = run_episode(sc, controller_factory("mppi", **kw), 5001)
    assert np.array_equal(a.tau_cmd, b.tau_cmd)
    assert not np.array_equal(a.tau_cmd[:5], c.tau_cmd[:5])


def test_ilqr_linearisation_matches_central_differences():
    from arena.controllers._ocp import NominalOCP
    ctx = Scenario("pole_balance").make_env().ctx
    o = NominalOCP(ctx)
    rng = np.random.default_rng(0)
    X, U = rng.normal(size=(3, o.nx)) * 0.3, rng.normal(size=(3, o.nu)) * 10
    A, B = o.linearize(X, U)
    e = 1e-5
    for t in range(3):
        for i in range(o.nx):
            d = np.zeros(o.nx); d[i] = e
            fd = (o.step(X[t] + d, U[t]) - o.step(X[t] - d, U[t])) / (2 * e)
            assert np.abs(A[t][:, i] - fd).max() < 1e-4 * (1 + np.abs(fd).max())
        for i in range(o.nu):
            d = np.zeros(o.nu); d[i] = e
            fd = (o.step(X[t], U[t] + d) - o.step(X[t], U[t] - d)) / (2 * e)
            assert np.abs(B[t][:, i] - fd).max() < 1e-4 * (1 + np.abs(fd).max())


def test_internal_step_matches_the_plant_without_perturbations():
    env = Scenario("tracking", {"ref": "glyph", "duration": 0.2}).make_env()
    ctrl = controller_factory("ilqr")(env.ctx)
    obs = env.reset(DEMO_SEEDS[0])
    x = ctrl.ocp.state_from_obs(obs)
    u = np.array([10.0, -5.0])
    nxt = ctrl.ocp.step(x, u)
    obs2, *_ = env.step(u)
    assert np.abs(nxt - ctrl.ocp.state_from_obs(obs2)).max() < 1e-5


def test_tracking_rmse_not_worse_than_twice_ctc_lqr():
    sc = _tracking(1.5)
    S = DEMO_SEEDS[:2]
    ctc = evaluate(sc, "ctc_lqr", S).metrics["rmse"]
    for name in ("ilqr", "ilqr_mpc"):
        r = evaluate(sc, name, S).metrics["rmse"]
        assert (r <= 2 * ctc).all(), (name, r, ctc)


def test_mppi_tracking_rmse_not_worse_than_twice_ctc_lqr():
    sc = _tracking(1.5)
    ctc = evaluate(sc, "ctc_lqr", DEMO_SEEDS[:1]).metrics["rmse"]
    r = evaluate(sc, controller_factory("mppi", samples=64, horizon=20), DEMO_SEEDS[:1]).metrics["rmse"]
    assert (r <= 2 * ctc).all(), (r, ctc)


def test_pole_balance_mpc_survives_full_horizon():
    sc = Scenario("pole_balance")
    r = evaluate(sc, "ilqr_mpc", DEMO_SEEDS[:1])
    assert r.metrics["fell"].sum() == 0 and r.outcomes[0] == "timeout"
    assert r.metrics["survival_time"][0] >= 4.99


def test_pole_balance_ilqr_and_mppi_survive():
    sc = Scenario("pole_balance", {"duration": 2.5})
    assert evaluate(sc, "ilqr", DEMO_SEEDS[:1]).metrics["fell"].sum() == 0
    assert evaluate(sc, controller_factory("mppi", samples=64, horizon=25), DEMO_SEEDS[:1]).metrics["fell"].sum() == 0


@pytest.mark.parametrize("name", NAMES)
def test_controller_only_gets_the_nominal_model(name):
    nominal = Scenario("tracking", {"ref": "glyph", "duration": 0.2}).make_env()
    sc = Scenario("tracking", {"ref": "glyph", "duration": 0.2}).with_perturb(PayloadError(rel=0.3))
    env = sc.make_env()
    assert env.model.body_mass.sum() > nominal.model.body_mass.sum() * 1.05          # the TRUE plant is heavier
    ctrl = controller_factory(name, **SMALL[name])(env.ctx)
    nom_mass = nominal.model.body_mass.copy()
    assert np.allclose(ctrl.model.body_mass, nom_mass) and ctrl.model is ctrl.ocp.model
    assert not np.allclose(ctrl.model.body_mass, env.model.body_mass)
    made = []

    def fac(ctx):
        made.append(controller_factory(name, **SMALL[name])(ctx))
        return made[-1]
    run_episode(sc, fac, DEMO_SEEDS[0], env=env)             # the controller that actually drives the perturbed plant
    assert np.allclose(made[0].model.body_mass, nom_mass) and not np.allclose(made[0].model.body_mass, env.model.body_mass)
