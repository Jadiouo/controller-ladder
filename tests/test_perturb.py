"""Each perturbation must change the thing it claims to change, and only that."""
import mujoco
import numpy as np
import pytest

from arena import Scenario
from arena.eval import evaluate, run_episode
from arena.perturb import Delay, Friction, PayloadError, SensorNoise, TorqueLimit, TorquePulses

SEEDS = [1000, 1001, 1002]


def test_payload_error_changes_true_plant_not_nominal():
    env = Scenario("tracking", perturbations=(PayloadError(rel=0.3),)).make_env()
    assert env.robot.payload_mass == pytest.approx(1.5 * 1.3) and env.nominal.payload_mass == 1.5
    base = Scenario("tracking").make_env()
    assert env.model.body_mass.sum() == pytest.approx(base.model.body_mass.sum() + 0.45)
    assert env.ctx.robot.payload_mass == 1.5                      # controller still sees the nominal robot
    assert PayloadError(mass=2.0).modify_robot(env.nominal).payload_mass == 2.0
    with pytest.raises(ValueError):
        PayloadError()


def test_payload_error_hurts_model_based_controller():
    clean = evaluate(Scenario("tracking"), "ctc_lqr", SEEDS).metrics["ss_rmse"].mean()
    bad = evaluate(Scenario("tracking").with_perturb(PayloadError(rel=0.5)), "ctc_lqr", SEEDS).metrics["ss_rmse"].mean()
    assert bad > 3 * clean


def test_torque_limit_clips_and_flags_saturation():
    sc = Scenario("tracking").with_perturb(TorqueLimit(10.0))
    tr = run_episode(sc, "ctc_lqr", 1000)
    assert np.abs(tr.tau_applied).max() <= 10.0 + 1e-12
    assert np.abs(tr.tau_cmd).max() > 10.0 and tr.sat.mean() > 0.3
    assert run_episode(Scenario("tracking"), "ctc_lqr", 1000).sat.sum() == 0


def test_delay_shifts_command_by_n_steps():
    d = 4
    tr = run_episode(Scenario("tracking").with_perturb(Delay(d)), "ctc_lqr", 1000)
    assert np.array_equal(tr.tau_applied[d:], tr.tau_cmd[:-d])
    assert np.array_equal(tr.tau_applied[0], tr.tau_cmd[0])      # warm-up holds the first command
    assert Delay.from_ms(30, dt=0.005).steps == 6


def test_friction_force_and_effect():
    f = Friction(coulomb=1.5, viscous=0.3)
    qd = np.array([1.0, -2.0])
    assert np.allclose(f.qfrc(qd, 0), -(1.5 * np.tanh(qd / 0.05) + 0.3 * qd))
    assert np.allclose(f.qfrc(np.zeros(2), 0), 0)
    base = evaluate(Scenario("tracking"), "pid", SEEDS).metrics["ss_rmse"].mean()
    fr = evaluate(Scenario("tracking").with_perturb(f), "pid", SEEDS).metrics["ss_rmse"].mean()
    assert fr > 1.5 * base


def test_sensor_noise_statistics_and_references_untouched():
    sc = Scenario("tracking", perturbations=(SensorNoise(q=0.01, qd=0.1),))
    env = sc.make_env()
    obs0 = env.reset(5)
    qs, noise_q, noise_qd = [], [], []
    obs = obs0
    for _ in range(400):
        noise_q.append(obs.q - env.data.qpos[:2])
        noise_qd.append(obs.qd - env.data.qvel[:2])
        obs, *_ = env.step(np.zeros(2))
    nq, nqd = np.array(noise_q), np.array(noise_qd)
    assert nq.std() == pytest.approx(0.01, rel=0.15) and nqd.std() == pytest.approx(0.1, rel=0.15)
    assert abs(nq.mean()) < 0.002
    ref_clean = Scenario("tracking").make_env()
    ref_clean.reset(5)
    assert np.array_equal(obs0.ref["q"], ref_clean.task.reference(0)["q"])


def test_pole_sensor_noise_applies_to_pole_channels():
    env = Scenario("pole_balance", perturbations=(SensorNoise(pole_theta=0.05),)).make_env()
    obs = env.reset(2)
    true = env.task.measure(env)
    assert float(obs.extra["pole_theta"]) != float(true["pole_theta"])
    assert float(obs.extra["pole_omega"]) == float(true["pole_omega"])


def test_pulses_active_mask_and_effect():
    sc = Scenario("tracking").with_perturb(TorquePulses(n=3, width=0.1, mag=(4, 8)))
    tr = run_episode(sc, "ctc_lqr", 1000)
    on = tr.pulse
    assert on.any() and 3 * 20 >= on.sum() >= 20          # 3 pulses of 20 steps (overlaps merge)
    clean = run_episode(Scenario("tracking"), "ctc_lqr", 1000)
    assert np.abs(tr.err[100:]).max() > 10 * np.abs(clean.err[100:]).max()     # after the initial transient
    env = sc.make_env()
    env.reset(1000)
    assert np.abs(env.perts[0].table).max() <= 8.0 and np.abs(env.perts[0].table).max() >= 4.0


def test_perturbations_compose_in_order():
    both = Scenario("tracking").with_perturb(Delay(6), TorqueLimit(32.0))
    assert both.make_env().tau_limit == 32.0
    tr = run_episode(both, "ctc_lqr", 1000)
    expect = np.clip(tr.tau_cmd[:-6], -32.0, 32.0)                  # delay first, then the cap
    assert np.array_equal(tr.tau_applied[6:], expect)
    assert np.abs(tr.tau_applied).max() <= 32.0 + 1e-12
