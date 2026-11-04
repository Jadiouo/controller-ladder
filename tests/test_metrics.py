"""Protocol v2 metrics on hand-made traces with known answers."""
import dataclasses

import numpy as np
import pytest

from arena.eval import Trace, compute_metrics, mean_ci, paired_diff, recovery_time, settle_time
from arena.eval.protocol import FAILURE_ERR
from arena.eval.stats import NonFiniteError

DT, N = 0.005, 800


def trace(err, tau=None, sat=None, pulse=None, **kw):
    err = np.asarray(err, float)
    T = len(err)
    return Trace(dt=DT, n_steps=N, t_end=T * DT, terminated=T < N, err=err,
                 tau_cmd=np.zeros((T, 2)), tau_applied=np.zeros((T, 2)) if tau is None else tau,
                 sat=np.zeros(T, bool) if sat is None else sat, pulse=np.zeros(T, bool) if pulse is None else pulse, **kw)


def test_rmse_constant_error():
    m = compute_metrics(trace(np.full((N, 2), 0.1)))
    assert m["rmse"] == pytest.approx(0.1) and m["ss_rmse"] == pytest.approx(0.1)


def test_rmse_averages_over_joints_and_time_like_spike():
    err = np.zeros((N, 2))
    err[:, 0] = 0.2                                  # one joint at 0.2 -> mean over joints of e^2 = 0.02
    assert compute_metrics(trace(err))["rmse"] == pytest.approx(np.sqrt(0.02))


def test_ss_rmse_drops_transient():
    err = np.zeros((N, 2))
    err[:100] = 1.0                                  # first 0.5 s
    m = compute_metrics(trace(err))
    assert m["ss_rmse"] == 0.0 and m["rmse"] == pytest.approx(np.sqrt(100 / N))


def test_settle_time():
    emax = np.zeros(N)
    assert settle_time(emax, DT) == 0.0
    emax[:200] = 0.05
    assert settle_time(emax, DT) == pytest.approx(200 * DT)
    emax[-1] = 0.05
    assert settle_time(emax, DT) == pytest.approx(N * DT)           # never settles -> horizon


def test_recovery_time():
    emax = np.zeros(N)
    pulse = np.zeros(N, bool)
    pulse[100:120] = True
    emax[100:150] = 0.05                                             # error stays up 30 steps after pulse end
    assert recovery_time(emax, pulse, DT) == pytest.approx(30 * DT)
    assert np.isnan(recovery_time(emax, np.zeros(N, bool), DT))      # no pulse -> undefined
    emax[:] = 0.0
    assert recovery_time(emax, pulse, DT) == 0.0                     # never left the band
    emax[100:] = 0.05                                                # never recovers: capped at the 1 s window
    assert recovery_time(emax, pulse, DT) == pytest.approx(1.0)


def test_energy_and_saturation():
    tau = np.zeros((N, 2))
    tau[:, 0] = 3.0
    sat = np.zeros(N, bool)
    sat[:200] = True
    m = compute_metrics(trace(np.zeros((N, 2)), tau=tau, sat=sat))
    assert m["energy"] == pytest.approx(9.0) and m["sat"] == pytest.approx(0.25)          # per-step mean, fraction


def test_failed_episode_is_padded_with_the_fixed_penalty_not_the_last_error():
    err = np.full((400, 2), 0.3)
    m = compute_metrics(trace(err))                                  # 400 steps at 0.3, then 400 at FAILURE_ERR
    assert m["rmse"] == pytest.approx(np.sqrt((0.09 + FAILURE_ERR ** 2) / 2))
    d = compute_metrics(dataclasses.replace(trace(err), diverged=True))
    assert d["rmse"] == pytest.approx(m["rmse"])                     # divergence and task failure are charged the same
    assert np.isfinite(compute_metrics(trace(np.full((N, 2), np.nan)))["rmse"])
    ok = compute_metrics(dataclasses.replace(trace(err), outcome="success"))
    assert ok["rmse"] == pytest.approx(0.3)                          # success: the terminal error is held


def test_mean_ci_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m, h = mean_ci(x)
    assert m == 3.0 and h == pytest.approx(2.7764451 * np.sqrt(2.5) / np.sqrt(5), rel=1e-6)
    assert np.isnan(mean_ci([1.0])[1])
    with pytest.raises(NonFiniteError):
        mean_ci([np.nan])


def test_paired_diff_win_rule():
    base = np.linspace(1.0, 2.0, 20)
    p = paired_diff(base * 0.8, base)                                # 20% better everywhere
    assert p["win"] and p["rel"] == pytest.approx(0.2) and p["hi"] < 0 and p["win_frac"] == 1.0
    p = paired_diff(base * 0.95, base)                               # better but < 10%
    assert not p["win"] and p["rel"] == pytest.approx(0.05)
    noisy = base + np.tile([-1.0, 0.8], 10)                          # mean diff -0.1, CI covers 0
    assert not paired_diff(noisy, base)["win"]
    # higher-is-better metric flips the direction
    assert paired_diff(base * 1.3, base, higher_is_better=True)["win"]
    assert not paired_diff(base * 1.3, base)["win"]


def test_pole_extra_metrics():
    from arena.tasks import make_task
    t = make_task("pole_balance")
    tr = trace(np.full((300, 1), 0.1))
    m = compute_metrics(tr, t)
    assert m["survival_time"] == pytest.approx(300 * DT) and m["fell"] == 1.0
