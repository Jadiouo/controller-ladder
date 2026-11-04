"""Protocol v2 metrics computed from one Trace (pure numpy; tested on hand-made traces).

Generic metrics (a task lists the ones it wants in `Task.metrics`; error-based ones need Task.error):
rmse          sqrt(mean_t mean_i e_i^2) over the whole horizon, transient included (primary for tracking)
ss_rmse       same, steps with t >= 0.5 s
settle_time   first time after which every |e_i| < THR_i for the rest of the episode (horizon if never)
recovery_time mean over disturbance pulses of the time from pulse end until |e| < THR (capped at 1 s window /
              next pulse); NaN if no pulse was observed
sat           FRACTION OF EXECUTED STEPS with some joint at the effective torque limit
energy        MEAN OVER EXECUTED STEPS of sum_j tau_j^2 (Nm^2; per-step so an early failure is not rewarded; the old
              integral over the horizon is energy * duration)
alive_frac    executed time / horizon if the episode failed, else 1 (use it to see whether two controllers are
              comparable on energy / sat)
+ task extras (ee_rmse, survival_time, fell, tip_rmse, success, ...) from Task.extra_metrics.

One rule for episodes that end early (Trace.series): the terminal state is included; the remaining steps are charged the
FIXED penalty protocol.FAILURE_ERR per error component after a failure (task failure, simulator or controller divergence)
and hold the terminal error after a success. Non-finite values are replaced by the same penalty.
"""
from __future__ import annotations

import numpy as np

from .protocol import FAILURE_ERR, RECOVERY_WINDOW_S, SETTLE_THR, SS_START_S

ERROR_METRICS = ("rmse", "ss_rmse", "settle_time", "recovery_time")
ALL_METRICS = ERROR_METRICS + ("sat", "energy", "alive_frac")


def settle_time(emax: np.ndarray, dt: float, thr: float = SETTLE_THR) -> float:
    bad = np.where(emax >= thr)[0]
    if len(bad) == 0:
        return 0.0
    return float(min(bad[-1] + 1, len(emax)) * dt)


def recovery_time(emax: np.ndarray, pulse: np.ndarray, dt: float, thr: float = SETTLE_THR,
                  window_s: float = RECOVERY_WINDOW_S) -> float:
    N = len(emax)
    edges = np.diff(np.concatenate([[0], pulse.astype(int), [0]]))
    st, en = np.where(edges == 1)[0], np.where(edges == -1)[0]
    vals = []
    for j, (s, e) in enumerate(zip(st, en)):
        wend = min(st[j + 1] if j + 1 < len(st) else N, e + int(window_s / dt))
        seg = emax[s:wend]
        if (seg < thr).all():
            vals.append(0.0)
            continue
        last = s + np.where(seg >= thr)[0][-1] + 1
        vals.append(0.0 if last <= e else min((last - e) * dt, (wend - e) * dt))
    return float(np.mean(vals)) if vals else float("nan")


def compute_metrics(trace, task=None) -> dict:
    names = tuple(task.metrics) if task is not None else ALL_METRICS
    unknown = set(names) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"unknown generic metrics {sorted(unknown)}; known: {ALL_METRICS}")
    dt, N = trace.dt, trace.n_steps
    out = {}
    if any(n in ERROR_METRICS for n in names):
        if trace.err.ndim != 2 or trace.err.shape[1] == 0:
            raise ValueError("error-based metrics requested but the task provides no error(); list only event metrics "
                             "in Task.metrics")
        err = trace.series("err", FAILURE_ERR)
        pulse = np.concatenate([trace.pulse, np.zeros(N - len(trace.pulse), bool)])[:N]
        thr = None if task is None else task.settle_thr
        if thr is None:
            emax, thr_v = np.abs(err).max(1), SETTLE_THR
        else:                                        # per-component thresholds in the units of each component
            emax, thr_v = (np.abs(err) / np.broadcast_to(np.asarray(thr, float), (err.shape[1],))).max(1), 1.0
        e2 = (err ** 2).mean(1)
        ss = int(round(SS_START_S / dt))
        ss = ss if ss < len(e2) else 0                 # horizon shorter than the transient window: use everything
        vals = dict(rmse=lambda: float(np.sqrt(e2.mean())), ss_rmse=lambda: float(np.sqrt(e2[ss:].mean())),
                    settle_time=lambda: settle_time(emax, dt, thr_v),
                    recovery_time=lambda: recovery_time(emax, pulse, dt, thr_v))
        out.update({n: vals[n]() for n in names if n in vals})
    T = max(len(trace.tau_applied), 1)
    if "sat" in names:
        out["sat"] = float(trace.sat.sum() / T)
    if "energy" in names:
        out["energy"] = float((trace.tau_applied ** 2).sum() / T)
    if "alive_frac" in names:
        out["alive_frac"] = float(min(1.0, trace.t_end / (N * dt))) if trace.outcome == "failure" else 1.0
    out = {n: out[n] for n in names if n in out}
    if task is not None:
        out.update(task.extra_metrics(trace))
    return out
