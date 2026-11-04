"""Demo: the classical ladder PID -> CTC+LQR -> iLQR -> iLQR-MPC -> MPPI on tracking and pole_balance (clean, nominal plant).
Writes results/classical_ladder/table.md: protocol-v2 metrics, per-step compute time (mean / p95, wall and CPU) and the
offline planning time of iLQR.

Run (heavy-job wrapper required on this machine):
  flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
      .venv/bin/python scripts/demo_classical_ladder.py [--n 5]
Seeds are the first --n DEMO_SEEDS (5000-5019, default 5): disjoint from calibration (1000-1049) and locked test (0-49) sets.
The model-based controllers only use the NOMINAL model (ctx.make_model()); in this demo the plant equals the nominal model.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from arena import Scenario
from arena.controllers import controller_factory
from arena.controllers._ocp import TimedController
from arena.eval import DEMO_SEEDS, evaluate, metrics_table, paired_table

OUT = "results/classical_ladder"
CTRLS = {"pid": "pid", "ctc_lqr": "ctc_lqr", "ilqr": "ilqr", "ilqr_mpc": "ilqr_mpc", "mppi": "mppi"}
SCENARIOS = [Scenario("tracking", {"ref": "glyph"}, name="tracking/clean"), Scenario("pole_balance", name="pole_balance/clean")]


class Probe:
    """Collects (wall, cpu) seconds per act() call of a controller that does not time itself (PID, CTC+LQR)."""

    def __init__(self, ctrl):
        self.ctrl, self.wall, self.cpu = ctrl, [], []

    def reset(self, seed=None):
        self.ctrl.reset(seed)

    def act(self, obs):
        t0, c0 = time.perf_counter(), time.thread_time()
        tau = self.ctrl.act(obs)
        self.wall.append(time.perf_counter() - t0)
        self.cpu.append(time.thread_time() - c0)
        return tau


def timing_of(instances):
    if isinstance(instances[0], Probe):
        w = np.concatenate([p.wall for p in instances]) * 1e3
        c = np.concatenate([p.cpu for p in instances]) * 1e3
        off = 0.0
    else:
        w = np.concatenate([i.compute_times for i in instances]) * 1e3
        c = np.concatenate([i.cpu_times for i in instances]) * 1e3
        off = float(np.mean([i.offline_time for i in instances]))
    return dict(n=len(w), wall_mean=w.mean(), wall_p95=np.percentile(w, 95), cpu_mean=c.mean(), cpu_p95=np.percentile(c, 95), offline=off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of DEMO seeds")
    args = ap.parse_args()
    seeds = DEMO_SEEDS[:args.n]
    os.makedirs(OUT, exist_ok=True)
    res, timings = {}, {}
    t0 = time.time()
    for sc in SCENARIOS:
        for name, c in CTRLS.items():
            inst = []

            def fac(ctx, c=c, inst=inst):
                ctrl = controller_factory(c)(ctx) if isinstance(c, str) else c(ctx)
                ctrl = ctrl if isinstance(ctrl, TimedController) else Probe(ctrl)
                inst.append(ctrl)
                return ctrl

            res[(sc.label, name)] = evaluate(sc, fac, seeds, label=name)
            timings[(sc.label, name)] = timing_of(inst)
            print(sc.label, name, f"{time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0

    md = ["# classical ladder: PID, CTC+LQR, iLQR, iLQR-MPC, MPPI (clean nominal plant)", "",
          f"Seeds {int(seeds[0])}-{int(seeds[-1])} (n={len(seeds)}, DEMO set; not calibration/test). Mean +- 95% t-CI over seeds. "
          "Tracking error = joint rad (glyph reference); pole error = theta (rad), tip distance is the separate tip_rmse (m). "
          "energy = mean sum(tau^2) per executed step, sat = fraction of executed steps. n/a = undefined. "
          "iLQR / iLQR-MPC / MPPI see only the NOMINAL model (ctx.make_model()) and the reference through obs.ref_preview.", "",
          "## Metrics", "", metrics_table(res, metrics=["rmse", "ss_rmse", "sat", "energy"]), "",
          "## Compute time per control step", "",
          "Measured around `act()` over all steps of all seeds. wall = elapsed time, cpu = thread CPU time (less sensitive to other jobs: "
          "the machine is shared and was heavily loaded, so wall numbers are inflated and noisy; cpu is the better estimate of the cost). "
          "The control period is 5 ms. `ilqr` plans the whole episode offline at the first call; that time is NOT in the per-step "
          "numbers and is listed in the last column (mean per episode, wall seconds). PID / CTC+LQR timings include only python overhead.", "",
          "| scenario | controller | steps | wall mean ms | wall p95 ms | cpu mean ms | cpu p95 ms | offline plan s/episode |",
          "|---|---|---|---|---|---|---|---|"]
    for (sc, c), t in timings.items():
        md.append(f"| {sc} | {c} | {t['n']} | {t['wall_mean']:.2f} | {t['wall_p95']:.2f} | {t['cpu_mean']:.2f} | {t['cpu_p95']:.2f} | "
                  f"{t['offline']:.2f} |")
    md += ["", "## Paired difference vs CTC+LQR (same seeds)", "",
           "diff = candidate - ctc_lqr; impr. > 0 means the candidate is better; rmse rows compare only seeds both survived.", "",
           paired_table(res, baseline="ctc_lqr", metrics=["survival_time", "rmse", "ss_rmse", "tip_rmse", "energy"],
                        common_survivors=True), "",
           f"Wall-clock of the whole script: {wall:.0f} s.", ""]
    open(f"{OUT}/table.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
