"""Demo: PID vs CTC+LQR on tracking and pole_balance, clean and perturbed. Writes results/arena_demo/.

Run (heavy-job wrapper required on this machine):
  flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
      .venv/bin/python scripts/demo_ladder.py
Seeds are DEMO_SEEDS (5000-5019): disjoint from the calibration (1000-1049) and locked test (0-49) sets.
"""
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena import Scenario
from arena.eval import DEMO_SEEDS, evaluate_grid, metrics_table, paired_table
from arena.perturb import Delay, PayloadError, SensorNoise, TorquePulses
from arena.viz import record_rollout

OUT = "results/arena_demo"
CTRLS = {"pid": "pid", "ctc_lqr": "ctc_lqr"}

tracking = Scenario("tracking", {"ref": "glyph"}, name="tracking/clean")
tracking_hard = tracking.with_perturb(PayloadError(rel=0.3), SensorNoise(q=0.001, qd=0.02), Delay(2), TorquePulses(),
                                      name="tracking/hard (payload+30%, noise, 10ms delay, pulses)")
pole = Scenario("pole_balance", name="pole_balance/clean")
pole_hard = pole.with_perturb(SensorNoise(pole_theta=0.005, pole_omega=0.03, q=0.0005, qd=0.01), Delay(1),
                              name="pole_balance/hard (noise, 5ms delay)")
SCENARIOS = [tracking, tracking_hard, pole, pole_hard]


def side_by_side(a, b, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a, "-i", b, "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2:shortest=0", "-pix_fmt", "yuv420p", out], check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = evaluate_grid(SCENARIOS, CTRLS, DEMO_SEEDS)
    wall = time.time() - t0
    md = ["# arena demo: PID vs CTC+LQR", "",
          f"Seeds {DEMO_SEEDS[0]}-{DEMO_SEEDS[-1]} (n={len(DEMO_SEEDS)}, demo set; not calibration/test). Mean +- 95% t-CI over seeds. "
          "Tracking error = joint rad; pole error = theta (rad) only, tip distance is the separate tip_rmse (m). "
          "energy = mean sum(tau^2) per executed step, sat = fraction of executed steps; early failures are charged a fixed "
          "error penalty (README, design decisions). n/a = undefined (no pulses in the scenario); "
          "n column = valid seeds; 0/1 metrics use a bootstrap CI.", "", "## Metrics", "", metrics_table(res), "",
          "## Paired difference, CTC+LQR vs PID (same seeds)", "",
          "diff = ctc_lqr - pid; impr. > 0 means CTC+LQR is better; WIN = CI entirely on the better side and >= 10%. "
          "rmse rows compare only seeds both controllers survived (n column); survival_time is the primary metric of pole_balance. "
          "Holm-corrected decision over all rows of this table in the extra column.", "",
          paired_table(res, baseline="pid", metrics=["survival_time", "rmse", "ss_rmse", "tip_rmse", "energy"],
                       common_survivors=True, holm=True),
          "", f"Wall-clock for the {len(res)} cells x {len(DEMO_SEEDS)} seeds: {wall:.0f} s.", ""]
    open(f"{OUT}/table.md", "w").write("\n".join(md))
    print("\n".join(md))

    for sc, seed, name in [(tracking_hard, int(DEMO_SEEDS[0]), "tracking_hard"), (pole_hard, int(DEMO_SEEDS[1]), "pole_hard")]:
        parts = []
        for c in CTRLS:
            p = f"{OUT}/_{name}_{c}.mp4"
            info = record_rollout(sc, c, seed, p, fps=25, size=(360, 480), label=f"{c} | {name} | seed {seed}")
            parts.append(p)
            print(name, c, info["frames"], "frames", {k: round(v, 4) for k, v in info["metrics"].items() if k in ("rmse", "survival_time")})
        side_by_side(parts[0], parts[1], f"{OUT}/{name}_pid_vs_ctc_lqr.mp4")
        for p in parts:
            os.remove(p)


if __name__ == "__main__":
    main()
