"""Evaluation protocol v2 constants (docs/03_hypotheses.md) and the seed-split guard."""
import datetime
import os
import sys
from pathlib import Path

import numpy as np

TRAIN_SEEDS = np.arange(10000, 10200)  # training of learned controllers / RL environments / data collection ONLY
CAL_SEEDS = np.arange(1000, 1050)   # calibration: weakness strengths, classical tuning, any hyper-parameter choice
TEST_SEEDS = np.arange(0, 50)       # locked test set: report only; never tune on it (every use is logged)
DEMO_SEEDS = np.arange(5000, 5020)  # smoke tests / demos / unit tests; disjoint from all of the above

SPLITS = {"train": TRAIN_SEEDS, "cal": CAL_SEEDS, "test": TEST_SEEDS, "demo": DEMO_SEEDS}

SS_START_S = 0.5        # steady-state window starts here (s)
SETTLE_THR = 0.02       # settling / recovery threshold on max-component |error| (error units: rad for angles)
RECOVERY_WINDOW_S = 1.0
WIN_REL = 0.10          # "win" = paired 95% CI entirely < 0 AND mean improvement >= 10%
# FIXED failure penalty: after a failure (task failure or simulator/controller divergence) every remaining step of the
# horizon is charged this error PER COMPONENT, whatever the last state was. 2.0 rad is far above any real error
# (the pole falls at 0.6 rad), so an early failure is always worse than finishing the horizon. Tasks that log other
# quantities define their own constant (pole_balance.FAIL_TIP_ERR for the tip distance in m).
FAILURE_ERR = 2.0
DIVERGED_ERR = FAILURE_ERR        # backward-compatible name

BINARY_METRICS = frozenset({"success", "fell", "failure"})                 # 0/1 per episode -> bootstrap CI
HIGHER_IS_BETTER = frozenset({"alive_frac", "success", "survival_time"})   # metrics where larger is better

_TEST_LOG_DEFAULT = Path(__file__).resolve().parents[2] / "results" / ".test_access.log"


def test_log_path() -> Path:
    """results/.test_access.log (override with the ARENA_TEST_LOG environment variable, e.g. in unit tests)."""
    return Path(os.environ.get("ARENA_TEST_LOG", _TEST_LOG_DEFAULT))


def log_test_access(condition: str) -> None:
    """Append one line (time, script, condition) to the test-set access log."""
    path = test_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(path, "a") as f:
        f.write(f"{stamp}\t{os.path.basename(sys.argv[0]) or 'python'}\t{condition}\n")


def resolve_split(seeds, split: str | None) -> str:
    """Name of the split `seeds` belongs to. With `split` given, every seed must belong to it. Without, the seeds must
    all lie in ONE named split (or none: 'adhoc'); mixing splits (e.g. CAL + TEST) is an error."""
    seeds = np.asarray(seeds)
    where = {name: np.isin(seeds, arr) for name, arr in SPLITS.items()}
    if split is not None:
        if split not in SPLITS and split != "adhoc":
            raise ValueError(f"unknown split '{split}'; use one of {sorted(SPLITS)} or 'adhoc'")
        if split != "adhoc" and not where[split].all():
            raise ValueError(f"split='{split}' but {int((~where[split]).sum())} seed(s) are not in it")
        if split == "adhoc" and any(w.any() for w in where.values()):
            raise ValueError("split='adhoc' is for seeds outside every named split")
        return split
    hit = [name for name, w in where.items() if w.any()]
    if len(hit) > 1:
        raise ValueError(f"seeds span several splits {hit}; evaluate each split separately")
    if hit and not where[hit[0]].all():
        raise ValueError(f"seeds partly in split '{hit[0]}', partly outside every split; give a clean seed set")
    return hit[0] if hit else "adhoc"
