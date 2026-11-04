import os

import pytest

from arena import Scenario
from arena.viz import record_rollout


@pytest.mark.parametrize("scenario", [Scenario("tracking"), Scenario("pole_balance")])
def test_record_mp4(tmp_path, scenario):
    out = tmp_path / "clip.mp4"
    info = record_rollout(scenario, "ctc_lqr", 5000, str(out), fps=25, size=(240, 320), max_seconds=1.0)
    assert out.exists() and os.path.getsize(out) > 3000
    assert info["frames"] >= 20
