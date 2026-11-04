import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tempfile

# Never write the repo's results/.test_access.log from the unit tests (the real log is for real experiments).
os.environ.setdefault("ARENA_TEST_LOG", os.path.join(tempfile.mkdtemp(prefix="arena_test_log_"), "test_access.log"))
