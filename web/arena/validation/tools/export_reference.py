"""Export the tracking references (glyphs, calibration seed 1000), initial states and the arm MJCF used by the web arena.
Run from anywhere with the repo's python environment (needs the `arena` package):  python export_reference.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))   # repo root
import numpy as np
from arena import Scenario
from arena.robots import RobotSpec, build_mjcf

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "assets"))
SEED = 1000
spec = RobotSpec.two_link()
open(os.path.join(OUT, "arm2.xml"), "w").write(build_mjcf(spec))
r = lambda a: np.round(np.asarray(a, float), 9).tolist()
data = {"seed": SEED, "dt": spec.dt, "dt_sim": spec.dt_sim, "substeps": spec.substeps, "tau_max": spec.tau_max,
        "payload_mass": spec.payload_mass, "duration": 4.0, "glyphs": {}}
for g in ["S", "8", "M", "Z", "heart", "circle"]:
    env = Scenario("tracking", {"ref": "glyph", "glyph": g}).make_env()
    env.reset(SEED)
    t = env.task
    data["glyphs"][g] = dict(Q=r(t.Q), Qd=r(t.Qd), Qdd=r(t.Qdd), EE=r(t.EE), q0=r(env.data.qpos[:2]), qd0=r(env.data.qvel[:2]))
json.dump(data, open(os.path.join(OUT, "tracking_refs.json"), "w"), separators=(",", ":"))
