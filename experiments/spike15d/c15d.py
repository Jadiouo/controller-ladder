"""spike15d shared code: classical control for spike15c approach B. Reuses spike15c (ProbeB, run_B, run_score, init_mem) unchanged; only the seed guard and the new-seed set differ.
Seeds allowed: CAL 1000-1049/2000-2149, holdout 1100-1599, new 704000-704199. Never TEST 0-49 / TEST-B 20000-20499 / TEST-C 30000-30499."""
import sys, os
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'spike15c'))
import c15c
from c15c import *
S15D = os.path.join(ROOT, '.scratch/s15d'); RES15D = os.path.join(ROOT, 'results/spike15d'); os.makedirs(S15D, exist_ok=True); os.makedirs(RES15D, exist_ok=True)
NEWD = list(range(704000, 704200)); SETS = dict(cal=CAL, hold=HOLD, new=NEWD)
def guard15d(seed):
    s = int(seed); ok = (1000 <= s < 1050) or (2000 <= s < 2150) or (1100 <= s <= 1599) or (704000 <= s <= 704199)
    assert ok, f'seed {s} not allowed in spike15d (never TEST 0-49 / TEST-B 20000-20499 / TEST-C 30000-30499)'
c15c.guard15c = guard15d; c13.guard13 = guard15d; c14.guard14 = guard15d; c13.ALLOW = ((704000, 704199),)
guard15c = guard15d
