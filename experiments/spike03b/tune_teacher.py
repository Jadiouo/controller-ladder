"""Tune the teacher ONCE on the nominal model (small sweep on tuning seeds 3000-3019 which are disjoint from calib 1000-1049 / test 0-49 / training)."""
import sys, os, itertools, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import env as E
seeds = np.arange(3000, 3020); ics = E.init_cond(seeds); rows = []
for kE, Kx in itertools.product([0.3, 0.5, 1.0, 1.5], [0.7, 1.1, 1.6]):
    E.TP['kE'] = kE; E.TP['Kx'] = Kx; env = E.JugEnv(20, E.Cond()); r = E.run_policy(env, None, ics, seeds)
    row = (kE, Kx, np.median(r['hits']), np.nanmean(r['apex']), np.nanmean(r['drift']), np.mean(r['tau2']))
    rows.append(row); print('kE %.1f Kx %.1f | median hits %d | apex err %.3f m | hit x-drift %.4f m | tau2 %.0f' % row, flush=True)
