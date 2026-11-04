"""Side-by-side clip (T-nom | best-classical | R-PPO-hist#0), same TEST episode. Episode rule (fixed before TEST): among TEST seeds where T-nom fails and best-classical succeeds, take the one whose
min_dist gap (T-nom minus best-classical) is closest to the median gap; fallback seed 0. Run AFTER eval9b.py."""
import sys, os, pickle, subprocess, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c9b import *
from arena.viz.record import record_rollout
R = pickle.load(open(os.path.join(OUT, 'test_raw.pkl'), 'rb')); F = pickle.load(open(os.path.join(OUT, 'final_methods.pkl'), 'rb')); best = R['_best']; Bs = R[f'T-{best}']; N = R['T-nom']
idx = np.where((N['success'] == 0) & (Bs['success'] == 1))[0]; gap = N['min_dist'] - Bs['min_dist']
ep = int(idx[np.argmin(np.abs(gap[idx] - np.median(gap[idx])))]) if len(idx) else 0; print('episode', ep, 'candidates', len(idx)); t = true_chain(ep); sc = scenario_of(t)
st = F['R-PPO-hist'][0]['layers']; facs = [('T-nom', make_factory('nom')), (f'T-{best} (best-classical)', make_factory(best)), ('R-PPO-hist seed 0', lambda ctx: ResCtl(ctx, best, True, False, st))]
os.makedirs(os.path.join(ROOT, '.scratch/v9b'), exist_ok=True); paths = []
for i, (lab, fac) in enumerate(facs):
    p = os.path.join(ROOT, f'.scratch/v9b/c{i}.mp4'); r = record_rollout(sc, fac, ep, p, fps=25, size=(360, 480), label=lab, camera=(0.7, 0.0, 0.1)); paths.append(p); print(lab, r['frames'], r['metrics'].get('success'), r['metrics'].get('min_dist'))
out = os.path.join(OUT, 'whip_test_episode.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + sum([['-i', p] for p in paths], []) + ['-filter_complex', 'hstack=inputs=3', '-pix_fmt', 'yuv420p', out], check=True); print('video', out)
