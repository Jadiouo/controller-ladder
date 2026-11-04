"""Calibrate weakness strength using ONLY the teacher (no students). Calibration trajectories: seeds 1000..1049 (disjoint from test 0..49)."""
import sys; sys.path.insert(0, 'spike02')
from sim2 import *
cs = np.arange(1000, 1050); P = sample_refs(cs); ie = sample_init_err(cs); ds = sample_dist(cs)
T = Teacher(0.0)
def run(w):
    a = rollout2(T.act, w, P, ie); b = rollout2(T.act, w, P, ie, dist=ds)
    return a, b
base = run(W['W0'])
print('W0', base[0]['rmse'].mean(), base[1]['rmse'].mean(), 'peak? sat', base[0]['sat'].mean())
cands = [W[k] for k in ('W1', 'W2', 'W3', 'W4')]
for w in cands:
    a, b = run(w)
    print(w, 'clean %.4f (x%.2f) ss %.4f dist %.4f (x%.2f) sat %.3f' % (a['rmse'].mean(), a['rmse'].mean()/base[0]['rmse'].mean(), a['ss_rmse'].mean(), b['rmse'].mean(), b['rmse'].mean()/base[1]['rmse'].mean(), a['sat'].mean()), flush=True)
