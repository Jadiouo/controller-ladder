"""LOCKED TEST SET evaluation (seeds 0-49, N=300 cap). Run once after calibration + training are final. Writes results/spike05/test_raw.pkl.
Plant error of each test episode is drawn from its seed (env5.draw_params) -> identical for every method."""
import sys, os, json, pickle, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env5 as V
torch.set_num_threads(2)
TEST = np.arange(50); out = {}; t0 = time.time()
def log(*x): print(*x, f'({time.time()-t0:.0f}s)', flush=True)
def net(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
class Res:
    def __init__(self, actor): self.actor = actor
    def __call__(self, env, tT, obs):
        with torch.no_grad(): m = self.actor(torch.from_numpy(obs)).numpy()
        env.prev_act = np.clip(m, -1, 1).astype(np.float32)
        return np.clip(tT + V.RES_BOUND * np.clip(m, -1, 1), -V.TAU_MAX, V.TAU_MAX)
only = sys.argv[1:]            # optional subset of method names
if os.path.exists('results/spike05/test_raw.pkl'): out = pickle.load(open('results/spike05/test_raw.pkl', 'rb'))
def run(name, P, policy=None, hist=False):
    if name in out or (only and name.split('#')[0] not in only): return
    t = time.time(); r = V.evaluate(P, TEST, policy=policy, hist=hist); r['eval_wall'] = time.time() - t; out[name] = r
    log(name, 'median hits', np.median(r['hits']), 'mean', r['hits'].mean(), 'maxres', round(r['maxres'], 2)); pickle.dump(out, open('results/spike05/test_raw.pkl', 'wb'))
run('T-nom', V.P_NOM)
for m in ('robust', 'adapt'):
    j = json.load(open(f'results/spike05/{m}.json')); run('T-' + m, j['params'])
for var, hist in (('dr', False), ('hist', True)):
    R = pickle.load(open(f'results/spike05/raw_rl_{var}.pkl', 'rb'))
    for s, v in sorted(R.items()):
        actor = net(v['obs'], v['hid'], 3); actor.load_state_dict(v['state']); run(f'R-{var}#{s}', V.P_NOM, Res(actor), hist)
        if f'R-{var}#{s}' in out: assert out[f'R-{var}#{s}']['maxres'] <= V.RES_BOUND + 1e-6
