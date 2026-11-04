"""LOCKED TEST SET evaluation (seeds 0-49, cap 300). Run once after calibration + training are frozen. Writes results/spike06/test_raw.pkl.
Plant error per test episode is drawn from its seed (identical for all methods); asserted below by comparing stored params across methods."""
import sys, os, json, pickle, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env6 as W; V = W.V; DIST = W.use_file()
import rl6; torch.set_num_threads(2)
TEST = np.arange(50); OUT = 'results/spike06/test_raw.pkl'; out = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}; t0 = time.time()
def run(name, P, policy=None, hist=False):
    if name in out: return
    t = time.time(); r = V.evaluate(P, TEST, policy=policy, hist=hist); r['eval_wall'] = time.time() - t; out[name] = r
    print(name, 'frac_cap', (r['hits'] >= 300).mean(), 'mean', r['hits'].mean(), 'median', np.median(r['hits']), 'maxres', round(r['maxres'], 2), f'({time.time()-t0:.0f}s)', flush=True); pickle.dump(out, open(OUT, 'wb'))
best = json.load(open('results/spike06/best.json')); out['_best'] = best['name']
run('T-nom', V.P_NOM)
for m in ('robust', 'adapt'): run('T-' + m, json.load(open(f'results/spike06/{m}.json'))['params']); run(f'T-{m}(spike05)', json.load(open(f'results/spike05/{m}.json'))['params'])
for var, hist in (('dr', False), ('hist', True)):
    R = pickle.load(open(f'results/spike06/final_{var}.pkl', 'rb'))
    for s, v in sorted(R.items()):
        actor = rl6.mk(v['obs'], 3, v['hid']); actor.load_state_dict(v['state']); run(f'R-{var}#{s}', best['params'], rl6.Res(actor), hist)
        assert out[f'R-{var}#{s}']['maxres'] <= V.RES_BOUND + 1e-6
ps = [tuple((p['e'], p['mb'], p['delay_ms']) for p in v['params']) for k, v in out.items() if k != '_best']; assert all(p == ps[0] for p in ps), 'plant errors differ across methods'
assert all(all(p['e'] >= DIST['e'][0] and p['e'] <= DIST['e'][1] for p in v['params']) for k, v in out.items() if k != '_best')
pickle.dump(out, open(OUT, 'wb')); print('done; identical plant errors across all methods: OK')
