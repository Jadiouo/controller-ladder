"""LOCKED TEST evaluation (seeds 0-49, cap 300) for spike07. RUN ONCE after everything is frozen; refuses to run if results/spike07/test_raw.pkl exists.
Appends one line to results/.test_access.log before touching TEST seeds. Plant errors identical across methods (asserted).
Methods: T-nom, T-robust(spike06), T-robust(kp10000), T-adapt(spike06), best-classical (alias), R-ppo#s, R-sac#s, R-oracle#s (residuals on best-classical, checkpoint chosen on CAL)."""
import sys, os, json, pickle, time, datetime, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env7 as W; V = W.V; DIST = W.use_file()
import rl7; torch.set_num_threads(2)
OUT = 'results/spike07/test_raw.pkl'; assert not os.path.exists(OUT), 'TEST already evaluated once; refusing to run again'
FINAL = json.load(open('results/spike07/final_methods.json'))      # {method: pkl path}, frozen before this script
with open('results/.test_access.log', 'a') as f: f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\tspike07/eval7.py\tH9/H10 single locked TEST run, all methods, seeds=0-49\n")
TEST = np.arange(50); out = {}; t0 = time.time()
best = json.load(open('results/spike07/best.json')); out['_best'] = best['name']
def run(name, P, actor=None, hist=False, oracle=False):
    t = time.time(); r = rl7.evaluate(P, actor, TEST, hist, oracle); r['eval_wall'] = time.time() - t; out[name] = r
    print(name, 'frac_cap', (r['hits'] >= 300).mean(), 'mean', r['hits'].mean(), 'median', np.median(r['hits']), 'res', r.get('res_mean'), r.get('res_max'), f'({time.time()-t0:.0f}s)', flush=True); pickle.dump(out, open(OUT + '.partial', 'wb'))
run('T-nom', V.P_NOM)
run('T-robust(spike06)', json.load(open('results/spike06/robust.json'))['params'])
run('T-robust(kp10000)', json.load(open('results/spike07/robust.json'))['params'])
run('T-adapt(spike06)', json.load(open('results/spike06/adapt.json'))['params'])
for m, path in FINAL.items():
    R = pickle.load(open(path, 'rb')); hist = m != 'R-nohist'
    for s, v in sorted(R.items()):
        assert v['classical'].endswith('results/spike07/best.json')
        run(f'{m}#{s}', best['params'], rl7.load_actor(v), hist=True, oracle=(v['variant'] == 'oracle'))
ps = [tuple((p['e'], p['mb'], p['delay_ms']) for p in v['params']) for k, v in out.items() if k != '_best']; assert all(p == ps[0] for p in ps), 'plant errors differ across methods'
assert all(all(DIST['e'][0] <= p['e'] <= DIST['e'][1] for p in v['params']) for k, v in out.items() if k != '_best')
pickle.dump(out, open(OUT, 'wb')); os.remove(OUT + '.partial'); print('done; identical plant errors across all methods: OK')
