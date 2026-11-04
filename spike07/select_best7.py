"""spike07 step 1b: best-classical = new T-robust (kp<=10000) if its CAL (frac_cap, mean hits) beats spike06's T-robust, else the spike06 one. CAL seeds only. Writes results/spike07/best.json, calib_final.json."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env7 as W; V = W.V; W.use_file()
seeds = np.arange(1000, 1050); out = {}
M = {'T-robust(spike06)': json.load(open('results/spike06/robust.json'))['params'], 'T-robust(kp10000)': json.load(open('results/spike07/robust.json'))['params'], 'T-adapt(spike06)': json.load(open('results/spike06/adapt.json'))['params']}
for n, P in M.items():
    r = V.evaluate(P, seeds); h = r['hits']; out[n] = dict(frac_cap=float((h >= 300).mean()), mean=float(h.mean()), median=float(np.median(h)), hits=h.tolist(), fails={f: r['fail'].count(f) for f in set(r['fail'])}, steps=r['steps'])
    print(n, {k: v for k, v in out[n].items() if k != 'hits'}, 'kp', P['kp'], flush=True)
key = lambda n: (out[n]['frac_cap'], out[n]['mean'])
best = 'T-robust(kp10000)' if key('T-robust(kp10000)') > key('T-robust(spike06)') else 'T-robust(spike06)'; print('best-classical =', best)
src = 'results/spike07/robust.json' if best == 'T-robust(kp10000)' else 'results/spike06/robust.json'
json.dump(out, open('results/spike07/calib_final.json', 'w'), indent=1)
json.dump(dict(name=best, params=M[best], cal=out[best], source=src), open('results/spike07/best.json', 'w'), indent=1)
