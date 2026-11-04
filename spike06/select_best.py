"""Step 2b: pick best-classical = higher CAL success (hits>=300) of the retuned T-robust / T-adapt (CAL seeds only; tie -> higher CAL mean hits, then T-adapt). Writes results/spike06/best.json
and results/spike06/calib_final.json (T-nom, spike05 params, retuned params on CAL at cap 300)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env6 as W; V = W.V; W.use_file()
seeds = np.arange(1000, 1050); out = {}
M = {'T-nom': V.P_NOM, 'T-robust(spike05)': json.load(open('results/spike05/robust.json'))['params'], 'T-adapt(spike05)': json.load(open('results/spike05/adapt.json'))['params'],
     'T-robust': json.load(open('results/spike06/robust.json'))['params'], 'T-adapt': json.load(open('results/spike06/adapt.json'))['params']}
for n, P in M.items():
    r = V.evaluate(P, seeds); h = r['hits']; out[n] = dict(frac_cap=float((h >= 300).mean()), mean=float(h.mean()), median=float(np.median(h)), hits=h.tolist(), fails={f: r['fail'].count(f) for f in set(r['fail'])})
    print(n, {k: v for k, v in out[n].items() if k != 'hits'}, flush=True)
key = lambda n: (out[n]['frac_cap'], out[n]['mean'], n == 'T-adapt')
best = max(['T-robust', 'T-adapt'], key=key); print('best-classical =', best)
json.dump(out, open('results/spike06/calib_final.json', 'w'), indent=1)
json.dump(dict(name=best, params=M[best], cal=out[best], source=f'results/spike06/{best[2:]}.json'), open('results/spike06/best.json', 'w'), indent=1)
