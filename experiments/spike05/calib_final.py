"""Calibration-set (seeds 1000-1049) evaluation of T-nom / T-robust / T-adapt at the full cap N=300 (reporting only; test set untouched)."""
import sys, os, json, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env5 as V
seeds = np.arange(1000, 1050); out = {}
for name, P in (('T-nom', V.P_NOM), ('T-robust', json.load(open('results/spike05/robust.json'))['params']), ('T-adapt', json.load(open('results/spike05/adapt.json'))['params'])):
    r = V.evaluate(P, seeds); out[name] = dict(median=float(np.median(r['hits'])), mean=float(r['hits'].mean()), frac_cap=float((r['hits'] >= 300).mean()), hits=r['hits'].tolist(), fails={f: r['fail'].count(f) for f in set(r['fail'])})
    print(name, {k: v for k, v in out[name].items() if k != 'hits'}, flush=True)
json.dump(out, open('results/spike05/calib_final.json', 'w'), indent=1)
