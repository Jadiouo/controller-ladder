"""merge per-seed pkl files into one pkl per method and write the frozen method list for eval7.py (no TEST access). R-SAC config (A or B) is chosen here from CAL numbers only (higher mean CAL success of the selected checkpoints; tie -> higher mean hits)."""
import pickle, json, glob, numpy as np
def load(prefix):
    d = {}
    for f in sorted(glob.glob(f'results/spike07/final_{prefix}_s*.pkl')): d.update(pickle.load(open(f, 'rb')))
    assert sorted(d) == [0, 1, 2], (prefix, sorted(d)); return d
sc = lambda d: (np.mean([v['selected']['frac_cap'] for v in d.values()]), np.mean([v['selected']['mean'] for v in d.values()]))
sacA, sacB = load('sacA'), load('sacB'); print('SAC config A mean CAL (frac_cap, mean hits):', sc(sacA), ' B:', sc(sacB)); pick = 'A' if sc(sacA) >= sc(sacB) else 'B'; print('R-SAC = config', pick)
D = {'R-ppo': ('ppo', load('ppo')), 'R-oracle': ('oracle', load('oracle')), 'R-sac': ('sac', sacA if pick == 'A' else sacB)}; out = {}
for m, (k, d) in D.items():
    pickle.dump(d, open(f'results/spike07/final_{k}.pkl', 'wb')); out[m] = f'results/spike07/final_{k}.pkl'
    print(m, {s: (v['selected']['it'], v['selected']['frac_cap'], round(v['selected']['res_mean'], 3)) for s, v in d.items()})
json.dump(dict(out, sac_config=pick), open('results/spike07/final_methods_meta.json', 'w'), indent=1); json.dump(out, open('results/spike07/final_methods.json', 'w'), indent=1)
