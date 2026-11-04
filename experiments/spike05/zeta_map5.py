"""Extend spike03b zeta map to e up to ~0.9 (drop test, same method as experiments/spike03b/sanity.py) -> experiments/spike05/zeta_map5.json"""
import sys, os, json, numpy as np
sys.path.insert(0, 'experiments/spike03b'); import sanity as S
old = json.load(open('experiments/spike03b/zeta_map.json')); zs = [0.01, 0.02, 0.03, 0.04, 0.05]; es = []
for z in zs:
    r = [S.drop(z, h) for h in (0.15, 0.5, 1.0)]; ee = [b/a for a, b in r]; es.append(float(np.mean(ee[:2]))); print(z, np.round(ee, 3), flush=True)
json.dump(dict(zeta=zs + old['zeta'], e=es + old['e']), open('experiments/spike05/zeta_map5.json', 'w'))
