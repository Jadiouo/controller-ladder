"""Extend the e->zeta map to lower restitution (drop test, same method as experiments/spike03b/sanity.py). Writes experiments/spike06/zeta_map6.json (spike05 map + new points)."""
import sys, os, json, numpy as np
sys.path.insert(0, 'experiments/spike03b'); import sanity as S
old = json.load(open('experiments/spike05/zeta_map5.json')); zs = [0.40]; es = []
for z in zs:
    r = [S.drop(z, h) for h in (0.15, 0.5, 1.0)]; ee = [b/a for a, b in r]; es.append(float(np.mean(ee[:2]))); print(z, np.round(ee, 3), flush=True)
json.dump(dict(zeta=old['zeta'] + zs, e=old['e'] + es), open('experiments/spike06/zeta_map6.json', 'w'))
