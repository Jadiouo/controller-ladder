"""Error-intensity calibration on TEACHER ONLY, calibration seeds 1000-1049. Rule: smallest grid intensity whose median consecutive-hit count <= 50% of nominal median."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import env as E
seeds = np.arange(1000, 1050); ics = E.init_cond(seeds)
GRID = {'restitution': [dict(e=v, name=f'e={v}') for v in (0.7, 0.6, 0.55, 0.52, 0.5)],
        'mass': [dict(mscale=v, name=f'mass x{v}') for v in (2.0, 3.0, 4.0, 5.0, 6.0)],
        'sensing': [dict(sigma=0.004, delay=d, name=f'sigma 4mm + delay {d*5}ms') for d in (6, 12, 16, 20, 24, 30)]}
# round 2 (teacher was found more robust than the first grid assumed after kE retune): extended grids, only for families with no hit in round 1
GRID2 = {'restitution': [dict(e=v, name=f'e={v}') for v in (0.48, 0.46, 0.44, 0.42, 0.4)],
         'mass': [dict(mscale=v, name=f'mass x{v}') for v in (3.0, 3.5, 4.0, 4.5, 5.0, 6.0)],
         'sensing': [dict(sigma=sg, delay=d, name=f'sigma {sg*1000:.0f}mm + delay {d*5}ms') for sg, d in ((0.004, 24), (0.004, 30), (0.008, 30), (0.008, 36), (0.012, 40), (0.012, 50))]}
prev = json.load(open('results/spike03b/calib.json')) if os.path.exists('results/spike03b/calib.json') else {}
if prev: GRID = {f: GRID2[f] for f in GRID if not prev.get(f + '_chosen')}
out = dict(prev); t0 = time.time()
r0 = E.run_policy(E.JugEnv(50, E.Cond()), None, ics, seeds); base = float(np.median(r0['hits'])); out['nominal'] = dict(median=base, hits=r0['hits'].tolist())
print('nominal median', base, flush=True)
for fam, lst in GRID.items():
    out[fam + '_round1'] = out.get(fam, []) if prev else None; out[fam] = []; chosen = None
    for kw in lst:
        c = E.Cond(**kw); r = E.run_policy(E.JugEnv(50, c), None, ics, seeds); m = float(np.median(r['hits']))
        out[fam].append(dict(kw=kw, median=m, mean=float(r['hits'].mean()), q25=float(np.percentile(r['hits'], 25)), fails={f: r['fail'].count(f) for f in set(r['fail'])}))
        print(fam, kw['name'], 'median', m, 'mean %.1f' % r['hits'].mean(), out[fam][-1]['fails'], '(%.0fs)' % (time.time()-t0), flush=True)
        if chosen is None and m <= 0.5*base: chosen = kw
    out[fam + '_chosen'] = chosen; print('CHOSEN', fam, chosen, flush=True)
json.dump(out, open('results/spike03b/calib.json', 'w'), indent=1)
