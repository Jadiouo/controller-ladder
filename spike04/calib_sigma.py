"""Step 2: noise strength from CALIBRATION seeds (1000-1049) only. Smallest sigma_q s.t. C1-old (spike03a W0 gains) RMSE >= 1.3x its no-noise value in BOTH clean and dist, in BOTH N2 and N1."""
import sys, json, time; sys.path.insert(0, 'spike04'); from lib4 import *
old = json.load(open('results/spike03a/tuned_W0_W3.json'))['W0']['C1']; sp, sd = old['sp'], old['sd']
GRID = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]; t0 = time.time(); out = dict(c1old=[sp, sd], grid=GRID, cond={})
for cn, w in COND.items():
    base = {m: evaluate4(lambda: Ctrl4(*a), w, CAL_SEEDS, 0.0) for m, a in (('C0', (1, 1)), ('C1old', (sp, sd)))}
    b = {m: (r['clean']['rmse'].mean(), r['dist']['rmse'].mean()) for m, r in base.items()}; out['cond'][cn] = dict(base=b, rows={})
    print(cn, 'no-noise calib RMSE', b, flush=True)
    for s in GRID:
        row = {}
        for m, a in (('C0', (1, 1)), ('C1old', (sp, sd))):
            r = evaluate4(lambda: Ctrl4(*a), w, CAL_SEEDS, s); v = (r['clean']['rmse'].mean(), r['dist']['rmse'].mean())
            row[m] = dict(rmse=v, ratio=(v[0]/b[m][0], v[1]/b[m][1]))
        out['cond'][cn]['rows'][str(s)] = row; print(cn, s, {m: [round(x, 4) for x in row[m]['rmse']] + [round(x, 2) for x in row[m]['ratio']] for m in row}, flush=True)
ok = lambda cn, s: all(min(out['cond'][cn]['rows'][str(s)]['C1old']['ratio']) >= 1.3 for _ in [0])
out['sigma'] = next((s for s in GRID if ok('N1', s) and ok('N2', s)), None); out['sigma_N2_only'] = next((s for s in GRID if ok('N2', s)), None); out['sigma_N1_only'] = next((s for s in GRID if ok('N1', s)), None)
out['wall'] = time.time() - t0; print('SIGMA', out['sigma'], 'N2-only', out['sigma_N2_only'], 'N1-only', out['sigma_N1_only'], 'wall', out['wall'])
json.dump(out, open('results/spike04/calib_sigma.json', 'w'), indent=1)
