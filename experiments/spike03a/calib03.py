"""Step 1: re-decide W1 cap and W2 delay on the CALIBRATION set only (seeds 1000-1049): mildest strength with C0 RMSE >= 1.3x W0 in BOTH clean and dist. Test seeds not touched."""
import sys, json, time; sys.path.insert(0, 'experiments/spike03a'); from lib import *
t0 = time.time(); base = evaluate(lambda: Ctrl(), W['W0'], CAL_SEEDS); b = (base['clean']['rmse'].mean(), base['dist']['rmse'].mean())
print('W0 calib', b, flush=True); out = dict(W0=b, cap={}, delay={})
def ratio(w):
    r = evaluate(lambda: Ctrl(), w, CAL_SEEDS); return r['clean']['rmse'].mean()/b[0], r['dist']['rmse'].mean()/b[1], float((r['clean']['rmse'] > 0.1).sum())
for cap in [80, 60, 50, 45, 40, 38, 36, 34, 32, 30, 28, 27, 26, 24, 22, 20]:
    out['cap'][cap] = ratio(dict(cap=float(cap), delay=0, fc=0, fv=0)); print('cap', cap, out['cap'][cap], flush=True)
for dl in range(0, 9):
    out['delay'][dl] = ratio(dict(cap=80.0, delay=dl, fc=0, fv=0)); print('delay steps', dl, out['delay'][dl], flush=True)
ok = lambda r: r[0] >= 1.3 and r[1] >= 1.3
out['new_cap'] = max([c for c, r in out['cap'].items() if ok(r)], default=None)
out['new_delay_steps'] = min([d for d, r in out['delay'].items() if ok(r)], default=None)
print('NEW', out['new_cap'], out['new_delay_steps'], 'sim steps', 100*NSTEP*(len(out['cap'])+len(out['delay'])+1)*50 // 100, 'wall', time.time()-t0)
json.dump(out, open('results/spike03a/calib_W.json', 'w'), indent=1)
