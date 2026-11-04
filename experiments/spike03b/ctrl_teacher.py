"""CONTROL (not part of the H6 rule): a classically re-tuned teacher per condition, tuned on CALIBRATION seeds 1000-1049 only (grid on the teacher's own model params
that correspond to the unmodelled error: restitution -> (e_model, kappa_max); mass -> CTC model mass scale; sensing -> ball-state extrapolation steps), evaluated once on the locked test set.
Purpose: is the student's gain just 'the teacher was mis-specified', i.e. recoverable by re-tuning the model parameter?"""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from stu import *
cs = load_conds(); out = {}; cal = np.arange(1000, 1050)
GR = {'restitution': [dict(E_NOM=en, KAPPA_MAX=km) for en in (0.8, 0.6, 0.5, 0.4) for km in (0.4, 0.8, 1.2)],
      'mass': [dict(MODEL_MS=ms) for ms in (1.0, 2.0, 3.0, 4.0, 5.0)],
      'sensing': [dict(pred=p) for p in (0, 10, 20, 30, 40)]}
def setp(d):
    for k, v in d.items():
        if k == 'pred': E.TP['pred'] = v
        else: setattr(E, k, v)
def reset():
    E.E_NOM = 0.8; E.KAPPA_MAX = 0.4; E.MODEL_MS = 1.0; E.TP['pred'] = 0
for c in GR:
    best = None
    for g in GR[c]:
        reset(); setp(g); r = E.run_policy(E.JugEnv(50, cs[c]), None, E.init_cond(cal), cal); m = (float(np.median(r['hits'])), float(r['hits'].mean()))
        print(c, g, 'calib median/mean', m, flush=True)
        if best is None or m > best[1]: best = (g, m)
    reset(); setp(best[0]); r = evaluate(cs[c], None)
    out[c] = dict(params=best[0], calib=best[1], test_hits=r['hits'].tolist(), test_median=float(np.median(r['hits'])), apex=float(np.nanmean(r['apex'])), tau2=float(r['tau2'].mean()))
    print(c, 'CHOSEN', best[0], 'TEST median', out[c]['test_median'], flush=True)
reset(); json.dump(out, open('results/spike03b/ctrl_teacher.json', 'w'), indent=1)
