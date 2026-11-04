"""Freeze the final method set before TEST-B / TEST: best-classical, tuned cfgs, L-est models, PPO checkpoints (chosen on CAL). Writes results/spike10b/final_methods.json (md5 of every artefact) + final_methods.pkl."""
import sys, os, json, pickle, hashlib, datetime; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
BC = jload('best_classical.json'); F = dict(best_classical=BC['best_classical'], best_spec=BC['spec'], ppo={}); files = {}
for fn in sorted(os.listdir(OUT)):
    if fn.startswith('lest_') and fn.endswith('.pkl') or fn in ('best_classical.json', 'probe_for_lest.json', 'tune_grad.json', 'tune_full.json', 'tune_ood.json', 'holdout1.json', 'dose.json', 'lest_train.json'): files['results/spike10b/' + fn] = md5(os.path.join(OUT, fn))
PPOF = sys.argv[1] if len(sys.argv) > 1 else 'ppo_hist.pkl'      # recipe adopted as R-PPO-hist (chosen on CAL only)
for extra in ('ppo_hist_diag.pkl', 'ppo_hist.pkl'): files['results/spike10b/' + extra] = md5(os.path.join(OUT, extra)) if os.path.exists(os.path.join(OUT, extra)) else None
F['ppo_file'] = PPOF
if os.path.exists(os.path.join(OUT, PPOF)):
    P = pickle.load(open(os.path.join(OUT, PPOF), 'rb'))
    for s, v in sorted(P.items()): F['ppo'][s] = dict(layers=v['selected_layers'], selected=v['selected'], init=v['init'], learned=v['learned'])
_H = os.path.dirname(os.path.abspath(__file__))
for fn in sorted(os.listdir(_H)):
    if fn.endswith('.py'): files['spike10b/' + fn] = md5(os.path.join(_H, fn))
pickle.dump(F, open(os.path.join(OUT, 'final_methods.pkl'), 'wb'))
json.dump(dict(frozen=datetime.datetime.now().isoformat(timespec='seconds'), best_classical=F['best_classical'], md5=files, ppo_selected={s: dict(sel=v['selected'], init=v['init'], learned=v['learned']) for s, v in F['ppo'].items()}), open(os.path.join(OUT, 'final_methods.json'), 'w'), indent=1, default=float); print('frozen', F['best_classical'], len(files), 'files')
