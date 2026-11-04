"""Freeze the final method set before TEST: best-classical (holdout), L-est models, residual PPO checkpoints chosen on CAL. Writes results/spike09b/final_methods.pkl + final_methods.json (md5 of every artefact)."""
import sys, os, json, pickle, hashlib, datetime; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c9b import *
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
HO = json.load(open(os.path.join(OUT, 'holdout.json'))); F = dict(best_classical=HO['best_classical'], lest={}, **{'R-PPO-hist': {}, 'R-oracle': {}}); files = {}
for s in range(3):
    p = os.path.join(OUT, f'lest_s{s}.pkl'); F['lest'][s] = pickle.load(open(p, 'rb')); files[os.path.basename(p)] = md5(p)
for m, f in (('R-PPO-hist', 'ppo_hist.pkl'), ('R-oracle', 'ppo_oracle.pkl')):
    P = pickle.load(open(os.path.join(OUT, f), 'rb')); files[f] = md5(os.path.join(OUT, f))
    for s, v in sorted(P.items()): F[m][s] = dict(layers=v['selected_layers'], selected=v['selected'], init=v['init'], learned=v['learned'])
assert all(len(F[k]) == 3 for k in ('lest', 'R-PPO-hist', 'R-oracle'))
for f in ('c9b.py', 'train9.py', 'eval9b.py', 'step3_lest.py', 'step1b_holdout.py'): files['experiments/spike09b/' + f] = md5(os.path.join(ROOT, 'experiments', 'spike09b', f))
pickle.dump(F, open(os.path.join(OUT, 'final_methods.pkl'), 'wb')); json.dump(dict(frozen=datetime.datetime.now().isoformat(timespec='seconds'), best_classical=F['best_classical'], md5=files,
    selected={m: {s: F[m][s]['selected'] for s in F[m]} for m in ('R-PPO-hist', 'R-oracle')}), open(os.path.join(OUT, 'final_methods.json'), 'w'), indent=1, default=float); print('frozen', F['best_classical'])
