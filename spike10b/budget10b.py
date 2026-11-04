"""Collect the three-axis budget (episodes, wall-clock, label source) into results/spike10b/budget.json from the saved logs."""
import sys, os, json, pickle, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
J_ = lambda p: json.load(open(os.path.join(OUT, p))) if os.path.exists(os.path.join(OUT, p)) else None
B = {}; a2 = json.load(open(os.path.join(ROOT, 'results/spike10a/adapt2_tuning.json')))
B['classical'] = {'T-robust episodes (spike09a)': 9050, 'T-adapt design episodes (spike09b, estimate)': 1300, 'T-adapt2 tuning episodes (spike10a)': a2['episodes'], 'T-adapt2 wall_s': a2.get('wall_s')}
for t in ('grad', 'full', 'ood'):
    T_ = J_(f'tune_{t}.json')
    if T_: B['classical'][f'T-adapt3-{t} tuning episodes (CAL, incl. 2 reference evals of T-adapt/T-adapt2)'] = T_['episodes']; B['classical'][f'T-adapt3-{t} tuning wall_s'] = T_['wall']
tot = sum(v for k, v in B['classical'].items() if 'T-robust' in k or ('T-adapt2 tuning' in k) or ('T-adapt3-grad tuning episodes' in k) or ('T-adapt3-full tuning episodes' in k)); B['classical']['total for the 2x rule (T-robust + T-adapt2 + T-adapt3 grad+full)'] = tot; B['classical']['2x'] = 2 * tot
LT = J_('lest_train.json'); fr = None
if LT:
    pr = LT['T'] / 2.5; B['L-est'] = {'probe segment T (s)': LT['T'], 'training probe episodes (full+eff share one set; ood its own)': 2 * LT['ntrain'], 'full-episode equivalents (x T/2.5)': 2 * LT['ntrain'] * pr, 'CAL probe episodes': 2 * 200, 'wall_s (data + sweep + 3 seeds x 3 variants)': LT['wall'],
        'label source': 'simulator ground truth (full plant theta: length scale, 10 per-link masses, damping, Coulomb friction) for every training episode; classical methods use no labels'}
PP = {}
for f, nm in (('ppo_hist.pkl', 'R-PPO-hist main recipe (adopted)'), ('ppo_hist_diag.pkl', 'R-PPO-hist diagnostic round (4x batch, not adopted)')):
    if os.path.exists(os.path.join(OUT, f)):
        for sd, v in pickle.load(open(os.path.join(OUT, f), 'rb')).items(): PP[f'{nm} seed {sd}'] = dict(train_episodes=v['train_episodes'], cal_eval_episodes=v['cal_eval_episodes'], total_episodes=v['train_episodes'] + v['cal_eval_episodes'], steps=v['train_steps'], wall_s=v['wall'], init_cal=v['init']['succ'], selected_cal=v['selected']['succ'], learned=v['learned'], res_mean_selected=v['selected']['res_mean'])
PB = J_('pool_base.json')
if PB: PP['pool cache-fill (zero residual, 300 scenarios)'] = PB
B['R-PPO-hist'] = PP; B['label sources'] = {'classical (T-nom/robust/adapt/adapt2/adapt3)': 'none: only measured arm q, qd and applied probe torques; tuned on CAL success', 'L-est-*': 'simulator ground-truth plant theta of every training episode (13 numbers for full/ood, 4 for eff)', 'R-PPO-hist': 'reward uses privileged chain-tip distance (simulator); observations are arm-only history', 'T-oracle': 'true plant given directly (reference)'}
json.dump(B, open(os.path.join(OUT, 'budget.json'), 'w'), indent=1, default=float); print(json.dumps(B, indent=1, default=float))
