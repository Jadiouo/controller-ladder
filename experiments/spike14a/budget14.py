"""Budget ledger (episodes, wall-clock, label source) -> results/spike14a/budget.json and a markdown block printed."""
import sys, os, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
PROBE_EQ = 0.6 / 2.5       # a probe-only episode counts as 0.24 of a full episode (spike10b convention)
def nnwall(tag):
    ws = []
    for s in range(3):
        f = os.path.join(DAT, f'pred_{tag}_s{s}.pkl')
        if os.path.exists(f): ws.append(pickle.load(open(f, 'rb'))['wall'])
    return max(ws) if ws else None       # wall is cumulative within one process (sweep + seeds)
B = {}
sens = json.load(open(os.path.join(OUT14, 'sens_cost.json'))) if os.path.exists(os.path.join(OUT14, 'sens_cost.json')) else {}
B['L-mlp1 (spike10b L-est-full)'] = dict(probe_episodes=12000, full_equiv=12000 * PROBE_EQ, cal_label_eps=200, wall_s='645 (whole spike10b lest pipeline incl. 2 other variants; upper bound)', labels='simulator theta13 for 12000 training episodes', note='frozen from spike10b; reused as is')
B['L-gru1'] = dict(probe_episodes=12000, full_equiv=12000 * PROBE_EQ, train_wall_s=nnwall('gru1'), data_wall_s='166 (8 workers, loaded machine)', labels='simulator theta13 x 12000')
B['L-gru4'] = dict(probe_episodes=48000, full_equiv=48000 * PROBE_EQ, train_wall_s=nnwall('gru4'), data_wall_s='166 + 807', labels='simulator theta13 x 48000')
B['L-mlp4'] = dict(probe_episodes=48000, full_equiv=48000 * PROBE_EQ, train_wall_s=nnwall('mlp4'), labels='simulator theta13 x 48000')
B['L-dec1'] = dict(probe_episodes=12000, full_equiv=12000 * PROBE_EQ + 1600, oracle_plans=1600, train_wall_s=nnwall('dec1_a0.9'), sens_wall_s=1585, labels='simulator theta13 x 12000 + true plant for 1600 oracle plans + finite-difference sensitivities (26 swing simulations each)')
B['L-dec4'] = dict(probe_episodes=48000, full_equiv=48000 * PROBE_EQ + 1600, oracle_plans=1600, train_wall_s=nnwall('dec4_a0.9_e38'), sens_wall_s=1585, labels='simulator theta13 x 48000 + true plant for 1600 oracle plans (sensitivities of the other 46400 episodes use the mean sensitivity)')
cc = json.load(open(os.path.join(OUT14, 'classical_cost.json'))) if os.path.exists(os.path.join(OUT14, 'classical_cost.json')) else {}
B['C0 (T-adapt3-full)'] = dict(tuning_episodes=750, tuning_wall_s=3007, labels='none (tuned on CAL success)', per_episode_fit_cpu_s=cc.get('C0|cal', {}).get('cpu_s', 0) / 200)
for k, v in cc.items():
    if k.endswith('|cal') and not k.startswith('C0'): B[f'{k.split("|")[0]} (CAL screening fits)'] = dict(fit_episodes=v['n'], fit_cpu_s=v['cpu_s'], fit_wall_s=v['wall_s'], per_episode_cpu_s=v['cpu_s'] / v['n'], labels='CAL label error used to screen; none for the fit itself')
json.dump(B, open(os.path.join(OUT14, 'budget.json'), 'w'), indent=1, default=str); print(json.dumps(B, indent=1, default=str))
