import json, numpy as np, sys
L = json.load(open('results/spike16a/cmp_machineA.json')); S = json.load(open('results/spike16a/cmp_machineB.json'))
print('machine-A', L['env'], '\nmachine-B', S['env'])
for t in L['tasks']:
    a, b = L['tasks'][t], S['tasks'][t]
    same_out = sum(x['outcome'] == y['outcome'] for x, y in zip(a, b)); same_steps = sum(x['steps'] == y['steps'] for x, y in zip(a, b))
    dq = [np.max(np.abs(np.array(x['qpos_all']) - np.array(y['qpos_all']))) for x, y in zip(a, b)]
    ds = [abs(x['smooth'] - y['smooth']) for x, y in zip(a, b)]
    print(f"{t}: outcome same {same_out}/10, same episode length {same_steps}/10, max|qpos diff| per episode {['%.1e' % d for d in dq]}, max|smooth diff| {max(ds):.1e}")
