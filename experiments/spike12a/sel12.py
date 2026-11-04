"""spike12a: CAL-based checkpoint selection for the extra training seeds (and seed 0 already in selection.json). Writes selection.json entries '<kind>_<tag>_s<seed>' and prints the final-eval spec list."""
import json, torch, sys
R = './'; c = json.load(open(R + 'results/spike12a/cal.json')); sel = json.load(open(R + 'results/spike12a/selection.json')); specs = []
for seed in (1, 2):
    for tag in ('multi', 'canon'):
        for k in ('bc', 'mip', 'diffusion', 'flow', 'flow1'):
            names = [f'{k}_{tag}_s{seed}_{s}' for s in (10000, 20000, 40000)]; best = max(names, key=lambda n: (c[n]['succ'], int(n.split('_')[-1]))); st = int(best.split('_')[-1]); kk = 'flow' if k == 'flow1' else k
            path = R + f'.scratch/s12/ckpt/{kk}_{tag}_s{seed}_{st}.pt'; sel[f'{k}_{tag}_s{seed}'] = dict(ckpt=f'{st//1000}k', cal=c[best]['succ'], params=torch.load(path, map_location='cpu', weights_only=False)['params'], path=path)
            specs.append(f'{k}_{tag}_s{seed}:{path}' + (':1' if k == 'flow1' else ''))
json.dump(sel, open(R + 'results/spike12a/selection.json', 'w'), indent=1); open(R + '.scratch/s12/specs2.txt', 'w').write(' '.join(specs))
