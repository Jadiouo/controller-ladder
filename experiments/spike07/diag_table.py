"""Markdown table of all diagnostic runs (parses results/spike07/diag/*.log) + sim-step accounting (train steps from the last progress line + CAL-checkpoint evaluation steps)."""
import re, ast, glob, sys
rows = []; tot = 0
for f in sorted(glob.glob('results/spike07/diag/*.log')):
    L = open(f).read().splitlines(); cks = [ast.literal_eval(l.strip()[5:]) for l in L if l.strip().startswith('ckpt ')]
    pr = [l for l in L if 'train-frac_cap' in l]
    if not pr: continue
    tr_steps = int(re.search(r'steps (\d+)', pr[-1]).group(1)); ev_steps = sum(c['steps'] for c in cks); tot += tr_steps + ev_steps
    tr = [float(re.search(r'last60\) (\S+)', l).group(1)) for l in pr]; half = tr[len(tr) // 2:]
    cs = ' / '.join(f"{c['frac_cap']*100:.0f}" for c in cks); rs = ' / '.join(f"{c['res_mean']:.2f}" for c in cks)
    rows.append(f"| {f.split('/')[-1][:-4]} | {cs} | {rs} | {min(half)*100:.0f}..{half[-1]*100:.0f} | {tr_steps/1e6:.2f}M + {ev_steps/1e6:.2f}M |")
print('| run | CAL success % at checkpoints | residual L2 mean at ckpts (Nm) | train success, 2nd half: min..last (%) | sim steps (train + ckpt eval) |\n|---|---|---|---|---|'); print('\n'.join(rows)); print(f'\nTotal diagnosis sim steps so far: {tot/1e6:.1f}M')
