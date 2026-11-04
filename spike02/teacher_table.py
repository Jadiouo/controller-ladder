"""Teacher-only check of all weakness conditions on the 50 TEST trajectories -> results/spike02/teacher_check.md (+ calibration set numbers)"""
import sys; sys.path.insert(0, 'spike02')
from sim2 import *
T = Teacher(0.0); base = None; L = ['| W | params | RMSE clean | x W0 | RMSE dist | x W0 | ss RMSE clean | sat frac clean | #traj RMSE>0.1 |', '|---|---|---|---|---|---|---|---|---|']
for k in ['W0', 'W1', 'W2', 'W3', 'W4']:
    r = eval_policy(T.act, W[k]); c, d = r['clean'], r['dist']
    if base is None: base = (c['rmse'].mean(), d['rmse'].mean())
    L.append(f"| {k} | {W[k]} | {c['rmse'].mean():.4f} | {c['rmse'].mean()/base[0]:.2f} | {d['rmse'].mean():.4f} | {d['rmse'].mean()/base[1]:.2f} | {c['ss_rmse'].mean():.4f} | {c['sat'].mean():.4f} | {(c['rmse']>0.1).sum()} |")
    print(L[-1], flush=True)
open('results/spike02/teacher_check.md', 'w').write('\n'.join(L) + '\n')
