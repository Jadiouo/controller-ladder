import pickle, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from stu import load_conds
rng = np.random.default_rng(0); NB = 4000
def ld(p): return pickle.load(open(p, 'rb')) if os.path.exists(p) else None
def ci(x): return np.percentile(x, [2.5, 97.5])
def pooled(res_list, key):   # (n_seeds, 50)
    return np.array([r[key] for r in res_list])
import glob
if not glob.glob('results/spike03b/raw_*.pkl'): sys.exit('results/spike03b/raw_*.pkl not found (per-episode data is not in the public release); table.md left unchanged')
conds = load_conds(); L = ['# spike03b table: mirror-law teacher vs students on ball juggling (locked test set, 50 episodes, hits capped at 100)\n']
verdict = []; rows = []
for c in conds:
    bc, rl = ld(f'results/spike03b/raw_bc_{c}.pkl'), ld(f'results/spike03b/raw_rl_{c}.pkl')
    if bc is None and rl is None: continue
    T = (bc or rl)['teacher']; th = T['hits'].astype(float)
    meths = {'teacher': [T]}
    if bc: meths['BC+DART'] = [v for k, v in bc.items() if isinstance(k, tuple)]
    if rl: meths['residual PPO'] = [v for k, v in rl.items() if isinstance(k, tuple)]
    idx = rng.integers(0, 50, (NB, 50))
    L.append(f'\n## {c}: {conds[c]}\n\n| method | n seeds | median hits [95% CI] | mean hits | rel. median vs teacher [95% CI, paired bootstrap] | seed-level rel. medians | apex err (m) | hit x-drift (m) | torque energy (Nm^2) |\n|---|---|---|---|---|---|---|---|---|')
    medT = np.median(th[idx], 1)
    ctl = json.load(open('results/spike03b/ctrl_teacher.json')) if os.path.exists('results/spike03b/ctrl_teacher.json') else {}
    if c in ctl: meths['CONTROL: classically re-tuned teacher ' + json.dumps(ctl[c]['params'])] = [dict(hits=np.array(ctl[c]['test_hits']), apex=np.array([ctl[c]['apex']]*50), drift=np.full(50, np.nan), tau2=np.array([ctl[c]['tau2']]*50))]
    for m, lst in meths.items():
        if not lst: continue
        H = pooled(lst, 'hits').astype(float)           # (S,50)
        med = np.array([np.median(H[:, ix]) for ix in idx])
        m0 = np.median(H); mT = np.median(th)
        if m == 'teacher': rel = '-'; sl = '-'
        else:
            relb = (med - medT) / np.maximum(medT, 1e-9); lo, hi = ci(relb); rel = f'{(m0-mT)/mT*100:+.0f}% [{lo*100:+.0f}, {hi*100:+.0f}]'
            sl = ', '.join(f'{(np.median(h)-mT)/mT*100:+.0f}%' for h in H)
            if m == 'residual PPO': rows.append((c, (m0-mT)/mT, lo, hi, mT, m0, [ (np.median(h)-mT)/mT for h in H]))
        lo2, hi2 = ci(med)
        ap = np.nanmean(pooled(lst, 'apex')); dr = np.nanmean(pooled(lst, 'drift')); te = np.mean(pooled(lst, 'tau2'))
        L.append(f'| {m} | {len(lst)} | {m0:.0f} [{lo2:.0f}, {hi2:.0f}] | {H.mean():.1f} | {rel} | {sl} | {ap:.3f} | {dr:.4f} | {te:.0f} |')
    if rl: L.append(f'\nresidual bound check: max |cmd - teacher| over all RL evals = {max(v["maxres"] for k, v in rl.items() if isinstance(k, tuple)):.2f} Nm (limit 12). RL train steps/seed: {list(rl.values())[1]["steps"] if len(rl)>1 else "-"}; train wall s: {[round(v["train_time"]) for k,v in rl.items() if isinstance(k,tuple)]}')
L.append('\n## H6 verdict\n')
if rows:
    for c, r, lo, hi, mT, m0, sl in rows:
        L.append(f'- {c}: teacher median {mT:.0f} -> residual PPO median {m0:.0f} ({r*100:+.0f}%, paired CI [{lo*100:+.0f}%, {hi*100:+.0f}%]); per-seed {[round(x*100) for x in sl]}%. Point estimate >= +30%: {"YES" if r >= .3 else "no"}; CI lower bound > 0: {"YES" if lo > 0 else "no"}.')
    ok = [c for c, r, lo, hi, *_ in rows if r >= .3 and lo > 0]
    L.append(f'\n**H6 {"SUPPORTED" if ok else "NOT supported"}** (rule: in at least one error condition residual-PPO median >= teacher median x 1.3, and the paired-bootstrap CI of the relative gain excludes 0). Conditions meeting it: {ok}.')
open('results/spike03b/table.md', 'w').write('\n'.join(L)); print('\n'.join(L))
