"""Report for spike10b -> results/spike10b/table.md. Reads raw_TESTB.pkl, raw_TEST.pkl, holdout1.json (+ holdout_extra.json), dose.json, final_methods.json, lest_train.json, tune_*.json, ident_cal.json, step0_nan_subf*.json, budget.json."""
import sys, os, json, pickle, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); OUT = os.environ.get('REPORT_OUT', os.path.join(ROOT, 'results/spike10b')); NB = 20000
def J(p, d=None): return json.load(open(os.path.join(OUT, p))) if os.path.exists(os.path.join(OUT, p)) else d
RB = pickle.load(open(os.path.join(OUT, 'raw_TESTB.pkl'), 'rb')); RT = pickle.load(open(os.path.join(OUT, 'raw_TEST.pkl'), 'rb')); best = RB['_best']; BN = f'T-{best}'
def boot(x, seed=0): x = np.asarray(x, float); r = np.random.default_rng(seed); return x[r.integers(0, len(x), (NB, len(x)))].mean(1)
def ci(x, seed=0): b = boot(x, seed); return float(np.quantile(b, .025)), float(np.quantile(b, .975))
def pval(d, seed=1): b = boot(d, seed); return float(min(1.0, max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0 / NB)))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(o): run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(run, 1.0)
    return adj
pp = lambda x: f'{100*x:.1f}'
FAM = {'L-est-full': 'L-est-full', 'L-est-eff': 'L-est-eff', 'L-est-OOD': 'L-est-ood', 'R-PPO-hist': 'R-PPO-hist'}
def per_seed(R, name, key='success'):
    """per-test-seed mean over training seeds for multi-seed families"""
    if name in FAM:
        ks = sorted(k for k in R if k.startswith(FAM[name] + '#') or k.startswith(FAM[name].replace('L-est-ood', 'L-est-ood') + '#'))
        ks = [k for k in R if k.lower().startswith(FAM[name].lower() + '#')]
        return np.mean([R[k][key] for k in sorted(ks)], 0) if ks else None
    return R[name][key] if name in R else None
CL = ['T-nom', 'T-robust', 'T-adapt', 'T-adapt2', 'T-adapt3_grad', 'T-adapt3_full', 'T-adapt3_ood', 'T-oracle']; LE = ['L-est-full', 'L-est-eff', 'L-est-OOD', 'R-PPO-hist']; ALL = [m for m in CL + LE if per_seed(RB, m) is not None]
L = []; A_ = L.append
FM = J('final_methods.json', {}); ppo_sel = FM.get('ppo_selected', {})
A_(f'# spike10b results (whip, realistic conditions). best-classical = {BN} (chosen on holdout 1100-1399, n=300)\n')
A_('Primary endpoint set: TEST-B seeds 20000-20499 (n=500, one locked run). Secondary: TEST seeds 0-49 (n=50, one locked run). Multi-seed methods: per-test-seed mean over the 3 training seeds, statistics over test seeds. Paired percentile bootstrap, 20000 resamples.\n')
def main_table(R, title):
    A_(f'## {title}\n'); A_('| method | success % [95% CI] | closest approach (m) | hit speed (m/s) | energy (Nm^2/step) | residual norm (Nm) | unstable-episode % (plant blow-up) | diverged % |'); A_('|---|---|---|---|---|---|---|---|')
    for m in ALL:
        s = per_seed(R, m); lo, hi = ci(s); hs = per_seed(R, m, 'hit_speed'); res = per_seed(R, m, 'res_mean') if m == 'R-PPO-hist' else None
        ks = [k for k in R if k.lower().startswith(FAM.get(m, m).lower() + '#')] if m in FAM else [m]; inst = 100 * np.mean([np.mean(R[k]['inst'] > 0) for k in ks]); dv = 100 * np.mean([np.mean(R[k]['diverged'] > 0) for k in ks])
        A_(f"| {m}{' (= best-classical)' if m == BN else ''} | {pp(s.mean())} [{pp(lo)}, {pp(hi)}] | {np.mean(per_seed(R, m, 'min_dist')):.4f} | {np.nanmean(hs):.2f} | {np.mean(per_seed(R, m, 'energy')):.1f} | {'-' if res is None else f'{res.mean():.3f}'} | {inst:.2f} | {dv:.2f} |")
    A_('')
main_table(RB, 'Main table, TEST-B (n=500)'); main_table(RT, 'Same table, TEST (n=50)')
A_('## Per training seed (success %, TEST-B | TEST)\n'); A_('| method | seed 0 | seed 1 | seed 2 |'); A_('|---|---|---|---|')
for m in LE:
    ks = sorted(k for k in RB if k.lower().startswith(FAM[m].lower() + '#'))
    if ks: A_(f'| {m} | ' + ' | '.join(f"{pp(RB[k]['success'].mean())} / {pp(RT[k]['success'].mean())}" for k in ks) + ' |')
def diffs(R, title):
    B = R[BN]['success']; A_(f'### {title}\n'); A_('| method | diff vs best-classical (pp) | 95% CI | p (boot) |'); A_('|---|---|---|---|')
    for m in ALL:
        if m == BN: continue
        d = per_seed(R, m) - B; lo, hi = ci(d, 2); A_(f'| {m} | {pp(d.mean())} | [{pp(lo)}, {pp(hi)}] | {pval(d):.4f} |')
    A_('')
A_('## Paired differences vs best-classical\n'); diffs(RB, 'TEST-B (n=500)'); diffs(RT, 'TEST (n=50)')
B = RB[BN]['success']
def verdict(R, names, title):
    Bq = R[BN]['success']; ds = [per_seed(R, n) - Bq for n in names]; ps = np.array([pval(d) for d in ds]); adj = holm(ps); ok = []
    A_(f'### {title}\n'); A_('| hypothesis | candidate | diff (pp) | 95% CI | p | Holm p | >= +10 pp and CI > 0 and Holm p < .05 |'); A_('|---|---|---|---|---|---|---|')
    for h, n, d, p_, a_ in zip(('H11', 'H12'), names, ds, ps, adj):
        lo, hi = ci(d, 2); o = bool(d.mean() >= 0.10 and lo > 0 and a_ < 0.05); ok.append(o); A_(f'| {h} | {n} | {pp(d.mean())} | [{pp(lo)}, {pp(hi)}] | {p_:.4f} | {a_:.4f} | {"yes: SUPPORTED" if o else "no: NOT SUPPORTED"} |')
    A_(''); return ok
A_('## Verdicts (primary endpoint: TEST-B; H11 and H12 Holm-corrected together)\n')
ok = verdict(RB, ['L-est-full', 'R-PPO-hist'] if per_seed(RB, 'R-PPO-hist') is not None else ['L-est-full'], 'TEST-B'); 
if per_seed(RT, 'R-PPO-hist') is not None: verdict(RT, ['L-est-full', 'R-PPO-hist'], 'TEST (n=50, reported alongside; the verdict uses TEST-B)')
A_(f'**H11: {"SUPPORTED" if ok[0] else "NOT SUPPORTED"}; H12: {"SUPPORTED" if len(ok) > 1 and ok[1] else "NOT SUPPORTED"} (TEST-B).**\n')
ceil = 1 - B.mean(); A_(f'Ceiling note: best-classical TEST-B success = {pp(B.mean())} %, largest conceivable gain {pp(ceil)} pp.\n')
A_('## Secondary comparisons on TEST-B\n'); A_('| comparison | diff (pp) | 95% CI | p |'); A_('|---|---|---|---|')
for a_, b_ in (('L-est-full', 'T-adapt3_grad'), ('L-est-full', 'T-adapt3_full'), ('L-est-full', 'L-est-eff'), ('L-est-OOD', 'T-adapt3_ood'), ('L-est-OOD', 'L-est-full'), ('T-adapt3_ood', 'T-adapt3_grad'), ('T-oracle', BN), ('T-oracle', 'L-est-full')):
    if per_seed(RB, a_) is None or per_seed(RB, b_) is None: continue
    d = per_seed(RB, a_) - per_seed(RB, b_); lo, hi = ci(d, 3); A_(f'| {a_} - {b_} | {pp(d.mean())} | [{pp(lo)}, {pp(hi)}] | {pval(d, 3):.4f} |')
A_('')
# holdout table
HO = J('holdout1.json', {}); HX = J('holdout_extra.json', {})
A_('## Holdout (seeds 1100-1399, n=300; selection set, so optimistic for the selected method)\n'); A_('| method | success % [95% CI] | unstable % |'); A_('|---|---|---|')
for n, v in {**HO, **HX}.items():
    x = np.array(v['success']); lo, hi = ci(x); A_(f"| {n} | {pp(x.mean())} [{pp(lo)}, {pp(hi)}] | {100*np.mean(np.array(v['inst'])>0):.2f} |")
A_('')
HALL = {**HO, **HX}
if HX and best in HALL:
    A_('### Holdout (n=300): learned methods (mean over 3 training seeds per holdout seed) vs best-classical, paired\n'); A_('| method | success % [95% CI] | diff vs best-classical (pp) | 95% CI | p |'); A_('|---|---|---|---|---|')
    Bh = np.array(HALL[best]['success'])
    for var, nm in (('lest_full', 'L-est-full'), ('lest_eff', 'L-est-eff'), ('lest_ood', 'L-est-OOD')):
        ks = [k for k in HX if k.startswith(var + '#')]
        if not ks: continue
        x = np.mean([HX[k]['success'] for k in ks], 0); d = x - Bh; lo, hi = ci(x); dlo, dhi = ci(d, 2); A_(f'| {nm} ({len(ks)} seeds) | {pp(x.mean())} [{pp(lo)}, {pp(hi)}] | {pp(d.mean())} | [{pp(dlo)}, {pp(dhi)}] | {pval(d):.4f} |')
        if var == 'lest_full' and 'adapt3_grad' in HALL:
            d2 = x - np.array(HALL['adapt3_grad']['success']); l2, h2 = ci(d2, 3); A_(f'| L-est-full minus T-adapt3_grad | | {pp(d2.mean())} | [{pp(l2)}, {pp(h2)}] | {pval(d2, 3):.4f} |')
    A_('')
# dose curve
DS = J('dose.json', {})
if DS:
    names = sorted({k.split('|')[0] for k in DS}); fam = {}
    for n in names: fam.setdefault(n.split('#')[0], []).append(n)
    def dget(f, c, key='success'):
        ks = [k for k in fam[f] if f'{k}|{c}' in DS]; return np.mean([DS[f'{k}|{c}'][key] for k in ks], 0) if ks else None
    Nn = len(next(iter(DS.values()))['success']); dn = lambda f: {'lest_full': 'L-est-full', 'lest_eff': 'L-est-eff', 'lest_ood': 'L-est-OOD'}.get(f, 'T-' + f)
    def dose_tables(levels, title):
        A_(f'## {title} (holdout seeds 1100-{1100+Nn-1}, n={Nn} per level; success %, 95% CI; seed-averaged for L-est)\n'); A_('| method | ' + ' | '.join(l for c, l in levels) + ' |'); A_('|---|' + '---|' * len(levels))
        for f in fam:
            row = []
            for c, l in levels:
                s_ = dget(f, c); row.append('-' if s_ is None else f'{pp(s_.mean())} [{pp(ci(s_)[0])}, {pp(ci(s_)[1])}]')
            A_(f'| {dn(f)}{" (= best-classical)" if f == best else ""} | ' + ' | '.join(row) + ' |')
        A_('\nPaired difference to best-classical (pp, 95% CI):\n'); A_('| method | ' + ' | '.join(l for c, l in levels) + ' |'); A_('|---|' + '---|' * len(levels))
        for f in fam:
            if f == best: continue
            row = []
            for c, l in levels:
                s_, b_ = dget(f, c), dget(best, c)
                if s_ is None or b_ is None: row.append('-'); continue
                d = s_ - b_; lo, hi = ci(d, 2); row.append(f'{pp(d.mean())} [{pp(lo)}, {pp(hi)}]')
            A_(f'| {dn(f)} | ' + ' | '.join(row) + ' |')
        A_('')
    dose_tables([('x0.5', 'x0.5 (all three factors)'), ('abc', 'x1 (all three)'), ('x2', 'x2 (all three)')], 'Dose curve')
    dose_tables([('a', 'a only: encoder noise'), ('b', 'b only: per-link mass + chain friction'), ('c', 'c only: actuator noise')], 'Single-factor table (each factor alone at x1)')
    A_('Interpretation rule (fixed in v7.1): if a learned method wins at x1 only, it is a single-point result, not a general conclusion.\n')
    A_('Unstable-episode % per method over all dose cells: ' + ', '.join(f"{n}: {100*np.mean([np.mean(np.array(DS[k]['inst'])>0) for k in DS if k.split('|')[0]==n]):.2f}" for n in names) + '\n')
# OOD table
A_('## OOD table (TEST-B, evaluation at x1; OOD variants trained/tuned on x0.5 and x2)\n'); A_('| method | train/tune distribution | TEST-B success % [CI] | TEST success % |'); A_('|---|---|---|---|')
for m, d_ in (('T-adapt3_full', 'x1 (CAL, in-distribution)'), ('T-adapt3_ood', 'x0.5 + x2 (CAL)'), ('L-est-full', 'x1 (training seeds, sim labels)'), ('L-est-eff', 'x1 (no per-link labels)'), ('L-est-OOD', 'x0.5 + x2 (training seeds, sim labels)')):
    if per_seed(RB, m) is None: continue
    s_ = per_seed(RB, m); lo, hi = ci(s_); A_(f'| {m} | {d_} | {pp(s_.mean())} [{pp(lo)}, {pp(hi)}] | {pp(per_seed(RT, m).mean())} |')
A_('')
# L-est training / identifiability / tuning
LT = J('lest_train.json'); ID = J('ident_cal.json')
if ID:
    A_('## T-adapt3 identifiability on CAL (default config, 50 seeds, condition abc; fraction of variance explained vs predicting the CAL mean)\n'); A_('| parametrisation | quantity | RMSE | prior SD | frac. variance explained |'); A_('|---|---|---|---|---|')
    for pm, v in ID.items():
        for q, e in v['errors'].items(): A_(f"| {pm} | {q} | {e['rmse']:.4g} | {e['prior_sd']:.4g} | {e['frac_var_explained']:.3f} |")
    A_('')
for t in ('grad', 'full', 'ood'):
    T_ = J(f'tune_{t}.json')
    if T_: A_(f"**tune_{t}**: episodes {T_['episodes']}, wall {T_['wall']:.0f}s, chosen CAL success {T_.get('chosen_cal')}, cfg `{json.dumps(T_.get('chosen'))}`\n")
if LT:
    A_('## L-est CAL errors (seed-0 model of the chosen config; RMSE in output units: ls, log m_i..., log ds, fric in 1e-3 Nm)\n')
    for v, e in LT['variants'].items(): A_(f"- {v}: chosen {e['chosen']['hid']}x{e['chosen']['hid']}, wd {e['chosen']['wd']}, epochs {e['chosen']['epochs']}; CAL z-MSE {e['chosen']['cal_mse_z']:.4f} (prior {e['cal_prior_mse_z']:.3f}); RMSE `{np.round(e['cal_rmse_phys'], 4).tolist()}`")
    A_('')
# PPO learned check
if ppo_sel:
    A_('## Did PPO learn? (CAL 1000-1049: selected checkpoint minus initialisation, need >= +5 pp)\n'); A_('| run | init CAL % | selected CAL % | diff pp | learned | residual norm at selected ckpt (Nm) |'); A_('|---|---|---|---|---|---|')
    for s, v in sorted(ppo_sel.items()): A_(f"| R-PPO-hist#{s} | {pp(v['init']['succ'])} | {pp(v['sel']['succ'])} (it {v['sel']['it']}) | {pp(v['sel']['succ']-v['init']['succ'])} | {'yes' if v['learned'] else 'NOT LEARNED'} | {v['sel']['res_mean']:.3f} |")
    A_('')
for fn, ttl in (('step0_nan_subf1.json', 'Step 0: plant blow-ups with the original integration step (SUBF=1, as spike09/10a), CAL n=50'), ('step0_nan_subf2.json', 'Step 0: integration step /2 (SUBF=2; used for CAL tuning), CAL n=50'), ('step0_nan_subf4.json', 'Step 0: integration step /4 (SUBF=4; used for holdout, TEST-B, TEST), CAL n=50')):
    S0 = J(fn)
    if S0:
        A_(f'### {ttl}\n'); A_('| method|cond | unstable episodes % | flagged diverged % | success % |'); A_('|---|---|---|---|')
        for k, v in S0.items(): A_(f"| {k} | {100*v['unstable_episodes']:.1f} | {100*v['diverged']:.1f} | {100*v['succ']:.0f} |")
        A_('')
CE = J('ceiling_cal.json')
if CE:
    A_('## Ceiling of the probe -> regress -> replay -> plan design (CAL 1000-1049, abc, n=50)\n'); A_('| estimator | success % [95% CI] |'); A_('|---|---|')
    for k, v in CE.items(): A_(f"| {k} | {pp(v['succ'])} [{pp(v['ci'][0])}, {pp(v['ci'][1])}] |")
    A_('')
IDC = J('ident_cal_chosen.json')
if IDC:
    A_('## T-adapt3 identifiability with the TUNED configs (CAL, 50 seeds, abc)\n'); A_('| parametrisation | quantity | RMSE | prior SD | frac. variance explained |'); A_('|---|---|---|---|---|')
    for pm, v in IDC.items():
        for q, e in v['errors'].items(): A_(f"| {pm} (T={v['cfg']['T']}) | {q} | {e['rmse']:.4g} | {e['prior_sd']:.4g} | {e['frac_var_explained']:.3f} |")
    A_('')
BG = J('budget.json')
if BG: A_('## Budget\n```\n' + json.dumps(BG, indent=1) + '\n```')
open(os.path.join(OUT, 'table.md'), 'w').write('\n'.join(L)); print('\n'.join(L)[:3000])
