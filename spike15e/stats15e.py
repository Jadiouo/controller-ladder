"""spike15e statistics (v9 rules). usage: stats15e.py hold  -> reproduction check (writes results/spike15e/repro_hold.json and .scratch/s15e/repro_ok iff both within 1 pp)
                                       stats15e.py testd -> table.md + testd_summary.json + testd_episodes.json in results/spike15e. Paired percentile bootstrap, 20000 resamples, one shared resample index for every contrast (as stats15.py)."""
import sys, os, json, pickle, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, '..')); sys.path.insert(0, os.path.join(ROOT, 'spike15c')); import analyzeB as AB
MODE = sys.argv[1]; S15E = os.path.join(ROOT, '.scratch/s15e'); SD = os.path.join(S15E, os.environ.get('SUB', MODE)); RES = os.environ.get('RESDIR', os.path.join(ROOT, 'results/spike15e')); os.makedirs(RES, exist_ok=True)
def load(name):
    B = pickle.load(open(os.path.join(SD, name + '.pkl'), 'rb')); r = B['res']; C = min(len(x['succ']) for x in r)
    return dict(S=np.stack([x['S'][:C] for x in r]), succ=np.stack([x['succ'][:C] for x in r]), base=np.array([x['base'] for x in r]), members=B['members'], seeds=np.array(B['seeds']), cpu=np.array([x['cpu_sel'] for x in r]), wall=B['wall'])
DB, DL = load('B'), load('LAP'); n = len(DB['base']); refs = json.load(open(os.path.join(SD, 'refs.json'))); R0 = np.array(refs['adapt3_full']['success'], float); R1 = np.array(refs['R1']['success'], float)
assert list(DB['seeds']) == list(DL['seeds'])
Lens = DB['succ'][np.arange(n), DB['base']]; Bs, _ = AB.vsucc(DB, ('c11', 'M3c', 'max')); Cs, chC = AB.vsucc(DL, ('c11', 'M3c', 'hit06')); Ctl, _ = AB.vsucc(DL, ('c11', 'center', 'mean'))
base_eq_R0 = float(np.mean(DL['succ'][np.arange(n), DL['base']] == R0)); Lbase_eq = float(np.mean(DL['succ'][np.arange(n), DL['base']] == R0))
if MODE == 'hold':
    out = dict(n=n, B=100 * Bs.mean(), Cens=100 * Cs.mean(), R0=100 * R0.mean(), R1=100 * R1.mean(), Lens=100 * Lens.mean(), ref_B=86.6, ref_Cens=79.6, ref_R0=69.2, ref_R1=86.0, ref_Lens=71.0, base_equals_R0_frac=base_eq_R0)
    old = pickle.load(open(os.path.join(ROOT, '.scratch/s15c/B_hold.pkl'), 'rb')); oldS = np.stack([x['S'][:, :4] for x in old['res']]); oldsucc = np.stack([x['succ'][:11] for x in old['res']])
    ob = AB.choose(oldS, 11, [0, 1, 2, 3], 'max'); out['B_episode_agreement_with_spike15c'] = float(np.mean(oldsucc[np.arange(n), ob] == Bs))
    oldD = pickle.load(open(os.path.join(ROOT, '.scratch/s15d/D_LAP_hold.pkl'), 'rb')); oS = np.stack([x['S'][:, :4] for x in oldD['res']]); osu = np.stack([x['succ'][:11] for x in oldD['res']]); oc = AB.choose(oS, 11, [0, 1, 2, 3], 'hit06')
    out['Cens_episode_agreement_with_spike15d'] = float(np.mean(osu[np.arange(n), oc] == Cs))
    out['diff_B_pp'] = out['B'] - 86.6; out['diff_Cens_pp'] = out['Cens'] - 79.6; out['pass'] = bool(abs(out['diff_B_pp']) <= 1.0 and abs(out['diff_Cens_pp']) <= 1.0 and n == 500)
    json.dump(out, open(os.path.join(RES, 'repro_hold.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
    if out['pass']: open(os.path.join(S15E, 'repro_ok'), 'w').write(json.dumps(out))
    sys.exit(0)
# ================================================================ TEST-D
B9 = load('B9'); B9s, _ = AB.vsucc(B9, ('c11', 'M9', 'mean'))
BN = 20000; IDX = np.random.default_rng(15).integers(0, n, (BN, n)); bs = lambda x: x[IDX].mean(1)
def ci(x, lev=95): b = bs(x); a = (100 - lev) / 2; return 100 * x.mean(), 100 * np.percentile(b, a), 100 * np.percentile(b, 100 - a)
def fmt(t): return f'{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]'
def fmtp(x): m, lo, hi = ci(x); return f'{m:.1f} [{lo:.1f}, {hi:.1f}]'
def p_le(x, thr): return max((bs(x) * 100 <= thr).mean(), 1 / (BN + 1))
def p_ge(x, thr): return max((bs(x) * 100 >= thr).mean(), 1 / (BN + 1))
dC, dB, dBC = Cs - R0, Bs - R0, Bs - Cs
cC, cB, cBC = ci(dC), ci(dB), ci(dBC)
v15 = cC[1] > 0; v15b = (cB[0] >= 10) and (cB[1] > 0); v16a = cBC[1] > 0; v16b = cBC[2] < 10
p = {'H15': p_le(dC, 0.0), 'H15b': p_le(dB, 0.0), 'H16a': p_le(dBC, 0.0), 'H16b': p_ge(dBC, 10.0)}
order = sorted(p, key=p.get); adj = {}; run = 0.0
for i, k in enumerate(order): run = max(run, min(1.0, (len(p) - i) * p[k])); adj[k] = run
V = {'H15': v15, 'H15b': v15b, 'H16a': v16a, 'H16b': v16b}; surv = {k: bool(V[k] and adj[k] < 0.05) for k in V}
if surv['H16a'] and surv['H16b']: sentence = '學習式估計器有小而真實的優勢，未達本文 10 pp 門檻'
elif surv['H16b']: sentence = '未偵測到學習式優勢'
else: sentence = '學習式優勢可能 ≥10 pp，論文照實改寫 K2'
# if only unadjusted (not Holm) support, also state it
sent_unadj = ('學習式估計器有小而真實的優勢，未達本文 10 pp 門檻' if (v16a and v16b) else ('未偵測到學習式優勢' if v16b else '學習式優勢可能 ≥10 pp，論文照實改寫 K2'))
M = {'R0 T-adapt3-full': R0, 'R1 true theta': R1, 'L-ens (3 frozen L-dec4, mean theta, base pick)': Lens, 'B  c11/M3c/max (learned)': Bs, 'C-ens LAP/c11/M3c/hit06 (classical)': Cs, 'B9 c11/M9/mean (descriptive)': B9s, 'R0 + c11/centre LAP (descriptive)': Ctl}
den = bs(R1 - R0)
def recf(x):
    nb = bs(x - R0); ok = np.abs(den) > 1e-9; r = nb[ok] / den[ok]; return (x - R0).mean() / (R1 - R0).mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)
cpuB = DB['cpu']; cpuC = DL['cpu']; cls = pickle.load(open(os.path.join(SD, 'cls.pkl'), 'rb')); infer = json.load(open(os.path.join(SD, 'infer.done')))['wall']
cpu_cls_ep = cls['cpu'] / n
L = [f'# spike15e TEST-D table (locked set seeds 40000-40499; n={n}; generated by spike15e/stats15e.py)', '', '## Verdicts (rules of hypothesis card v9, applied verbatim)', '',
     '| hypothesis | statistic | CI | rule | verdict (unadjusted) | one-sided bootstrap p | Holm-adjusted p | survives Holm (alpha 0.05) |', '|---|---|---|---|---|---|---|---|',
     f'| H15 robust selection works for classical | C-ens - R0 = {cC[0]:+.1f} pp | 95% [{cC[1]:+.1f}, {cC[2]:+.1f}] | lower > 0 | **{"SUPPORTED" if v15 else "NOT supported"}** | {p["H15"]:.5f} (H0: diff<=0) | {adj["H15"]:.5f} | {"yes" if surv["H15"] else "no"} |',
     f'| H15b robust selection + learned estimator eats headroom | B - R0 = {cB[0]:+.1f} pp | 95% [{cB[1]:+.1f}, {cB[2]:+.1f}] | point >= +10 and lower > 0 | **{"SUPPORTED" if v15b else "NOT supported"}** | {p["H15b"]:.5f} (H0: diff<=0, as H13 in v8) | {adj["H15b"]:.5f} | {"yes" if surv["H15b"] else "no"} |',
     f'| H16a learned advantage | B - C-ens = {cBC[0]:+.1f} pp | 95% [{cBC[1]:+.1f}, {cBC[2]:+.1f}] | lower > 0 | **{"SUPPORTED" if v16a else "NOT supported"}** | {p["H16a"]:.5f} (H0: diff<=0) | {adj["H16a"]:.5f} | {"yes" if surv["H16a"] else "no"} |',
     f'| H16b learned advantage < 10 pp | B - C-ens = {cBC[0]:+.1f} pp | 95% [{cBC[1]:+.1f}, {cBC[2]:+.1f}] | upper < +10 | **{"SUPPORTED" if v16b else "NOT supported"}** | {p["H16b"]:.5f} (H0: diff>=10) | {adj["H16b"]:.5f} | {"yes" if surv["H16b"] else "no"} |', '',
     'Holm: step-down on the four one-sided bootstrap p-values (family of 4), family-wise alpha = 0.05 (v8.1 convention, same code as stats15.py). Unadjusted p floor = 1/(20000+1). H15b p uses H0: diff<=0 (the v8 H13 precedent); P*(B-R0 <= 10) = ' + f'{p_le(dB, 10.0):.5f} (descriptive).', '',
     f'**Interpretation sentence (v9, pre-written; applied on Holm-surviving verdicts): {sentence}.** (Unadjusted verdicts alone would give: {sent_unadj}.)', '',
     '## Success rate per method (95% bootstrap CI, %)', '', '| method | success % [95% CI] | vs R0 pp [95% CI] | recovered (X-R0)/(R1-R0) [95% CI] |', '|---|---|---|---|']
for k, v in M.items():
    r = recf(v) if not k.startswith(('R0 T', 'R1')) else None
    L.append(f'| {k} | {fmtp(v)} | ' + ('-' if k.startswith('R0 T') else fmt(ci(v - R0))) + ' | ' + (f'{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]' if r else '') + ' |')
L += ['', '## Further paired contrasts (pp [95% CI]; descriptive)', '', '| contrast | diff |', '|---|---|']
for a, x in (('B - L-ens', Bs - Lens), ('C-ens - L-ens', Cs - Lens), ('L-ens - R0', Lens - R0), ('B9 - R0', B9s - R0), ('B9 - C-ens', B9s - Cs), ('B9 - B', B9s - Bs), ('R0+c11/centre LAP - R0', Ctl - R0), ('R1 - R0', R1 - R0)): L.append(f'| {a} | {fmt(ci(x))} |')
c90 = ci(dBC, 90); L += ['', f'B - C-ens 90% CI [{c90[1]:+.1f}, {c90[2]:+.1f}].', '',
     '## CPU seconds per episode (mean over 500 episodes; process CPU of the selection pipeline)', '',
     f'* B: planning + scoring in centre and 3 member models + 3 member-own plans (first, ordinary `_swing_start`): {cpuB.mean():.1f} s (median {np.median(cpuB):.1f}); plus L-dec4 inference (9 nets incl. descriptive ones, wall {infer:.0f} s for the whole set, {infer/n:.2f} s/episode wall, 3 nets would be ~1/3).',
     f'* C-ens: Laplace ensemble fit (12-start R0 fit + Jacobian posterior, `cls15d.ens_fit`) {cpu_cls_ep:.1f} s + same planning/scoring {cpuC.mean():.1f} s (median {np.median(cpuC):.1f}) = {cpu_cls_ep + cpuC.mean():.1f} s.',
     f'* B total with inference approx. {cpuB.mean() + infer/n/3:.1f} s (3 nets share of inference wall). Not counted: the R0 baseline episode itself, and the offline execution of all 11 candidates.',
     f'* Stage wall (8 workers): B collect {DB["wall"]:.0f} s, LAP collect {DL["wall"]:.0f} s, cls fits {cls["wall"]:.0f} s (cpu {cls["cpu"]:.0f} s).', '',
     f'## Controls', '', f'* base candidate of the LAP run reproduces the independent R0 episode outcome in {100*base_eq_R0:.1f} % of episodes (B-run base candidate vs L-ens by construction).', f'* C-ens changes the R0 base pick in {100*np.mean(chC != DL["base"]):.0f} % of episodes; ceiling (oracle best executed candidate): B-run {100*DB["succ"].max(1).mean():.1f} %, LAP-run {100*DL["succ"].max(1).mean():.1f} %.']
open(os.path.join(RES, 'table.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
json.dump(dict(n=n, verdict=V, p=p, holm_adj=adj, survives_holm=surv, sentence=sentence, sentence_unadjusted=sent_unadj, ci=dict(H15=cC, H15b=cB, BminusC=cBC), means={k: 100 * float(v.mean()) for k, v in M.items()}, cpu=dict(B=float(cpuB.mean()), Cens_sel=float(cpuC.mean()), Cens_fit=float(cpu_cls_ep)), base_eq_R0=base_eq_R0), open(os.path.join(RES, 'testd_summary.json'), 'w'), indent=1, default=float)
json.dump(dict(seeds=[int(s) for s in DB['seeds']], **{k: [float(x) for x in v] for k, v in {'R0': R0, 'R1': R1, 'L_ens': Lens, 'B': Bs, 'C_ens': Cs, 'B9': B9s, 'R0_c11_centre_LAP': Ctl}.items()}, cpu_sel_B=[float(x) for x in cpuB], cpu_sel_Cens=[float(x) for x in cpuC]), open(os.path.join(RES, 'testd_episodes.json'), 'w'))
