"""Generates results/spike14a/table.md and summary.json from pipe_<set>.json, est files and raw data. usage: report14.py   (selection of best learned / best classical is read from results/spike14a/selection.json, written by select14.py from CAL only)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
LV = {'L-mlp1 (spike10b L-est-full)': [f'lmlp1_s{s}' for s in range(3)], 'L-mlp1 spike13b original run (holdout only)': [f'lmlp1_13b_s{s}' for s in range(3)], 'L-gru1': [f'gru1_s{s}' for s in range(3)], 'L-gru4': [f'gru4_s{s}' for s in range(3)], 'L-mlp4': [f'mlp4_s{s}' for s in range(3)],
      'L-dec1 a0.9': [f'dec1_a0.9_s{s}' for s in range(3)], 'L-dec4 a0.9': [f'dec4_a0.9_e38_s{s}' for s in range(3)], 'L-ens(dec4)': ['ens_dec4'], 'L-ens(gru4)': ['ens_gru4']}
CV = {'C0 T-adapt3-full (=R0)': ['C0'], 'C-ms10 (K=10 step-ahead)': ['C_ms10'], 'C-map1-ms (MAP + multi-start)': ['C_map1_ms'], 'HYB (dec4-ensemble init + MAP polish)': ['C_hyb']}
REF = {'R1 true theta': ['R1']}
ALL = {**LV, **CV, **REF}
def load_pipe(st): f = os.path.join(OUT14, f'pipe_{st}.json'); return json.load(open(f)) if os.path.exists(f) else {}
def per_ep(P, comps, st=None):
    if not all(c in P for c in comps): return None
    return np.mean([np.array(P[c]['success'], float) for c in comps], 0)
def boot_diff(a, b, n=20000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float); r = np.random.default_rng(seed); bs = d[r.integers(0, len(d), (n, len(d)))].mean(1); return 100 * d.mean(), 100 * float(np.quantile(bs, .025)), 100 * float(np.quantile(bs, .975))
def ci(a, **k): m, lo, hi = boot_diff(a, np.zeros_like(a), **k); return f'{m:.1f} [{lo:.1f}, {hi:.1f}]'
def perr(st, comps):
    D = pickle.load(open(os.path.join(DAT, f'raw_{st}.pkl'), 'rb')); fs = [os.path.join(DAT, 'est', f'{c}_{st}.npy') for c in comps]
    if not all(os.path.exists(f) for f in fs): return None
    rs = [err_summary(D['TH'], np.load(f)) for f in fs]; return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
def main():
    sel = json.load(open(os.path.join(OUT14, 'selection.json'))) if os.path.exists(os.path.join(OUT14, 'selection.json')) else None; L = []; S = {}
    pipes = {st: load_pipe(st) for st in ('cal', 'hold', 'new')}
    for st, title in (('cal', 'CAL (1000-1049 + 2000-2149, n=200): selection set'), ('hold', 'Holdout 1100-1599 (n=500)'), ('new', 'New seeds 701000-701199 (n=200)')):
        P = pipes[st]; R0 = per_ep(P, ['C0'], st); R1 = per_ep(P, ['R1'], st)
        L += [f'## {title}', '', '| method | success % [95% CI] | vs R0 (pp [CI]) | recovered fraction of R0->R1 | unstable ep % | param RMS: ls / mass(link) / log-damp / fric(1e-3Nm) |', '|---|---|---|---|---|---|']
        for nm, comps in ALL.items():
            v = per_ep(P, comps, st)
            if v is None: continue
            vs = boot_diff(v, R0) if R0 is not None and nm != 'C0 T-adapt3-full (=R0)' else None; rec = ''
            if R0 is not None and R1 is not None and nm not in ('R1 true theta',) and nm != 'C0 T-adapt3-full (=R0)':
                den = (R1 - R0).mean(); rec = f'{(v - R0).mean() / den:.2f}' if abs(den) > 1e-9 else ''
            pe = perr(st, comps); pes = f"{pe['ls']:.3f} / {pe['lm_rms_link']:.3f} / {pe['ld']:.3f} / {pe['fr']:.2f}" if pe else ''
            inst = 100 * np.mean([np.mean(np.array(P[c]['inst']) > 0) for c in comps]); S[f'{st}|{nm}'] = dict(mean=100 * float(v.mean()), vs_R0=vs, rec=rec, perr=pe)
            L.append(f"| {nm} | {ci(v)} | {'-' if vs is None else f'{vs[0]:+.1f} [{vs[1]:+.1f}, {vs[2]:+.1f}]'} | {rec} | {inst:.1f} | {pes} |")
        L.append('')
        if st != 'cal':
            L += [f'Per training seed ({st}):', '', '| model | success % | param RMS ld |', '|---|---|---|']
            for nm, comps in LV.items():
                for c in comps:
                    if c in P and len(comps) > 1: pe = perr(st, [c]); L.append(f"| {c} | {100*np.mean(P[c]['success']):.1f} | {'' if pe is None else round(pe['ld'], 3)} |")
            L.append('')
    L += ['## Dose curve: training data x1 (12000 probe episodes) vs x4 (48000); label error = mean over the 3 training seeds, success = mean over the seeds that were run through the pipeline (n_seeds given)', '', '| model family | data | holdout RMS ls / mass(link) / log-damp / fric | holdout success % (n_seeds) | new-seed success % (n_seeds) | CAL RMS log-damp |', '|---|---|---|---|---|---|']
    for fam, a, b in (('MLP (spike10b recipe)', 'lmlp1', 'mlp4'), ('GRU, plain MSE', 'gru1', 'gru4'), ('GRU, decision-weighted loss', 'dec1_a0.9', 'dec4_a0.9_e38')):
        for lab, tg in (('x1', a), ('x4', b)):
            comps = [f'{tg}_s{k}' for k in range(3)]; pe = perr('hold', comps); pc = perr('cal', comps); cells = []
            for st in ('hold', 'new'):
                P = pipes[st]; have = [c for c in comps if c in P]
                cells.append(f"{100*np.mean([np.mean(P[c]['success']) for c in have]):.1f} ({len(have)})" if have else '-')
            pes = '' if pe is None else '%.3f / %.3f / %.3f / %.2f' % (pe['ls'], pe['lm_rms_link'], pe['ld'], pe['fr'])
            L.append(f"| {fam} | {lab} | {pes} | {cells[0]} | {cells[1]} | {'' if pc is None else round(pc['ld'], 3)} |")
    L.append('')
    if sel:
        L += ['## Primary judgement (rule fixed in README section 0)', '']
        for st in ('hold', 'new'):
            P = pipes[st]; a = per_ep(P, ALL[sel['learned']], st); b = per_ep(P, ALL[sel['classical']], st)
            if a is None or b is None: continue
            d = boot_diff(a, b); S[f'primary|{st}'] = d; L.append(f"* {st}: best learned [{sel['learned']}] - best classical [{sel['classical']}] = {d[0]:+.1f} pp [{d[1]:+.1f}, {d[2]:+.1f}]  (learned {100*a.mean():.1f}, classical {100*b.mean():.1f})")
        h = S.get('primary|hold')
        if h:
            win = 'LEARNED WINS' if (h[0] >= 10 and h[1] > 0) else ('CLASSICAL WINS' if (h[0] <= -10 and h[2] < 0) else 'no side wins at the 10 pp bar')
            n_ = S.get('primary|new'); conf = ''
            if n_ and win != 'no side wins at the 10 pp bar': conf = ' (confirmed on new seeds)' if np.sign(n_[0]) == np.sign(h[0]) else ' (NOT confirmed on new seeds)'
            L += ['', f'**Verdict: {win}{conf}**']; S['verdict'] = win + conf
    open(os.path.join(OUT14, 'table.md'), 'w').write('# spike14a tables (generated by experiments/spike14a/report14.py)\n\n' + '\n'.join(L) + '\n'); json.dump(S, open(os.path.join(OUT14, 'summary.json'), 'w'), indent=1, default=float); print('\n'.join(L))
main()
