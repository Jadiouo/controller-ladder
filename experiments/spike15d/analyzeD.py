"""spike15d offline analysis (pure numpy on stored per-episode matrices). usage: analyzeD.py select   (CAL only -> results/spike15d/D_selection.json, BEFORE hold/new are looked at)
                                                                  analyzeD.py report   (needs selection -> results/spike15d/D_table.md, D_summary.json)"""
import sys, os, json, pickle, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', 'spike15c')); import analyzeB as AB
ROOT = os.path.abspath(os.path.join(H, '..', '..')); S15C = os.path.join(ROOT, '.scratch/s15c'); S15D = os.path.join(ROOT, '.scratch/s15d'); RES = os.path.join(ROOT, 'results/spike15d')
def _load(p):
    B = pickle.load(open(p, 'rb')); r = B['res']; C = min(len(x['succ']) for x in r)
    return dict(S=np.stack([x['S'][:C] for x in r]), succ=np.stack([x['succ'][:C] for x in r]), base=np.array([x['base'] for x in r]), members=B['members'], seeds=np.array(B['seeds']))
def loadF(fam, SET):
    if fam == 'B': return _load(os.path.join(S15C, f'B_{SET}.pkl') if SET != 'new' else os.path.join(S15D, 'B_new.pkl'))
    return _load(os.path.join(S15D, f'D_{fam}_{SET}.pkl'))
def refs(SET): return json.load(open(os.path.join(S15C, f'refs_{SET}.json') if SET != 'new' else os.path.join(S15D, 'refs_new.json')))
FAMS = ['MS', 'LAP']; ORDER = [(f, p, 'M3c', a) for f in FAMS for p in ('c8', 'c11') for a in AB.AGGS]; CTRL = [(f, 'c11', 'center', 'mean') for f in FAMS]; ALLV = ORDER + CTRL
def vs(D, v): return AB.vsucc(D, v[1:])
def vn(v): return '/'.join(v)
def ci(x, q=(2.5, 97.5), n=20000): 
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1); return 100 * np.percentile(b, q[0]), 100 * np.percentile(b, q[1])
def d_ci(a, b, n=20000):
    d = np.asarray(a, float) - np.asarray(b, float); r = np.random.default_rng(0); m = d[r.integers(0, len(d), (n, len(d)))].mean(1); q = np.percentile(m, [2.5, 5, 95, 97.5]) * 100
    return dict(d=100 * d.mean(), lo95=q[0], hi95=q[3], lo90=q[1], hi90=q[2])
if __name__ == '__main__':
    mode = sys.argv[1]; os.makedirs(RES, exist_ok=True)
    if mode == 'select':
        DD = {f: loadF(f, 'cal') for f in FAMS}; rows = []
        for v in ALLV: s, ch = vs(DD[v[0]], v); rows.append((float(s.mean()), v, float((ch != DD[v[0]]['base']).mean())))
        order = {v: i for i, v in enumerate(ALLV)}; cand = [r for r in rows if r[1] in ORDER]; cand.sort(key=lambda r: (-r[0], order[r[1]])); sel = cand[0]
        base = {f: float(DD[f]['succ'][np.arange(len(DD[f]['base'])), DD[f]['base']].mean()) for f in FAMS}
        json.dump(dict(rule='highest CAL (n=200) success among the 20 ensemble variants (2 families x {c8,c11} x M3c x 5 aggregators); ties -> listed order; 2 c11/center/mean controls are descriptive', n_variants=len(ALLV), selected=vn(sel[1]), selected_family=sel[1][0], selected_cal=sel[0], selected_changed_frac=sel[2],
                       cal_base_R0_candidate=base, all=[(vn(r[1]), r[0], r[2]) for r in rows], top5=[(vn(r[1]), r[0]) for r in cand[:5]]), open(os.path.join(RES, 'D_selection.json'), 'w'), indent=1)
        print('selected', vn(sel[1]), sel[0], 'bases', base); [print(vn(r[1]), round(r[0], 3), round(r[2], 2)) for r in rows]
    else:
        S = json.load(open(os.path.join(RES, 'D_selection.json'))); sv = tuple(S['selected'].split('/')); fam = sv[0]; Bv = ('c11', 'M3c', 'max'); out = {}; L = [f"# spike15d: B (learned L-dec4, c11/M3c/max) vs classical control\n\nC-ens (CAL-selected) = **{S['selected']}** (CAL {100*S['selected_cal']:.1f} %); {S['n_variants']} variants incl. 2 controls.\n"]
        for SET, nm in (('hold', 'holdout 1100-1599 n=500'), ('new', 'new seeds 704000-704199 n=200'), ('cal', 'CAL n=200 (selection set)')):
            DB = loadF('B', SET); DC = {f: loadF(f, SET) for f in FAMS}; R = refs(SET); n = len(DB['base']); r0 = np.array(R['adapt3_full']['success'], float); r1 = np.array(R['R1']['success'], float)
            Lens = DB['succ'][np.arange(n), DB['base']]; Bs, _ = AB.vsucc(DB, Bv); Cs, ch = vs(DC[fam], sv); Ctl = {f: vs(DC[f], (f, 'c11', 'center', 'mean'))[0] for f in FAMS}
            ctrl_chk = {f: float(np.mean(DC[f]['succ'][np.arange(n), DC[f]['base']] == r0)) for f in FAMS}      # base candidate reproduces R0 episode?
            o = dict(n=n, R0=r0.mean(), R1=r1.mean(), Lens=Lens.mean(), B=Bs.mean(), Cens=Cs.mean(), ctrl={f: Ctl[f].mean() for f in FAMS}, base_equals_R0_frac=ctrl_chk, ci_mean={k: ci(v) for k, v in dict(R0=r0, R1=r1, Lens=Lens, B=Bs, Cens=Cs).items()})
            o['B_minus_Cens'] = d_ci(Bs, Cs); o['Cens_minus_R0'] = d_ci(Cs, r0); o['B_minus_R0'] = d_ci(Bs, r0); o['B_minus_Lens'] = d_ci(Bs, Lens)
            o['rec'] = {k: AB.rec(v, r0, r1) for k, v in dict(Lens=Lens, B=Bs, Cens=Cs, **{f'ctl_{f}': Ctl[f] for f in FAMS}).items()}
            o['Cens_changed_frac'] = float((ch != DC[fam]['base']).mean())
            o['all'] = {vn(v): float(vs(DC[v[0]], v)[0].mean()) for v in ALLV}
            if len(DC['LAP']['members']) >= 9:
                o['LAP_M9c'] = {a: dict(succ=float(AB.vsucc(DC['LAP'], ('c11', 'M9c', a))[0].mean()), vs_B=d_ci(Bs, AB.vsucc(DC['LAP'], ('c11', 'M9c', a))[0])) for a in AB.AGGS}
            out[SET] = o
            L.append(f'\n## {nm}\n\n| method | success % [95% CI] | vs R0 pp | recovered (X-R0)/(R1-R0) [CI] |\n|---|---|---|---|')
            def row(lab, v, key):
                dd = d_ci(v, r0); rr = o['rec'].get(key); L.append(f"| {lab} | {100*np.mean(v):.1f} [{ci(v)[0]:.1f}, {ci(v)[1]:.1f}] | " + ('-' if key == 'R0' else f"{dd['d']:+.1f} [{dd['lo95']:+.1f}, {dd['hi95']:+.1f}]") + ' | ' + (f"{rr[0]:.2f} [{rr[1]:.2f}, {rr[2]:.2f}]" if rr else '') + ' |')
            row('R0 T-adapt3-full', r0, 'R0'); row('R1 true theta', r1, 'R1'); row('L-ens (3 L-dec4 nets, base pick)', Lens, 'Lens'); row('**B c11/M3c/max (learned, fixed)**', Bs, 'B'); row(f"**C-ens {S['selected']} (CAL-selected)**", Cs, 'Cens')
            for f in FAMS: row(f'R0 + c11/centre/mean, family {f} (no ensemble)', Ctl[f], f'ctl_{f}')
            d = o['B_minus_Cens']; L.append(f"\n**B - C-ens = {d['d']:+.1f} pp, 95 % CI [{d['lo95']:+.1f}, {d['hi95']:+.1f}], 90 % CI [{d['lo90']:+.1f}, {d['hi90']:+.1f}]**; C-ens - R0 = {o['Cens_minus_R0']['d']:+.1f} [{o['Cens_minus_R0']['lo95']:+.1f}, {o['Cens_minus_R0']['hi95']:+.1f}]; B - R0 = {o['B_minus_R0']['d']:+.1f} [{o['B_minus_R0']['lo95']:+.1f}, {o['B_minus_R0']['hi95']:+.1f}]. Base candidate == R0 outcome fraction: {ctrl_chk}. C-ens changes the R0 pick in {100*o['Cens_changed_frac']:.0f} % of episodes.\n")
            L.append('| variant | success % |\n|---|---|'); [L.append(f'| {k} | {100*x:.1f} |') for k, x in o['all'].items()]
            if 'LAP_M9c' in o: L.append('\nLAP M9c (descriptive, 9 members) c11:\n\n| aggregator | success % | B - this pp [95% CI] |\n|---|---|---|'); [L.append(f"| {a} | {100*x['succ']:.1f} | {x['vs_B']['d']:+.1f} [{x['vs_B']['lo95']:+.1f}, {x['vs_B']['hi95']:+.1f}] |") for a, x in o['LAP_M9c'].items()]
        h, nw = out['hold']['B_minus_Cens'], out['new']['B_minus_Cens']
        win = all(x['d'] >= 10 and x['lo95'] > 0 for x in (h, nw)); tie = all(x['lo90'] > -10 and x['hi90'] < 10 for x in (h, nw)); w5 = all(x['lo90'] > -5 and x['hi90'] < 5 for x in (h, nw))
        verdict = 'LEARNED ESTIMATOR WINS under robust selection' if win else ('CLASSICAL CONTROL CATCHES UP (TOST +-10 pp)' if tie else 'UNCERTAIN'); out['verdict'] = dict(verdict=verdict, win=bool(win), tie10=bool(tie), within5=bool(w5))
        L.insert(2, f"\n**Verdict (pre-registered rule): {verdict}.** holdout B - C-ens {h['d']:+.1f} (95 % [{h['lo95']:+.1f}, {h['hi95']:+.1f}], 90 % [{h['lo90']:+.1f}, {h['hi90']:+.1f}]); new {nw['d']:+.1f} (95 % [{nw['lo95']:+.1f}, {nw['hi95']:+.1f}], 90 % [{nw['lo90']:+.1f}, {nw['hi90']:+.1f}]). Both 90 % CIs inside +-5 pp: {w5}.\n")
        json.dump(out, open(os.path.join(RES, 'D_summary.json'), 'w'), indent=1, default=float); open(os.path.join(RES, 'D_table.md'), 'w').write('\n'.join(L)); print('\n'.join(L[:14]))
