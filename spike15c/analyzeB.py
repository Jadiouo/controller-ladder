"""spike15c approach B offline analysis. usage: analyzeB.py select   (CAL only -> results/spike15c/B_selection.json, written BEFORE holdout/new are looked at)
                                  analyzeB.py report   (needs selection.json; evaluates the selected + all variants on cal/hold/new -> results/spike15c/B_table.md, B_summary.json)
Pure numpy on the stored per-episode matrices (.scratch/s15c/B_<set>.pkl, refs_<set>.json)."""
import sys, os, json, pickle, itertools, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); S15C = os.path.join(ROOT, '.scratch/s15c'); RES = os.path.join(ROOT, 'results/spike15c'); TAG = os.environ.get('OUTTAG', 'B')
def load(SET):
    B = pickle.load(open(os.path.join(S15C, f'{os.environ.get("PKL", "B")}_{SET}.pkl'), 'rb')); r = B['res']; C = min(len(x['succ']) for x in r)
    S = np.stack([x['S'][:C] for x in r]); succ = np.stack([x['succ'][:C] for x in r]); base = np.array([x['base'] for x in r]); return dict(S=S, succ=succ, base=base, members=B['members'], seeds=np.array(B['seeds']))
def refs(SET):
    p = os.path.join(S15C, f'refs_{SET}.json'); return json.load(open(p)) if os.path.exists(p) else None
def colset(name, members):
    idx = {m: 1 + i for i, m in enumerate(members)}; m3 = [idx[m] for m in (0, 1, 2)]
    if name == 'center': return [0]
    if name == 'M3': return m3
    if name == 'M3c': return [0] + m3
    big = [idx[m] for m in members]
    if name == 'M9': return big
    if name == 'M9c': return [0] + big
def choose(S, pool, cols, agg):
    X = S[:, :pool, :][:, :, cols]; X = np.minimum(X, 1.0); mean = X.mean(2)
    if agg == 'mean': key = mean
    elif agg == 'cmean': key = np.minimum(X, 0.2).mean(2)
    elif agg == 'max': key = X.max(2) + 1e-3 * mean
    elif agg.startswith('hit'): thr = float(agg[3:]) / 100; key = -(X < thr).mean(2) + 1e-3 * mean
    return np.argmin(key, 1)
POOLS = {'c8': 8, 'c11': 11}; AGGS = ['mean', 'cmean', 'hit04', 'hit06', 'max']
def variants(members):
    cs = ['center', 'M3', 'M3c'] + (['M9', 'M9c'] if len(members) >= 9 else []); out = []
    for pn in POOLS:
        for cn in cs:
            for ag in (['mean'] if cn == 'center' else AGGS): out.append((pn, cn, ag))
    return out
def vname(v): return '/'.join(v)
def vsucc(D, v):
    pool = POOLS[v[0]]; pool = min(pool, D['succ'].shape[1]); ch = choose(D['S'], pool, colset(v[1], D['members']), v[2]); return D['succ'][np.arange(len(ch)), ch], ch
def boot(x, n=20000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed); b = x[r.integers(0, len(x), (n, len(x)))].mean(1); return float(np.quantile(b, .025)), float(np.quantile(b, .975))
def pdiff(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); lo, hi = boot(d); return 100 * d.mean(), 100 * lo, 100 * hi
def rec(a, r0, r1, n=20000):
    r = np.random.default_rng(1); N = len(a); I = r.integers(0, N, (n, N)); da = (np.asarray(a, float) - r0); d1 = (np.asarray(r1, float) - r0); f = da[I].mean(1) / np.where(np.abs(d1[I].mean(1)) < 1e-9, np.nan, d1[I].mean(1))
    return float(da.mean() / d1.mean()), float(np.nanquantile(f, .025)), float(np.nanquantile(f, .975))
if __name__ == '__main__':
    mode = sys.argv[1]; os.makedirs(RES, exist_ok=True)
    if mode == 'select':
        D = load('cal'); V = variants(D['members']); rows = []
        for v in V: s, ch = vsucc(D, v); rows.append((float(s.mean()), v, float((ch != D['base']).mean())))
        base = D['succ'][np.arange(len(D['base'])), D['base']].mean(); order = {v: i for i, v in enumerate(V)}
        cand = [r for r in rows if r[1] != ('c8', 'center', 'mean')]; cand.sort(key=lambda r: (-r[0], order[r[1]])); sel = cand[0]
        ceil = float(D['succ'].max(1).mean()); ceil8 = float(D['succ'][:, :8].max(1).mean())
        json.dump(dict(rule='highest CAL (n=200) success among all variants except the baseline, ties -> earlier in the listed order (simpler first)', n_variants=len(V), cal_base_Lens=float(base), selected=vname(sel[1]), selected_cal=sel[0], selected_changed_frac=sel[2],
                       ceiling_any_candidate_all=ceil, ceiling_any_candidate_first8=ceil8, top10=[(vname(r[1]), r[0]) for r in cand[:10]], all=[(vname(r[1]), r[0], r[2]) for r in rows], members=D['members']), open(os.path.join(RES, f'{TAG}_selection.json'), 'w'), indent=1)
        print('selected', vname(sel[1]), sel[0], 'baseline', base, 'ceilings', ceil8, ceil, 'n variants', len(V))
    else:
        S = json.load(open(os.path.join(RES, f'{TAG}_selection.json'))); selv = tuple(S['selected'].split('/')); out = {}; L = ['# spike15c approach B: ensemble/posterior-robust plan selection\n', f"selected on CAL: **{S['selected']}** (CAL {100*S['selected_cal']:.1f} %, baseline L-ens {100*S['cal_base_Lens']:.1f} %, {S['n_variants']} variants tried)\n"]
        for SET, nm in (('hold', 'holdout 1100-1599 n=500'), ('new', 'new seeds 703000-703199 n=200'), ('cal', 'CAL n=200 (selection set)')):
            D = load(SET); R = refs(SET); n = len(D['base']); Lens = D['succ'][np.arange(n), D['base']]; v = selv; s, ch = vsucc(D, v)
            r0 = np.array(R['adapt3_full']['success']) if R else None; r1 = np.array(R['R1']['success']) if R else None
            o = dict(n=n, selected=float(s.mean()), Lens=float(Lens.mean()), changed_frac=float((ch != D['base']).mean()), ceiling_any=float(D['succ'].max(1).mean()), ceiling_first8=float(D['succ'][:, :8].max(1).mean()))
            o['sel_vs_Lens'] = pdiff(s, Lens)
            if R: o.update(R0=float(r0.mean()), R1=float(r1.mean()), sel_vs_R0=pdiff(s, r0), Lens_vs_R0=pdiff(Lens, r0), sel_recovered=rec(s, r0, r1), Lens_recovered=rec(Lens, r0, r1))
            allv = {}
            for vv in variants(D['members']):
                sv, _ = vsucc(D, vv); allv[vname(vv)] = dict(succ=float(sv.mean()), vs_Lens=pdiff(sv, Lens), vs_R0=pdiff(sv, r0) if R else None)
            o['all'] = allv; out[SET] = o
            L.append(f'\n## {nm}\n'); L.append(f"| method | success % | vs R0 pp [95% CI] | vs L-ens pp [95% CI] | recovered fraction of R0->R1 [CI] |\n|---|---|---|---|---|")
            if R: L.append(f"| R0 T-adapt3-full | {100*o['R0']:.1f} | - | | |"); L.append(f"| R1 true theta | {100*o['R1']:.1f} | | | |")
            L.append(f"| L-ens (3 nets, baseline pick) | {100*o['Lens']:.1f} | " + (f"{o['Lens_vs_R0'][0]:+.1f} [{o['Lens_vs_R0'][1]:+.1f}, {o['Lens_vs_R0'][2]:+.1f}]" if R else '') + " | - | " + (f"{o['Lens_recovered'][0]:.2f} [{o['Lens_recovered'][1]:.2f}, {o['Lens_recovered'][2]:.2f}]" if R else '') + ' |')
            L.append(f"| **B selected {S['selected']}** | {100*o['selected']:.1f} | " + (f"{o['sel_vs_R0'][0]:+.1f} [{o['sel_vs_R0'][1]:+.1f}, {o['sel_vs_R0'][2]:+.1f}]" if R else '') + f" | {o['sel_vs_Lens'][0]:+.1f} [{o['sel_vs_Lens'][1]:+.1f}, {o['sel_vs_Lens'][2]:+.1f}] | " + (f"{o['sel_recovered'][0]:.2f} [{o['sel_recovered'][1]:.2f}, {o['sel_recovered'][2]:.2f}]" if R else '') + ' |')
            L.append(f"\nepisodes where the selected rule changes the baseline pick: {100*o['changed_frac']:.1f} %. Ceiling (best executed candidate per episode, oracle selection): first 8 candidates {100*o['ceiling_first8']:.1f} %, all {100*o['ceiling_any']:.1f} %.\n")
            L.append('| variant (pool/members/aggregator) | success % | vs L-ens | vs R0 |\n|---|---|---|---|')
            for k, x in allv.items(): L.append(f"| {k} | {100*x['succ']:.1f} | {x['vs_Lens'][0]:+.1f} [{x['vs_Lens'][1]:+.1f}, {x['vs_Lens'][2]:+.1f}] | " + (f"{x['vs_R0'][0]:+.1f} [{x['vs_R0'][1]:+.1f}, {x['vs_R0'][2]:+.1f}]" if R else '') + ' |')
        h, nw = out['hold'], out['new']
        if 'sel_vs_R0' in h:
            ok_R0 = h['sel_vs_R0'][0] >= 5 and h['sel_vs_R0'][1] > 0 and nw['sel_vs_R0'][0] >= 5 and nw['sel_vs_R0'][1] > 0
            ok_L = h['sel_vs_Lens'][0] >= 5 and h['sel_vs_Lens'][1] > 0 and nw['sel_vs_Lens'][0] >= 5 and nw['sel_vs_Lens'][1] > 0
            L.insert(2, f"\n**Verdict (pre-registered rule):** vs R0: {'EATS headroom' if ok_R0 else 'does NOT eat headroom'} (holdout {h['sel_vs_R0'][0]:+.1f} [{h['sel_vs_R0'][1]:+.1f}, {h['sel_vs_R0'][2]:+.1f}], new {nw['sel_vs_R0'][0]:+.1f} [{nw['sel_vs_R0'][1]:+.1f}, {nw['sel_vs_R0'][2]:+.1f}]). vs its own natural base L-ens: {'EATS headroom beyond L-ens' if ok_L else 'does NOT eat headroom beyond L-ens'} (holdout {h['sel_vs_Lens'][0]:+.1f} [{h['sel_vs_Lens'][1]:+.1f}, {h['sel_vs_Lens'][2]:+.1f}], new {nw['sel_vs_Lens'][0]:+.1f} [{nw['sel_vs_Lens'][1]:+.1f}, {nw['sel_vs_Lens'][2]:+.1f}]).\n")
            out['verdict'] = dict(vs_R0=bool(ok_R0), vs_Lens=bool(ok_L))
        json.dump(out, open(os.path.join(RES, f'{TAG}_summary.json'), 'w'), indent=1); open(os.path.join(RES, f'{TAG}_table.md'), 'w').write('\n'.join(L)); print('\n'.join(L[:12]))
