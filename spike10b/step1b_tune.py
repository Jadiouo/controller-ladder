"""Step 1b: T-adapt3 greedy tuning on CAL 1000-1049 (condition given by argv[2], default abc), for parametrisation argv[1] in {grad, full}. Budget (episodes) logged.
Fitness key = (CAL success, -min_dist). Stage A probe (T, amp) / B loss / C residual scales / D priors / E optimiser effort / F probe refinement. Resumable (evaluations cached by config hash).
Also evaluates T-adapt (spike09b) and T-adapt2 (spike10a chosen cfg) on CAL under the SAME plant (integration fix) as comparison / abandon rule baseline."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c10b import *
if __name__ == '__main__':
    pm = sys.argv[1]; cond = sys.argv[2] if len(sys.argv) > 2 else 'abc'; tag = sys.argv[3] if len(sys.argv) > 3 else pm
    if cond == 'x0.5+x2': conds = ['x0.5', 'x2']
    else: conds = [cond]
    S = list(range(1000, 1050)); guard(S); f = os.path.join(OUT, f'tune_{tag}.json'); st = json.load(open(f)) if os.path.exists(f) else dict(cache={}, log=[], episodes=0, wall=0.0)
    pool = mp.Pool(8); t0 = time.time(); w0 = st['wall']; items = [(s, conds[i % len(conds)]) for i, s in enumerate(S)]
    def ev(spec, name):
        k = name + '|' + json.dumps(spec, sort_keys=True, default=str)
        if k not in st['cache']:
            r = pool.map(run10, [(spec, s, None, c) for s, c in items], chunksize=1); su = np.array([x['success'] for x in r])
            st['cache'][k] = dict(name=name, spec=spec, succ=float(su.mean()), min_dist=float(np.nanmean([x['min_dist'] for x in r])), inst=float(np.mean([x['inst'] > 0 for x in r])), diverged=float(np.mean([x['diverged'] for x in r])))
            st['episodes'] += len(S); st['log'].append(dict(name=name, succ=st['cache'][k]['succ'], min_dist=st['cache'][k]['min_dist'], episodes_so_far=st['episodes'])); st['wall'] = w0 + time.time() - t0
            json.dump(st, open(f, 'w'), indent=1, default=str); print(name, st['cache'][k]['succ'], round(st['cache'][k]['min_dist'], 4), st['episodes'], f'{st["wall"]:.0f}s', flush=True)
        return st['cache'][k]
    key = lambda e: (round(e['succ'], 6), -e['min_dist'])
    if pm in ('grad', 'full') and tag == pm:      # baselines only once (in-distribution run)
        ev('adapt', 'ref T-adapt'); a2 = json.load(open(os.path.join(ROOT, 'results/spike10a/adapt2_tuning.json')))['chosen']; ev(('adapt2', a2), 'ref T-adapt2')
    B = {**A3_BASE, 'pm': pm, 'fr_sd': 1.0, **({'g_sd': 0.15} if pm == 'grad' else {'preg': 1.0})}; best = dict(cfg=B, **ev(('adapt3', B), 'base'))
    def tryc(name, **kw):
        global best
        cfg = {**best['cfg'], **kw}; e = ev(('adapt3', cfg), name); e['cfg'] = cfg
        if key(e) > key(best): best = e
    for T, amp in [(0.2, 0.6), (0.4, 1.0), (0.6, 0.6), (0.6, 1.0)]: tryc(f'A T={T} amp={amp}', T=T, amp=amp)
    xs_ = len(sys.argv) > 4 and sys.argv[4] == 'xshort'      # reduced protocol for the OOD control (wall-clock): A, B-linear, D only
    tryc('B linear', loss='linear')
    if not xs_: tryc('B soft_l1 fs=3', f_scale=3.0); tryc('C sq=.003 sv=.03', sq=0.003, sv=0.03); tryc('C sq=.03 sv=.3', sq=0.03, sv=0.3)
    if not xs_:
        if pm == 'grad': tryc('D g_sd=0.05', g_sd=0.05); tryc('D g_sd=0 (free slope)', g_sd=0.0)
        else: tryc('D preg=0.3', preg=0.3); tryc('D preg=3', preg=3.0)
    tryc('D fr_sd=0.5', fr_sd=0.5)
    if not xs_: tryc('D fr_sd=0 (free friction)', fr_sd=0.0)
    if not (len(sys.argv) > 4 and sys.argv[4] in ('short', 'xshort')):
        tryc('E ngrid=9 nstart=6 nfev=50', ngrid=9, nstart=6, nfev=50)
        b = best['cfg']; tryc('F T/2', T=round(b['T'] * 0.75, 3)); tryc('F amp x1.5', amp=b['amp'] * 1.5)
    st['chosen'] = best['cfg']; st['chosen_cal'] = best['succ']; st['wall'] = w0 + time.time() - t0; json.dump(st, open(f, 'w'), indent=1, default=str); print('chosen', best['cfg'], best['succ'], 'episodes', st['episodes'])
