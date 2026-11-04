"""Step 3: T-adapt2 greedy tuning on CAL 1000-1049, condition abc. Stage A probe (T, amp); B robust loss; C residual weights; D combo. Budget logged."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c10a import *
BASE = dict(T=0.2, amp=0.3, loss='linear', f_scale=1.0, sq=0.01, sv=0.1, prefixes=[0.5, 0.75, 1.0], ngrid=7, nstart=4)
if __name__ == '__main__':
    S = list(range(1000, 1050)); guard(S); pool = mp.Pool(8); log = []; t0 = time.time(); neps = [0]
    def ev(cfg, tag):
        r = pool.map(run10, [(('adapt2', cfg), s, 'abc') for s in S], chunksize=1); su = np.array([x['success'] for x in r]); neps[0] += len(S)
        e = dict(tag=tag, cfg=cfg, succ=float(su.mean()), min_dist=float(np.nanmean([x['min_dist'] for x in r])), episodes_so_far=neps[0], wall=time.time() - t0); log.append(e); print(tag, e['succ'], e['min_dist'], flush=True)
        json.dump(dict(log=log, episodes=neps[0]), open(os.path.join(OUT, 'adapt2_tuning.json'), 'w'), indent=1); return e
    key = lambda e: (round(e['succ'], 6), -e['min_dist'])
    best = ev(BASE, 'base(=T-adapt)')
    for T, amp in [(0.2, 0.6), (0.2, 1.0), (0.3, 0.3), (0.3, 0.6), (0.4, 0.6)]:
        e = ev({**BASE, 'T': T, 'amp': amp}, f'A T={T} amp={amp}'); best = max([best, e], key=key)
    b = best['cfg']
    for loss, fs in [('soft_l1', 1.0), ('soft_l1', 3.0), ('huber', 2.0)]:
        e = ev({**b, 'loss': loss, 'f_scale': fs}, f'B {loss} fs={fs}'); best = max([best, e], key=key)
    b = best['cfg']
    for sq, sv in [(0.003, 0.03), (0.03, 0.3)]:
        e = ev({**b, 'sq': sq, 'sv': sv}, f'C sq={sq} sv={sv}'); best = max([best, e], key=key)
    b = best['cfg']
    e = ev({**b, 'ngrid': 9, 'nstart': 6}, 'D denser start grid'); best = max([best, e], key=key)
    json.dump(dict(log=log, episodes=neps[0], wall_s=time.time() - t0, chosen=best['cfg'], chosen_cal=best['succ']), open(os.path.join(OUT, 'adapt2_tuning.json'), 'w'), indent=1); print('chosen', best)
