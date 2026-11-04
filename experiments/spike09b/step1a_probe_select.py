"""Step 1a: choose the T-adapt probe (duration T, amplitude scale of [0.5,0.5,0.4] rad) on CAL 1000-1049 only. 4 configs x 50 episodes = 200 episodes. Fitness = success - 0.2*mean(min(min_dist,0.5)); ties -> shorter probe."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import c9b; from c9b import *
def one(s):
    t = true_chain(s); sc = scenario_of(t); h = []
    def fac(ctx): c = Probe(ctx, fit_est, 'fit'); h.append(c); return c
    tr = run_episode(sc, fac, s); m = episode_metrics(sc, tr); m['err'] = np.log(np.array(h[0].est) / tvec(t)).tolist(); return m
if __name__ == '__main__':
    S = list(range(1000, 1050)); res = {}
    for T, sc in [(0.1, 0.1), (0.2, 0.1), (0.2, 0.3), (0.3, 0.5)]:
        c9b.PROBE_T = T; c9b.PROBE_AMP = np.array([0.5, 0.5, 0.4]) * sc; pool = mp.Pool(8); t0 = time.time(); r = pool.map(one, S, chunksize=1); pool.close()
        E = np.array([x['err'] for x in r]); su = np.array([x['success'] for x in r]); md = np.array([x['min_dist'] for x in r])
        res[f'{T}_{sc}'] = dict(T=T, amp_scale=sc, succ=float(su.mean()), min_dist=float(md.mean()), fit=float(su.mean() - 0.2 * np.minimum(md, .5).mean()), rmse_log_err=np.sqrt((E ** 2).mean(0)).tolist(), wall=time.time() - t0)
        print(res[f'{T}_{sc}'], flush=True)
    best = max(res.values(), key=lambda r: (round(r['fit'], 6), -r['T'])); res['chosen'] = dict(T=best['T'], amp_scale=best['amp_scale'], episodes=200)
    json.dump(res, open(os.path.join(OUT, 'adapt_probe_cal.json'), 'w'), indent=1); print('chosen', res['chosen'])
