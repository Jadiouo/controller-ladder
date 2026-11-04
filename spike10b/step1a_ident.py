"""Step 1a: identifiability of the T-adapt3 parametrisations on CAL (seeds 1000-1049, condition abc): probe only, estimate theta, report parameter-estimation errors
against the label-free prior spread (RMSE of predicting the CAL mean). Default T-adapt3 config (T=0.4, amp=0.6 from T-adapt2). No swing, so cheap."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c10b import *
def one(a):
    cfg, s, cond = a; guard([s]); sc = scen(s, cond); env = sc.make_env(); ctx = env.ctx; obs = env.reset(s); q0, taus, Qm, Vm = probe_run3(env, ctx, obs, cfg['T'], np.array([0.5, 0.5, 0.4]) * cfg['amp'])
    t0 = time.time(); th = fit3(ctx, q0, taus, Qm, Vm, cfg); return th, true_theta(s, cond), time.time() - t0, fit3.nfev
def summarize(TH, TT):
    lm_t = np.log(TT[:, 1:11]); lm_e = np.log(TH[:, 1:11]); x = (np.arange(10) - 4.5) / 4.5; slope = lambda L: (L * x).sum(1) / (x ** 2).sum()
    rows = {}
    def row(name, e, t):
        rows[name] = dict(rmse=float(np.sqrt(np.mean((e - t) ** 2))), prior_sd=float(np.std(t)), frac_var_explained=float(1 - np.mean((e - t) ** 2) / max(np.var(t), 1e-30)), corr=float(np.corrcoef(e, t)[0, 1]) if np.std(e) > 0 else float('nan'))
    row('length_scale', TH[:, 0], TT[:, 0]); row('log_total_mass', lm_e.mean(1), lm_t.mean(1)); row('mass_log_slope', slope(lm_e - lm_e.mean(1, keepdims=True)), slope(lm_t - lm_t.mean(1, keepdims=True)))
    row('log_damping', np.log(TH[:, 11]), np.log(TT[:, 11])); row('friction_1e-3Nm', TH[:, 12] * 1e3, TT[:, 12] * 1e3)
    dev_e = lm_e - lm_e.mean(1, keepdims=True); dev_t = lm_t - lm_t.mean(1, keepdims=True); rows['per_link_mass_deviation(log, mean-removed)'] = dict(rmse=float(np.sqrt(np.mean((dev_e - dev_t) ** 2))), prior_sd=float(np.sqrt(np.mean(dev_t ** 2))), frac_var_explained=float(1 - np.mean((dev_e - dev_t) ** 2) / np.mean(dev_t ** 2)))
    return rows
if __name__ == '__main__':
    S = list(range(1000, 1050)); pool = mp.Pool(8); out = {}; chosen = len(sys.argv) > 1 and sys.argv[1] == 'chosen'
    for pm in ('grad', 'full'):
        cfg = json.load(open(os.path.join(OUT, f'tune_{pm}.json')))['chosen'] if chosen else {**A3_BASE, 'pm': pm, 'preg': 1.0 if pm == 'full' else 0.0}; t0 = time.time(); r = pool.map(one, [(cfg, s, 'abc') for s in S], chunksize=1)
        TH = np.array([x[0] for x in r]); TT = np.array([x[1] for x in r]); out[pm] = dict(cfg=cfg, errors=summarize(TH, TT), fit_wall_mean_s=float(np.mean([x[2] for x in r])), nfev_mean=float(np.mean([x[3] for x in r])), wall=time.time() - t0)
        print(pm, json.dumps(out[pm]['errors'], indent=1), out[pm]['fit_wall_mean_s'], flush=True); json.dump(out, open(os.path.join(OUT, 'ident_cal_chosen.json' if chosen else 'ident_cal.json'), 'w'), indent=1)
