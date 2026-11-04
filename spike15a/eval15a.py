"""spike15a: v8 confirmation of K3 on the locked set TEST-C (seeds 30000-30499). All methods frozen (spike13b / spike14a / spike10b implementations imported, nothing retuned).
usage: eval15a.py smoke   -> CAL seeds 1000-1009 only (never TEST-C); results in .scratch/s15/smoke_*.json
       eval15a.py testc [--relog "reason"]  -> ONE locked run on 30000-30499. Writes ONE results/.test_access.log line before the first TEST-C seed is touched (marker .scratch/s15/testc_logged;
                                              a resume after a crash does not log again; --relog adds a second line and requires a reason, to be explained in the README).
Pipeline per set: stage A raw probe data; stage B L-dec4 inference (subprocess, CPU); stage C estimators / episodes. Everything resumable per method."""
import sys, os, json, time, pickle, datetime, subprocess, multiprocessing as mp
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', 'spike13b')); sys.path.insert(0, os.path.join(H, '..', 'spike14a'))
import c13, c14
from classical import *          # fit4 / CFGS / estimate_set (spike14a), re-exports c14 + r10b
MODE = sys.argv[1]; assert MODE in ('smoke', 'testc')
SEEDS = list(range(1000, 1010)) if MODE == 'smoke' else list(range(30000, 30500))
S15 = os.path.join(ROOT, '.scratch/s15'); os.makedirs(S15, exist_ok=True); RES = os.path.join(ROOT, 'results/spike15a'); os.makedirs(RES, exist_ok=True)
STORE = os.path.join(S15 if MODE == 'smoke' else RES, f'{"smoke" if MODE == "smoke" else "testc"}_episodes.json')
def guard15(seed):
    s = int(seed); assert not (0 <= s < 50) and not (20000 <= s <= 20499), f'seed {s} forbidden (TEST / TEST-B)'
    assert (1000 <= s < 1010) if MODE == 'smoke' else (30000 <= s <= 30499), f'seed {s} not allowed in mode {MODE}'
c13.guard13 = guard15; c14.guard14 = guard15; c13.ALLOW = ((30000, 30499),) if MODE == 'testc' else ((1000, 1009),)
[guard15(s) for s in SEEDS]
LOG = os.path.join(ROOT, 'results/.test_access.log'); MARK = os.path.join(S15, 'testc_logged')
def log_access(reason=None):
    if MODE != 'testc': return
    first = not os.path.exists(MARK)
    if not first and not reason: print('TEST-C already logged; resuming without a new log line'); return
    msg = 'v8 single locked TEST-C run, seeds=30000-30499' + ('' if first else f' (RE-RUN after crash: {reason})')
    with open(LOG, 'a') as f: f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\tspike15a/eval15a.py\t{msg}\n")
    open(MARK, 'a').write(datetime.datetime.now().isoformat() + ' ' + msg + '\n'); print('logged:', msg, flush=True)
# ---------------------------------------------------------------- reverse-order rung (chain state first): fitted theta (R0 estimator), TRUE simulator state at the end of the probe
class RevProbe(c13.Probe3True):
    def _swing_start(self, obs):
        taus, Qm, Vm = self.history(); self.pq0 = self.hq0; key = est_key(self.tag, self.pq0, taus, Qm)
        if key not in EST_CACHE:
            f = os.path.join(PL, f'est3_{key}.npy')
            if os.path.exists(f): EST_CACHE[key] = tuple(float(x) for x in np.load(f))
            else: EST_CACHE[key] = tuple(float(x) for x in self.est_fn(self.ctx, self.pq0, taus, Qm, Vm)); np.save(f + f'.{os.getpid()}.npy', np.array(EST_CACHE[key])); os.replace(f + f'.{os.getpid()}.npy', f)
        th = np.array(EST_CACHE[key]); self.est = th; m = model_theta(self.ctx, th); d = self.env.data; x0 = (np.array(d.qpos[:m.nq], float), np.array(d.qvel[:m.nv], float))
        inner = Tuned3S(self.ctx, th, x0, self.ctx.task_info['duration'] - self.T, self.qp, self.qv); inner.reset(self.seed)
        self.q0 = np.asarray(obs.q, float).copy(); self.knots = inner.plan(self.q0, np.asarray(obs.ref['target'], float)); self.plan_info = dict(inner.plan_info)
def run_rrev(a):
    seed, cond = a; guard15(seed); reset_probe_globals(); INST[0] = 0; sc = scen(seed, cond, None); cfg = spec_of('adapt3_full')[1]; env = sc.make_env()
    est = lambda ctx, q0, taus, Qm, Vm: fit3(ctx, q0, taus, Qm, Vm, cfg)
    def fac(ctx): p = RevProbe(ctx, est, cfg_tag(cfg), cfg['T'], np.array([0.5, 0.5, 0.4]) * cfg['amp']); p.env = env; return p
    tr = run_episode(sc, fac, int(seed), env=env); m = episode_metrics(sc, tr); return dict(success=m['success'], min_dist=m['min_dist'], inst=int(INST[0]), diverged=m['diverged'])
def run_any(a):
    kind, meth, seed = a
    if kind == 'c13': r = c13.run13((meth, seed, 'abc'))
    elif kind == 'c14': r = c14.run14((meth, seed))
    else: r = run_rrev((seed, 'abc'))
    return {k: r[k] for k in ('success', 'min_dist', 'inst', 'diverged')}
def save_ep(name, r, wall):
    res = json.load(open(STORE)) if os.path.exists(STORE) else {}
    res[name] = dict(seeds=SEEDS, success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], inst=[x['inst'] for x in r], diverged=[x['diverged'] for x in r], wall=wall); json.dump(res, open(STORE, 'w'))
    print(name, f"{100*np.mean(res[name]['success']):.1f}%", 'unstable', np.mean(np.array(res[name]['inst']) > 0), f'{wall:.0f}s', flush=True)
def done(name): return os.path.exists(STORE) and name in json.load(open(STORE))
def tabinit(tabpath): c14._init_tab(tabpath)
if __name__ == '__main__':
    reason = sys.argv[sys.argv.index('--relog') + 1] if '--relog' in sys.argv else None
    log_access(reason)
    NW = int(os.environ.get('NW', 8)); t00 = time.time()
    # ---- stage A: raw probe data
    rawp = os.path.join(S15, f'raw_{MODE}.pkl'); pool = mp.Pool(NW)
    if not os.path.exists(rawp):
        D = c14.gen_set(pool, [(s, None, 'abc') for s in SEEDS], chunk=10); assert list(D['seed']) == SEEDS; pickle.dump(D, open(rawp, 'wb')); print('raw probe data', len(SEEDS), f'{time.time()-t00:.0f}s', flush=True)
    D = pickle.load(open(rawp, 'rb')); os.makedirs(os.path.join(S15, 'est'), exist_ok=True)
    # ---- stage B: L-dec4 inference (CPU subprocess)
    if MODE == 'smoke' and not os.path.exists(os.path.join(DAT, 'model_r15dec4_s0.pt')):      # smoke only, before the re-trained weights exist: use the stored spike14a predictions of the same CAL seeds (tests the plumbing, not the inference)
        for s in range(3): np.save(os.path.join(S15, 'est', f'dec4_smoke_s{s}.npy'), np.load(os.path.join(DAT, 'est', f'dec4_a0.9_e38_s{s}_cal.npy'))[:len(SEEDS)])
        print('SMOKE: using stored spike14a dec4 predictions (weights not yet available)')
    if not all(os.path.exists(os.path.join(S15, 'est', f'dec4_{MODE}_s{s}.npy')) for s in range(3)):
        subprocess.run([sys.executable, os.path.join(H, 'infer_dec4.py'), rawp, os.path.join(S15, 'est', f'dec4_{MODE}')], check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    TE = [np.load(os.path.join(S15, 'est', f'dec4_{MODE}_s{s}.npy')) for s in range(3)]; ENS = np.mean(TE, 0)
    # ---- stage C: estimators needing a fit (C-MAP = spike14a C_map1_ms, frozen cfg)
    fp = os.path.join(S15, 'est', f'C_map1_ms_{MODE}.npy')
    if not os.path.exists(fp):
        t0 = time.time(); cm, cpu, nf = estimate_set(pool, 'C_map1_ms', D); np.save(fp, cm); json.dump(dict(cpu_s=cpu, wall_s=time.time() - t0, n=len(cm)), open(os.path.join(S15, 'est', f'C_map1_ms_{MODE}_cost.json'), 'w')); print('C-MAP fits', f'{time.time()-t0:.0f}s', flush=True)
    CM = np.load(fp)
    tab = {f'dec4_s{s}': {k: TE[s][i] for i, k in enumerate(D['key'])} for s in range(3)}; tab['ens_dec4'] = {k: ENS[i] for i, k in enumerate(D['key'])}; tab['C_map1_ms'] = {k: CM[i] for i, k in enumerate(D['key'])}
    tp = os.path.join(S15, f'tab_{MODE}.pkl'); pickle.dump(tab, open(tp, 'wb')); pool.close(); pool.join()
    # episodes (fresh pool so the lookup table is inherited; priority order)
    c14._init_tab(tp); pool = mp.Pool(NW)
    jobs = [('R0', 'c13', 'adapt3_full'), ('R1', 'c13', 'R1'), ('R2', 'c13', 'R2'), ('R4', 'c13', 'oracle')] + [(f'lest#{s}', 'c13', f'lest_full#{s}') for s in range(3)] + \
           [(f'dec4_s{s}', 'c14', f'dec4_s{s}') for s in range(3)] + [('ens_dec4', 'c14', 'ens_dec4'), ('C_map1_ms', 'c14', 'C_map1_ms'), ('Rrev1', 'rev', None), ('R5', 'c13', 'R5')]
    for name, kind, meth in jobs:
        if done(name): continue
        t0 = time.time(); r = pool.map(run_any, [(kind, meth, s) for s in SEEDS], chunksize=1); save_ep(name, r, time.time() - t0)
    print('ALL DONE', f'{time.time()-t00:.0f}s', flush=True)
