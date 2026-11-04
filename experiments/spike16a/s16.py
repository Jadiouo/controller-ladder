"""spike16a CLI. usage: s16.py {cal|gate|real|stats} TASK   (all stages resumable: finished outputs are skipped).  See results/spike16a/README.md section 0."""
import sys, os, time, json, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h16 import *
import torch, torch.nn as nn
torch.set_num_threads(2)
TUNE = dict(catch=dict(gens=8, set=CAL), toss=dict(gens=5, set=CAL), push_t=dict(gens=2, set=CAL_S))
RLB = dict(catch=16200, toss=10200, push_t=2100)          # real-run budget in episodes (training + CAL-S checkpoint evaluation)
LAM, MU = 10, 3
SUF = os.environ.get('S16_SUFFIX', '')          # extension runs write to results/spike16a/<task><SUF>/
def P(task, name): return os.path.join(RES, task + SUF, name)
def log(*a, **k): print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------------------------------------------------------- classical: calibration + ES tuning
def eval_many(task, s, seeds, plist, n=6):
    guard(seeds)
    jobs = [(task, int(sd), s, p, None, None) for p in plist for sd in seeds]
    r = pool(n).map(run_ep, jobs, chunksize=1); out = []
    for i in range(len(plist)):
        rr = r[i * len(seeds):(i + 1) * len(seeds)]
        out.append(dict(succ=np.array([x['success'] for x in rr]), smooth=np.array([x['smooth'] for x in rr]), err=int(sum(x['err'] for x in rr))))
    return out

def tune(task, s, tag):
    path = P(task, f'tune_{tag}.json')
    if os.path.exists(path): return json.load(open(path))
    cfg = TUNE[task]; seeds = cfg['set']; t0 = time.time(); rng = np.random.default_rng([16, {'catch': 1, 'toss': 2, 'push_t': 3}[task], int(s * 100)])
    d0 = default_params(task); e0 = eval_many(task, s, seeds, [d0])[0]
    best = dict(fit=fitness(task, e0), succ=float(e0['succ'].mean()), params=d0); default = dict(best); mean = p2u(task, d0); sigma = 0.25; hist = [dict(gen=0, succ=best['succ'], fit=best['fit'])]
    log(task, 's', s, 'default CAL succ', best['succ'], 'fit', best['fit'], flush=True); neval, errs = 1, e0['err']
    for g in range(1, cfg['gens'] + 1):
        U = np.clip(mean + sigma * rng.standard_normal((LAM, len(mean))), 0, 1); pl = [u2p(task, u) for u in U]; es = eval_many(task, s, seeds, pl); neval += LAM
        fits = np.array([fitness(task, e) for e in es]); errs += sum(e['err'] for e in es); order = np.argsort(-fits); mean = U[order[:MU]].mean(0)
        imp = fits[order[0]] > best['fit']
        if imp: best = dict(fit=float(fits[order[0]]), succ=float(es[order[0]]['succ'].mean()), params=pl[order[0]])
        sigma *= 0.85 if imp else 1.15; hist.append(dict(gen=g, gen_best_fit=float(fits[order[0]]), gen_best_succ=float(es[order[0]]['succ'].mean()), best_fit=best['fit'], best_succ=best['succ'], sigma=sigma))
        log(task, tag, 'gen', g, hist[-1], f'{time.time()-t0:.0f}s', flush=True)
    use = 'tuned' if best['succ'] >= default['succ'] else 'default'
    out = dict(task=task, s=s, tuning_set='CAL' if seeds is CAL else 'CAL-S', n_set=len(seeds), evals=neval, episodes=neval * len(seeds), wall_s=time.time() - t0, errors=errs,
               default=default, tuned=best, strongest=use, params=best['params'] if use == 'tuned' else d0, succ_cal=max(best['succ'], default['succ']), hist=hist)
    jdump(out, path); return out

def cmd_cal(task):
    cp = P(task, 'calib_s.json')
    if SUF and not os.path.exists(cp): jdump(dict(grid_eval=[], s0=float(os.environ['S16_S']), note='strength fixed by the extension, no calibration'), cp)
    if os.path.exists(cp): c = json.load(open(cp))
    else:
        c = []; d0 = default_params(task); s0 = None
        for s in GRID:
            e = eval_many(task, s, CAL_A, [d0])[0]; c.append(dict(s=s, default_cal_a=float(e['succ'].mean()), err=e['err'])); log(task, 'calib s', s, c[-1], flush=True)
            if e['succ'].mean() < 0.55: s0 = s; break
        if s0 is None: s0 = GRID[-1]
        jdump(dict(grid_eval=c, s0=s0), cp); c = json.load(open(cp))
    s0 = c['s0']; A = tune(task, s0, 'A'); fin = dict(s=s0, tune='A', rule='accepted')
    if not SUF and not (0.60 <= A['succ_cal'] <= 0.85):
        i = GRID.index(s0); j = min(i + 1, len(GRID) - 1) if A['succ_cal'] > 0.85 else max(i - 1, 0)
        if j != i:
            B = tune(task, GRID[j], 'B'); fin = dict(s=GRID[j], tune='B', rule=f'A at s={s0} had CAL {A["succ_cal"]:.3f} outside [0.60,0.85]; re-tuned once at s={GRID[j]}')
        else: fin = dict(s=s0, tune='A', rule=f'A had CAL {A["succ_cal"]:.3f} outside the band but no neighbour grid value exists')
    T = json.load(open(P(task, f'tune_{fin["tune"]}.json')))
    fin.update(params=T['params'], strongest=T['strongest'], succ_cal=T['succ_cal'], tuning_set=T['tuning_set'], tuning_episodes=T['episodes'], tuning_wall_s=T['wall_s'])
    jdump(fin, P(task, 'classical.json')); log(task, 'FINAL classical', {k: v for k, v in fin.items() if k != 'params'}, flush=True)

# ---------------------------------------------------------------- PPO
def mk(i, o, hid, sc=1.0):
    n = nn.Sequential(nn.Linear(i, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, o))
    for m in n:
        if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(n[-1].weight, sc); return n
def layers_of(net): return [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]

def worker(conn, wid, E, task, s, params, bias, seed):
    rng = np.random.default_rng([seed, wid, 77]); envs = [None] * E
    def new(i): envs[i] = Ep(task, 510000 + int(rng.integers(0, 10 ** 7)), s, params, bias, True)
    while True:
        cmd, a = conn.recv()
        if cmd == 'reset':
            for i in range(E): new(i)
            conn.send(np.stack([e.x for e in envs]))
        elif cmd == 'step':
            xs, rs, ds, eps = [], [], [], []
            for i in range(E):
                x, r, d, info = envs[i].step(a[i])
                if d: eps.append(info['success']); new(i); x = envs[i].x
                xs.append(x); rs.append(r); ds.append(d)
            conn.send((np.stack(xs), np.array(rs, np.float32), np.array(ds, np.float32), eps))
        elif cmd == 'close': break

def train(task, s, params, bias, logstd0, budget, seed, hid=256, nenv=48, horizon=128, lr=1e-4, epochs=4, mb=2048, logstd_min=-5.0):
    bl = None if bias is None else list(bias); torch.manual_seed(seed); np.random.seed(seed); OBS = feat_dim(task); N, H, GAM = nenv, horizon, GAMMA[task]; nw = 6; E = N // nw
    nA = 2 if task == 'push_t' else 3; t0 = time.time()
    init = evaluate(task, CAL_S, s, params, None, bl); cal_eps = len(CAL_S); log('  init CAL-S', float(init['succ'].mean()), flush=True)
    target = budget - 7 * len(CAL_S) - len(CAL_S); thr = [(k + 1) / 7 * target for k in range(7)]
    conns, procs = [], []
    for w in range(nw):
        pc, cc = mp.Pipe(); p = mp.Process(target=worker, args=(cc, w, E, task, s, params, bl, seed * 1000 + 7), daemon=True); p.start(); conns.append(pc); procs.append(p)
    for c in conns: c.send(('reset', None))
    x = np.concatenate([c.recv() for c in conns]); actor, critic = mk(OBS, nA, hid, 0.01), mk(OBS, 1, hid); logstd = nn.Parameter(torch.full((nA,), logstd0))
    with torch.no_grad(): actor[-1].weight.zero_(); actor[-1].bias.zero_()
    params_ = list(actor.parameters()) + list(critic.parameters()) + [logstd]; opt = torch.optim.Adam(params_, lr=lr)
    cks, best, eps_done, succ_hist, it, steps = [], None, 0, [], 0, 0
    while eps_done < target:
        bx, ba, blp, bv, br, bd = [], [], [], [], [], []
        for h in range(H):
            xt = torch.from_numpy(x)
            with torch.no_grad(): mu = actor(xt); sd_ = logstd.exp(); act = mu + sd_ * torch.randn_like(mu); lp = (-0.5 * ((act - mu) / sd_) ** 2 - logstd - 0.9189385).sum(1); v = critic(xt).squeeze(1)
            an = act.numpy(); an_c = np.clip(an, -1, 1)
            for w, c in enumerate(conns): c.send(('step', an_c[w * E:(w + 1) * E]))
            outs = [c.recv() for c in conns]; xn = np.concatenate([o[0] for o in outs]); r = np.concatenate([o[1] for o in outs]); d = np.concatenate([o[2] for o in outs])
            for o in outs:
                for e in o[3]: eps_done += 1; succ_hist.append(e)
            bx.append(x); ba.append(an); blp.append(lp.numpy()); bv.append(v.numpy()); br.append(r); bd.append(d); x = xn
        steps += N * H
        with torch.no_grad(): lastv = critic(torch.from_numpy(x)).squeeze(1).numpy()
        R_, V_, D_ = np.array(br), np.array(bv), np.array(bd); adv = np.zeros_like(R_); g = 0
        for h in reversed(range(H)):
            nv = lastv if h == H - 1 else V_[h + 1]; nd = D_[h]; delta = R_[h] + GAM * nv * (1 - nd) - V_[h]; g = delta + GAM * 0.95 * (1 - nd) * g; adv[h] = g
        ret = adv + V_; X = torch.from_numpy(np.concatenate(bx)); A_ = torch.from_numpy(np.concatenate(ba)); LP = torch.from_numpy(np.concatenate(blp)); ADV = torch.from_numpy(adv.reshape(-1)); RET = torch.from_numpy(ret.reshape(-1))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); n = len(X); kl = []
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]; mu = actor(X[idx]); lp = (-0.5 * ((A_[idx] - mu) / logstd.exp()) ** 2 - logstd - 0.9189385).sum(1); ratio = (lp - LP[idx]).exp()
                loss = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]).mean() + 0.5 * ((critic(X[idx]).squeeze(1) - RET[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params_, 0.5); opt.step()
                with torch.no_grad(): logstd.clamp_(min=logstd_min); kl.append(float(((ratio - 1) - ratio.log()).mean()))
        it += 1
        if len(cks) < 7 and eps_done >= thr[len(cks)]:
            lay = layers_of(actor); c = evaluate(task, CAL_S, s, params, lay, bl); cal_eps += len(CAL_S)
            rec = dict(it=it, train_eps=eps_done, succ=float(c['succ'].mean()), smooth=float(c['smooth'].mean()), res=c['res'], err=c['err']); cks.append(rec)
            if best is None or (rec['succ'], rec['smooth']) > (best[0]['succ'], best[0]['smooth']): best = (rec, lay)
            log('  ckpt', rec, flush=True)
        if it % 5 == 0:
            rec = succ_hist[-200:]; log(f'  [{task} s{seed}] it {it} eps {eps_done}/{target} train-succ(200) {np.mean(rec) if rec else float("nan"):.3f} kl {np.mean(kl):.4f} std {float(logstd.exp().mean().detach()):.3f} t {time.time()-t0:.0f}s', flush=True)
    for c in conns: c.send(('close', None))
    for p in procs: p.join(timeout=5)
    if best is None:
        lay = layers_of(actor); c = evaluate(task, CAL_S, s, params, lay, bl); cal_eps += len(CAL_S); best = (dict(it=it, train_eps=eps_done, succ=float(c['succ'].mean()), smooth=float(c['smooth'].mean()), res=c['res'], err=c['err']), lay)
    return dict(seed=seed, init_cal_s=float(init['succ'].mean()), ckpts=cks, selected=best[0], layers=best[1], train_episodes=eps_done, cal_eval_episodes=cal_eps, total_episodes=eps_done + cal_eps,
                control_steps=steps * REP[task], decisions=steps, wall_s=time.time() - t0, logstd=logstd.detach().numpy().copy(), args=dict(logstd0=logstd0, budget=budget, nenv=nenv, horizon=horizon, lr=lr))

def run_train(task, tag, s, params, bias, logstd0, budget, seeds):
    path = P(task, f'ppo_{tag}.pkl'); res = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    for sd in seeds:
        if sd in res: continue
        res[sd] = train(task, s, params, bias, logstd0, budget, sd); os.makedirs(os.path.dirname(path), exist_ok=True); pickle.dump(res, open(path, 'wb'))
        r = res[sd]; log(f'{task} {tag} seed {sd} done {r["wall_s"]:.0f}s init {r["init_cal_s"]:.3f} sel {r["selected"]["succ"]:.3f} eps {r["total_episodes"]}', flush=True)
    return res

# ---------------------------------------------------------------- gate and real condition
def cmd_gate(task):
    C = json.load(open(P(task, 'classical.json'))); s, params = C['s'], C['params']; gp = P(task, 'gate.json')
    G = json.load(open(gp)) if os.path.exists(gp) else {}
    if 'bias' not in G:
        cal = {}; b_sel = None
        for b in [1, 2, 3, 4, 6, 8, 10]:
            e = evaluate(task, CAL_A, s, params, None, list(bias_vec(task, b))); cal[b] = float(e['succ'].mean()); log(task, 'bias', b, 'CAL-A zero-residual', cal[b], flush=True)
            if cal[b] <= 0.40: b_sel = b; break
        G['bias_calib'] = cal; G['bias'] = b_sel if b_sel is not None else 10; G['bias_fallback'] = b_sel is None; jdump(G, gp)
    bias = bias_vec(task, G['bias']); full = RLB[task]; half = full // 2
    if 'stageA' not in G:
        runs = {}
        for ls in (-2.0, -1.5):
            r = run_train(task, f'gateA_ls{ls}', s, params, bias, ls, half, [0])[0]; runs[ls] = dict(cal_s_sel=r['selected']['succ'], init=r['init_cal_s'], episodes=r['total_episodes'], wall_s=r['wall_s'])
        G['stageA'] = {str(k): v for k, v in runs.items()}; G['chosen'] = -2.0 if runs[-2.0]['cal_s_sel'] >= runs[-1.5]['cal_s_sel'] else -1.5; jdump(G, gp)
    init = evaluate(task, GATE, s, params, None, list(bias))['succ']
    def test(tag, ls):
        r = pickle.load(open(P(task, f'ppo_{tag}.pkl'), 'rb'))[0]; e = evaluate(task, GATE, s, params, r['layers'], list(bias)); b = boot_diff(e['succ'] - init)
        return dict(tag=tag, logstd0=ls, init=float(init.mean()), rl=float(e['succ'].mean()), res=e['res'], **b, passed=bool(b['diff'] >= 0.10 and b['lo95'] > 0))
    if 'gate_tests' not in G:
        G['gate_tests'] = []; ch = G['chosen']; t = test(f'gateA_ls{ch}', ch); G['gate_tests'].append(t); log(task, 'GATE stage A', t, flush=True)
        if init.mean() > 0.90: G['gate_status'] = 'not testable (zero-residual init > 90% on GATE: +10 pp impossible)'; G['recipe_logstd0'], G['recipe_budget'] = ch, full
        elif t['passed']: G['gate_status'] = 'PASS (stage A)'
        else:
            oth = -1.5 if ch == -2.0 else -2.0; t2 = test(f'gateA_ls{oth}', oth); G['gate_tests'].append(t2); log(task, 'GATE other', t2, flush=True); win = None
            if t2['passed']: win = (oth, 'stage B: other log-std0 at 0.5x')
            else:
                for ls in (ch, oth):
                    run_train(task, f'gateB_ls{ls}', s, params, bias, ls, full, [0]); t3 = test(f'gateB_ls{ls}', ls); t3['stage'] = 'B 1x'; G['gate_tests'].append(t3); log(task, 'GATE B', t3, flush=True)
                    if t3['passed']: win = (ls, 'stage B: 1x budget'); break
            G['gate_status'] = f'PASS ({win[1]})' if win else 'FAILED'; G['recipe_logstd0'] = win[0] if win else None; G['recipe_budget'] = full
        if G['gate_tests'][0]['passed']: G['recipe_logstd0'], G['recipe_budget'] = ch, full
        jdump(G, gp)
    log(task, 'GATE status', G['gate_status'], flush=True)

def cmd_real(task):
    G = json.load(open(P(task, 'gate.json')))
    forced = False
    if not (G['gate_status'].startswith('PASS') or G['gate_status'].startswith('not testable')):
        if not os.environ.get('S16_FORCE'): log(task, 'gate not passed; real condition NOT run'); return
        forced = True; G['recipe_logstd0'] = G['chosen']; log(task, 'gate FAILED but S16_FORCE set: DESCRIPTIVE run only (not a verdict)')
    C = json.load(open(P(task, 'classical.json'))); s, params = C['s'], C['params']; ls = G['recipe_logstd0']
    seeds_ = [int(v) for v in os.environ['S16_SEEDS'].split(',')] if os.environ.get('S16_SEEDS') else ([0, 1] if task == 'push_t' else [0, 1, 2])     # push_t: 2 seeds because of the 8 h wall clock (README deviations)
    res = run_train(task, f'real_ls{ls}', s, params, None, ls, RLB[task], seeds_); out = dict(task=task, s=s, logstd0=ls, sets={}, seeds=seeds_, descriptive_only_gate_failed=forced)
    for name, seeds in (('hold', HOLD), ('new', NEW)):
        cp = P(task, f'classical_{name}.npy')
        if os.path.exists(cp): c = np.load(cp)
        else: c = evaluate(task, seeds, s, params, None, None)['succ']; np.save(cp, c)
        pers = []; ps = {}
        for sd, r in sorted(res.items()):
            sp = P(task, f'learned_{name}_seed{sd}.json')
            if os.path.exists(sp): z = json.load(open(sp)); e = dict(succ=np.array(z['succ']), res=z['res'])
            else: e = evaluate(task, seeds, s, params, r['layers'], None); jdump(dict(succ=e['succ'], res=e['res']), sp)
            pers.append(e['succ']); b = boot_diff(e['succ'] - c); ps[sd] = dict(succ=float(e['succ'].mean()), res=e['res'], **b); log(task, name, 'seed', sd, ps[sd], flush=True)
        pool_ = np.mean(pers, 0); b = boot_diff(pool_ - c); out['sets'][name] = dict(n=len(seeds), classical=float(c.mean()), learned=float(pool_.mean()), per_seed=ps, **b); np.save(P(task, f'learned_{name}.npy'), np.array(pers))
        log(task, name, 'POOLED', out['sets'][name]['classical'], out['sets'][name]['learned'], b, flush=True)
    out['budget'] = {str(sd): dict(total_episodes=r['total_episodes'], train_episodes=r['train_episodes'], cal_eval_episodes=r['cal_eval_episodes'], control_steps=r['control_steps'], wall_s=r['wall_s'], selected=r['selected'], init_cal_s=r['init_cal_s']) for sd, r in res.items()}
    jdump(out, P(task, 'real.json'))

if __name__ == '__main__':
    cmd, task = sys.argv[1], sys.argv[2]
    {'cal': cmd_cal, 'gate': cmd_gate, 'real': cmd_real}[cmd](task)
