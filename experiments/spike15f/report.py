"""spike15f report: per-method warning counts (plan / model / plant) and diverged fraction of the ACTUALLY EXECUTED episode. cwd-independent."""
import sys, os, pickle, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, '..', '..')); sys.path.insert(0, os.path.join(ROOT, 'experiments', 'spike15c')); import analyzeB as AB
S = os.path.join(ROOT, '.scratch/s15f'); L = lambda n: pickle.load(open(os.path.join(S, n + '.pkl'), 'rb'))
R0, B, LAP = L('R0'), L('B'), L('LAP'); n = len(R0['seeds'])
def Dof(P): return dict(S=np.stack([x['r']['S'] for x in P['res']]), succ=np.stack([x['r']['succ'] for x in P['res']]), members=[0, 1, 2])
def chosen_call(x, ci):
    r = x['r']; tgt = r['cands'][ci]
    for c in x['calls']:
        f = c['forced'] if c['forced'] is not None else r['cands'][r['base']]
        if np.abs(f - tgt).max() < 1e-12: return c
    raise RuntimeError('no matching call')
_, chB = AB.vsucc(Dof(B), ('c11', 'M3c', 'max')); _, chC = AB.vsucc(Dof(LAP), ('c11', 'M3c', 'hit06'))
rows = {}
def add(name, P, calls_exec, succ, dist):
    mw = lambda k: int(sum(c0[k] for c0 in [P_c[0] for P_c in calls_exec['sel']])) if False else None
    rows[name] = None
def summ(name, sel_calls, exec_calls, succ, dist):
    ex = lambda k: int(sum(c[k] for c in exec_calls))
    bad = np.array([c['plant_reset'] > 0 or c['plant_nonfinite'] > 0 or c['diverged'] or not c['err_finite'] or not c['q_finite'] for c in exec_calls])
    pw = np.array([c['plant'] > 0 for c in exec_calls]); dfin = np.isfinite(np.asarray(dist, float)).all()
    rows[name] = dict(n=n, model_plan_warn=int(sum(c['plan'] for c in sel_calls)), model_other_warn=int(sum(c['model'] for c in sel_calls)), plant_warn=ex('plant'), plant_reset=ex('plant_reset'), plant_nonfinite=ex('plant_nonfinite'),
        episodes_with_plant_warn=int(pw.sum()), episodes_diverged=int(bad.sum()), diverged_pct=100 * float(bad.mean()), dist_all_finite=bool(dfin), success_pct=100 * float(np.mean(succ)),
        success_pct_excl_bad=(100 * float(np.mean(np.asarray(succ)[~bad])) if (~bad).any() else None), success_pct_bad_as_fail=100 * float(np.mean(np.asarray(succ) * ~bad)), pass_lt_0p5pct=bool(100 * bad.mean() < 0.5), n_succ_in_bad=int(np.asarray(succ)[bad].sum()))
summ('R0', [x['calls'][0] for x in R0['res']], [x['calls'][0] for x in R0['res']], [x['r']['success'] for x in R0['res']], [x['r']['min_dist'] for x in R0['res']])
summ('L-ens', [x['calls'][0] for x in B['res']], [x['calls'][0] for x in B['res']], [x['r']['succ'][x['r']['base']] for x in B['res']], [x['r']['dist'][x['r']['base']] for x in B['res']])
summ('B', [x['calls'][0] for x in B['res']], [chosen_call(x, c) for x, c in zip(B['res'], chB)], [x['r']['succ'][c] for x, c in zip(B['res'], chB)], [x['r']['dist'][c] for x, c in zip(B['res'], chB)])
summ('C-ens', [x['calls'][0] for x in LAP['res']], [chosen_call(x, c) for x, c in zip(LAP['res'], chC)], [x['r']['succ'][c] for x, c in zip(LAP['res'], chC)], [x['r']['dist'][c] for x, c in zip(LAP['res'], chC)])
# offline-only: all 11 candidate executions per pipeline
off = {}
for nm, P in (('B/L-ens pipeline', B), ('C-ens pipeline', LAP)):
    cs = [c for x in P['res'] for c in x['calls']]; off[nm] = dict(executions=len(cs), plant_warn=sum(c['plant'] for c in cs), exec_with_reset=sum(c['plant_reset'] > 0 for c in cs), exec_nonfinite=sum(c['plant_nonfinite'] > 0 or c['diverged'] for c in cs),
        model_warn_total=sum(c['plan'] + c['model'] for c in cs), model_plan=sum(c['plan'] for c in cs), model_other=sum(c['model'] for c in cs))
json.dump(dict(methods=rows, offline_all_candidates=off, other_nonnan_warn=dict(R0=sum(x['calls'][0]['other_warn'] for x in R0['res']))), open(os.path.join(ROOT, 'results/spike15f/summary.json'), 'w'), indent=1); print(json.dumps(dict(methods=rows, off=off), indent=1))
