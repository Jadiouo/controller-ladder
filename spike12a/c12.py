"""spike12a shared code. Imports spike11a (s11/c11) unchanged. Teacher data with several CMA-ES streams per scenario (D-multi) or canonical warm-started plans (D-canon);
chunked students (arena.learn.models) that output a CHUNK of residual references on top of the canonical prior (same prior-residual recipe as spike11a 'canon_dagger3');
inner loop = same computed-torque controller. Run with cwd = .scratch/run."""
import sys, os; _H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'spike11a')); sys.path.insert(0, os.path.join(_H, '..'))
from s11 import *
OUT12 = os.path.join(ROOT, 'results/spike12a'); os.makedirs(OUT12, exist_ok=True); SD12 = os.path.join(ROOT, '.scratch/s12'); os.makedirs(SD12, exist_ok=True)
TA, EXEC, H, STRIDE = 16, 8, 6, 5            # chunk length, executed steps per sample, history window (as spike11a)
def guard12(seed, cal): assert (seed >= 500000) or (cal and 1000 <= seed < 1100), 'seed not allowed in spike12a'
_M = {}
def load_model(path, steps_override=None):
    import torch; torch.set_num_threads(1)
    if path not in _M:
        from arena.learn.models import build_policy
        ck = torch.load(path, map_location='cpu', weights_only=False); c = ck['config']; m = build_policy(c['kind'], c['obs_dim'], c['n'], c['chunk'], c['width'], c['depth']); m.load_state_dict(ck['state']); m.eval(); _M[path] = m
    return _M[path]
def run_ep12(a):
    """a: seed, cond, rep, [teacher: plan_j, canon] | [student: model(path), steps, kind override, noise ('fresh'|'fixed')], cal, record, log_ref, salt"""
    import torch; torch.set_num_threads(1)
    seed, rep, cond, cal = a['seed'], a.get('rep'), a.get('cond', 'abc'), a.get('cal', False); guard12(seed, cal); INST[0] = 0; sc, env, obs = new_env(seed, cond, rep); ctx = env.ctx
    arm = ArmModel(ctx.robot); Kp, Kd = lqr_double_integrator(DEF['qp'], DEF['qv']); fz = Feat(H, STRIDE); target = np.asarray(obs.ref['target'], float); Pk = canon_x0(target)
    teach = a.get('model') is None; T = None
    if teach: T = Teacher(env, seed, cond, rep, None, canon=a.get('canon', False)); T.seed = int(seed) * 10 + int(a.get('plan_j', 0))
    else:
        m = load_model(a['model']); gen = torch.Generator().manual_seed(int(seed) * 131 + int(a.get('salt', 0)) + 7); zgen = None
        if a.get('noise') == 'fixed': zgen = True
        kw = {}
        if a.get('steps'): kw['steps'] = int(a['steps'])
    X, R, QR = [], [], []; tips = [env.task.chain_tip(env).copy()]; prev = np.zeros(3); done = False; info = {}; fz.push(obs); chunk = None; k0 = 0
    while not done:
        k = obs.k; x = fz(obs, prev, target); pq, pv, pa = (z[0] for z in spline_g(fz.q1, np.zeros(3), Pk, 1.6, np.array([k * DT]))); x = np.concatenate([x, np.concatenate([pq - obs.q, pv, pa]).astype(np.float32)])
        if teach:
            T.maybe_plan(k, target); qr, qdr, qddr = T.ref(k)
            if a.get('record'): X.append(x); R.append(np.concatenate([qr - pq, qdr - pv, qddr - pa]).astype(np.float32))
        else:
            if chunk is None or k - k0 >= EXEC:
                if zgen:
                    gen.manual_seed(int(seed) * 131 + int(a.get('salt', 0)) + 7)       # fixed noise: identical z for every chunk of the episode
                t_ = time.time(); chunk = m.act_chunk(torch.from_numpy(x)[None], gen, **kw)[0].numpy().astype(np.float64); k0 = k
            y = chunk[k - k0]; qr, qdr, qddr = pq + y[:3], pv + y[3:6], pa + y[6:9]
        if a.get('log_ref'): QR.append(qr.copy())
        tau = ctc(arm, Kp, Kd, ctx.tau_limit, obs.q, obs.qd, qr, qdr, qddr); obs, _, term, trunc, info = env.step(tau); prev = tau; tips.append(env.task.chain_tip(env).copy()); done = term or trunc
        if not done: fz.push(obs)
    dist = np.linalg.norm(np.array(tips) - env.task.target, axis=1)
    out = dict(success=float(info['outcome'] == 'success'), min_dist=float(dist.min()), diverged=float(info.get('diverged', False)), inst=int(INST[0]), steps=env.k)
    if a.get('record'): out['X'] = np.array(X, np.float32); out['R'] = np.array(R, np.float32)
    if a.get('log_ref'): out['QR'] = np.array(QR)
    return out
