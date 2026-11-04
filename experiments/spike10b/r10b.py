"""spike10b: method factories, residual-RL environment/eval with realistic conditions (extends c10b). Run with cwd = .scratch/run."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c10b import *
def factory_of(spec):
    """controller factory for a method spec: 'nom'|'robust'|'adapt'|('adapt2',cfg)|('adapt3',cfg)|('est',tag,estfn,T,amp)"""
    reset_probe_globals()
    if spec in ('nom', 'robust', 'adapt'): return make_factory(spec)
    if spec[0] == 'adapt2': return make_adapt2_10b(spec[1])
    if spec[0] == 'adapt3': return make_adapt3(spec[1])
    if spec[0] == 'est': return lambda ctx: Probe3(ctx, spec[2], spec[1], spec[3], np.array([0.5, 0.5, 0.4]) * spec[4])
    raise ValueError(spec)
class ResEnv10(c9b.ResEnv):
    def __init__(self, base, fz, seed, rep, cond='abc'):
        self.sc = scen(seed, cond, rep); self.env = self.sc.make_env(); self.ctx = self.env.ctx; self.base = factory_of(base)(self.ctx); self.fz = fz; self.yt = np.zeros(3, np.float32); self.seed = seed
        self.obs = self.env.reset(seed); self.base.reset(seed); self.fz.reset(); self.fz.push(self.obs); self.task = self.env.task
        self.dmin = float(np.linalg.norm(self.task.chain_tip(self.env) - self.task.target)); self.rn = []; self.tips = [self.task.chain_tip(self.env).copy()]; self.en = []
        self.tb = self.base.act(self.obs); self.x = self.fz(self.obs, self.tb, self.base, self.yt)
def run_residual_episode10(args):
    """args = (base_spec, state_layers_or_None, seed, rep, cond, allow) -> metrics (same definitions as run10 / arena)"""
    base, state, seed, rep, cond, allow = args; guard([seed], allow); INST[0] = 0; fz = c9b.Feat(True, False); e = ResEnv10(base, fz, seed, rep, cond); done = False; x = e.x
    while not done:
        a = np.zeros(3) if state is None else np.clip(c9b.np_mlp(state, x.astype(np.float64)), -1, 1); x, r, done, info = e.step(a)
    tip = np.array(e.tips); dist = np.linalg.norm(tip - e.task.target, axis=1); hit = np.where(dist < e.task.r_hit)[0]
    hs = float(np.linalg.norm(tip[max(int(hit[0]), 1)] - tip[max(int(hit[0]), 1) - 1]) / e.ctx.dt) if len(hit) and len(tip) > 1 else float('nan')
    return dict(success=float(info['success']), min_dist=float(dist.min()), hit_speed=hs, energy=float(np.mean(e.en)), diverged=0.0, inst=int(INST[0]), res_mean=float(np.mean(e.rn)) if e.rn else 0.0, res_max=float(np.max(e.rn)) if e.rn else 0.0, steps=len(e.en), theta=true_theta(seed, cond, rep).tolist())
import pickle, glob
def jload(p): return json.load(open(os.path.join(OUT, p)))
def probe_of_lest(): return jload('probe_for_lest.json')       # {'T','amp'} written by step1c (probe of the better T-adapt3 parametrisation)
def lest_models(var):
    return {s: pickle.load(open(os.path.join(OUT, f'lest_{var}_s{s}.pkl'), 'rb')) for s in range(3)}
def spec_of(name):
    """method name -> run10 spec. 'lest_<var>#<s>' -> ('est', tag, MLPEst10, T, amp)"""
    if name in ('nom', 'robust', 'adapt', 'oracle'): return name
    if name == 'adapt2': return ('adapt2', json.load(open(os.path.join(ROOT, 'results/spike10a/adapt2_tuning.json')))['chosen'])
    if name == 'adapt3_grad': return ('adapt3', jload('tune_grad.json')['chosen'])
    if name == 'adapt3_full': return ('adapt3', jload('tune_full.json')['chosen'])
    if name == 'adapt3_ood': return ('adapt3', jload('tune_ood.json')['chosen'])
    if name.startswith('lest_'):
        var, s = name[5:].split('#'); st = pickle.load(open(os.path.join(OUT, f'lest_{var}_s{s}.pkl'), 'rb')); pr = probe_of_lest()
        return ('est', f'lest_{var}{s}_' + hashlib.md5(np.concatenate([w.ravel() for w, b in st['layers']]).tobytes()).hexdigest()[:8], MLPEst10(st), pr['T'], pr['amp'])
    raise ValueError(name)
