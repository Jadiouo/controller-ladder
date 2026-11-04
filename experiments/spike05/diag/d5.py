import sys; sys.path.insert(0,'experiments/spike05/diag'); from d1 import *
class ResNo(Res):
    def __call__(s, env, tT, obs):
        r = super().__call__(env, tT, obs); env.prev_act[:] = 0; return r
for tag, noact in (('lr1e-4', False), ('noact', True)):
    v = pickle.load(open(f'results/spike05/diag/fix_{tag}.pkl','rb'))[0]
    t = net(v['obs'],256,3); t.load_state_dict(v['state'])
    p = (ResNo if noact else Res)(t); ev(f'fix-hist {tag} mean', p, True); print('  mean|res| Nm', 12*np.mean(p.rs))
