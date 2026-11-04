import sys, pickle; sys.path.insert(0,'spike03b'); from stu import *; from run_rl_net import make_actor
c = load_conds()['restitution']; R = pickle.load(open('results/spike03b/raw_rl_restitution.pkl','rb'))[('rl',0)]
a = make_actor(); a.load_state_dict(R['state'])
for name, pol in (('teacher', None), ('student', ResPolicy(a))):
    env = E.JugEnv(10, c); tl = []; 
    def rec(e):
        for i in range(10):
            if e.active[i] and e.d[i].qvel[4] < -0.1 and e.d[i].qpos[4]-E.OFF < 0.1: tl.append((abs(e.d[i].qpos[:3].sum()), abs(e.d[i].qpos[3]-E.fk(e.d[i].qpos[:3])[0])))
    r = E.run_policy(env, pol, E.init_cond(np.arange(10)), np.arange(10), record=rec, maxk=int(15/E.DT))
    t = np.array(tl); print(name, 'hits', r['hits'], 'pre-contact |tilt| median/90%', np.median(t[:,0]), np.percentile(t[:,0],90), '|dx| median/90%', np.median(t[:,1]), np.percentile(t[:,1],90))
