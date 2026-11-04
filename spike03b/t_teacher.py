import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import env as E
cond = E.Cond(**eval(sys.argv[1])) if len(sys.argv) > 1 else E.Cond()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
seeds = np.arange(1000, 1000+n); ics = E.init_cond(seeds); env = E.JugEnv(n, cond)
t = time.time(); r = E.run_policy(env, None, ics, seeds); dt = time.time()-t
print(cond, 'time %.1fs'%dt, 'steps', r['dur'].sum()/E.DT)
print('hits', r['hits'], r['fail']); print('median', np.median(r['hits']), 'apex err', np.nanmean(r['apex']), 'drift', np.nanmean(r['drift']))
