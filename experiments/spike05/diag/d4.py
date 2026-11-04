import sys; sys.path.insert(0,'experiments/spike05/diag'); from d1 import *
v = pickle.load(open('results/spike05/diag/fix_hist.pkl','rb'))[0]
t = net(v['obs'],256,3); t.load_state_dict(v['state'])
p = Res(t); r=ev('fix-hist (logstd0=-3.5, 1M) mean', p, True); print('  mean|res| Nm', 12*np.mean(p.rs))
ev('fix-hist sample std.03', Res(t,0.03), True)
