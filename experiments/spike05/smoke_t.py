import sys, time, numpy as np; sys.path.insert(0,'experiments/spike05'); import env5 as V
import env as E
# 1) T-nom parity with spike03b teacher on a plain nominal-like config (no randomisation differences): compare torque of one step
seeds=np.arange(1000,1010)
t=time.time(); r=V.evaluate(V.P_NOM, seeds, cap=60); dt=time.time()-t
print('nom hits',r['hits'],'steps',r['steps'],'time',round(dt,1),'steps/s',round(r['steps']/dt)); print(r['fail'])
print([ (round(p['e'],2),round(p['mb'],1),round(p['delay_ms'])) for p in r['params']])
P=dict(V.P_NOM); P['adapt']=1
t=time.time(); r=V.evaluate(P, seeds, cap=60); dt=time.time()-t
print('adapt hits',r['hits'],'time',round(dt,1)); print(r['fail'])
for i in range(10): print(i, 'true e %.2f d %.1f'%(r['params'][i]['e'], r['params'][i]['delay']), 'e_obs', np.round(r['e_est'][i][:8],2), 'd_obs', np.round(r['d_est'][i][:8],1))
