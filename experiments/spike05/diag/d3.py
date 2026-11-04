# history alignment / NaN / episode-boundary checks
import sys; sys.path.insert(0,'experiments/spike05'); import numpy as np, env5 as V
seeds=np.array([1000,1001]); env=V.JugEnv5(2,V.P_NOM,hist=True); env.reset(V.init_cond(seeds),seeds)
obs=[]
for k in range(300):
    env.measure_update(); tT=env.teacher(); o=env.obs(tT); obs.append(o.copy()); env.step(tT)
obs=np.array(obs); print('nan',np.isnan(obs).any(),'abs max per hist-feature block (first 11):',np.abs(obs[:,23:34]).max(0).round(2))
# lag j (1-based) at step k must equal base feature of step k-16*j; fresh-fill before that equals step 0
f=lambda o:o[:,V.HF]
ok=True
for k in (5,40,100,250):
    for j in (1,2,5,16):
        src=max(k-16*j,0); a=obs[k,23+(j-1)*11:23+(j-1)*11+8]; b=f(obs[src])[0] if False else obs[src,0][V.HF]
        ok&=np.allclose(a[:8] if a.ndim==1 else a[0],b[:8],atol=1e-6) if False else np.allclose(obs[k,0,23+(j-1)*11:23+(j-1)*11+8],obs[src,0][V.HF])
print('lag alignment ok',ok)
sc=obs[:,0,:23]; print('base obs scale (max abs):',np.abs(sc).max(0).round(2))
# boundary: reset_one then first obs history == own first obs
env.reset_one(0,V.init_cond([2000])[0],2000); env.measure_update(); tT=env.teacher(); o=env.obs(tT)
print('post-reset lags all equal current:',np.allclose(o[0,23:34],o[0,34:45]) and np.allclose(o[0,23:31],o[0,V.HF]) )
