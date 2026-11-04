import sys, os, pickle, numpy as np, torch, torch.nn as nn, json
sys.path.insert(0, 'experiments/spike05'); import env5 as V
torch.set_num_threads(2)
CAL = np.arange(1000, 1020)
def net(i, h, o, sc=None):
    n = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    if sc is not None:
        for m in n:
            if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(n[-1].weight, sc)
    return n
class Res:
    def __init__(s, actor, std=0.0, seed=0): s.a = actor; s.std = std; s.g = torch.Generator().manual_seed(seed); s.rs=[]
    def __call__(s, env, tT, obs):
        with torch.no_grad():
            m = s.a(torch.from_numpy(obs))
            if s.std > 0: m = m + s.std * torch.randn(m.shape, generator=s.g)
            m = m.numpy()
        s.rs.append(np.abs(np.clip(m,-1,1)).mean())
        env.prev_act = np.clip(m, -1, 1).astype(np.float32)
        return np.clip(tT + V.RES_BOUND * np.clip(m, -1, 1), -V.TAU_MAX, V.TAU_MAX)
def ev(name, pol=None, hist=False, cap=300):
    r = V.evaluate(V.P_NOM, CAL, policy=pol, hist=hist, cap=cap)
    print(f'{name:40s} median {np.median(r["hits"]):5.1f} mean {r["hits"].mean():6.1f} hits {sorted(r["hits"].tolist())[:20]}', flush=True); return r
if __name__ == '__main__':
    ev('T-nom CAL')
    for var, hist in (('dr', False), ('hist', True)):
        OBS = 23 + (176 if hist else 0)
        torch.manual_seed(0); a = net(OBS, 256, 3, 0.01)
        p = Res(a); ev(f'init-{var} mean-action', p, hist); print('  mean|act|', np.mean(p.rs), '-> Nm', 12*np.mean(p.rs))
        ev(f'init-{var} sample std.135', Res(a, np.exp(-2.0)), hist)
        R = pickle.load(open(f'results/spike05/raw_rl_{var}.pkl', 'rb'))
        for s, v in sorted(R.items()):
            t = net(v['obs'], 256, 3); t.load_state_dict(v['state'])
            p = Res(t); ev(f'trained-{var} s{s} mean', p, hist); print('  mean|act| Nm', 12*np.mean(p.rs))
            ev(f'trained-{var} s{s} sample std.09', Res(t, 0.085), hist)
