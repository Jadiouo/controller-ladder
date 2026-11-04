import sys, os, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__)); import env as E
torch.set_num_threads(2)
OBS = 23
def load_conds():
    c = json.load(open('results/spike03b/calib.json')); out = {'nominal': E.Cond()}
    for fam in ('restitution', 'mass', 'sensing'):
        kw = dict(c[fam + '_chosen']) if c.get(fam + '_chosen') else None
        if kw: out[fam] = E.Cond(**kw)
    return out
TEST_SEEDS = np.arange(50)
def evaluate(cond, policy, maxk=None):
    env = E.JugEnv(50, cond); return E.run_policy(env, policy, E.init_cond(TEST_SEEDS), TEST_SEEDS, maxk=maxk)
def mlp(i, o, h=128, act=nn.ReLU, layers=3):
    mods = []; d = i
    for _ in range(layers): mods += [nn.Linear(d, h), act()]; d = h
    mods.append(nn.Linear(d, o)); return nn.Sequential(*mods)
class BCPolicy:
    def __init__(self, net, xm, xs, ym, ys): self.net, self.xm, self.xs, self.ym, self.ys = net, xm, xs, ym, ys
    def __call__(self, env, tT, obs):
        with torch.no_grad(): y = self.net(torch.from_numpy((obs - self.xm) / self.xs).float()).numpy()
        return np.clip(y * self.ys + self.ym, -E.TAU_MAX, E.TAU_MAX)
class ResPolicy:
    def __init__(self, actor, B=E.RES_BOUND): self.actor, self.B = actor, B
    def __call__(self, env, tT, obs):
        with torch.no_grad(): m = self.actor(torch.from_numpy(obs)).numpy()
        return np.clip(tT + self.B * np.clip(m, -1, 1), -E.TAU_MAX, E.TAU_MAX)
