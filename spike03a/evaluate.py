"""Step 4: evaluate C0, C1, C2 (tuned params from tuned_W0_W3.json) and spike02 residual PPO (re-evaluated from the saved actor weights) on the locked TEST set (seeds 0-49), W0 and W3."""
import sys, json, pickle, time; sys.path.insert(0, 'spike03a'); from lib import *
import torch, torch.nn as nn
torch.set_num_threads(2)
tuned = json.load(open('results/spike03a/tuned_W0_W3.json'))
def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
def feats(q, qd, qr, qdr, qddr, tT):
    return np.concatenate([q, qd, qr, qdr, qddr / 3, tT / 30, (q - qr) * 20, (qd - qdr) * 3], 1).astype(np.float32)
class RL:
    def __init__(self, state): self.a = mlp(16, 2); self.a.load_state_dict(state); self.T = Teacher(0.0); self.maxres = 0.
    def __call__(self, q, qd, qr, qdr, qddr):
        tT = self.T.act(q, qd, qr, qdr, qddr)
        with torch.no_grad(): m = self.a(torch.from_numpy(feats(q, qd, qr, qdr, qddr, tT))).numpy()
        out = np.clip(tT + BOUND*np.clip(m, -1, 1), -TAU_MAX, TAU_MAX); self.maxres = max(self.maxres, float(np.abs(out-tT).max())); return out
out = {}
for wk in ['W0', 'W3']:
    w = W[wk]; t = tuned[wk]; R = {}
    R['C0'] = [evaluate(lambda: Ctrl(), w, TEST)]
    R['C1'] = [evaluate(lambda: Ctrl(t['C1']['sp'], t['C1']['sd']), w, TEST)]
    R['C2'] = [evaluate(lambda: Ctrl(t['C2']['sp'], t['C2']['sd'], t['C2']['ki']), w, TEST)]
    sp2 = pickle.load(open(f'results/spike02/raw_rl_{wk}.pkl', 'rb')); R['RL'] = []
    for s in range(5):
        pol = RL(sp2[('rl', s)]['state']); r = evaluate(lambda: pol, w, TEST); assert pol.maxres <= BOUND + 1e-6
        old = sp2[('rl', s)]; d = abs(r['clean']['rmse'] - old['clean']['rmse']).max(); print(wk, 'RL seed', s, 're-eval vs spike02 stored max|diff| rmse clean', d, flush=True)
        r['train_time'] = old['train_time']; r['steps'] = old['steps']; R['RL'].append(r)
    out[wk] = R; print(wk, 'done', flush=True)
pickle.dump(out, open('results/spike03a/eval.pkl', 'wb'))
