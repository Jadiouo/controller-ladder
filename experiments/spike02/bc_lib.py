import numpy as np, torch, torch.nn as nn
from sim2 import TAU_MAX
HID, BS, LR = 128, 1024, 1e-3   # same as spike01

class NPPolicy:
    def __init__(self, net, xm, xs, ym, ys):
        self.W = [(l.weight.detach().numpy().T.astype(np.float64), l.bias.detach().numpy().astype(np.float64)) for l in net if isinstance(l, nn.Linear)]
        self.xm, self.xs, self.ym, self.ys = xm, xs, ym, ys
    def __call__(self, q, qd, qr, qdr, qddr):
        h = (np.concatenate([q, qd, qr, qdr, qddr], 1) - self.xm) / self.xs
        for i, (W, b) in enumerate(self.W):
            h = h @ W + b
            if i < len(self.W) - 1: h = np.maximum(h, 0)
        return np.clip(h * self.ys + self.ym, -TAU_MAX, TAU_MAX)

def new_net():
    return nn.Sequential(nn.Linear(10, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 2))

def norm_stats(X, Y):
    return X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6

def fit(net, X, Y, stats, epochs):
    xm, xs, ym, ys = stats
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32); Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=LR); n = len(Xt); steps = epochs * ((n + BS - 1) // BS)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]; loss = ((net(Xt[idx]) - Yt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return float(loss.detach())
