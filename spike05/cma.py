"""Minimal (mu/mu_w, lambda)-CMA-ES (Hansen 2016 defaults), maximisation, in the unit hypercube with clipping. No external dependency."""
import numpy as np
class CMA:
    def __init__(self, x0, sigma0, popsize, seed=0):
        n = len(x0); self.n = n; self.lam = popsize; self.mu = popsize // 2; self.rng = np.random.default_rng(seed)
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); self.w = w / w.sum(); self.mueff = 1 / (self.w**2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n); self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3)**2 + self.mueff); self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs; self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.array(x0, float); self.sigma = sigma0; self.C = np.eye(n); self.pc = np.zeros(n); self.ps = np.zeros(n); self.gen = 0
    def ask(self):
        D2, B = np.linalg.eigh(self.C); D = np.sqrt(np.maximum(D2, 1e-20)); self.B, self.D = B, D
        self.z = self.rng.standard_normal((self.lam, self.n)); self.y = (self.z * D) @ B.T
        return np.clip(self.m + self.sigma * self.y, 0, 1)
    def tell(self, X, f):
        idx = np.argsort(-np.asarray(f)); old = self.m.copy(); Xs = X[idx[:self.mu]]
        yw = ((Xs - old) / self.sigma).T @ self.w; self.m = old + self.sigma * yw
        Cinvsqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cinvsqrt @ yw
        self.gen += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.gen)) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        Y = (Xs - old) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * (Y.T * self.w) @ Y
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)); self.sigma = min(self.sigma, 0.5)
