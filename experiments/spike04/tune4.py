"""Step 3: re-tune C1 (Kp,Kd scale) and C1F (C1 + first-order LPF cutoff) under a noise condition with the same CMA-ES / pop 8 / box as spike03a, calibration seeds only.
Usage: python experiments/spike04/tune4.py N1|N2"""
import sys, json, time; sys.path.insert(0, 'experiments/spike04'); from lib4 import *
LO = np.array([np.log(0.2), np.log(0.2), np.log(1.0)]); HI = np.array([np.log(100.), np.log(100.), np.log(100.)])

def cmaes(f, x0, sigma0, lam, gens, lo, hi, seed=0):
    rng = np.random.default_rng(seed); n = len(x0); mu = lam//2
    wts = np.log(mu + .5) - np.log(np.arange(1, mu+1)); wts /= wts.sum(); mueff = 1/(wts**2).sum()
    cs = (mueff+2)/(n+mueff+5); ds = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + cs
    cc = (4+mueff/n)/(n+4+2*mueff/n); c1 = 2/((n+1.3)**2+mueff); cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
    chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
    m = np.array(x0, float); sig = sigma0; C = np.eye(n); ps = np.zeros(n); pc = np.zeros(n); hist = []; best = (np.inf, None)
    for g in range(gens):
        D2, B = np.linalg.eigh(C); D = np.sqrt(np.maximum(D2, 1e-20))
        Z = rng.standard_normal((lam, n)); Y = Z*D @ B.T; X = np.clip(m + sig*Y, lo, hi)
        Y = (X - m)/sig; fs = np.array([f(x) for x in X]); idx = np.argsort(fs)
        for i in range(lam): hist.append((X[i].tolist(), float(fs[i])))
        if fs[idx[0]] < best[0]: best = (float(fs[idx[0]]), X[idx[0]].copy())
        yw = (wts[:, None]*Y[idx[:mu]]).sum(0); m = m + sig*yw
        Cinv2 = B @ np.diag(1/D) @ B.T; ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff)*Cinv2@yw
        hs = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chiN < 1.4+2/(n+1))
        pc = (1-cc)*pc + hs*np.sqrt(cc*(2-cc)*mueff)*yw
        C = (1-c1-cmu)*C + c1*(np.outer(pc, pc) + (1-hs)*cc*(2-cc)*C) + cmu*sum(wts[i]*np.outer(Y[idx[i]], Y[idx[i]]) for i in range(mu))
        sig *= np.exp((cs/ds)*(np.linalg.norm(ps)/chiN - 1)); print(f'  gen {g} best {best[0]:.5f} sigma {sig:.3f}', flush=True)
    return best, hist

cn = sys.argv[1]; w = COND[cn]; sigma = json.load(open('results/spike04/calib_sigma.json'))['sigma']; res = dict(sigma=sigma, cond=cn); nev = [0]
def mk(kind):
    def f(p):
        nev[0] += 1
        try: v = objective4(p, w, kind, sigma)[0]
        except Exception as ex: v = 1.0
        return v if np.isfinite(v) else 1.0
    return f
t = time.time(); f1 = mk('C1'); b1, h1 = cmaes(f1, [0, 0], 0.5, 8, 10, LO[:2], HI[:2], seed=1)
res['C1'] = dict(params=b1[1].tolist(), sp=float(np.exp(b1[1][0])), sd=float(np.exp(b1[1][1])), calib_obj=b1[0], evals=nev[0], sim_steps=nev[0]*100*NSTEP, wall=time.time()-t, dof=2)
nev[0] = 0; res['C0_calib_obj'] = float(f1(np.zeros(2))); nev[0] = 0
old = json.load(open('results/spike03a/tuned_W0_W3.json'))['W0']['C1']; res['C1old_calib_obj'] = float(f1(np.log([old['sp'], old['sd']]))); nev[0] = 0
print(cn, 'C1', res['C1'], 'C0 obj', res['C0_calib_obj'], 'C1old obj', res['C1old_calib_obj'], flush=True)
t = time.time(); f2 = mk('C1F'); x0 = np.append(b1[1], np.log(30.)); b2, h2 = cmaes(f2, x0, 0.5, 8, 14, LO, HI, seed=2)
res['C1F'] = dict(params=b2[1].tolist(), sp=float(np.exp(b2[1][0])), sd=float(np.exp(b2[1][1])), fc=float(np.exp(b2[1][2])), calib_obj=b2[0], evals=nev[0], sim_steps=nev[0]*100*NSTEP, wall=time.time()-t, dof=3)
print(cn, 'C1F', res['C1F'], flush=True)
res['Cstar'] = 'C1F' if res['C1F']['calib_obj'] < res['C1']['calib_obj'] else 'C1'
print('C* =', res['Cstar']); json.dump(res, open('results/spike04/tuned_%s.json' % cn, 'w'), indent=1)
