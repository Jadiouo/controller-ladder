"""Steps 2-3: tune C1 (Kp,Kd scale) and C2 (C1-style gains + integral, extra torque clamped +-16 Nm) with CMA-ES on the CALIBRATION set only (seeds 1000-1049). Objective = 0.5*mean RMSE(clean)+0.5*mean RMSE(dist), transient included."""
import sys, json, time; sys.path.insert(0, 'spike03a'); from lib import *
LO = np.array([np.log(0.2), np.log(0.2), np.log(10.)]); HI = np.array([np.log(100.), np.log(100.), np.log(1e6)])

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

out = {}
for wk in sys.argv[1:] or ['W0', 'W3']:
    w = W[wk]; res = {}
    nev = [0]
    def mk(kind):
        def f(p):
            nev[0] += 1
            try: v = objective(np.append(p, 0) if kind == 'C1' else p, w, kind)[0]
            except Exception as ex: v = 1.0
            return v if np.isfinite(v) else 1.0
        return f
    t = time.time(); nev[0] = 0
    f1 = mk('C1'); b1, h1 = cmaes(f1, [0, 0], 0.5, 8, 10, LO[:2], HI[:2], seed=1)
    res['C1'] = dict(params=b1[1].tolist(), sp=float(np.exp(b1[1][0])), sd=float(np.exp(b1[1][1])), calib_obj=b1[0], evals=nev[0], sim_steps=nev[0]*100*NSTEP, wall=time.time()-t, dof=2)
    # C0 reference objective on the calibration set
    res['C0_calib_obj'] = float(f1(np.zeros(2))); nev[0] -= 1
    print(wk, 'C1', res['C1'], 'C0 obj', res['C0_calib_obj'], flush=True)
    t = time.time(); nev[0] = 0
    f2 = mk('C2'); x0 = np.append(b1[1], np.log(3000.)); b2, h2 = cmaes(f2, x0, 0.5, 8, 14, LO, HI, seed=2)
    res['C2'] = dict(params=b2[1].tolist(), sp=float(np.exp(b2[1][0])), sd=float(np.exp(b2[1][1])), ki=float(np.exp(b2[1][2])), calib_obj=b2[0], evals=nev[0], sim_steps=nev[0]*100*NSTEP, wall=time.time()-t, dof=3)
    print(wk, 'C2', res['C2'], flush=True)
    out[wk] = res; json.dump(out, open('results/spike03a/tuned_%s.json' % '_'.join(sys.argv[1:] or ['W0', 'W3']), 'w'), indent=1)
