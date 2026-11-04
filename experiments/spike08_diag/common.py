"""spike08_diag shared helpers. CAL-only (seeds >= 1000). Run from repo root. Uses spike06 env (dist.json) and spike06 T-robust params (best-classical)."""
import sys, os, json, numpy as np, multiprocessing as mp
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.join(ROOT, 'experiments', 'spike06')); sys.path.insert(0, os.path.join(ROOT, 'experiments', 'spike05'))
import env6 as W; V = W.V; DIST = W.use_file(os.path.join(ROOT, 'results/spike06/dist.json'))
PROB = json.load(open(os.path.join(ROOT, 'results/spike06/robust.json')))['params']
OUT = os.path.join(ROOT, 'results/spike08_diag')
def guard(seeds): assert np.min(seeds) >= 1000, 'TEST seeds forbidden'
def boot_ci(x, n=10000, seed=0, a=0.05):
    x = np.asarray(x, float); r = np.random.default_rng(seed); m = r.choice(len(x), (n, len(x))); b = x[m].mean(1); return float(np.quantile(b, a / 2)), float(np.quantile(b, 1 - a / 2))
def _ev(args):
    P, seeds, cap, polf = args; guard(seeds); pol = polf(seeds) if polf else None; r = V.evaluate(P, np.asarray(seeds), cap=cap, policy=pol); return r['hits'], r['steps']
def peval(pool, P, seeds, nw, cap=300, polf=None):
    """parallel evaluate over chunks of seeds; returns hits array (order of seeds), total steps"""
    seeds = np.asarray(seeds); ch = np.array_split(np.arange(len(seeds)), min(nw, len(seeds))); out = pool.map(_ev, [(P, seeds[c], cap, polf) for c in ch if len(c)])
    return np.concatenate([o[0] for o in out]), sum(o[1] for o in out)
