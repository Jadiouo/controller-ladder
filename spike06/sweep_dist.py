"""Step 1: coarse sweep of widened error distributions. CAL seeds 1000-1049 ONLY, spike05 T-robust / T-adapt params (no retuning), cap 300.
usage: sweep_dist.py name e_lo e_hi mb_lo mb_hi d_lo d_hi   -> appends to results/spike06/sweep_dist.jsonl"""
import sys, os, json, time, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env6 as W; V = W.V
SEEDS = np.arange(1000, 1050)
def chunk(a):
    P, s, dist = a; assert s.min() >= 1000; W.use(dist); r = V.evaluate(P, s); return r['hits'].tolist(), r['steps']
if __name__ == '__main__':
    n = sys.argv[1]; v = [float(x) for x in sys.argv[2:8]]; dist = dict(e=v[0:2], mb=v[2:4], delay=v[4:6]); pool = mp.Pool(2)
    for m in ('robust', 'adapt'):
        P = json.load(open(f'results/spike05/{m}.json'))['params']; t = time.time()
        out = pool.map(chunk, [(P, c, dist) for c in np.array_split(SEEDS, 2)]); h = np.array(sum((o[0] for o in out), [])); st = sum(o[1] for o in out)
        rec = dict(name=n, dist=dist, method='T-' + m + '(spike05 params)', frac_cap=float((h >= 300).mean()), mean=float(h.mean()), median=float(np.median(h)), steps=st, wall=time.time() - t)
        print(rec, flush=True); open('results/spike06/sweep_dist.jsonl', 'a').write(json.dumps(rec) + '\n')
