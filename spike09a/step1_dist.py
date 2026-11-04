"""Step 1: coarse width sweep of the error distribution on CAL 1000-1049 with T-nom; pick the mildest width whose nominal success is in [20%, 60%]."""
import time, multiprocessing as mp; from common import *
if __name__ == '__main__':
    pool = mp.Pool(8); S = np.arange(1000, 1050); rows = {}; t0 = time.time()
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        s, md, hs = peval(pool, S, w, None); rows[w] = dict(succ=float(s.mean()), ci=boot_ci(s), min_dist=float(np.mean(md))); print(w, rows[w], f'{time.time()-t0:.0f}s', flush=True)
    pick = next((w for w in sorted(rows) if w > 0 and 0.2 <= rows[w]['succ'] <= 0.6), None); print('pick', pick)
    dist = dict(w=pick, form='per-episode seed rng([seed,9109]): len_rel~U(-w,w) (all links), mass_factor=exp(U(-5w,5w)), damp_factor=exp(U(-10w,10w)); chain_len/mass/damp scaled uniformly over links',
                sweep=rows, rule='mildest width with T-nom success on CAL 1000-1049 in [20%,60%]', cal_seeds='1000-1049', holdout='1100-1599')
    json.dump(dist, open(os.path.join(OUT, 'dist.json'), 'w'), indent=1)
