"""usage: run_classical.py <cfgname> <set,set,..> [init_est_name]  -> DAT/est/<cfg>_<set>.npy and results/spike14a/classical_cost.json (cpu-seconds, nfev)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from classical import *
if __name__ == '__main__':
    name = sys.argv[1]; pool = mp.Pool(8); f = os.path.join(OUT14, 'classical_cost.json')
    for st in sys.argv[2].split(','):
        D = pickle.load(open(os.path.join(DAT, f'raw_{st}.pkl'), 'rb')); t0 = time.time(); inits = np.load(os.path.join(DAT, 'est', sys.argv[3] + f'_{st}.npy')) if len(sys.argv) > 3 else None; TE, cpu, nf = estimate_set(pool, name, D, inits=inits); wall = time.time() - t0
        np.save(os.path.join(DAT, 'est', f'{name}_{st}.npy'), TE); c = json.load(open(f)) if os.path.exists(f) else {}; c[f'{name}|{st}'] = dict(n=len(TE), cpu_s=cpu, wall_s=wall, nfev=nf); json.dump(c, open(f, 'w'), indent=1)
        print(name, st, f'wall {wall:.0f}s cpu {cpu:.0f}s', 'err', {k: round(v, 3) for k, v in err_summary(D['TH'], TE).items()}, flush=True)
