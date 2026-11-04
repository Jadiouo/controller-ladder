"""C0 (= T-adapt3-full) estimates for the holdout from the spike13b est3 caches (same probe data, same frozen cfg) -> est/C0_hold.npy ; missing episodes are listed (then computed by run_classical.py)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from classical import *
tag = cfg_tag(cfg0()); out = []; miss = []
for st in sys.argv[1:]:
    D = pickle.load(open(os.path.join(DAT, f'raw_{st}.pkl'), 'rb')); out = []; miss = []
    for i in range(len(D['key'])):
        f = os.path.join(PL, 'est3_' + est_key(tag, D['q0'][i], D['taus'][i], D['Qm'][i]) + '.npy')
        if os.path.exists(f): out.append(np.load(f))
        else: out.append(np.full(13, np.nan)); miss.append(i)
    out = np.array(out); print(st, 'missing', len(miss), 'of', len(out))
    if not miss:
        np.save(os.path.join(DAT, 'est', f'C0_{st}.npy' if st != 'cal' else 'C0cache_cal.npy'), out); print({k: round(v, 3) for k, v in err_summary(D['TH'], out).items()})
