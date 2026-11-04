"""pred_<tag>_s<k>.pkl (nn_train.py) -> est/<tag>_s<k>_<set>.npy ; prints label error summary on CAL. usage: export_preds.py tag[,tag...]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
for tag in sys.argv[1].split(','):
    for s in range(3):
        f = os.path.join(DAT, f'pred_{tag}_s{s}.pkl')
        if not os.path.exists(f): continue
        P = pickle.load(open(f, 'rb')); out = []
        for st in ('cal', 'hold', 'new'):
            D = pickle.load(open(os.path.join(DAT, f'raw_{st}.pkl'), 'rb')); np.save(os.path.join(DAT, 'est', f'{tag}_s{s}_{st}.npy'), P['preds'][st])
            if st == 'cal': out = err_summary(D['TH'], P['preds'][st])
        print(tag, s, 'best_ep', P['best_ep'], 'cal', {k: round(v, 3) for k, v in out.items()})
