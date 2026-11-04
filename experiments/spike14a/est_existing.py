"""baseline estimators computed offline on the stored probe data: L-est-full models (spike10b, 3 training seeds) -> est/lmlp1_s<k>_<set>.npy ; R1 (true theta) -> est/R1_<set>.npy"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
for st in ('cal', 'hold', 'new'):
    D = pickle.load(open(os.path.join(DAT, f'raw_{st}.pkl'), 'rb')); np.save(os.path.join(DAT, 'est', f'R1_{st}.npy'), D['TH'])
    F = np.clip(np.nan_to_num(np.concatenate([(D['Qm'] - D['Q0n'])[:, 1::2].reshape(len(D['Qm']), -1), (D['Vm'] - D['V0n'])[:, 1::2].reshape(len(D['Vm']), -1)], 1)), -2, 2)
    for s in range(3):
        sd = pickle.load(open(os.path.join(OUT, f'lest_full_s{s}.pkl'), 'rb')); X = (F - sd['mu']) / sd['sd']
        for i, (Wt, b) in enumerate(sd['layers']): X = X @ Wt.T + b; X = np.tanh(X) if i < len(sd['layers']) - 1 else X
        TE = np.array([y_to_theta(x * sd['ysd'] + sd['ymu'], 'full') for x in X]); np.save(os.path.join(DAT, 'est', f'lmlp1_s{s}_{st}.npy'), TE); print(st, s, {k: round(v, 3) for k, v in err_summary(D['TH'], TE).items()})
