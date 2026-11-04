"""Gate reference: constant residual c*(+1,-1,+1)... applied as fixed action vectors; residual = -bias exactly restores the base. CAL (n=200), bias 6 Nm."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rl13 import *
if __name__ == '__main__':
    pool = mp.Pool(8); bl = bias_vec(6).tolist(); out = {}
    for c in [0.0, 0.1, 0.25, 0.5, 0.75]:
        st = [(np.zeros((3, 19 + 6 * len(c9b.LAGS))), -c * np.array([1.0, -1.0, 1.0]))]      # linear layer, weights 0, bias = -c*dir  -> const action
        e = eval_set(pool, st, 'ppo', CAL, 12.0, bl); out[c] = e['succ']; print('const residual = -%.2f*bias (%.1f Nm): CAL succ %.3f' % (c, 6 * c, e['succ']), flush=True)
    e = eval_set(pool, None, 'ppo', CAL, 12.0, None); out['nobias'] = e['succ']; print('no bias, zero residual: CAL succ', e['succ'])
    json.dump(out, open(os.path.join(ROOT, 'results/spike13a/gate_ref.json'), 'w'))
