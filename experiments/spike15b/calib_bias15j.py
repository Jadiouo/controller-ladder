"""S5 gate calibration: zero-residual success (300-hit cap) of the best classical (T-robust spike06) under a constant torque bias b*(1,-1,1) on CAL-A (n=50); also b=0 sanity (CAL 76% in spike07)."""
import sys, os, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rl15j import *
def f(b):
    PC = json.load(open(os.path.join(ROOT, 'results/spike07/best.json')))['params']; r = evaluate(PC, None, CAL_A, None if b == 0 else bias_vec(b)); return b, summ(r)
if __name__ == '__main__':
    with mp.Pool(6) as p: out = dict(p.map(f, [0, 1, 2, 3, 4, 6]))
    for b in sorted(out): print('bias', b, out[b], flush=True)
    json.dump(out, open(os.path.join(ROOT, 'results/spike15b/s5_gate_calib.json'), 'w'), indent=1)
