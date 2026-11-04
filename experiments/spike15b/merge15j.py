import pickle, sys, os, glob
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results/spike15b/runs_s5'); out = {}
for s in range(3): out.update(pickle.load(open(os.path.join(RUNS, f'real_ls-2.0_s{s}.pkl'), 'rb')))
pickle.dump(out, open(os.path.join(RUNS, 'real_ls-2.0.pkl'), 'wb')); print(sorted(out))
