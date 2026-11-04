"""spike15c: extra L-dec4 ensemble members (seeds 3..8), IDENTICAL recipe to experiments/spike15a/retrain_dec4.py (dec4, H=192, lr 2e-3, alpha 0.9, 38 epochs, checkpoint = lowest CAL loss).
Weights -> .scratch/s15c/model_c15dec4_s{seed}.pt. Training data: seeds >= 500000 only; CAL labels for the checkpoint. No TEST seed involved."""
import os, sys, types, torch
H = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(H, '..', 'spike14a', 'nn_train.py')).read()
old = "net.load_state_dict(best[1]); net.eval(); preds = {}"; assert src.count(old) == 1
OUTD = os.path.join(H, '..', '..', '.scratch', 's15c'); os.makedirs(OUTD, exist_ok=True)
new = old + f"; torch.save(dict(sd=best[1], mu=mu, sd_in=sd, ymu=ymu, ysd=ysd, H=H, best_ep=best[2]), os.path.join({OUTD!r}, f'model_{{tag}}_s{{seed}}.pt'))"
m = types.ModuleType('nn_train14'); m.__file__ = os.path.join(H, '..', 'spike14a', 'nn_train.py'); m.__name__ = 'nn_train14'
exec(compile(src.replace(old, new), m.__file__, 'exec'), m.__dict__)
m.DAT = m.DAT   # data dir unchanged (reads raw_*.pkl, sens_train.pkl); predictions pickles go to .scratch/s14 under the new tag
torch.use_deterministic_algorithms(False)
m.run('dec4', 192, 0.002, 0.9, [3, 4, 5, 6, 7, 8], 'c15dec4', 38)
