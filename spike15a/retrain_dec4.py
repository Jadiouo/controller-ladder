"""spike15a: the L-dec4 networks of spike14a were not saved (only their predictions on cal/hold/new). Re-train them with the IDENTICAL recipe
(spike14a/nn_train.py run('dec4', H=192, lr=2e-3, alpha=0.9, seeds 0,1,2, EP=38); checkpoint = lowest CAL loss) and additionally save the weights.
nn_train.py is not modified: its source is loaded with one patched line (weights are also saved). Output tag r15dec4 (stored spike14a predictions are not overwritten).
Uses only training seeds >= 500000 and CAL 1000-1049/2000-2149 labels; no TEST-C seed is touched here."""
import os, sys, types, torch
H = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(H, '..', 'spike14a', 'nn_train.py')).read()
old = "net.load_state_dict(best[1]); net.eval(); preds = {}"
assert src.count(old) == 1
new = old + "; torch.save(dict(sd=best[1], mu=mu, sd_in=sd, ymu=ymu, ysd=ysd, H=H, best_ep=best[2]), os.path.join(DAT, f'model_{tag}_s{seed}.pt'))"
m = types.ModuleType('nn_train14'); m.__file__ = os.path.join(H, '..', 'spike14a', 'nn_train.py'); m.__name__ = 'nn_train14'
exec(compile(src.replace(old, new), m.__file__, 'exec'), m.__dict__)
torch.use_deterministic_algorithms(False)
m.run('dec4', 192, 0.002, 0.9, [0, 1, 2], 'r15dec4', 38)
