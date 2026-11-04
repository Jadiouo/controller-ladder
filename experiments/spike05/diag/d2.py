import sys; sys.path.insert(0,'experiments/spike05/diag'); from d1 import *
torch.manual_seed(0); a = net(23, 256, 3, 0.01)
for sd in (0.05, 0.02, 0.01):
    ev(f'init-dr sample std={sd} ({12*sd:.2f} Nm)', Res(a, sd), False)
