"""spike15e: CPU inference of the 9 frozen L-dec4 nets (results/spike15c/models) on a raw probe-data pickle. usage: infer15e.py <raw.pkl> <out_prefix> -> <out_prefix>_m{0..8}.npy. Nothing is chosen or tuned here."""
import sys, os, pickle, numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, '..')); sys.path.insert(0, os.path.join(ROOT, 'spike14a')); import nn_train as N
raw, outp = sys.argv[1], sys.argv[2]; D = pickle.load(open(raw, 'rb')); X = N.seq_feats(D); torch.set_num_threads(2)
for mi in range(9):
    f = os.path.join(ROOT, 'results/spike15c/models', f'model_r15dec4_s{mi}.pt' if mi < 3 else f'model_c15dec4_s{mi}.pt')
    M = torch.load(f, map_location='cpu', weights_only=False); net = N.GRUNet(X.shape[2], M['H']); net.load_state_dict(M['sd']); net.eval()
    with torch.no_grad(): Y = net(torch.from_numpy(((X - M['mu']) / M['sd_in']).astype(np.float32))).numpy().astype(np.float64) * M['ysd'] + M['ymu']
    np.save(f'{outp}_m{mi}.npy', N.y_to_theta(Y)); print('net', mi, Y.shape, flush=True)
