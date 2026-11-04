"""spike15a: L-dec4 inference (re-trained weights, see retrain_dec4.py) on a stored raw probe-data pickle -> theta13 per episode.
usage: infer_dec4.py <raw.pkl> <out_prefix>   writes <out_prefix>_s{0,1,2}.npy. CPU only; nothing here chooses or tunes anything."""
import sys, os, pickle, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spike14a')); import nn_train as N
raw, outp = sys.argv[1], sys.argv[2]; D = pickle.load(open(raw, 'rb')); X = N.seq_feats(D); torch.set_num_threads(2)
for s in range(3):
    M = torch.load(os.path.join(N.DAT, f'model_r15dec4_s{s}.pt'), map_location='cpu', weights_only=False); net = N.GRUNet(X.shape[2], M['H']); net.load_state_dict(M['sd']); net.eval()
    with torch.no_grad(): Y = net(torch.from_numpy(((X - M['mu']) / M['sd_in']).astype(np.float32))).numpy().astype(np.float64) * M['ysd'] + M['ymu']
    np.save(f'{outp}_s{s}.npy', N.y_to_theta(Y)); print('dec4 seed', s, 'done', Y.shape, flush=True)
