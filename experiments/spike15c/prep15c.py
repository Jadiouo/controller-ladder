"""spike15c prep: raw probe data for new seeds (703000-703199) and the PPO training pool (600000-600299, rep 0); inference of the frozen L-dec4 nets (spike15a weights) and of the extra members (.scratch/s15c) on cal/hold/new/pool.
Writes .scratch/s15c/raw_{cal,hold,new,pool}.pkl (cal/hold copied from spike14a raw), est_{set}_m{0..8}.npy. Usage: prep15c.py raw | infer"""
import sys, os, pickle, subprocess, multiprocessing as mp, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c15c import *
what = sys.argv[1]; NW = int(os.environ.get('NW', 8))
RAW = {k: os.path.join(S15C, f'raw_{k}.pkl') for k in ('cal', 'hold', 'new', 'pool')}
if what == 'raw':
    for k in ('cal', 'hold'): shutil.copy(os.path.join(ROOT, f'.scratch/s14/raw_{k}.pkl'), RAW[k])
    pool = mp.Pool(NW)
    for k, seeds, rep in (('new', NEW, None), ('pool', POOL, 0)):
        if os.path.exists(RAW[k]): continue
        D = c14.gen_set(pool, [(s, rep, 'abc') for s in seeds], chunk=10); assert list(D['seed']) == seeds; pickle.dump(D, open(RAW[k], 'wb')); print('raw', k, len(seeds), flush=True)
    sys.exit(0)
# ---- inference (torch, CPU)
import torch; import nn_train as N
torch.set_num_threads(2)
MODELS = [os.path.join(ROOT, f'.scratch/s14/model_r15dec4_s{s}.pt') for s in range(3)] + [os.path.join(S15C, f'model_c15dec4_s{s}.pt') for s in range(3, 9)]
for k, p in RAW.items():
    D = pickle.load(open(p, 'rb')); X = N.seq_feats(D)
    for mi, mp_ in enumerate(MODELS):
        f = os.path.join(S15C, f'est_{k}_m{mi}.npy')
        if os.path.exists(f) or not os.path.exists(mp_): continue
        M = torch.load(mp_, map_location='cpu', weights_only=False); net = N.GRUNet(X.shape[2], M['H']); net.load_state_dict(M['sd']); net.eval()
        with torch.no_grad(): Y = net(torch.from_numpy(((X - M['mu']) / M['sd_in']).astype(np.float32))).numpy().astype(np.float64) * M['ysd'] + M['ymu']
        np.save(f, N.y_to_theta(Y)); print('infer', k, mi, Y.shape, flush=True)
