"""spike15d prep: raw probe data of the new seeds 704000-704199 and CPU inference of the 3 frozen L-dec4 nets on it. usage: prep15d.py raw | infer"""
import sys, os, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c15d import *
what = sys.argv[1]; NW = int(os.environ.get('NW', 8)); RAWF = os.path.join(S15D, 'raw_new.pkl')
if what == 'raw':
    pool = mp.Pool(NW); D = c14.gen_set(pool, [(s, None, 'abc') for s in NEWD], chunk=10); assert list(D['seed']) == NEWD; pickle.dump(D, open(RAWF, 'wb')); print('raw new', len(NEWD), flush=True)
else:
    import torch; import nn_train as N
    torch.set_num_threads(2)
    for k, p in (('new', RAWF), ('hold', os.path.join(S15C, 'raw_hold.pkl'))):
        D = pickle.load(open(p, 'rb')); X = N.seq_feats(D)
        for mi in range(3):
            M = torch.load(os.path.join(RES15C, 'models', f'model_r15dec4_s{mi}.pt'), map_location='cpu', weights_only=False); net = N.GRUNet(X.shape[2], M['H']); net.load_state_dict(M['sd']); net.eval()
            with torch.no_grad(): Y = net(torch.from_numpy(((X - M['mu']) / M['sd_in']).astype(np.float32))).numpy().astype(np.float64) * M['ysd'] + M['ymu']
            th = N.y_to_theta(Y); f = os.path.join(S15D, f'est_{k}_m{mi}.npy')
            if k == 'hold': print('hold reproduces spike15c est (max abs diff)', np.abs(th - np.load(os.path.join(S15C, f'est_hold_m{mi}.npy'))).max()); continue
            np.save(f, th); print('infer', k, mi, th.shape, flush=True)
