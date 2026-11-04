"""spike12a: (a) same-observation output dispersion on held-out scenarios (seeds 530000-530049, 4 teacher plans each), (b) CPU inference latency.
python disp12.py name:ckpt[:steps] ...   -> results/spike12a/disp.json, latency.json.  Light CPU job, 1 thread."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from arena.learn.models import build_policy
torch.set_num_threads(1); R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'spike12a'); SD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.scratch', 's12')
TA = 16; KS = (0, 25, 50, 100)
def J(n):
    p = os.path.join(R, n); return json.load(open(p)) if os.path.exists(p) else {}
def teacher_spread(tag):
    d = np.load(os.path.join(SD, f'D_{tag}_heldout.npz')); ends = d['ends']; seed, plan = d['seed'], d['plan']; starts = np.r_[0, np.unique(ends)[:-1] + 1]; ue = np.unique(ends); out = {}
    for k in KS:
        sp = []
        for s in np.unique(seed):
            ch = []
            for j in range(4):
                e = [i for i in range(len(ue)) if seed[ue[i]] == s and plan[ue[i]] == j]
                if not e: continue
                i = e[0]; a, b = starts[i], ue[i]
                if a + k > b: continue
                idx = np.minimum(a + k + np.arange(TA), b); ch.append(d['R'][idx, :3])
            if len(ch) >= 2: sp.append(np.sqrt(np.var(np.array(ch), axis=0).mean()))
        out[k] = float(np.mean(sp)) if sp else float('nan')
    return out
def obs_rows(tag):
    d = np.load(os.path.join(SD, f'D_{tag}_heldout.npz')); ends = d['ends']; seed, plan = d['seed'], d['plan']; ue = np.unique(ends); starts = np.r_[0, ue[:-1] + 1]; rows = {k: [] for k in KS}
    for i in range(len(ue)):
        if plan[ue[i]] != 0: continue
        for k in KS:
            if starts[i] + k <= ue[i]: rows[k].append(d['X'][starts[i] + k])
    return {k: np.array(v) for k, v in rows.items()}
if __name__ == '__main__':
    disp, lat = J('disp.json'), J('latency.json')
    disp['teacher_spread_rad'] = {'multi': teacher_spread('multi'), 'canon': teacher_spread('canon')}
    for spec in sys.argv[1:]:
        name, path = spec.split(':')[:2]; st = int(spec.split(':')[2]) if spec.count(':') >= 2 else None
        ck = torch.load(path, map_location='cpu', weights_only=False); c = ck['config']; m = build_policy(c['kind'], c['obs_dim'], c['n'], c['chunk'], c['width'], c['depth']); m.load_state_dict(ck['state']); m.eval(); kw = {'steps': st} if st else {}
        rows = obs_rows('multi'); g = torch.Generator().manual_seed(0); res = {}
        for k in KS:
            sds = []
            for x in rows[k]:
                o = torch.from_numpy(x)[None].repeat(16, 1); y = m.act_chunk(o, g, **kw)[:, :, :3].numpy(); sds.append(np.sqrt(np.var(y, axis=0).mean()))
            res[k] = float(np.mean(sds))
        disp[name] = res; x1 = torch.from_numpy(rows[0][0])[None]; g = torch.Generator().manual_seed(0)
        for _ in range(20): m.act_chunk(x1, g, **kw)
        t = time.perf_counter()
        for _ in range(300): m.act_chunk(x1, g, **kw)
        ms = (time.perf_counter() - t) / 300 * 1e3; lat[name] = dict(ms_per_chunk_cpu1=ms, ms_per_control_step_amortised=ms / 8, params=ck['params']); print(name, res, lat[name], flush=True)
    json.dump(disp, open(os.path.join(R, 'disp.json'), 'w'), indent=1); json.dump(lat, open(os.path.join(R, 'latency.json'), 'w'), indent=1)
