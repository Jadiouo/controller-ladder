"""spike12a GPU trainer: one student (bc|mip|diffusion|flow) on one dataset npz; saves EMA checkpoints at several step counts. Same optimiser/batch/steps/EMA for all kinds.
python train12.py --data D.npz --kind mip --seed 0 --out DIR --tag multi [--device cuda]  (submit through a GPU job queue)"""
import argparse, copy, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from arena.learn.models import build_policy, n_params
TA = 16
ap = argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--kind'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out'); ap.add_argument('--tag'); ap.add_argument('--device', default='cuda')
ap.add_argument('--steps', default='10000,20000,40000,60000'); ap.add_argument('--batch', type=int, default=1024); ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--match', type=int, default=0)
a = ap.parse_args(); dev = torch.device(a.device); steps = [int(s) for s in a.steps.split(',')]; torch.manual_seed(a.seed)
d = np.load(a.data); X = torch.from_numpy(d['X']).to(dev); R = torch.from_numpy(d['R']).to(dev); ends = torch.from_numpy(d['ends']).to(dev); N = len(X)
om, os_ = X.mean(0), X.std(0) + 1e-6; am, as_ = R.mean(0), R.std(0) + 1e-6; Xn = (X - om) / os_; Rn = (R - am) / as_; off = torch.arange(TA, device=dev)
def chunk(idx): return Rn[torch.minimum(idx[:, None] + off[None], ends[idx][:, None])]       # (B, TA, 9), padded with the last row of the episode
model = build_policy(a.kind, X.shape[1], 9, TA, 256, 4, match_to=a.match or None); model.set_norm(om.cpu(), os_.cpu(), am.cpu(), as_.cpu()); model.to(dev)
ema = copy.deepcopy(model).eval().requires_grad_(False); gen = torch.Generator(device=dev).manual_seed(a.seed * 7919 + 13); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
print(f'kind={a.kind} tag={a.tag} seed={a.seed} rows={N} params={n_params(model)} dev={dev}', flush=True); os.makedirs(a.out, exist_ok=True); t0 = time.time(); run = torch.zeros((), device=dev); log = []
for s in range(max(steps)):
    for g in opt.param_groups: g['lr'] = a.lr * min(1.0, (s + 1) / 200)
    idx = torch.randint(0, N, (a.batch,), generator=gen, device=dev); loss = model.loss(Xn[idx], chunk(idx), gen); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        dd = min(0.999, (1 + s) / (10 + s))
        for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - dd)
    run += loss.detach()
    if (s + 1) % 1000 == 0: log.append((s + 1, float(run) / 1000)); run.zero_(); print(a.kind, s + 1, log[-1][1], f'{time.time()-t0:.0f}s', flush=True)
    if (s + 1) in steps:
        torch.save(dict(config=ema.config(), state={k: v.cpu() for k, v in ema.state_dict().items()}, params=n_params(model), steps=s + 1, seed=a.seed, tag=a.tag), os.path.join(a.out, f'{a.kind}_{a.tag}_s{a.seed}_{s+1}.pt'))
json.dump(dict(kind=a.kind, tag=a.tag, seed=a.seed, params=n_params(model), seconds=time.time() - t0, rows=N, log=log, gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0), open(os.path.join(a.out, f'{a.kind}_{a.tag}_s{a.seed}.json'), 'w'))
