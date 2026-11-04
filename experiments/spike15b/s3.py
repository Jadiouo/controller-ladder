"""spike15b S3: rerun of the learn_demo students on tracking/clean with the FIXED reference; hyper-parameters selected on CAL only; final on holdout 1100-1599 and fresh 702500-702699. TEST never touched.
Stages: collect | train_grid | evalcal | select | train_final | evalfinal | report.  Uses arena/learn unchanged."""
import argparse, json, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from arena import Scenario
from arena.controllers import controller_factory
from arena.eval import TRAIN_SEEDS, CAL_SEEDS, evaluate
OUT = "results/spike15b/s3"; HIST, CHUNK, EXEC_K = 3, 8, 4; STEPS, BATCH = 15000, 256
KINDS = ["bc", "mip", "diffusion", "flow"]; DARTS = [0.05, 0.1, 0.2]; LRS = [3e-4, 1e-3]
VARIANTS = [("bc", "bc", {}), ("mip", "mip", {}), ("diffusion (DDIM-10)", "diffusion", {"steps": 10}), ("flow (10 steps)", "flow", {"steps": 10}), ("flow (1 step)", "flow", {"steps": 1})]
MAIN = {"bc": ("bc", {}), "mip": ("mip", {}), "diffusion": ("diffusion", {"steps": 10}), "flow": ("flow", {"steps": 10})}
SETS = dict(cal=list(range(1000, 1050)), holdout=list(range(1100, 1600)), fresh=list(range(702500, 702700)))
SC = Scenario("tracking", {"ref": "lissajous"}, history=HIST, name="tracking/clean")
def dd(d): return f"{OUT}/data_d{d}"
def ck(kind, d, lr, s): return f"{OUT}/ckpt/{kind}_d{d}_lr{lr:g}_s{s}.pt"
def stage_collect(a):
    from arena.learn import collect
    for d in DARTS:
        if os.path.exists(dd(d) + "/meta.json"): continue
        t = time.time(); collect(SC, "ctc_lqr", TRAIN_SEEDS[:200], dd(d), dart_sigma=d, shard_episodes=25, verbose=False); print(f"collected dart {d} in {time.time()-t:.0f}s", flush=True)
def train_one(kind, d, lr, s, device):
    from arena.learn.dataset import ShardedDataset
    from arena.learn.train import TrainCfg, train
    from arena.learn.models import build_policy, n_params
    p = ck(kind, d, lr, s)
    if os.path.exists(p): return None
    cfg = TrainCfg(steps=STEPS, batch=BATCH, chunk=CHUNK, seed=s, lr=lr, rel_ref=False, wrap_angles=True); ds = ShardedDataset(dd(d), CHUNK, False, True)
    ref = n_params(build_policy("mip", ds.spec.obs_dim, ds.n, CHUNK, cfg.width, cfg.depth))
    info = train(ds, kind, p, cfg, device, match_params=ref if kind == "bc" else None, verbose=False)
    print(kind, d, lr, s, "loss", info["final_loss"], f"{info['seconds']:.0f}s params {info['params']}", flush=True); return {k: v for k, v in info.items() if k != "log"}
def stage_train_grid(a):
    summ = json.load(open(f"{OUT}/train_summary.json")) if os.path.exists(f"{OUT}/train_summary.json") else {}
    for kind in KINDS:
        for d in DARTS:
            for lr in LRS:
                r = train_one(kind, d, lr, 0, a.device)
                if r: summ[f"{kind}_d{d}_lr{lr:g}_s0"] = r; json.dump(summ, open(f"{OUT}/train_summary.json", "w"), indent=1)
def stage_train_final(a):
    sel = json.load(open(f"{OUT}/selected.json")); summ = json.load(open(f"{OUT}/train_summary.json"))
    for kind in KINDS:
        d, lr = sel[kind]["dart"], sel[kind]["lr"]
        for s in (1, 2):
            r = train_one(kind, d, lr, s, a.device)
            if r: summ[f"{kind}_d{d}_lr{lr:g}_s{s}"] = r; json.dump(summ, open(f"{OUT}/train_summary.json", "w"), indent=1)
# ---- parallel evaluation (CPU). task = (kind of controller, ckpt, kw, seeds, split, label)
def _job(t):
    import torch; torch.set_num_threads(1)
    ckpt, kw, seeds, split = t
    if ckpt is None: r = evaluate(SC, "ctc_lqr", seeds, label="teacher", split=split)
    else: r = evaluate(SC, controller_factory("student", ckpt=ckpt, exec_k=EXEC_K, sample_kw=kw), seeds, label="s", split=split)
    return {m: r.metrics[m].tolist() for m in r.metrics} | {"outcome": r.outcomes.tolist()}
def run_eval(pool, ckpt, kw, sname, nchunk=8):
    seeds = SETS[sname]; split = "cal" if sname == "cal" else "adhoc"; ch = [seeds[i::nchunk] for i in range(nchunk)]
    outs = pool.map(_job, [(ckpt, kw, c, split) for c in ch], chunksize=1)
    res = {}
    for k in outs[0]:
        v = np.empty(len(seeds), dtype=object if k == "outcome" else float)
        for i, o in enumerate(outs): v[i::nchunk] = o[k]
        res[k] = v.tolist()
    return res
def stage_evalcal(a):
    import multiprocessing as mp; pool = mp.get_context("spawn").Pool(a.workers); path = f"{OUT}/cal_eval.json"; res = json.load(open(path)) if os.path.exists(path) else {}
    if "teacher" not in res: res["teacher"] = run_eval(pool, None, {}, "cal"); print("teacher CAL rmse", np.mean(res["teacher"]["rmse"]), flush=True)
    for kind in KINDS:
        for d in DARTS:
            for lr in LRS:
                key = f"{kind}_d{d}_lr{lr:g}"
                if key in res: continue
                t = time.time(); vk, kw = MAIN[kind]; res[key] = run_eval(pool, ck(kind, d, lr, 0), kw, "cal"); print(key, "CAL rmse", np.mean(res[key]["rmse"]), f"{time.time()-t:.0f}s", flush=True); json.dump(res, open(path, "w"))
def stage_select(a):
    res = json.load(open(f"{OUT}/cal_eval.json")); sel = {}
    for kind in KINDS:
        c = [(np.mean(res[f"{kind}_d{d}_lr{lr:g}"]["rmse"]), d, lr) for d in DARTS for lr in LRS]; c.sort(); sel[kind] = dict(dart=c[0][1], lr=c[0][2], cal_rmse=c[0][0], grid=[list(x) for x in c])
        print(kind, "selected dart", c[0][1], "lr", c[0][2], "CAL rmse %.5f" % c[0][0], "teacher %.5f" % np.mean(res["teacher"]["rmse"]))
    json.dump(sel, open(f"{OUT}/selected.json", "w"), indent=1)
def stage_evalfinal(a):
    import multiprocessing as mp; pool = mp.get_context("spawn").Pool(a.workers); sel = json.load(open(f"{OUT}/selected.json"))
    for sname in ("holdout", "fresh"):
        path = f"{OUT}/final_{sname}.json"; res = json.load(open(path)) if os.path.exists(path) else {}
        if "teacher" not in res: t = time.time(); res["teacher"] = run_eval(pool, None, {}, sname); print(sname, "teacher rmse", np.mean(res["teacher"]["rmse"]), f"{time.time()-t:.0f}s", flush=True); json.dump(res, open(path, "w"))
        for label, kind, kw in VARIANTS:
            for s in (0, 1, 2):
                key = f"{label}|{s}"
                if key in res: continue
                t = time.time(); res[key] = run_eval(pool, ck(kind, sel[kind]["dart"], sel[kind]["lr"], s), kw, sname); print(sname, key, "rmse", np.mean(res[key]["rmse"]), f"{time.time()-t:.0f}s", flush=True); json.dump(res, open(path, "w"))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--device", default="auto"); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    os.makedirs(OUT + "/ckpt", exist_ok=True); globals()["stage_" + a.stage](a)
