"""L6 demo: BC-MLP, MIP, Diffusion Policy and flow matching (+ one-step flow) imitating the CTC+LQR teacher on tracking
(2-DoF arm, clean plant).  Same data (200 DART-noised teacher rollouts on TRAIN_SEEDS), same observation features, same
optimiser / step budget / batch / EMA for every student; 3 training seeds each; evaluated once on TEST_SEEDS
(split="test") and compared with the teacher seed by seed.

Stages (run in this order; the heavy ones need the wrappers shown):
  collect  CPU : flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
                     .venv/bin/python scripts/demo_students.py collect
  train    GPU : a GPU job queue systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
                     python scripts/demo_students.py train [--rel-ref]
  eval     CPU : flock ... .venv-gpu/bin/python scripts/demo_students.py eval [--rel-ref]       (checkpoints run on CPU)
  latency  GPU : a GPU job queue ... .venv-gpu/bin/python scripts/demo_students.py latency                (batch=1, GPU and CPU)
  report   --  : .venv/bin/python scripts/demo_students.py report   -> results/learn_demo/table.md
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from arena import Scenario
from arena.controllers import controller_factory
from arena.eval import TEST_SEEDS, TRAIN_SEEDS, evaluate
from arena.eval.stats import paired_diff

OUT = "results/learn_demo"
DATA = f"{OUT}/data"
HIST, CHUNK, EXEC_K = 3, 8, 4          # observation window = 3 past rows + current, action chunk 8, execute 4 then re-plan
DART = 0.1                             # DART noise: sigma = 0.1 * tau_limit on the executed action (chosen on CAL seeds, README 54)
KINDS = ["bc", "mip", "diffusion", "flow"]
TRAIN_SEEDS_STUDENT = [0, 1, 2]
STEPS, BATCH = 15000, 256
# student variants evaluated: (label, kind, sample_kw)
VARIANTS = [("bc", "bc", {}), ("mip", "mip", {}), ("diffusion (DDIM-10)", "diffusion", {"steps": 10}),
            ("flow (10 steps)", "flow", {"steps": 10}), ("flow (1 step)", "flow", {"steps": 1})]
SC = Scenario("tracking", {"ref": "lissajous"}, history=HIST, name="tracking/clean")


def ckdir(rel):
    return f"{OUT}/ckpt{'_rel' if rel else ''}"


def stage_collect(a):
    from arena.learn import collect
    t = time.time()
    collect(SC, "ctc_lqr", TRAIN_SEEDS[:200], DATA, dart_sigma=DART, shard_episodes=25, verbose=True)
    print(f"collected in {time.time() - t:.0f}s -> {DATA}")


def stage_train(a):
    import torch
    from arena.learn.dataset import ShardedDataset
    from arena.learn.train import TrainCfg, train
    from arena.learn.models import build_policy, n_params
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(), flush=True)
    os.makedirs(ckdir(a.rel_ref), exist_ok=True)
    summary = {}
    for seed in TRAIN_SEEDS_STUDENT:
        for kind in KINDS:
            cfg = TrainCfg(steps=STEPS, batch=BATCH, chunk=CHUNK, seed=seed, rel_ref=a.rel_ref, wrap_angles=True)
            ds = ShardedDataset(DATA, CHUNK, a.rel_ref, True)
            ref = n_params(build_policy("mip", ds.spec.obs_dim, ds.n, CHUNK, cfg.width, cfg.depth))
            info = train(ds, kind, f"{ckdir(a.rel_ref)}/{kind}_s{seed}.pt", cfg, a.device, match_params=ref if kind == "bc" else None)
            summary[f"{kind}_s{seed}"] = {k: v for k, v in info.items() if k != "log"} | {"log": info["log"][-3:]}
            json.dump(summary, open(f"{ckdir(a.rel_ref)}/train_summary.json", "w"), indent=1)


def _timing(c):
    return c.timing() if hasattr(c, "timing") else {}


def stage_eval(a):
    import torch
    torch.set_num_threads(1)
    res = {}
    path = f"{ckdir(a.rel_ref)}/eval.json"
    if a.rel_ref is False:
        t0 = time.time()
        r = evaluate(SC, "ctc_lqr", TEST_SEEDS, label="teacher", split="test")
        res["teacher"] = {m: r.metrics[m].tolist() for m in r.metrics} | {"outcome": r.outcomes.tolist()}
        print("teacher rmse", r.metrics["rmse"].mean(), f"{time.time() - t0:.0f}s", flush=True)
    for label, kind, kw in VARIANTS:
        for seed in TRAIN_SEEDS_STUDENT:
            made = []
            f = controller_factory("student", ckpt=f"{ckdir(a.rel_ref)}/{kind}_s{seed}.pt", exec_k=EXEC_K, sample_kw=kw)
            t0 = time.time()
            r = evaluate(SC, lambda ctx: made.append(f(ctx)) or made[-1], TEST_SEEDS, label=f"{label}_s{seed}", split="test")
            tm = np.concatenate([m.infer_times for m in made]) * 1e3
            res[f"{label}|{seed}"] = {m: r.metrics[m].tolist() for m in r.metrics} | {
                "outcome": r.outcomes.tolist(), "infer_ms_mean_cpu_in_loop": float(tm.mean())}
            print(label, seed, "rmse", r.metrics["rmse"].mean(), f"{time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(path, "w"))


def stage_latency(a):
    import torch
    from arena.learn.train import load_checkpoint
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(), flush=True)
    out = {}
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        torch.set_num_threads(1)
        for label, kind, kw in VARIANTS:
            L = load_checkpoint(f"{ckdir(False)}/{kind}_s0.pt", dev)
            x = torch.randn(1, L.spec.obs_dim, device=dev)
            g = torch.Generator(device=dev).manual_seed(0)
            ts = []
            for i in range(30 + 300):
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                L.policy.act_chunk(x, g, **kw).cpu()          # includes the copy of the result to the host
                if dev == "cuda":
                    torch.cuda.synchronize()
                if i >= 30:
                    ts.append((time.perf_counter() - t0) * 1e3)
            out[f"{label}|{dev}"] = dict(mean=float(np.mean(ts)), p95=float(np.percentile(ts, 95)), median=float(np.median(ts)))
            print(label, dev, out[f"{label}|{dev}"], flush=True)
    json.dump(out, open(f"{OUT}/latency.json", "w"), indent=1)


def _mean(res, label, metric):
    return np.mean([res[f"{label}|{s}"][metric] for s in TRAIN_SEEDS_STUDENT], axis=0)     # (n_test,) mean over train seeds


def section_eval(res, title):
    L = [f"### {title}", "",
         "| student | rmse per training seed (rad) | rmse mean +- sd over training seeds | ss_rmse | sat | paired diff vs teacher (seed-averaged student - teacher), 95% CI | verdict |",
         "|---|---|---|---|---|---|---|"]
    t = np.array(res["teacher"]["rmse"])
    L.append(f"| **teacher (ctc_lqr)** | - | {t.mean():.5f} | {np.mean(res['teacher']['ss_rmse']):.5f} | {np.mean(res['teacher']['sat']):.4f} | - | - |")
    for label, _, _ in VARIANTS:
        if f"{label}|0" not in res:
            continue
        per = [np.mean(res[f"{label}|{s}"]["rmse"]) for s in TRAIN_SEEDS_STUDENT]
        avg = _mean(res, label, "rmse")
        d = paired_diff(avg, t, undefined="mark")
        verdict = "WIN" if d["win"] else ("worse" if d["lo"] > 0 else "no difference")
        fails = sum(o != "timeout" for s in TRAIN_SEEDS_STUDENT for o in res[f"{label}|{s}"]["outcome"])
        L.append(f"| {label} | {', '.join(f'{p:.5f}' for p in per)} | {np.mean(per):.5f} +- {np.std(per):.5f} | "
                 f"{np.mean(_mean(res, label, 'ss_rmse')):.5f} | {np.mean(_mean(res, label, 'sat')):.4f} | "
                 f"{d['diff']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}] ({d['rel'] * 100:+.0f}% improvement; negative = worse) | {verdict}{'; failures ' + str(fails) if fails else ''} |")
    return L


def stage_report(a):
    L = ["# L6 imitation students on tracking (2-DoF, CTC+LQR teacher)", "",
         f"Scenario `tracking/clean` (Lissajous reference, 4 s, nominal plant), TEST_SEEDS 0-49 (n = 50, split=\"test\"), "
         f"3 training seeds per student. Data: 200 teacher rollouts on TRAIN_SEEDS (DART sigma = {DART} tau_limit), same file for every student. "
         f"Observation: window of {HIST} past rows + current row of `Obs.to_vector()` (q, qd, ref.ee, ref.q, ref.qd, ref.qdd). Chunk {CHUNK}, "
         f"executes {EXEC_K} then re-plans. Steps {STEPS}, batch {BATCH}, AdamW + cosine, EMA 0.999, fp32.", ""]
    res = json.load(open(f"{ckdir(False)}/eval.json"))
    ts = json.load(open(f"{ckdir(False)}/train_summary.json"))
    L += ["## Parameters and training", "", "| student | params | final train loss (mean of 3 seeds) | train device | train time / run (s) |", "|---|---|---|---|---|"]
    for kind in KINDS:
        rows = [ts[f"{kind}_s{s}"] for s in TRAIN_SEEDS_STUDENT]
        L.append(f"| {kind} | {rows[0]['params']:,} | {np.mean([r['final_loss'] for r in rows]):.5f} (loss is not comparable across kinds) | "
                 f"{rows[0]['device']} | {np.mean([r['seconds'] for r in rows]):.0f} |")
    L += [""] + section_eval(res, "Raw features (as specified)") + [""]
    rp = f"{ckdir(True)}/eval.json"
    if os.path.exists(rp):
        res2 = json.load(open(rp))
        res2["teacher"] = res["teacher"]
        L += section_eval(res2, "Extra input: tracking errors (q - ref.q, qd - ref.qd) appended to every row (`--rel-ref`)") + [""]
    lp = f"{OUT}/latency.json"
    if os.path.exists(lp):
        lat = json.load(open(lp))
        L += ["## Inference time (latency axis B), batch = 1, one chunk of 8 x 2 torques, 1 CPU thread, fp32", "",
              "| student | network evaluations | GPU (RTX 5070 Ti) mean / p95 (ms) | CPU mean / p95 (ms) |", "|---|---|---|---|"]
        nfe = {"bc": 1, "mip": 2, "diffusion (DDIM-10)": 10, "flow (10 steps)": 10, "flow (1 step)": 1}
        for label, _, _ in VARIANTS:
            g, c = lat.get(f"{label}|cuda"), lat.get(f"{label}|cpu")
            fm = lambda x: "n/a" if x is None else f"{x['mean']:.2f} / {x['p95']:.2f}"
            L.append(f"| {label} | {nfe[label]} | {fm(g)} | {fm(c)} |")
        L += ["", f"The controller calls the network once per {EXEC_K} control steps (period 5 ms): per-step cost = mean / {EXEC_K}. "
              "The machine is shared and heavily loaded, wall times are inflated and noisy.", ""]
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/table.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "train", "eval", "latency", "report"])
    ap.add_argument("--rel-ref", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    globals()["stage_" + a.stage](a)
