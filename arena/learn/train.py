"""Training and checkpoints.  Same data, same observation features, same optimiser / step budget / batch / EMA for all
students; only the model class differs (see models.py).  Works on cpu or cuda.

  python -m arena.learn.train --data DIR --kind mip --seed 0 --steps 30000 --out ckpt/mip_s0.pt [--device cuda]
Submit GPU runs through a GPU job queue (see README).  fp32 throughout (no TF32/bf16: the regression targets are precision-
hungry, and the nets are tiny so the GPU is latency-bound, not FLOP-bound).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import ShardedDataset
from .features import FeatureSpec
from .models import CLASSES, build_policy, n_params


@dataclass
class TrainCfg:
    steps: int = 30000
    batch: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    wd: float = 1e-6
    ema: float = 0.999
    warmup: int = 200
    chunk: int = 8
    width: int = 256
    depth: int = 4
    seed: int = 0
    rel_ref: bool = False
    wrap_angles: bool = False
    log_every: int = 1000


def _pick_device(device):
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(data, kind: str, out, cfg: TrainCfg | None = None, device="auto", match_params: int | None = None,
          verbose: bool = True) -> dict:
    """Train one student on `data` (a ShardedDataset or a directory) and save `out` (.pt).  Returns the log dict.
    The EMA weights are what is saved.  Same (data, kind, cfg) -> same result on cpu."""
    cfg = cfg or TrainCfg()
    dev = _pick_device(device)
    ds = data if isinstance(data, ShardedDataset) else ShardedDataset(data, cfg.chunk, cfg.rel_ref, cfg.wrap_angles)
    assert ds.chunk == cfg.chunk and ds.spec.rel_ref == cfg.rel_ref and ds.spec.wrap_angles == cfg.wrap_angles
    torch.manual_seed(cfg.seed)
    model = build_policy(kind, ds.spec.obs_dim, ds.n, cfg.chunk, cfg.width, cfg.depth, match_to=match_params)
    norm = ds.stats()
    model.set_norm(*norm)
    model.to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 7919 + 13)
    sampler = ds.sampler(cfg.batch, dev, gen, norm=tuple(torch.as_tensor(x) for x in norm))
    opt = torch.optim.AdamW([p for p in model.parameters()], lr=cfg.lr, weight_decay=cfg.wd)

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        p = (s - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + np.cos(np.pi * p))

    if verbose:
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
        print(f"[train] kind={kind} seed={cfg.seed} device={dev} ({name}) rows={len(ds)} params={n_params(model)} "
              f"resident={sampler.resident}", flush=True)
    log, run, t0 = [], torch.zeros((), device=dev), time.time()
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        o, a = sampler.next()
        loss = model.loss(o, a, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(cfg.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        run += loss.detach()
        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
            n_acc = cfg.log_every if (step + 1) % cfg.log_every == 0 else (step + 1) % cfg.log_every
            val = float(run) / n_acc
            run.zero_()
            log.append((step + 1, val))
            if verbose:
                print(f"[train] {kind} s{cfg.seed} step {step + 1}/{cfg.steps} loss {val:.5f} {time.time() - t0:.0f}s", flush=True)
    info = dict(kind=kind, seed=cfg.seed, device=str(dev), params=n_params(model), seconds=time.time() - t0,
                final_loss=log[-1][1], log=log, cfg=cfg.__dict__.copy())
    ckpt = dict(config=ema.config(), state=ema.state_dict(), spec=ds.spec.to_dict(), train=info,
                data=dict(path=str(ds.path), rows=len(ds), episodes=ds.episodes, teacher=ds.meta["teacher"],
                          scenario=ds.meta["scenario"], tau_limit=ds.meta["tau_limit"]))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(ckpt, out)
    return info


@dataclass
class Loaded:
    policy: torch.nn.Module
    spec: FeatureSpec
    meta: dict
    device: torch.device


_CACHE: dict = {}


def load_checkpoint(path, device="cpu") -> Loaded:
    dev = torch.device(device)
    key = (str(Path(path).resolve()), os.path.getmtime(path), str(dev))
    if key not in _CACHE:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c = ck["config"]
        pol = CLASSES[c["kind"]](c["obs_dim"], c["n"], c["chunk"], c["width"], c["depth"])
        pol.load_state_dict(ck["state"], strict=False)       # `abar` (diffusion) is a non-persistent buffer
        pol.to(dev).eval().requires_grad_(False)
        _CACHE[key] = Loaded(pol, FeatureSpec.from_dict(ck["spec"]), dict(train=ck["train"], data=ck["data"]), dev)
    return _CACHE[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", required=True, choices=list(CLASSES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    for f, v in TrainCfg().__dict__.items():
        if isinstance(v, bool):
            ap.add_argument("--" + f.replace("_", "-"), action="store_true", default=v)
        else:
            ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    cfg = TrainCfg(**{f: getattr(a, f) for f in TrainCfg().__dict__})
    train(a.data, a.kind, a.out, cfg, a.device)


if __name__ == "__main__":
    main()
