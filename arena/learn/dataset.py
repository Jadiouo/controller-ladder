"""Demonstration data: teacher rollouts -> npz shards -> batched reading.

Shard file `shard_XXXX.npz` holds WHOLE episodes (so action chunks never cross a shard):
  win (N, H+1, d) float32   observation window ([history..., current] rows of Obs.to_vector())
  act (N, n)      float32   the TEACHER's action at that observation (the label), clipped to the torque limit
  ep  (N,)        int64     episode seed        k (N,) int32  control step in the episode
`meta.json` (written atomically after every shard) holds the feature layout, history H, dt, tau limit, teacher name,
scenario label and one record per shard.  Collection options:
  dart_sigma : DART - the EXECUTED action is the teacher action + N(0, (dart_sigma * tau_limit)^2) noise, the label is
               still the clean teacher action (Laskey et al. 2017), so the data covers a tube around the teacher path.
  student/beta : DAgger - the executed action is the student's with probability 1 - beta per step (teacher with beta);
               the teacher labels every visited state.  Use `append=True` to add the round to an existing directory.
All randomness comes from np.random.default_rng([episode seed, 0xDA27, round]).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..controllers.base import as_factory
from .features import FeatureSpec, layout_of


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def collect(scenario, teacher, seeds, out_dir, *, dart_sigma: float = 0.0, student=None, beta: float = 0.0,
            shard_episodes: int = 25, append: bool = False, round_id: int = 0, tag: str = "teacher",
            verbose: bool = False) -> Path:
    """Roll the teacher out on `seeds` (use TRAIN_SEEDS) and write shards to `out_dir`.  Returns the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.json"
    if meta_p.exists() and not append:
        raise FileExistsError(f"{out} already has data; pass append=True (DAgger round) or use a new directory")
    env = scenario.make_env()
    ctx = env.ctx
    teach = as_factory(teacher)(ctx)
    stud = None if student is None else as_factory(student)(ctx)
    H = scenario.history
    obs = env.reset(int(seeds[0]))
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else dict(
        layout=[[n, s] for n, s in layout_of(obs)], history=H, n=env.n, dt=float(env.dt), tau_limit=float(ctx.tau_limit),
        n_steps=int(env.n_steps), scenario=scenario.label, task=scenario.task,
        teacher=str(getattr(teacher, "registry_name", teacher)), shards=[])
    if meta["history"] != H:
        raise ValueError("history of the scenario differs from the existing data")
    tau_lim = float(ctx.tau_limit)
    buf = dict(win=[], act=[], ep=[], k=[])
    n_ep_in_buf, idx = 0, len(meta["shards"])

    def flush():
        nonlocal buf, n_ep_in_buf, idx
        if not buf["win"]:
            return
        name = f"shard_{idx:04d}.npz"
        arrs = dict(win=np.asarray(buf["win"], np.float32), act=np.asarray(buf["act"], np.float32),
                    ep=np.asarray(buf["ep"], np.int64), k=np.asarray(buf["k"], np.int32))
        tmp = out / (name + ".tmp.npz")
        np.savez(tmp, **arrs)
        os.replace(tmp, out / name)
        meta["shards"].append(dict(file=name, rows=int(len(arrs["act"])), episodes=int(len(set(arrs["ep"].tolist()))),
                                   tag=tag, round=round_id, dart_sigma=dart_sigma, beta=beta if student is not None else None))
        _atomic_json(meta_p, meta)
        buf = dict(win=[], act=[], ep=[], k=[])
        n_ep_in_buf, idx = 0, idx + 1

    spec = FeatureSpec([(n, s) for n, s in meta["layout"]], H)
    for s in seeds:
        s = int(s)
        rng = np.random.default_rng([s, 0xDA27, round_id])
        obs = env.reset(s)
        teach.reset(s)
        if stud is not None:
            stud.reset(s)
        for k in range(env.n_steps):
            a_t = np.clip(np.asarray(teach.act(obs), float), -tau_lim, tau_lim)
            buf["win"].append(spec.window(obs))
            buf["act"].append(a_t)
            buf["ep"].append(s)
            buf["k"].append(k)
            a = a_t
            if stud is not None:
                a_s = np.asarray(stud.act(obs), float)           # always called so its chunk buffer stays consistent
                a = a_t if rng.random() < beta else a_s
            if dart_sigma > 0:
                a = a + dart_sigma * tau_lim * rng.standard_normal(a.shape)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        n_ep_in_buf += 1
        if n_ep_in_buf >= shard_episodes:
            flush()
            if verbose:
                print(f"  shard {idx} written ({s})", flush=True)
    flush()
    return out


class ShardedDataset:
    """Reads the shards of a directory lazily.  `chunk` = length Ta of the action-chunk label: the label of row i is
    act[i], act[i+1], ..., act[i+Ta-1] of the same episode (the last action repeated after the episode end)."""

    def __init__(self, path, chunk: int = 1, rel_ref: bool = False, wrap_angles: bool = False):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.chunk = int(chunk)
        self.spec = FeatureSpec([(n, s) for n, s in self.meta["layout"]], self.meta["history"], rel_ref, wrap_angles)
        self.shards = [self.path / s["file"] for s in self.meta["shards"]]
        self.rows = [s["rows"] for s in self.meta["shards"]]
        self.n = int(self.meta["n"])

    def __len__(self):
        return int(sum(self.rows))

    @property
    def episodes(self) -> int:
        return int(sum(s["episodes"] for s in self.meta["shards"]))

    def nbytes(self) -> int:
        return len(self) * (self.spec.obs_dim + self.chunk * self.n) * 4

    def load_shard(self, i: int):
        """-> obs (N, obs_dim) float32, act chunk (N, Ta, n) float32, plus the raw arrays dict"""
        z = np.load(self.shards[i])
        win, act, ep = z["win"], z["act"], z["ep"]
        N = len(act)
        last = np.empty(N, np.int64)            # index of the last row of each row's episode
        brk = np.flatnonzero(np.r_[ep[1:] != ep[:-1], True] | np.r_[z["k"][1:] == 0, True])
        start = 0
        for b in brk:
            last[start:b + 1] = b
            start = b + 1
        idx = np.minimum(np.arange(N)[:, None] + np.arange(self.chunk)[None], last[:, None])
        return self.spec.expand(win), act[idx].astype(np.float32), dict(win=win, act=act, ep=ep, k=z["k"])

    def stats(self):
        """Streaming mean / std: obs per feature column (shared by the H+1 window rows), action per joint."""
        d = self.spec.d_row
        s1 = s2 = 0.0
        a1 = a2 = 0.0
        cnt = acnt = 0
        for i in range(len(self.shards)):
            o, a, _ = self.load_shard(i)
            r = o.reshape(-1, d).astype(np.float64)
            s1, s2, cnt = s1 + r.sum(0), s2 + (r ** 2).sum(0), cnt + len(r)
            aa = a[:, 0].astype(np.float64)
            a1, a2, acnt = a1 + aa.sum(0), a2 + (aa ** 2).sum(0), acnt + len(aa)
        om = s1 / cnt
        os_ = np.sqrt(np.maximum(s2 / cnt - om ** 2, 0)) + 1e-6
        am = a1 / acnt
        as_ = np.sqrt(np.maximum(a2 / acnt - am ** 2, 0)) + 1e-6
        H1 = self.spec.history + 1
        return (np.tile(om, H1).astype(np.float32), np.tile(os_, H1).astype(np.float32),
                am.astype(np.float32), as_.astype(np.float32))

    def sampler(self, batch: int, device, gen, norm=None, max_resident_bytes: int = 1 << 30, group: int = 4):
        return BatchSampler(self, batch, device, gen, norm, max_resident_bytes, group)


class BatchSampler:
    """Random mini-batches on `device`.  If the whole (normalised) dataset fits in `max_resident_bytes` it is loaded
    once; otherwise `group` randomly chosen shards at a time are resident and replaced after one pass over them."""

    def __init__(self, ds, batch, device, gen, norm, max_resident_bytes, group):
        import torch
        self.torch, self.ds, self.batch, self.device, self.gen = torch, ds, batch, device, gen
        self.norm = None if norm is None else tuple(torch.as_tensor(x, device=device) for x in norm)
        self.resident = ds.nbytes() <= max_resident_bytes
        self.group = max(1, group)
        self._order, self._left, self.loads = [], 0, 0
        if self.resident:
            self._load(list(range(len(ds.shards))))

    def _load(self, ids):
        t = self.torch
        os_, as_ = [], []
        for i in ids:
            o, a, _ = self.ds.load_shard(i)
            os_.append(o)
            as_.append(a)
        o = t.as_tensor(np.concatenate(os_), device=self.device)
        a = t.as_tensor(np.concatenate(as_), device=self.device)
        if self.norm is not None:
            om, osd, am, asd = self.norm
            o, a = (o - om) / osd, (a - am) / asd
        self.o, self.a = o, a
        self.loads += 1
        self._left = max(1, len(o) // self.batch)

    def next(self):
        t = self.torch
        if not self.resident and self._left <= 0:
            if not self._order:
                perm = t.randperm(len(self.ds.shards), generator=self.gen, device=self.gen.device).cpu().tolist()
                self._order = perm
            ids, self._order = self._order[:self.group], self._order[self.group:]
            self._load(ids)
        self._left -= 1
        i = t.randint(0, len(self.o), (self.batch,), generator=self.gen, device=self.gen.device)
        return self.o[i], self.a[i]
