"""Observation features for learned students: everything comes from the `Obs` object (no plant, no model).

A demonstration row stores the WINDOW  [history rows (H), current row]  of `Obs.to_vector()` (so the env must be built
with `Scenario(history=H)`), shape (H+1, d).  `FeatureSpec.expand` turns windows into the flat network input.
Optional `rel_ref`: appends the tracking errors (q - ref.q, qd - ref.qd) to each row.  This is NOT new information
(it is a linear function of the raw row) but it changes what an MLP can resolve easily; it is off by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..types import Obs


def layout_of(obs: Obs) -> list:
    """[(name, size)] of the columns of `obs.to_vector()` (same order: q, qd, ref keys sorted, extra keys sorted)."""
    out = [("q", len(obs.q)), ("qd", len(obs.qd))]
    out += [("ref." + k, int(np.size(obs.ref[k]))) for k in sorted(obs.ref)]
    out += [("extra." + k, int(np.size(obs.extra[k]))) for k in sorted(obs.extra)]
    return out


@dataclass
class FeatureSpec:
    layout: list                  # [(name, size)] of the raw row
    history: int = 0              # H: number of previous observations in the window
    rel_ref: bool = False
    wrap_angles: bool = False     # shift q and ref.q of each row by the SAME multiple of 2 pi so that ref.q is in [-pi, pi]

    @property
    def d_raw(self) -> int:
        return int(sum(s for _, s in self.layout))

    def _slices(self):
        o, out = 0, {}
        for name, s in self.layout:
            out[name] = slice(o, o + s)
            o += s
        return out

    @property
    def d_row(self) -> int:
        return self.d_raw + (self._rel_dim() if self.rel_ref else 0)

    def _rel_dim(self) -> int:
        sl = self._slices()
        return sum(sl[k].stop - sl[k].start for k in ("q", "qd") if "ref." + k in sl)

    @property
    def obs_dim(self) -> int:
        return (self.history + 1) * self.d_row

    def row_features(self, rows: np.ndarray) -> np.ndarray:
        """(..., d_raw) -> (..., d_row)"""
        sl = self._slices()
        if self.wrap_angles and "q" in sl and "ref.q" in sl:
            # (defensive no-op since the tracking reference is wrapped/unwrapped at the source; formerly joint 0 sat at -7 ... +81 rad on some seeds) the plant and the teacher
            # are 2 pi periodic and only use q - ref.q, so a common shift is information-preserving and removes the huge inputs
            rows = np.array(rows, copy=True)
            shift = 2 * np.pi * np.round(rows[..., sl["ref.q"]] / (2 * np.pi))
            rows[..., sl["q"]] -= shift
            rows[..., sl["ref.q"]] -= shift
        if not self.rel_ref:
            return rows
        extra = [rows[..., sl[k]] - rows[..., sl["ref." + k]] for k in ("q", "qd") if "ref." + k in sl]
        return np.concatenate([rows] + extra, axis=-1)

    def expand(self, win: np.ndarray) -> np.ndarray:
        """(..., H+1, d_raw) -> (..., (H+1)*d_row) float32"""
        f = self.row_features(np.asarray(win, np.float32))
        return f.reshape(f.shape[:-2] + (-1,)).astype(np.float32)

    def window(self, obs: Obs) -> np.ndarray:
        """The (H+1, d_raw) window of one Obs.  Requires Scenario(history >= H)."""
        cur = obs.to_vector()[None]
        if self.history == 0:
            return cur
        h = obs.history
        if h is None or len(h) < self.history:
            raise ValueError(f"student needs Obs.history of length {self.history}: build the env with "
                             f"Scenario(history={self.history})")
        return np.concatenate([np.asarray(h, np.float32)[-self.history:], cur], 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layout"] = [[n, int(s)] for n, s in self.layout]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(layout=[(n, int(s)) for n, s in d["layout"]], history=int(d["history"]), rel_ref=bool(d["rel_ref"]),
                   wrap_angles=bool(d.get("wrap_angles", False)))
