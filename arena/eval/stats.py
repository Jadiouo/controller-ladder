"""Statistics over seeds (protocol v2): t-interval for continuous metrics, percentile bootstrap for binary ones,
paired differences with a win rule, optional Holm correction.

Non-finite values are never dropped silently: mean_ci / paired_diff / bootstrap_ci raise NonFiniteError. `summarize`
(used by the report tables) and `paired_diff(..., undefined='mark')` accept NaN as 'undefined' (e.g. recovery_time
of an episode without a pulse) but report how many values were valid; +-inf always raises.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .protocol import WIN_REL


class NonFiniteError(ValueError):
    """A statistic was asked for data containing NaN / inf."""


def _clean(x, name="values"):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise NonFiniteError(f"{name} contain {int((~np.isfinite(x)).sum())} non-finite entries of {x.size}; "
                             "fix the cause (diverged run?) or use summarize()/undefined='mark' to report them")
    return x


def mean_ci(x):
    """mean and half-width of the 95% t interval. Raises NonFiniteError on NaN/inf."""
    x = _clean(x)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarize(x) -> dict:
    """mean / t half-width over the finite entries PLUS the counts: n (all), n_valid, n_undefined (NaN). inf raises."""
    x = np.asarray(x, float)
    if np.isinf(x).any():
        raise NonFiniteError("values contain inf")
    ok = ~np.isnan(x)
    if ok.any():
        m, h = mean_ci(x[ok])
    else:
        m, h = float("nan"), float("nan")
    return dict(mean=m, half=h, n=int(x.size), n_valid=int(ok.sum()), n_undefined=int((~ok).sum()))


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """(mean, lo, hi): percentile bootstrap of the mean (deterministic: fixed seed). For 0/1 metrics."""
    x = _clean(x)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed)
    means = x[r.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff(cand, base, higher_is_better: bool = False, win_rel: float = WIN_REL, method: str = "t",
                undefined: str = "raise", mask=None, win_abs: float = 0.0, n_boot: int = 2000) -> dict:
    """cand vs base on the same seeds. diff = cand - base. `rel` = improvement of cand over base as a fraction of
    |mean(base)| (positive = cand better). win: CI entirely on the 'better' side AND rel >= win_rel.
    If mean(base) == 0 the relative rule is meaningless: rel = NaN and the absolute improvement must exceed `win_abs`
    (default 0, i.e. any improvement whose CI excludes 0).
    method 't' (default) or 'bootstrap' (paired percentile bootstrap; use for 0/1 metrics).
    undefined: 'raise' (NaN -> NonFiniteError) or 'mark' (pairs with NaN are excluded and counted: n < n_total).
    mask: optional boolean array, only these seeds are compared (e.g. episodes both controllers survived).
    Returns dict(n, n_total, diff, lo, hi, rel, win, win_frac, p)."""
    a, b = np.asarray(cand, float), np.asarray(base, float)
    if a.shape != b.shape:
        raise ValueError("cand and base must have the same shape (same seeds)")
    sel = np.ones(a.shape, bool) if mask is None else np.asarray(mask, bool)
    n_total = int(sel.sum())
    a, b = a[sel], b[sel]
    if np.isinf(a).any() or np.isinf(b).any():
        raise NonFiniteError("values contain inf")
    ok = ~(np.isnan(a) | np.isnan(b))
    if undefined == "raise" and not ok.all():
        raise NonFiniteError(f"{int((~ok).sum())} NaN pair(s) of {len(ok)}")
    d = (a - b)[ok]
    if method == "bootstrap" and len(d):
        m, lo, hi = bootstrap_ci(d, n_boot=n_boot)
        r = np.random.default_rng(1).integers(0, len(d), (n_boot, len(d)))
        bm = d[r].mean(1)
        p = float(min(1.0, 2 * min(((bm <= 0).sum() + 1) / (n_boot + 1), ((bm >= 0).sum() + 1) / (n_boot + 1))))
    else:
        m, h = mean_ci(d)
        lo, hi = m - h, m + h
        if len(d) < 2:
            p = float("nan")
        elif d.std(ddof=1) == 0:
            p = 1.0 if m == 0 else 0.0
        else:
            p = float(stats.ttest_1samp(d, 0.0).pvalue)
    bmean = float(np.mean(b[ok])) if ok.any() else float("nan")
    sign = 1.0 if higher_is_better else -1.0           # improvement direction
    better_ci = (lo > 0) if higher_is_better else (hi < 0)
    finite = np.isfinite(lo) and np.isfinite(hi)
    if np.isfinite(bmean) and abs(bmean) < 1e-12:
        rel = float("nan")
        win = bool(finite and better_ci and sign * m > win_abs)
    else:
        rel = sign * m / abs(bmean) if np.isfinite(bmean) else float("nan")
        win = bool(finite and better_ci and rel >= win_rel)
    return dict(n=int(ok.sum()), n_total=n_total, diff=m, lo=lo, hi=hi, rel=rel, win=win, p=p,
                win_frac=float(((d > 0) if higher_is_better else (d < 0)).mean()) if len(d) else float("nan"))


def holm_adjust(pvals) -> list:
    """Holm-Bonferroni adjusted p-values (same order as the input; NaN stays NaN and is not counted in the family)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out.tolist()
