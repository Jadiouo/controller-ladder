"""Markdown tables: per-condition metrics (mean +- 95% CI over seeds; bootstrap CI for 0/1 metrics) and paired
differences vs a baseline (same seeds, optional Holm correction)."""
from __future__ import annotations

import numpy as np

from .stats import bootstrap_ci, holm_adjust, paired_diff, summarize

DEFAULT_METRICS = ["rmse", "ss_rmse", "settle_time", "recovery_time", "sat", "energy"]
FMT = {"energy": "{:.1f}", "sat": "{:.3f}", "settle_time": "{:.2f}", "recovery_time": "{:.2f}", "survival_time": "{:.2f}",
       "fell": "{:.2f}", "alive_frac": "{:.2f}", "success": "{:.2f}"}
ERROR_BASED = frozenset({"rmse", "ss_rmse", "settle_time", "recovery_time", "ee_rmse", "tip_rmse", "tip_settle_time"})
SURVIVAL_SENSITIVE = frozenset({"energy", "sat"})      # depend on how long the episode lasted


def _f(name, v):
    return "n/a" if not np.isfinite(v) else FMT.get(name, "{:.4f}").format(v)


def _cell(r, m):
    x = r.metrics[m]
    s = summarize(x)
    if s["n_valid"] == 0:
        return "n/a", 0, s["n"]
    if m in r.binary and s["n_undefined"] == 0:
        mu, lo, hi = bootstrap_ci(x)
        txt = f"{_f(m, mu)} [{_f(m, lo)}, {_f(m, hi)}]"
    else:
        txt = _f(m, s["mean"]) + ("" if not np.isfinite(s["half"]) else " ± " + _f(m, s["half"]))
    if s["n_undefined"]:
        txt += f" ({s['n_valid']}/{s['n']})"
    return txt, s["n_valid"], s["n"]


def metrics_table(results: dict, metrics=None) -> str:
    """results: {(scenario, controller): EvalResult} (an evaluate_grid result also lists its skipped combinations).
    n = number of VALID seeds (the minimum over the shown metrics that are defined at all, 'valid/total' if some are
    NaN); a metric that is NaN for every seed (e.g. recovery_time without pulses) shows n/a and does not count."""
    metrics = metrics or DEFAULT_METRICS
    extra = sorted({k for r in results.values() for k in r.metrics} - set(DEFAULT_METRICS))
    cols = [m for m in metrics if any(m in r.metrics for r in results.values())] + \
           ([e for e in extra if metrics is DEFAULT_METRICS or e in metrics])
    L = ["| scenario | controller | n | " + " | ".join(cols) + " |", "|" + "---|" * (3 + len(cols))]
    for (sc, c), r in results.items():
        cells, valid, total = [], [], len(r.seeds)
        for m in cols:
            if m not in r.metrics:
                cells.append("")
                continue
            txt, nv, nt = _cell(r, m)
            cells.append(txt)
            if nv > 0:
                valid.append(nv)
        nv = min(valid) if valid else 0
        L.append(f"| {sc} | {c} | {total if nv == total else f'{nv}/{total}'} | " + " | ".join(cells) + " |")
    skipped = getattr(results, "skipped", [])
    if skipped:
        L += ["", "Skipped (controller does not support the task): " + "; ".join(f"{c} on {sc}" for sc, c in skipped)]
    return "\n".join(L)


def paired_table(results: dict, baseline: str, metrics=None, holm: bool = False, common_survivors: bool = False,
                 alpha: float = 0.05) -> str:
    """Each controller vs `baseline` on the same seeds (a seed mismatch raises). diff = cand - base; impr. > 0 means
    the candidate is better.
    * binary metrics (success, fell): paired bootstrap CI instead of the t interval.
    * energy / sat are per-step quantities of episodes of different length when survival differs: if alive_frac differs
      on any seed the row is never a WIN and is annotated.
    * common_survivors=True: error-based metrics (rmse, ...) are compared only on seeds where BOTH controllers
      survived (n column shows how many); survival_time / success are always compared on all seeds.
    * holm=True: adds a column with the Holm-corrected decision over ALL rows of this table (family = the table)."""
    pairs = [(sc, c, r, results[(sc, baseline)]) for (sc, c), r in results.items()
             if c != baseline and (sc, baseline) in results]
    for sc, c, r, base in pairs:
        if not np.array_equal(r.seeds, base.seeds):
            raise ValueError(f"paired comparison needs identical seeds; '{c}' and '{baseline}' on {sc} differ")
    first = pairs[0][2] if pairs else None
    if metrics is None:
        metrics = list(dict.fromkeys([first.primary_metric if first else "rmse", "rmse", "ss_rmse", "energy"]))
    rows = []
    for sc, c, r, base in pairs:
        for m in metrics:
            if m not in r.metrics:
                continue
            note, mask = "", None
            alive_c, alive_b = r.metrics.get("alive_frac"), base.metrics.get("alive_frac")
            differ = alive_c is not None and alive_b is not None and not np.allclose(alive_c, alive_b)
            if common_survivors and m in ERROR_BASED and alive_c is not None and alive_b is not None:
                mask = (alive_c >= 1.0) & (alive_b >= 1.0)
                note = f"common survivors only ({int(mask.sum())}/{len(mask)})"
            p = paired_diff(r.metrics[m], base.metrics[m], m in r.higher_is_better,
                            method="bootstrap" if m in r.binary else "t", undefined="mark", mask=mask)
            if m in SURVIVAL_SENSITIVE and differ:
                p["win"] = False
                note = "survival differs: not judged"
            rows.append((sc, c, m, p, note))
    if holm:
        adj = holm_adjust([p["p"] for *_, p, _ in rows])
        for (sc, c, m, p, note), a in zip(rows, adj):
            p["p_holm"], p["win_holm"] = a, bool(p["win"] and np.isfinite(a) and a < alpha)
    head = "| scenario | candidate vs " + baseline + " | metric | n | diff [95% CI] | impr. | WIN (CI & >=10%) |"
    head += " WIN (Holm) |" if holm else ""
    head += " note |"
    L = [head, "|" + "---|" * (head.count("|") - 1)]
    for sc, c, m, p, note in rows:
        n = str(p["n"]) if p["n"] == p["n_total"] else f"{p['n']}/{p['n_total']}"
        rel = "n/a (base 0: abs)" if not np.isfinite(p["rel"]) and np.isfinite(p["diff"]) else f"{100 * p['rel']:+.1f}%"
        line = (f"| {sc} | {c} | {m} | {n} | {_f(m, p['diff'])} [{_f(m, p['lo'])}, {_f(m, p['hi'])}] | {rel} | "
                f"{'YES' if p['win'] else 'no'} |")
        if holm:
            line += f" {'YES' if p['win_holm'] else 'no'} (p={_f('x', p['p_holm'])}) |"
        L.append(line + f" {note} |")
    return "\n".join(L)
