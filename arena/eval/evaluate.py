"""Run a controller on a scenario over a list of seeds -> EvalResult (per-seed protocol metrics)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..controllers.base import as_factory
from ..scenario import Scenario
from .metrics import compute_metrics
from .protocol import BINARY_METRICS, HIGHER_IS_BETTER, log_test_access, resolve_split
from .rollout import run_episode
from .stats import mean_ci


@dataclass
class EvalResult:
    scenario: str
    controller: str
    seeds: np.ndarray
    metrics: dict                    # name -> (n_seeds,) array
    higher_is_better: frozenset = frozenset()
    traces: list = field(default_factory=list)
    primary_metric: str = "rmse"
    binary: frozenset = BINARY_METRICS
    outcomes: np.ndarray | None = None      # per seed: 'success' | 'failure' | 'timeout'
    split: str = "adhoc"

    def mean(self, name):
        return mean_ci(self.metrics[name])


def evaluate(scenario: Scenario, controller, seeds, label: str | None = None, keep_traces: bool = False,
             split: str | None = None) -> EvalResult:
    """split: 'train' | 'cal' | 'test' | 'demo' | 'adhoc' (default: inferred from the seeds). Every evaluation on
    TEST_SEEDS appends a line to results/.test_access.log (time, script, condition)."""
    seeds = np.asarray(seeds)
    name = label or getattr(controller, "registry_name", None) or str(controller)
    split = resolve_split(seeds, split)
    if split == "test":
        log_test_access(f"scenario={scenario.label} controller={name} n_seeds={len(seeds)} "
                        f"seeds={int(seeds.min())}-{int(seeds.max())}")
    factory = as_factory(controller)
    env = scenario.make_env()                       # one plant reused across seeds (reset(seed) re-initialises)
    rows, traces, outcomes = [], [], []
    for s in seeds:
        tr = run_episode(scenario, factory, int(s), env=env)
        rows.append(compute_metrics(tr, env.task))
        outcomes.append(tr.outcome)
        if keep_traces:
            traces.append(tr)
    names = rows[0].keys()
    return EvalResult(scenario.label, name, seeds, {k: np.array([r[k] for r in rows]) for k in names},
                      frozenset(env.task.higher_is_better) | HIGHER_IS_BETTER, traces, env.task.primary_metric,
                      frozenset(env.task.binary_metrics) | BINARY_METRICS, np.array(outcomes), split)


class GridResults(dict):
    """{(scenario.label, controller_name): EvalResult} plus `.skipped` = [(scenario.label, controller_name)] of the
    combinations whose controller declared (at construction) that it does not support the task."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.skipped: list = []


def evaluate_grid(scenarios, controllers: dict, seeds, split: str | None = None) -> GridResults:
    """controllers = {name: name_or_factory}. A controller whose CONSTRUCTION raises NotImplementedError (the declared
    way to say 'not for this task') is skipped and listed in result.skipped; a NotImplementedError raised later,
    inside act(), is a bug and propagates."""
    out = GridResults()
    for sc in scenarios:
        ctx = sc.make_env().ctx
        for name, c in controllers.items():
            try:
                as_factory(c)(ctx)
            except NotImplementedError:
                out.skipped.append((sc.label, name))
                continue
            out[(sc.label, name)] = evaluate(sc, c, seeds, label=name, split=split)
    return out
