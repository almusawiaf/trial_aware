"""
tune.py
=======
Randomised hyperparameter search with an identical budget for every model.

Why an equal budget matters
---------------------------
The most common way a model-comparison paper reaches a wrong conclusion is
unequal tuning effort: the proposed model gets a careful sweep, the baselines
get library defaults. Every model here draws the *same* number of random
configurations from its own space, is selected on the *same* validation
quadrant, by the *same* metric, and the winning configuration is then refit
once and scored on test. Models whose `hyperparameter_space` is empty (the
heuristics) simply skip the search.

Selection uses validation PR-AUC. Because the validation quadrant can be large,
it is class-stratified down to a cap for tuning only -- the test set is never
subsampled.

Search, not Bayesian optimisation, is deliberate: random search is trivially
parallel, has no internal state that could differ between models, and with ~20
draws is competitive with more sophisticated methods over these low-dimensional
spaces. Trading a little sample efficiency for an obviously fair protocol is
the right trade here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import subsample_pairs
from .metrics import safe_pr_auc, safe_roc_auc
from .models.base import BaseMatcher, TrainContext, build_model

log = logging.getLogger(__name__)

TUNE_VAL_CAP = 300_000


@dataclass
class TuningResult:
    model_key: str
    best_params: Dict
    best_score: float
    history: List[Tuple[Dict, float]]

    def summary(self) -> str:
        lines = [f"Tuning {self.model_key}: best val PR-AUC = {self.best_score:.4f}"]
        for params, score in sorted(self.history, key=lambda x: -x[1])[:3]:
            lines.append(f"    {score:.4f}  {params}")
        return "\n".join(lines)


def tune_model(
    model_key: str,
    ctx: TrainContext,
    n_trials: int = 20,
    metric: str = "pr_auc",
    seed: int = 0,
) -> TuningResult:
    """Draw `n_trials` configurations, keep the best on validation."""
    rng = np.random.default_rng(seed)
    scorer = safe_pr_auc if metric == "pr_auc" else safe_roc_auc

    probe = build_model(model_key, ctx.cfg, seed=seed)
    space_probe = probe.hyperparameter_space(rng)
    if not space_probe:
        log.info("[tune] %s has no tunable parameters; using defaults.", model_key)
        return TuningResult(model_key, {}, float("nan"), [])

    val = subsample_pairs(ctx.split.val, TUNE_VAL_CAP, seed=seed)
    history: List[Tuple[Dict, float]] = []
    best_params: Dict = {}
    best_score = -np.inf

    for i in range(n_trials):
        params = probe.hyperparameter_space(rng)
        try:
            model = build_model(model_key, ctx.cfg, seed=seed, **params)
            model.fit(ctx)
            score = scorer(val.y, model.score(ctx, val))
        except Exception as e:  # a bad draw must not kill the sweep
            log.warning("[tune] %s draw %d failed (%s: %s)", model_key, i, type(e).__name__, e)
            continue

        history.append((params, float(score)))
        if np.isfinite(score) and score > best_score:
            best_score, best_params = float(score), params
        log.info("[tune] %s draw %d/%d val_%s=%.4f (best %.4f)",
                 model_key, i + 1, n_trials, metric, score, best_score)

    if not history:
        log.warning("[tune] every draw failed for %s; falling back to defaults.", model_key)
        return TuningResult(model_key, {}, float("nan"), [])

    result = TuningResult(model_key, best_params, best_score, history)
    log.info("\n%s", result.summary())
    return result


def refit_best(model_key: str, ctx: TrainContext, tuning: TuningResult, seed: int) -> BaseMatcher:
    """Refit the winning configuration on the training fold."""
    model = build_model(model_key, ctx.cfg, seed=seed, **tuning.best_params)
    return model.fit(ctx)
