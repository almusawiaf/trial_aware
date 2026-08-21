"""
train_eval.py
=============
Orchestration: one call builds the labelled dataset, the splits and the feature
bundle for a given seed and regime, then fits and evaluates each requested
model against identical inputs.

Order of operations is load once -> label once -> split per seed -> fit feature
transforms on the training entities of *that* seed. The feature transforms must
be re-fitted per seed, because their vocabularies and SVD bases are estimated
from training data; fitting them once outside the seed loop would leak held-out
structure into every fold and quietly inflate all five "independent" runs in
the same direction.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import BenchmarkConfig
from .data import Dataset, load_dataset
from .features import FeatureBuilder, FeatureRegime
from .labeling import LabelMatrices, RuleLabeler, audit_leakage
from .metrics import EvalResult, evaluate_scores
from .models import MODEL_GROUPS, build_model
from .models.base import TrainContext
from .splits import SplitBundle, check_split_health, make_splits
from .tune import refit_best, tune_model

log = logging.getLogger(__name__)


@dataclass
class PreparedData:
    dataset: Dataset
    labels: LabelMatrices
    audit: Dict[str, float]


def prepare_data(cfg: BenchmarkConfig, data_mode: str = "auto") -> PreparedData:
    """Load, label and audit. Done once and reused across seeds."""
    dataset = load_dataset(cfg, mode=data_mode)
    log.info("%s", dataset.describe())

    labeler = RuleLabeler(
        inc_threshold=cfg.label.inc_threshold,
        exc_threshold=cfg.label.exc_threshold,
        sigmoid_delta=cfg.label.sigmoid_delta,
        drop_unresolved=cfg.label.drop_unmapped_only_trials,
    )
    labels = labeler.fit(dataset.patients).build(
        dataset, drop_trials_without_inclusion=cfg.label.drop_trials_without_inclusion
    )

    audit = audit_leakage(labels)
    log.info("Label statistics: %s", json.dumps(labels.stats(), indent=2))
    log.warning(
        "LEAKAGE AUDIT -- scoring pairs by the rule's own inputs (M_inc - M_exc) "
        "gives ROC-AUC=%.4f, PR-AUC=%.4f at prevalence %.4f. Any model handed "
        "overlap features can reach this; it is a property of the label, not a result.",
        audit["oracle_score_roc_auc"], audit["oracle_score_pr_auc"], audit["prevalence"],
    )
    return PreparedData(dataset, labels, audit)


def build_context(
    cfg: BenchmarkConfig, prepared: PreparedData, seed: int, regime: FeatureRegime
) -> TrainContext:
    split = make_splits(prepared.labels, cfg, seed)
    for w in check_split_health(split):
        log.warning("SPLIT HEALTH: %s", w)

    builder = FeatureBuilder(cfg, regime=regime).fit(prepared.dataset, split)
    bundle = builder.build(prepared.dataset)

    return TrainContext(
        dataset=prepared.dataset,
        labels=prepared.labels,
        split=split,
        bundle=bundle,
        cfg=cfg,
        seed=seed,
    )


def evaluate_model_on_all_splits(
    model, ctx: TrainContext, model_key: str, include_diagnostics: bool = True
) -> List[EvalResult]:
    """Score the fitted model on test, validation and the diagnostic quadrants."""
    results: List[EvalResult] = []
    targets = [("test", ctx.split.test), ("val", ctx.split.val)]
    if include_diagnostics:
        targets += [(name, ps) for name, ps in ctx.split.diagnostics.items()]

    for split_name, pairs in targets:
        if len(pairs) == 0 or pairs.y.max() == 0:
            log.warning("[%s] split '%s' has no positives; skipping.", model.name, split_name)
            continue
        scores = model.score(ctx, pairs)
        probs = model.proba(ctx, pairs)
        results.append(
            evaluate_scores(
                y=pairs.y.astype(np.int32),
                scores=scores,
                p_idx=pairs.p_idx,
                t_idx=pairs.t_idx,
                cfg=ctx.cfg,
                model=model.name,
                regime=ctx.regime.value,
                split=split_name,
                seed=ctx.seed,
                probabilities=probs,
                fit_seconds=model.fit_seconds,
            )
        )
    return results


def run_single(
    cfg: BenchmarkConfig,
    prepared: PreparedData,
    model_key: str,
    seed: int,
    regime: FeatureRegime,
    tune: bool = True,
    ctx: Optional[TrainContext] = None,
) -> List[EvalResult]:
    """Tune (optionally), fit and evaluate one model for one seed and regime."""
    ctx = ctx or build_context(cfg, prepared, seed, regime)

    if tune and cfg.n_tuning_trials > 0:
        tuning = tune_model(model_key, ctx, n_trials=cfg.n_tuning_trials, seed=seed)
        model = refit_best(model_key, ctx, tuning, seed)
    else:
        model = build_model(model_key, cfg, seed=seed).fit(ctx)

    return evaluate_model_on_all_splits(model, ctx, model_key)


def run_benchmark(
    cfg: BenchmarkConfig,
    model_keys: Sequence[str],
    regimes: Sequence[FeatureRegime] = (FeatureRegime.HONEST,),
    seeds: Optional[Sequence[int]] = None,
    data_mode: str = "auto",
    tune: bool = True,
    prepared: Optional[PreparedData] = None,
) -> "pd.DataFrame":
    """Full sweep: models x regimes x seeds."""
    import pandas as pd

    prepared = prepared or prepare_data(cfg, data_mode)
    seeds = list(seeds if seeds is not None else cfg.seeds)

    rows: List[Dict] = []
    for regime in regimes:
        for seed in seeds:
            log.info("=" * 78)
            log.info("REGIME=%s  SEED=%d", regime.value, seed)
            log.info("=" * 78)
            # Context is built once per (regime, seed) and shared by all models,
            # so every model sees byte-identical splits and features.
            ctx = build_context(cfg, prepared, seed, regime)

            for key in model_keys:
                log.info("--- model=%s regime=%s seed=%d ---", key, regime.value, seed)
                try:
                    results = run_single(
                        cfg, prepared, key, seed, regime, tune=tune, ctx=ctx
                    )
                    rows.extend(r.flat() for r in results)
                except Exception as e:
                    log.exception("Model %s failed on seed %d: %s", key, seed, e)
                    rows.append(
                        {"model": key, "regime": regime.value, "split": "test",
                         "seed": seed, "error": f"{type(e).__name__}: {e}"}
                    )

    df = pd.DataFrame(rows)
    out = os.path.join(cfg.paths.output_dir, "raw_results.csv")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Raw results written to %s (%d rows)", out, len(df))
    return df


def resolve_model_keys(spec: Sequence[str]) -> List[str]:
    """Expand group names ('all', 'gnn', ...) into concrete model keys."""
    out: List[str] = []
    for s in spec:
        if s in MODEL_GROUPS:
            out.extend(MODEL_GROUPS[s])
        else:
            out.append(s)
    seen, unique = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique
