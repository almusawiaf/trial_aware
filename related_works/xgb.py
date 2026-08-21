"""
models/xgb.py
=============
Gradient-boosted trees, in two forms.

`XGBoostMatcher` (classification)
    Standard binary objective with `scale_pos_weight` and early stopping on the
    *validation* fold. Early stopping is deliberately driven by `aucpr`, not
    `logloss` or `auc`: at ~2% prevalence, logloss is minimised by predicting
    the prior and ROC-AUC barely moves, so both stop far too early or far too
    late relative to the metric the benchmark actually reports.

`XGBRankerMatcher` (learning to rank)
    Trains `rank:pairwise` with one query group per trial. This matches the
    deployed task -- rank patients *within* a trial -- and removes the
    inter-trial score offset that lets a classifier score well by learning
    which trials are permissive. Rows must be sorted by group for XGBoost, and
    the sort is applied to the labels too; getting that wrong silently trains
    on shuffled groups, so it is asserted below.

`scale_pos_weight` is computed from the training fold *after* the negative
subsampling in `splits.py`, not from the raw prevalence -- applying the raw
ratio on top of already-subsampled data double-counts the correction and
pushes every prediction towards 1.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from ..features import materialize, subsample_pairs
from ..splits import PairSet
from .base import BaseMatcher, TrainContext, chunked_predict, positive_class_weight, register

log = logging.getLogger(__name__)

MAX_VAL_PAIRS_FOR_EARLY_STOP = 400_000


@register("xgboost")
class XGBoostMatcher(BaseMatcher):
    name = "XGBoost"

    def _fit(self, ctx: TrainContext) -> None:
        import xgboost as xgb

        xc = ctx.cfg.xgb
        X, y = materialize(ctx.bundle, ctx.split.train)

        val_pairs = subsample_pairs(
            ctx.split.val, MAX_VAL_PAIRS_FOR_EARLY_STOP, seed=self.seed
        )
        Xv, yv = materialize(ctx.bundle, val_pairs)

        spw = positive_class_weight(y, cap=1000.0)

        params = dict(
            n_estimators=int(self.params.get("n_estimators", xc.n_estimators)),
            max_depth=int(self.params.get("max_depth", xc.max_depth)),
            learning_rate=float(self.params.get("learning_rate", xc.learning_rate)),
            subsample=float(self.params.get("subsample", xc.subsample)),
            colsample_bytree=float(self.params.get("colsample_bytree", xc.colsample_bytree)),
            min_child_weight=float(self.params.get("min_child_weight", xc.min_child_weight)),
            reg_lambda=float(self.params.get("reg_lambda", xc.reg_lambda)),
            reg_alpha=float(self.params.get("reg_alpha", xc.reg_alpha)),
            scale_pos_weight=spw,
            tree_method=xc.tree_method,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=xc.early_stopping_rounds,
            random_state=self.seed,
            n_jobs=ctx.cfg.n_jobs,
        )
        self._clf = xgb.XGBClassifier(**params)

        fit_kwargs = {"eval_set": [(Xv, yv)], "verbose": False}
        if yv.max() == 0:
            # No positives in validation -> aucpr is undefined and early
            # stopping would silently pick iteration 0. Train without it and
            # say so, rather than reporting a model that never grew any trees.
            log.warning(
                "[XGBoost] validation fold has no positives; disabling early stopping."
            )
            self._clf.set_params(early_stopping_rounds=None)
            fit_kwargs.pop("eval_set")

        self._clf.fit(X, y, **fit_kwargs)
        self.best_iteration = getattr(self._clf, "best_iteration", None)
        self.feature_importances_ = self._clf.feature_importances_

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        self.check_fitted()
        return chunked_predict(
            lambda X: self._clf.predict_proba(X)[:, 1], ctx.bundle, pairs
        )

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return self.score(ctx, pairs)

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        return {
            "max_depth": int(rng.integers(3, 11)),
            "learning_rate": float(10 ** rng.uniform(-2.3, -0.7)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "min_child_weight": float(10 ** rng.uniform(-0.3, 1.7)),
            "reg_lambda": float(10 ** rng.uniform(-1, 2)),
            "reg_alpha": float(10 ** rng.uniform(-3, 1)),
        }

    def top_features(self, ctx: TrainContext, k: int = 20):
        names = ctx.bundle.feature_names()
        order = np.argsort(-self.feature_importances_)[:k]
        return [(names[i] if i < len(names) else f"f{i}",
                 float(self.feature_importances_[i])) for i in order]


@register("xgboost_rank")
class XGBRankerMatcher(BaseMatcher):
    """Pairwise learning-to-rank with one query group per trial."""

    name = "XGBoost-Rank"

    def _fit(self, ctx: TrainContext) -> None:
        import xgboost as xgb

        xc = ctx.cfg.xgb
        pairs = ctx.split.train

        order = np.argsort(pairs.t_idx, kind="stable")
        p_sorted = PairSet(pairs.p_idx[order], pairs.t_idx[order], pairs.y[order], "train[sorted]")
        assert np.all(np.diff(p_sorted.t_idx) >= 0), "group ids must be sorted for XGBRanker"

        X, y = materialize(ctx.bundle, p_sorted)
        qid = p_sorted.t_idx.astype(np.int64)

        val = subsample_pairs(ctx.split.val, MAX_VAL_PAIRS_FOR_EARLY_STOP, seed=self.seed)
        v_order = np.argsort(val.t_idx, kind="stable")
        v_sorted = PairSet(val.p_idx[v_order], val.t_idx[v_order], val.y[v_order], "val[sorted]")
        Xv, yv = materialize(ctx.bundle, v_sorted)
        qid_v = v_sorted.t_idx.astype(np.int64)

        self._rk = xgb.XGBRanker(
            objective="rank:pairwise",
            n_estimators=int(self.params.get("n_estimators", 600)),
            max_depth=int(self.params.get("max_depth", xc.max_depth)),
            learning_rate=float(self.params.get("learning_rate", xc.learning_rate)),
            subsample=float(self.params.get("subsample", xc.subsample)),
            colsample_bytree=float(self.params.get("colsample_bytree", xc.colsample_bytree)),
            reg_lambda=float(self.params.get("reg_lambda", xc.reg_lambda)),
            tree_method=xc.tree_method,
            eval_metric="ndcg@50",
            early_stopping_rounds=xc.early_stopping_rounds,
            random_state=self.seed,
            n_jobs=ctx.cfg.n_jobs,
        )
        try:
            self._rk.fit(X, y, qid=qid, eval_set=[(Xv, yv)], eval_qid=[qid_v], verbose=False)
        except Exception as e:      # groups too small / no positives in a group
            log.warning("[XGBRanker] early stopping failed (%s); refitting without it.", e)
            self._rk.set_params(early_stopping_rounds=None)
            self._rk.fit(X, y, qid=qid, verbose=False)

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        self.check_fitted()
        return chunked_predict(lambda X: self._rk.predict(X), ctx.bundle, pairs)

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        return {
            "max_depth": int(rng.integers(3, 9)),
            "learning_rate": float(10 ** rng.uniform(-2.0, -0.7)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "reg_lambda": float(10 ** rng.uniform(-1, 2)),
        }
