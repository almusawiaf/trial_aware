"""
models/trees.py
===============
Bagged tree ensembles (random forest, extremely randomised trees).

These sit between logistic regression and boosting. They capture interactions
without the tuning sensitivity of boosting, and because they are bagged rather
than boosted they do not chase the noise in the small positive class as
aggressively -- useful evidence when the boosted model's advantage turns out to
be an artefact of overfitting the ~2% positives.

`class_weight='balanced_subsample'` reweights within each bootstrap sample,
which is the right variant here: with global balancing, a bootstrap that
happens to draw very few positives gets the same weights as one that draws
many, and the ensemble members become inconsistent with each other.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from ..features import materialize
from ..splits import PairSet
from .base import BaseMatcher, TrainContext, chunked_predict, register


class _ForestBase(BaseMatcher):
    _cls = RandomForestClassifier

    def _fit(self, ctx: TrainContext) -> None:
        rc = ctx.cfg.rf
        X, y = materialize(ctx.bundle, ctx.split.train)
        self._clf = self._cls(
            n_estimators=int(self.params.get("n_estimators", rc.n_estimators)),
            max_depth=self.params.get("max_depth", rc.max_depth),
            min_samples_leaf=int(self.params.get("min_samples_leaf", rc.min_samples_leaf)),
            max_features=self.params.get("max_features", rc.max_features),
            class_weight=rc.class_weight,
            n_jobs=ctx.cfg.n_jobs,
            random_state=self.seed,
            bootstrap=True,
        ).fit(X, y)
        self.feature_importances_ = getattr(self._clf, "feature_importances_", None)

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        self.check_fitted()
        return chunked_predict(
            lambda X: self._clf.predict_proba(X)[:, 1], ctx.bundle, pairs
        )

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return self.score(ctx, pairs)

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        return {
            "n_estimators": int(rng.choice([200, 400, 800])),
            "max_depth": int(rng.choice([8, 12, 20, 32])),
            "min_samples_leaf": int(rng.choice([1, 3, 5, 10])),
            "max_features": str(rng.choice(["sqrt", "log2"])),
        }

    def top_features(self, ctx: TrainContext, k: int = 20):
        """Most important features, for the interpretability section of a paper."""
        if self.feature_importances_ is None:
            return []
        names = ctx.bundle.feature_names()
        order = np.argsort(-self.feature_importances_)[:k]
        return [(names[i] if i < len(names) else f"f{i}",
                 float(self.feature_importances_[i])) for i in order]


@register("rf")
class RandomForestMatcher(_ForestBase):
    name = "RandomForest"
    _cls = RandomForestClassifier


@register("extratrees")
class ExtraTreesMatcher(_ForestBase):
    name = "ExtraTrees"
    _cls = ExtraTreesClassifier

    def _fit(self, ctx: TrainContext) -> None:
        # ExtraTrees defaults to bootstrap=False; keep it that way, which is
        # what makes it "extremely randomised" rather than a slower RF.
        rc = ctx.cfg.rf
        X, y = materialize(ctx.bundle, ctx.split.train)
        self._clf = ExtraTreesClassifier(
            n_estimators=int(self.params.get("n_estimators", rc.n_estimators)),
            max_depth=self.params.get("max_depth", rc.max_depth),
            min_samples_leaf=int(self.params.get("min_samples_leaf", rc.min_samples_leaf)),
            max_features=self.params.get("max_features", rc.max_features),
            class_weight="balanced",
            n_jobs=ctx.cfg.n_jobs,
            random_state=self.seed,
            bootstrap=False,
        ).fit(X, y)
        self.feature_importances_ = self._clf.feature_importances_
