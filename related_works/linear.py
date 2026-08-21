"""
models/linear.py
================
Regularised logistic regression.

Purpose in the benchmark: it is the reference that tells you how much of the
signal is *linearly* available in the feature space. If XGBoost and the MLP
barely beat it, the interaction structure they are supposed to be learning
isn't there, and the honest conclusion is that the extra capacity bought
nothing -- a result worth reporting rather than hiding.

`class_weight='balanced'` is used rather than resampling inside the model,
because the training pairs have already been negative-subsampled once in
`splits.py`; stacking a second imbalance correction on top would distort the
decision threshold twice over.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..features import materialize
from ..splits import PairSet
from .base import BaseMatcher, TrainContext, chunked_predict, register


def _sklearn_ge(major: int, minor: int) -> bool:
    import sklearn

    parts = sklearn.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):
        return False


def _logreg_kwargs(penalty: str, l1_ratio: float, **common) -> dict:
    """Build LogisticRegression kwargs across scikit-learn versions.

    `penalty` was deprecated in scikit-learn 1.8 in favour of expressing the
    same thing through `l1_ratio` (0 = ridge, 1 = lasso). The benchmark is
    expected to run on cluster environments pinned to older releases, so both
    spellings are supported rather than pinning the whole project to one.
    """
    if _sklearn_ge(1, 8):
        ratio = {"l2": 0.0, "l1": 1.0}.get(penalty, l1_ratio)
        return dict(solver="saga", l1_ratio=ratio, **common)

    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
    kwargs = dict(penalty=penalty, solver=solver, **common)
    if solver != "saga":
        kwargs.pop("n_jobs", None)
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = l1_ratio
    return kwargs


@register("logreg")
class LogisticRegressionMatcher(BaseMatcher):
    name = "LogisticRegression"

    def _fit(self, ctx: TrainContext) -> None:
        lc = ctx.cfg.logreg
        X, y = materialize(ctx.bundle, ctx.split.train)

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        penalty = self.params.get("penalty", lc.penalty)
        l1_ratio = float(self.params.get("l1_ratio", 0.5))
        kwargs = _logreg_kwargs(
            penalty=penalty,
            l1_ratio=l1_ratio,
            C=float(self.params.get("C", lc.C)),
            max_iter=int(self.params.get("max_iter", lc.max_iter)),
            class_weight=lc.class_weight,
            random_state=self.seed,
            n_jobs=ctx.cfg.n_jobs,
        )
        self._clf = LogisticRegression(**kwargs).fit(Xs, y)

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        self.check_fitted()
        return chunked_predict(
            lambda X: self._clf.decision_function(self._scaler.transform(X)),
            ctx.bundle, pairs,
        )

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        self.check_fitted()
        return chunked_predict(
            lambda X: self._clf.predict_proba(self._scaler.transform(X))[:, 1],
            ctx.bundle, pairs,
        )

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        return {
            "C": float(10 ** rng.uniform(-3, 2)),
            "penalty": str(rng.choice(["l2", "l1", "elasticnet"])),
            "l1_ratio": float(rng.uniform(0.1, 0.9)),
        }
