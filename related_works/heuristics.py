"""
models/heuristics.py
====================
Reference points without which none of the learned numbers can be interpreted.

`RandomScorer`
    The floor. Pooled ROC-AUC 0.5 by construction; its PR-AUC equals the
    prevalence and is the number every other PR-AUC should be read against.

`TrialPrevalencePrior` / `PatientPrevalencePrior`
    Score depends only on the trial, or only on the patient -- never on the
    pair. These are the "degenerate solution" detectors. A model that beats
    random but not the prevalence priors has learned who is generally eligible,
    not who matches *this* trial. In a doubly-cold split the priors must fall
    back to a global constant for unseen ids, which is exactly the point: the
    gap between their warm and cold behaviour bounds how much of any model's
    score is popularity memorisation.

`EmbeddingCosine`
    Unsupervised: cosine similarity between patient and trial vectors in the
    shared SVD space, never touching `y`. This is the structural analogue of
    the upstream Stage A/Stage B cosine scorer, and it is the fair comparator
    for the GCL model -- comparing an unsupervised retriever against supervised
    XGBoost without also showing this row would flatter the supervised side for
    reasons that have nothing to do with architecture.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..splits import PairSet
from .base import BaseMatcher, TrainContext, register


@register("random")
class RandomScorer(BaseMatcher):
    name = "Random"
    unsupervised = True

    def _fit(self, ctx: TrainContext) -> None:
        self._rng = np.random.default_rng(self.seed)

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        return self._rng.random(len(pairs)).astype(np.float32)

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        prior = float(ctx.split.train.y.mean())
        return np.full(len(pairs), prior, dtype=np.float32)


@register("prior_trial")
class TrialPrevalencePrior(BaseMatcher):
    name = "TrialPrior"

    def _fit(self, ctx: TrainContext) -> None:
        y = ctx.labels.y
        tr_p = ctx.split.patients["train"]
        self._global = float(y[np.ix_(tr_p, ctx.split.trials["train"])].mean())
        rates = np.full(y.shape[1], self._global, dtype=np.float32)
        for t in ctx.split.trials["train"]:
            rates[t] = float(y[tr_p, t].mean())
        self._rates = rates
        self._seen = set(ctx.split.trials["train"].tolist())

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        s = self._rates[pairs.t_idx].copy()
        unseen = ~np.isin(pairs.t_idx, list(self._seen))
        s[unseen] = self._global
        return s.astype(np.float32)

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return self.score(ctx, pairs)


@register("prior_patient")
class PatientPrevalencePrior(BaseMatcher):
    name = "PatientPrior"

    def _fit(self, ctx: TrainContext) -> None:
        y = ctx.labels.y
        tr_t = ctx.split.trials["train"]
        self._global = float(y[np.ix_(ctx.split.patients["train"], tr_t)].mean())
        rates = np.full(y.shape[0], self._global, dtype=np.float32)
        for p in ctx.split.patients["train"]:
            rates[p] = float(y[p, tr_t].mean())
        self._rates = rates
        self._seen = set(ctx.split.patients["train"].tolist())

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        s = self._rates[pairs.p_idx].copy()
        unseen = ~np.isin(pairs.p_idx, list(self._seen))
        s[unseen] = self._global
        return s.astype(np.float32)

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return self.score(ctx, pairs)


@register("cosine")
class EmbeddingCosine(BaseMatcher):
    """Unsupervised cosine in the shared SVD space -- the GCL-style comparator."""

    name = "SVD-Cosine (unsup.)"
    unsupervised = True

    def _fit(self, ctx: TrainContext) -> None:
        P = ctx.bundle.P_svd
        T = ctx.bundle.T_svd
        self._Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)
        self._Tn = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-8)

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        return np.einsum(
            "ij,ij->i", self._Pn[pairs.p_idx], self._Tn[pairs.t_idx]
        ).astype(np.float32)
