"""
models/base.py
==============
Common interface every model in the benchmark implements, plus the shared
training context.

The contract is deliberately narrow: a model receives a `TrainContext`
(dataset, labels, split, fitted feature bundle, config, seed) and must produce
a real-valued score per pair. Higher = more likely eligible. Models that emit
calibrated probabilities additionally implement `proba`; the rest return None
and are simply excluded from the calibration columns.

Everything a model needs is on the context, and nothing on the context is
split-specific beyond what the split object exposes, which makes it structurally
awkward to accidentally touch the test fold during fitting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from ..data import Dataset
from ..features import FeatureBundle, FeatureRegime
from ..labeling import LabelMatrices
from ..splits import PairSet, SplitBundle

log = logging.getLogger(__name__)


@dataclass
class TrainContext:
    dataset: Dataset
    labels: LabelMatrices
    split: SplitBundle
    bundle: FeatureBundle
    cfg: object
    seed: int

    @property
    def regime(self) -> FeatureRegime:
        return self.bundle.regime

    @property
    def device(self) -> str:
        return self.cfg.resolve_device()


class BaseMatcher:
    """Abstract patient-trial scorer."""

    name: str = "base"
    #: set False for models that build their own representation from the graph
    uses_pair_features: bool = True
    #: set True for models that never see `y` (zero-shot / unsupervised)
    unsupervised: bool = False

    def __init__(self, cfg, seed: int = 0, **kwargs):
        self.cfg = cfg
        self.seed = seed
        self.params: Dict = dict(kwargs)
        self.fit_seconds: float = 0.0
        self._is_fit = False

    # -- to implement --------------------------------------------------
    def _fit(self, ctx: TrainContext) -> None:
        raise NotImplementedError

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        raise NotImplementedError

    # -- optional ------------------------------------------------------
    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return None

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        """Draw one random hyperparameter configuration. Empty = no tuning."""
        return {}

    # -- driver --------------------------------------------------------
    def fit(self, ctx: TrainContext) -> "BaseMatcher":
        t0 = time.time()
        self._fit(ctx)
        self.fit_seconds = time.time() - t0
        self._is_fit = True
        log.info("[%s] fitted in %.1fs", self.name, self.fit_seconds)
        return self

    def check_fitted(self) -> None:
        if not self._is_fit:
            raise RuntimeError(f"{self.name} used before fit()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[..., BaseMatcher]] = {}


def register(name: str):
    def deco(factory):
        if name in _REGISTRY:
            raise KeyError(f"Model '{name}' already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def build_model(name: str, cfg, seed: int = 0, **kwargs) -> BaseMatcher:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg=cfg, seed=seed, **kwargs)


def available_models() -> List[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Helpers shared by the tabular models
# ---------------------------------------------------------------------------
def chunked_predict(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    bundle: FeatureBundle,
    pairs: PairSet,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """Apply a row-wise predictor over a pair set without materialising it."""
    from ..features import pair_features

    out = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        X = pair_features(bundle, pairs.p_idx[start:end], pairs.t_idx[start:end])
        out[start:end] = np.asarray(predict_fn(X), dtype=np.float32).ravel()
    return out


def positive_class_weight(y: np.ndarray, cap: float = 100.0) -> float:
    n_pos = float(max(y.sum(), 1))
    n_neg = float(max(y.size - y.sum(), 1))
    return float(min(n_neg / n_pos, cap))
