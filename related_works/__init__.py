"""Model registry. Importing this module registers every matcher."""

from .base import (  # noqa: F401
    BaseMatcher,
    TrainContext,
    available_models,
    build_model,
    register,
)
from . import heuristics  # noqa: F401
from . import linear  # noqa: F401
from . import trees  # noqa: F401
from . import xgb  # noqa: F401
from . import mlp  # noqa: F401
from . import gnn  # noqa: F401

#: Groups used by run_all.py's --models shortcut
MODEL_GROUPS = {
    "baselines": ["random", "prior_trial", "prior_patient", "cosine"],
    "classical": ["logreg", "rf", "extratrees"],
    "boosting": ["xgboost", "xgboost_rank"],
    "neural": ["mlp", "two_tower"],
    "gnn": ["gcn", "graphsage", "gat", "rgcn", "graphsage_gcl"],
}
MODEL_GROUPS["all"] = [m for g in ("baselines", "classical", "boosting", "neural", "gnn")
                       for m in MODEL_GROUPS[g]]

__all__ = [
    "BaseMatcher",
    "TrainContext",
    "build_model",
    "available_models",
    "register",
    "MODEL_GROUPS",
]
