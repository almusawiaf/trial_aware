"""
metrics.py
==========
Evaluation for an extremely imbalanced, structured retrieval problem.

Choices worth defending
-----------------------
* **PR-AUC is primary, ROC-AUC secondary.** At ~1-3% prevalence, ROC-AUC is
  dominated by the vast negative mass and moves very little between a useless
  and a useful model. Average precision responds to what actually matters here:
  where the positives sit near the top of the ranking.

* **Ranking metrics are computed per trial, then averaged.** The deployed
  question is "given this trial, which patients should a coordinator screen?"
  A global pooled AUC over the flattened matrix silently rewards a model that
  merely learns which *trials* are permissive, because inter-trial score
  offsets shift the whole column. Per-trial (grouped) metrics remove that
  degree of freedom. Both are reported; a large gap between pooled and grouped
  AUC is a red flag, not a rounding difference.

* **Bootstrap resamples trials, not pairs.** Pairs sharing a trial are strongly
  dependent, so a pair-level bootstrap understates the interval, often by a
  factor of several. Resampling whole trials (a cluster bootstrap) respects the
  dependency structure.

* **Calibration is reported** because training used negative subsampling, which
  shifts the class prior. Brier and ECE make that distortion visible instead of
  letting a well-ranked but badly-calibrated model pass as production-ready.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def safe_roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.size == 0 or y.max() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, y.size)
    top = np.argpartition(-s, k - 1)[:k]
    return float(y[top].sum() / n_pos)


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    if y.size == 0:
        return float("nan")
    k = min(k, y.size)
    top = np.argpartition(-s, k - 1)[:k]
    return float(y[top].sum() / k)


def ndcg_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, y.size)
    order = np.argsort(-s, kind="stable")[:k]
    gains = y[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal_n = min(n_pos, k)
    idcg = float(discounts[:ideal_n].sum())
    return dcg / idcg if idcg > 0 else float("nan")


def reciprocal_rank(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    hits = np.where(y[order] == 1)[0]
    return float(1.0 / (hits[0] + 1))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    if y.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / y.size) * abs(y[m].mean() - p[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Grouped (per-trial / per-patient) evaluation
# ---------------------------------------------------------------------------
def grouped_metrics(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    ks: Sequence[int] = (10, 50, 100),
    min_group_pos: int = 1,
) -> Dict[str, float]:
    """Average per-group metrics over groups that contain at least one positive.

    Groups with no positive (or no negative) are skipped for AUC because the
    metric is undefined there; the count of usable groups is returned so a
    result computed over three trials is not mistaken for a stable estimate.
    """
    out_lists: Dict[str, List[float]] = {}
    uniq = np.unique(groups)
    n_used = 0

    for g in uniq:
        m = groups == g
        yg, sg = y[m], s[m]
        if yg.sum() < min_group_pos:
            continue
        n_used += 1
        out_lists.setdefault("roc_auc", []).append(safe_roc_auc(yg, sg))
        out_lists.setdefault("pr_auc", []).append(safe_pr_auc(yg, sg))
        out_lists.setdefault("mrr", []).append(reciprocal_rank(yg, sg))
        for k in ks:
            out_lists.setdefault(f"recall@{k}", []).append(recall_at_k(yg, sg, k))
            out_lists.setdefault(f"precision@{k}", []).append(precision_at_k(yg, sg, k))
            out_lists.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(yg, sg, k))

    out = {k: float(np.nanmean(v)) if v else float("nan") for k, v in out_lists.items()}
    out["n_groups_used"] = float(n_used)
    out["n_groups_total"] = float(len(uniq))
    return out


@dataclass
class EvalResult:
    model: str
    regime: str
    split: str
    seed: int
    pooled: Dict[str, float] = field(default_factory=dict)
    per_trial: Dict[str, float] = field(default_factory=dict)
    per_patient: Dict[str, float] = field(default_factory=dict)
    calibration: Dict[str, float] = field(default_factory=dict)
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    n_pairs: int = 0
    n_positive: int = 0
    fit_seconds: float = 0.0

    def flat(self) -> Dict[str, float]:
        row: Dict[str, float] = {
            "model": self.model,
            "regime": self.regime,
            "split": self.split,
            "seed": self.seed,
            "n_pairs": self.n_pairs,
            "n_positive": self.n_positive,
            "fit_seconds": round(self.fit_seconds, 2),
        }
        row.update({f"pooled_{k}": v for k, v in self.pooled.items()})
        row.update({f"trial_{k}": v for k, v in self.per_trial.items()})
        row.update({f"patient_{k}": v for k, v in self.per_patient.items()})
        row.update({f"calib_{k}": v for k, v in self.calibration.items()})
        for k, (lo, hi) in self.ci.items():
            row[f"ci_{k}_lo"], row[f"ci_{k}_hi"] = lo, hi
        return row


def evaluate_scores(
    y: np.ndarray,
    scores: np.ndarray,
    p_idx: np.ndarray,
    t_idx: np.ndarray,
    cfg,
    model: str,
    regime: str,
    split: str,
    seed: int,
    probabilities: Optional[np.ndarray] = None,
    fit_seconds: float = 0.0,
) -> EvalResult:
    """Full evaluation of one score vector."""
    ec = cfg.eval
    res = EvalResult(
        model=model, regime=regime, split=split, seed=seed,
        n_pairs=int(y.size), n_positive=int(y.sum()), fit_seconds=fit_seconds,
    )

    res.pooled = {
        "roc_auc": safe_roc_auc(y, scores),
        "pr_auc": safe_pr_auc(y, scores),
        "prevalence": float(y.mean()) if y.size else float("nan"),
        "lift_over_prevalence": (
            safe_pr_auc(y, scores) / float(y.mean()) if y.size and y.mean() > 0 else float("nan")
        ),
    }
    res.per_trial = grouped_metrics(y, scores, t_idx, ec.ks)
    res.per_patient = grouped_metrics(y, scores, p_idx, ec.ks)

    if ec.compute_calibration and probabilities is not None:
        p = np.clip(probabilities, 1e-7, 1 - 1e-7)
        res.calibration = {
            "brier": float(brier_score_loss(y, p)) if y.size else float("nan"),
            "ece": expected_calibration_error(y, p),
            "mean_pred": float(p.mean()),
            "mean_label": float(y.mean()) if y.size else float("nan"),
        }

    if ec.n_bootstrap > 0:
        res.ci = cluster_bootstrap_ci(
            y, scores, t_idx if ec.bootstrap_unit == "trial" else p_idx,
            n_boot=ec.n_bootstrap, alpha=ec.ci_alpha, seed=seed,
        )
    return res


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    metrics: Sequence[str] = ("roc_auc", "pr_auc"),
) -> Dict[str, Tuple[float, float]]:
    """Percentile CI from resampling whole groups (trials) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]

    draws: Dict[str, List[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        yb, sb = y[idx], s[idx]
        if yb.min() == yb.max():
            continue
        if "roc_auc" in draws:
            draws["roc_auc"].append(safe_roc_auc(yb, sb))
        if "pr_auc" in draws:
            draws["pr_auc"].append(safe_pr_auc(yb, sb))

    out: Dict[str, Tuple[float, float]] = {}
    for m, vals in draws.items():
        if len(vals) < 20:
            out[m] = (float("nan"), float("nan"))
            continue
        arr = np.asarray(vals, dtype=float)
        out[m] = (
            float(np.nanpercentile(arr, 100 * alpha / 2)),
            float(np.nanpercentile(arr, 100 * (1 - alpha / 2))),
        )
    return out


def paired_bootstrap_test(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    groups: np.ndarray,
    metric: str = "pr_auc",
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Two-sided paired cluster bootstrap on the metric difference (A - B).

    Both score vectors are evaluated on the *same* resampled clusters, so the
    test controls for which trials happened to be drawn -- much tighter than
    comparing two independent CIs, and the correct way to ask "is A better than
    B on this data".
    """
    fn = safe_pr_auc if metric == "pr_auc" else safe_roc_auc
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]

    observed = fn(y, s_a) - fn(y, s_b)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        diffs.append(fn(yb, s_a[idx]) - fn(yb, s_b[idx]))

    if len(diffs) < 20:
        return {"diff": observed, "p_value": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    arr = np.asarray(diffs, dtype=float)
    centred = arr - np.nanmean(arr)
    p = float((np.abs(centred) >= abs(observed)).mean())
    return {
        "diff": float(observed),
        "p_value": p,
        "ci_lo": float(np.nanpercentile(arr, 2.5)),
        "ci_hi": float(np.nanpercentile(arr, 97.5)),
        "n_boot_used": float(len(arr)),
    }


def across_seed_test(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank across seeds, the right test for a multi-seed sweep.

    With five seeds the minimum attainable two-sided p is 0.0625, so a
    'significant' result at 0.05 is impossible by construction. That is stated
    rather than hidden: use the paired bootstrap for within-split evidence and
    treat the seed-level test as a consistency check.
    """
    from scipy import stats

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.allclose(a, b):
        return {"mean_diff": float(np.mean(a - b)) if a.size else float("nan"),
                "p_value": float("nan"), "n": float(a.size)}
    try:
        stat, p = stats.wilcoxon(a, b)
    except ValueError:
        return {"mean_diff": float(np.mean(a - b)), "p_value": float("nan"), "n": float(a.size)}
    return {
        "mean_diff": float(np.mean(a - b)),
        "p_value": float(p),
        "statistic": float(stat),
        "n": float(a.size),
        "min_attainable_p": float(2.0 / (2 ** a.size)),
    }
