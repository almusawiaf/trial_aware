"""
splits.py
=========
Doubly-disjoint, grouped train/validation/test splits.

Why not a random split over pairs
---------------------------------
The natural unit here is the (patient, trial) pair, but pairs are not
independent: the same patient appears in ~n_trials pairs and the same trial in
~n_patients pairs. A uniform random split over pairs therefore leaks both the
patient and the trial across folds, and a model can score well by memorising
"this patient is eligible for most things" without learning anything about
matching. Reported AUCs under that design are optimistic by a wide margin.

What is done instead
--------------------
Patients and trials are partitioned *independently*, producing a 3x3 grid of
quadrants. Three are used:

    train      = train_patients  x  train_trials
    validation = val_patients    x  val_trials      (both-cold; tuning + early stop)
    test       = test_patients   x  test_trials     (both-cold; PRIMARY result)

and two more are computed as diagnostics, because the gap between them tells
you which axis a model is failing to generalise along:

    cold_patient_only = test_patients x train_trials
    cold_trial_only   = train_patients x test_trials

The trial partition is stratified by each trial's positive rate, so selective
trials (few eligible patients) and permissive trials are represented in every
fold. Without that, a random 20% trial holdout routinely lands with a wildly
different prevalence from training and the PR-AUC becomes incomparable across
seeds.

Negatives are subsampled in *training only*. Evaluation always runs on the full
dense quadrant, so reported PR-AUC reflects the true operating prevalence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .labeling import LabelMatrices

log = logging.getLogger(__name__)


@dataclass
class PairSet:
    """A set of (patient, trial) pairs with labels, addressed by matrix index."""

    p_idx: np.ndarray
    t_idx: np.ndarray
    y: np.ndarray
    name: str = ""

    def __len__(self) -> int:
        return int(self.p_idx.shape[0])

    @property
    def prevalence(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self):,} pairs, "
            f"{int(self.y.sum()):,} positive ({self.prevalence:.4%})"
        )


@dataclass
class SplitBundle:
    seed: int
    patients: Dict[str, np.ndarray]   # 'train' | 'val' | 'test' -> patient row indices
    trials: Dict[str, np.ndarray]     # 'train' | 'val' | 'test' -> trial col indices
    train: PairSet
    val: PairSet
    test: PairSet
    diagnostics: Dict[str, PairSet]
    # Full (un-subsampled) training quadrant, used by models that need the
    # complete dense matrix (e.g. per-trial ranking losses).
    train_full: PairSet

    def summary(self) -> str:
        lines = [f"Split(seed={self.seed})"]
        for k in ("train", "val", "test"):
            lines.append(
                f"  patients[{k}]={len(self.patients[k]):5d}  trials[{k}]={len(self.trials[k]):4d}"
            )
        lines.append("  " + self.train.describe())
        lines.append("  " + self.val.describe())
        lines.append("  " + self.test.describe())
        for name, ps in self.diagnostics.items():
            lines.append("  " + ps.describe())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def _partition(
    n: int, fractions: Tuple[float, float, float], rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    idx = rng.permutation(n)
    f_tr, f_va, _ = fractions
    n_tr = int(round(f_tr * n))
    n_va = int(round(f_va * n))
    return {
        "train": np.sort(idx[:n_tr]),
        "val": np.sort(idx[n_tr : n_tr + n_va]),
        "test": np.sort(idx[n_tr + n_va :]),
    }


def _stratified_partition(
    values: np.ndarray,
    fractions: Tuple[float, float, float],
    n_bins: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Partition indices into train/val/test balanced across bins of `values`."""
    n = values.shape[0]
    if n_bins <= 1 or n < 3 * n_bins:
        return _partition(n, fractions, rng)

    # Quantile bins; ties collapse harmlessly into fewer bins.
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(values, edges)

    out = {"train": [], "val": [], "test": []}
    for b in np.unique(bins):
        members = np.where(bins == b)[0]
        sub = _partition(len(members), fractions, rng)
        for k in out:
            out[k].append(members[sub[k]])
    return {k: np.sort(np.concatenate(v)) if v else np.array([], int) for k, v in out.items()}


def _dense_pairs(
    labels: LabelMatrices, p_idx: np.ndarray, t_idx: np.ndarray, name: str
) -> PairSet:
    """Every pair in the p_idx x t_idx quadrant."""
    pp, tt = np.meshgrid(p_idx, t_idx, indexing="ij")
    pp = pp.ravel()
    tt = tt.ravel()
    y = labels.y[pp, tt].astype(np.int8)
    return PairSet(pp, tt, y, name=name)


def _subsample_negatives(
    pairs: PairSet,
    neg_per_pos: int,
    max_pairs: int,
    rng: np.random.Generator,
    name: str,
) -> PairSet:
    """Keep all positives, sample negatives at the requested ratio.

    Note for anyone reading probabilities off a model trained on this: the
    class prior has been shifted. Ranking metrics are unaffected, but if you
    need calibrated probabilities apply the standard prior correction
        p_true = p / (p + (1 - p) / r)
    with r = (sampled negative rate) / (true negative rate). `metrics.py`
    reports Brier/ECE so the miscalibration is at least visible.
    """
    pos = np.where(pairs.y == 1)[0]
    neg = np.where(pairs.y == 0)[0]
    n_neg_target = min(len(neg), max(neg_per_pos * max(len(pos), 1), 1000))
    if len(neg) > n_neg_target:
        neg = rng.choice(neg, size=n_neg_target, replace=False)
    keep = np.concatenate([pos, neg])

    if len(keep) > max_pairs:
        keep = rng.choice(keep, size=max_pairs, replace=False)
    keep = rng.permutation(keep)

    return PairSet(pairs.p_idx[keep], pairs.t_idx[keep], pairs.y[keep], name=name)


def make_splits(labels: LabelMatrices, cfg, seed: int) -> SplitBundle:
    """Build the split bundle for one seed."""
    rng = np.random.default_rng(seed)
    sc = cfg.split

    n_p, n_t = labels.y.shape

    p_parts = _partition(
        n_p, (sc.patient_train, sc.patient_val, sc.patient_test), rng
    )

    if sc.stratify_trials_by_prevalence:
        trial_prev = labels.y.mean(axis=0)
        t_parts = _stratified_partition(
            trial_prev,
            (sc.trial_train, sc.trial_val, sc.trial_test),
            sc.n_prevalence_bins,
            rng,
        )
    else:
        t_parts = _partition(n_t, (sc.trial_train, sc.trial_val, sc.trial_test), rng)

    for split_name, part in (("patient", p_parts), ("trial", t_parts)):
        for k, v in part.items():
            if len(v) == 0:
                raise ValueError(
                    f"{split_name} split '{k}' is empty (n={n_p if split_name=='patient' else n_t}). "
                    "Reduce the number of folds or supply more data."
                )

    train_full = _dense_pairs(labels, p_parts["train"], t_parts["train"], "train_full")
    train = _subsample_negatives(
        train_full, sc.neg_per_pos_train, sc.max_train_pairs, rng, "train"
    )
    val = _dense_pairs(labels, p_parts["val"], t_parts["val"], "val(both-cold)")
    test = _dense_pairs(labels, p_parts["test"], t_parts["test"], "test(both-cold)")

    diagnostics = {
        "cold_patient_only": _dense_pairs(
            labels, p_parts["test"], t_parts["train"], "diag:cold-patient-only"
        ),
        "cold_trial_only": _dense_pairs(
            labels, p_parts["train"], t_parts["test"], "diag:cold-trial-only"
        ),
    }

    bundle = SplitBundle(
        seed=seed,
        patients=p_parts,
        trials=t_parts,
        train=train,
        val=val,
        test=test,
        diagnostics=diagnostics,
        train_full=train_full,
    )
    _assert_disjoint(bundle)
    if cfg.verbose:
        log.info("\n%s", bundle.summary())
    return bundle


def _assert_disjoint(b: SplitBundle) -> None:
    """Fail loudly rather than silently reporting a leaked number."""
    for axis, parts in (("patients", b.patients), ("trials", b.trials)):
        for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = np.intersect1d(parts[a], parts[c])
            if overlap.size:
                raise AssertionError(
                    f"{axis} overlap between {a} and {c}: {overlap.size} shared ids"
                )
    # And no pair may appear in more than one evaluated set.
    def key(ps: PairSet) -> set:
        return set(zip(ps.p_idx.tolist(), ps.t_idx.tolist()))

    if key(b.train_full) & key(b.test):
        raise AssertionError("train/test pair overlap detected")
    if key(b.val) & key(b.test):
        raise AssertionError("val/test pair overlap detected")


def check_split_health(b: SplitBundle, min_positives: int = 20) -> List[str]:
    """Return human-readable warnings about a split that is too small to trust."""
    warnings: List[str] = []
    for ps in (b.val, b.test):
        n_pos = int(ps.y.sum())
        if n_pos < min_positives:
            warnings.append(
                f"{ps.name} has only {n_pos} positive pairs; PR-AUC will be very "
                f"high variance. Consider more trials, more patients, or fewer folds."
            )
        if n_pos == 0:
            warnings.append(f"{ps.name} has NO positives; metrics are undefined.")
    return warnings
