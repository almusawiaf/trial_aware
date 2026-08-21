"""
features.py
===========
Builds patient-side and trial-side feature blocks and assembles pair features
under an explicit *feature regime*.

The regimes exist because the benchmark's label is a closed-form rule over the
same inputs the model sees (see `labeling.py`). Which features you allow
decides whether you are measuring learning or measuring the labeller.

    FeatureRegime.ORACLE
        Adds exact overlap statistics: |inc n patient| / |inc|, exclusion hit
        indicator, Jaccard, weighted match counts. These *are* M_inc and M_exc
        up to a monotone transform, so any model given them reproduces the
        label. Runs in this regime are a leakage audit and an upper bound.
        Never quote them as performance.

    FeatureRegime.HONEST                                     [default]
        Patient block + trial block + interactions computed only in the
        low-rank SVD space. No exact code-overlap count is ever exposed. A
        model must recover the interaction from training pairs. Cross-basis
        SVD products are lossy and do not reconstruct the intersection
        cardinality, so this is learning, not lookup.

    FeatureRegime.HONEST_NOINT
        As above with every interaction term removed -- plain concatenation
        [patient | trial]. The strictest setting, and the cleanest contrast
        with the two-tower and GNN models, which build their own interactions.

Vocabularies, IDF weights and SVD bases are fit on **training patients and
training trials only**. Fitting them on the full matrix would leak test-fold
co-occurrence structure into the representation, which is a subtle but real
form of transductive leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .data import Dataset, PatientRecord, TrialSpec
from .splits import PairSet, SplitBundle

log = logging.getLogger(__name__)


class FeatureRegime(str, Enum):
    ORACLE = "oracle"
    HONEST = "honest"
    HONEST_NOINT = "honest_noint"


@dataclass
class FeatureBundle:
    """Per-entity feature matrices plus the metadata needed to assemble pairs."""

    P: np.ndarray                 # (n_patients, d_p)
    T: np.ndarray                 # (n_trials, d_t)
    P_svd: np.ndarray             # (n_patients, k)
    T_svd: np.ndarray             # (n_trials, k)
    patient_names: List[str]
    trial_names: List[str]
    regime: FeatureRegime
    # ORACLE-only ingredients
    P_code: Optional[sparse.csr_matrix] = None   # patient x shared-code space
    T_inc: Optional[sparse.csr_matrix] = None    # trial x shared-code space (inclusion)
    T_exc: Optional[sparse.csr_matrix] = None    # trial x shared-code space (exclusion)

    @property
    def dim(self) -> int:
        d = self.P.shape[1] + self.T.shape[1]
        if self.regime == FeatureRegime.HONEST:
            d += 2 * self.P_svd.shape[1] + 3
        elif self.regime == FeatureRegime.ORACLE:
            d += 2 * self.P_svd.shape[1] + 3 + 8
        return d

    def feature_names(self) -> List[str]:
        names = list(self.patient_names) + list(self.trial_names)
        k = self.P_svd.shape[1]
        if self.regime in (FeatureRegime.HONEST, FeatureRegime.ORACLE):
            names += [f"inter_prod_{i}" for i in range(k)]
            names += [f"inter_absdiff_{i}" for i in range(k)]
            names += ["inter_cos", "inter_l2", "inter_dot"]
        if self.regime == FeatureRegime.ORACLE:
            names += [
                "orc_inc_match_frac", "orc_inc_match_count", "orc_inc_size",
                "orc_exc_any_hit", "orc_exc_match_count", "orc_exc_size",
                "orc_jaccard", "orc_inc_minus_exc",
            ]
        return names


class FeatureBuilder:
    """Fit on training entities, then transform any pair set."""

    def __init__(self, cfg, regime: FeatureRegime = FeatureRegime.HONEST):
        self.cfg = cfg
        self.fc = cfg.features
        self.regime = regime

        self._dx_vocab: Dict[str, int] = {}
        self._rx_vocab: Dict[str, int] = {}
        self._lab_vocab: Dict[str, int] = {}
        self._crit_vocab: Dict[str, int] = {}
        self._shared_vocab: Dict[str, int] = {}
        self._tfidf: Optional[TfidfVectorizer] = None
        self._svd_p: Optional[TruncatedSVD] = None
        self._svd_t: Optional[TruncatedSVD] = None
        self._scaler_p: Optional[StandardScaler] = None
        self._scaler_t: Optional[StandardScaler] = None
        self._fitted = False

    # -- vocabulary ----------------------------------------------------
    @staticmethod
    def _top_k(counter: Dict[str, int], k: int) -> Dict[str, int]:
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return {code: i for i, (code, _) in enumerate(items)}

    def fit(self, dataset: Dataset, split: SplitBundle) -> "FeatureBuilder":
        patients = dataset.patients
        trials = dataset.trials
        tr_p = [patients[i] for i in split.patients["train"]]
        tr_t = [trials[i] for i in split.trials["train"]]

        dx_c, rx_c, lab_c, crit_c = {}, {}, {}, {}
        for p in tr_p:
            for c in p.diagnosis_codes:
                dx_c[c] = dx_c.get(c, 0) + 1
            for c in p.medication_codes:
                rx_c[c] = rx_c.get(c, 0) + 1
            for c in p.lab_values:
                lab_c[c] = lab_c.get(c, 0) + 1
        for t in tr_t:
            for c in t.criteria:
                if c.is_resolved:
                    key = f"{c.entity_type[:2]}:{c.entity_code}"
                    crit_c[key] = crit_c.get(key, 0) + 1

        self._dx_vocab = self._top_k(dx_c, self.fc.top_k_diagnoses)
        self._rx_vocab = self._top_k(rx_c, self.fc.top_k_medications)
        self._lab_vocab = self._top_k(lab_c, self.fc.top_k_labs)
        self._crit_vocab = self._top_k(crit_c, self.fc.top_k_criterion_codes)

        # Shared code space, used only for the ORACLE overlap features.
        shared = set()
        for p in tr_p:
            shared |= {f"di:{c}" for c in p.diagnosis_codes}
            shared |= {f"me:{c}" for c in p.medication_codes}
            shared |= {f"la:{c}" for c in p.lab_values}
        for t in tr_t:
            for c in t.criteria:
                if c.is_resolved:
                    shared.add(f"{c.entity_type[:2]}:{c.entity_code}")
        self._shared_vocab = {c: i for i, c in enumerate(sorted(shared))}

        # Text model over training trials only.
        if self.fc.use_text_svd:
            corpus = [t.text() for t in tr_t]
            corpus = [c if c.strip() else "empty" for c in corpus]
            self._tfidf = TfidfVectorizer(
                max_features=20000,
                min_df=min(self.fc.min_df, max(1, len(corpus) - 1)),
                stop_words="english",
                sublinear_tf=True,
            )
            try:
                self._tfidf.fit(corpus)
            except ValueError:
                log.warning("TF-IDF fit failed (degenerate corpus); disabling text block.")
                self._tfidf = None

        # SVD bases, fit on training entities only.
        P_raw_tr = self._patient_block(tr_p, dense=False)
        T_raw_tr = self._trial_block(tr_t, dense=False)

        k_p = min(self.fc.svd_components_patient, max(1, P_raw_tr.shape[1] - 1), max(1, len(tr_p) - 1))
        k_t = min(self.fc.svd_components_trial, max(1, T_raw_tr.shape[1] - 1), max(1, len(tr_t) - 1))
        k = max(2, min(k_p, k_t))   # shared width so interactions are elementwise

        self._svd_p = TruncatedSVD(n_components=k, random_state=0).fit(P_raw_tr)
        self._svd_t = TruncatedSVD(n_components=k, random_state=0).fit(T_raw_tr)

        self._scaler_p = StandardScaler().fit(self._svd_p.transform(P_raw_tr))
        self._scaler_t = StandardScaler().fit(self._svd_t.transform(T_raw_tr))

        self._fitted = True
        log.info(
            "FeatureBuilder fitted [regime=%s]: |dx|=%d |rx|=%d |lab|=%d |crit|=%d svd_k=%d",
            self.regime.value, len(self._dx_vocab), len(self._rx_vocab),
            len(self._lab_vocab), len(self._crit_vocab), k,
        )
        return self

    # -- entity blocks -------------------------------------------------
    def _patient_block(self, patients: Sequence[PatientRecord], dense: bool = True):
        n = len(patients)
        n_dx, n_rx, n_lab = len(self._dx_vocab), len(self._rx_vocab), len(self._lab_vocab)
        rows, cols, vals = [], [], []
        extra = np.zeros((n, 6), dtype=np.float32)

        off_rx = n_dx
        off_lab = n_dx + n_rx
        off_labmask = off_lab + n_lab

        for i, p in enumerate(patients):
            for c in p.diagnosis_codes:
                j = self._dx_vocab.get(c)
                if j is not None:
                    rows.append(i); cols.append(j); vals.append(1.0)
            for c in p.medication_codes:
                j = self._rx_vocab.get(c)
                if j is not None:
                    rows.append(i); cols.append(off_rx + j); vals.append(1.0)
            lab_vals = []
            for c, v in p.lab_values.items():
                j = self._lab_vocab.get(c)
                if j is not None:
                    rows.append(i); cols.append(off_lab + j); vals.append(float(v))
                    rows.append(i); cols.append(off_labmask + j); vals.append(1.0)
                    lab_vals.append(float(v))
            extra[i] = [
                len(p.diagnosis_codes),
                len(p.medication_codes),
                len(p.lab_values),
                float(p.n_admissions),
                float(np.mean(lab_vals)) if lab_vals else 0.0,
                float(np.std(lab_vals)) if lab_vals else 0.0,
            ]

        width = off_labmask + n_lab
        block = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(n, max(width, 1))
        )
        out = sparse.hstack([block, sparse.csr_matrix(extra)], format="csr")
        return np.asarray(out.todense(), dtype=np.float32) if dense else out

    def _trial_block(self, trials: Sequence[TrialSpec], dense: bool = True):
        n = len(trials)
        n_cr = len(self._crit_vocab)
        rows, cols, vals = [], [], []
        n_meta = 14
        meta = np.zeros((n, n_meta), dtype=np.float32)

        op_list = ["EXISTS", "NOT_EXISTS", "GT", "GTE", "LT", "LTE", "EQ", "BETWEEN"]
        et_list = ["diagnosis", "medication", "lab", "procedure"]

        for i, t in enumerate(trials):
            inc, exc = t.inclusion, t.exclusion
            for c in t.criteria:
                if not c.is_resolved:
                    continue
                j = self._crit_vocab.get(f"{c.entity_type[:2]}:{c.entity_code}")
                if j is None:
                    continue
                # Inclusion codes occupy the first half, exclusion the second,
                # so the model can tell "required" from "disqualifying".
                col = j if c.is_inclusion else n_cr + j
                rows.append(i); cols.append(col); vals.append(float(c.severity_weight))

            resolved = [c for c in t.criteria if c.is_resolved]
            op_counts = [sum(1 for c in resolved if c.operator == o) for o in op_list[:4]]
            et_counts = [sum(1 for c in resolved if c.entity_type == e) for e in et_list]
            meta[i] = [
                len(inc), len(exc), len(resolved),
                len(t.criteria) - len(resolved),
                float(np.mean([c.severity_weight for c in inc])) if inc else 0.0,
                float(t.sample_size),
                float(len(t.conditions)),
                *op_counts,
                *et_counts,
            ][:n_meta]

        block = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(n, max(2 * n_cr, 1)),
        )
        parts = [block, sparse.csr_matrix(meta)]

        if self._tfidf is not None:
            corpus = [t.text() if t.text().strip() else "empty" for t in trials]
            parts.append(self._tfidf.transform(corpus))

        out = sparse.hstack(parts, format="csr")
        return np.asarray(out.todense(), dtype=np.float32) if dense else out

    def _shared_code_matrix(
        self, entities, kind: str
    ) -> sparse.csr_matrix:
        """Binary matrix in the shared code space (ORACLE features only)."""
        V = max(len(self._shared_vocab), 1)
        rows, cols, vals = [], [], []
        for i, e in enumerate(entities):
            if kind == "patient":
                keys = (
                    [f"di:{c}" for c in e.diagnosis_codes]
                    + [f"me:{c}" for c in e.medication_codes]
                    + [f"la:{c}" for c in e.lab_values]
                )
                weights = [1.0] * len(keys)
            else:
                crits = e.inclusion if kind == "inclusion" else e.exclusion
                crits = [c for c in crits if c.is_resolved]
                keys = [f"{c.entity_type[:2]}:{c.entity_code}" for c in crits]
                weights = [c.severity_weight for c in crits]
            for key, w in zip(keys, weights):
                j = self._shared_vocab.get(key)
                if j is not None:
                    rows.append(i); cols.append(j); vals.append(float(w))
        return sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(entities), V),
        )

    # -- assembly ------------------------------------------------------
    def build(self, dataset: Dataset) -> FeatureBundle:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.fit must be called before build")

        P_raw = self._patient_block(dataset.patients, dense=False)
        T_raw = self._trial_block(dataset.trials, dense=False)

        P_svd = self._scaler_p.transform(self._svd_p.transform(P_raw)).astype(np.float32)
        T_svd = self._scaler_t.transform(self._svd_t.transform(T_raw)).astype(np.float32)

        bundle = FeatureBundle(
            P=P_svd,          # dense compressed patient view used for pair features
            T=T_svd,
            P_svd=P_svd,
            T_svd=T_svd,
            patient_names=[f"p_svd_{i}" for i in range(P_svd.shape[1])],
            trial_names=[f"t_svd_{i}" for i in range(T_svd.shape[1])],
            regime=self.regime,
        )

        if self.regime == FeatureRegime.ORACLE:
            bundle.P_code = self._shared_code_matrix(dataset.patients, "patient")
            bundle.T_inc = self._shared_code_matrix(dataset.trials, "inclusion")
            bundle.T_exc = self._shared_code_matrix(dataset.trials, "exclusion")
        return bundle


# ---------------------------------------------------------------------------
# Pair assembly
# ---------------------------------------------------------------------------
def pair_features(
    bundle: FeatureBundle, p_idx: np.ndarray, t_idx: np.ndarray
) -> np.ndarray:
    """Assemble the design matrix for the given pairs under the active regime."""
    P = bundle.P[p_idx]
    T = bundle.T[t_idx]
    parts = [P, T]

    if bundle.regime in (FeatureRegime.HONEST, FeatureRegime.ORACLE):
        Ps, Ts = bundle.P_svd[p_idx], bundle.T_svd[t_idx]
        prod = Ps * Ts
        absdiff = np.abs(Ps - Ts)
        dot = prod.sum(axis=1, keepdims=True)
        npn = np.linalg.norm(Ps, axis=1, keepdims=True)
        ntn = np.linalg.norm(Ts, axis=1, keepdims=True)
        cos = dot / np.maximum(npn * ntn, 1e-8)
        l2 = np.linalg.norm(Ps - Ts, axis=1, keepdims=True)
        parts += [prod, absdiff, cos, l2, dot]

    if bundle.regime == FeatureRegime.ORACLE:
        parts.append(_oracle_block(bundle, p_idx, t_idx))

    return np.hstack(parts).astype(np.float32)


def _oracle_block(
    bundle: FeatureBundle, p_idx: np.ndarray, t_idx: np.ndarray
) -> np.ndarray:
    """Exact overlap statistics -- i.e. the label rule. Diagnostic use only."""
    Pc = bundle.P_code[p_idx]
    Ti = bundle.T_inc[t_idx]
    Te = bundle.T_exc[t_idx]

    inc_hits = np.asarray(Pc.multiply(Ti.sign()).sum(axis=1)).ravel()
    inc_size = np.asarray(Ti.sign().sum(axis=1)).ravel()
    exc_hits = np.asarray(Pc.multiply(Te.sign()).sum(axis=1)).ravel()
    exc_size = np.asarray(Te.sign().sum(axis=1)).ravel()

    inc_frac = inc_hits / np.maximum(inc_size, 1.0)
    p_size = np.asarray(Pc.sign().sum(axis=1)).ravel()
    union = np.maximum(p_size + inc_size - inc_hits, 1.0)
    jaccard = inc_hits / union

    return np.column_stack(
        [
            inc_frac,
            inc_hits,
            inc_size,
            (exc_hits > 0).astype(np.float32),
            exc_hits,
            exc_size,
            jaccard,
            inc_frac - (exc_hits > 0).astype(np.float32),
        ]
    ).astype(np.float32)


def iter_pair_chunks(
    bundle: FeatureBundle, pairs: PairSet, chunk_size: int = 200_000
) -> Iterator[Tuple[np.ndarray, np.ndarray, slice]]:
    """Yield (X_chunk, y_chunk, slice) so large quadrants never go fully dense."""
    n = len(pairs)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sl = slice(start, end)
        X = pair_features(bundle, pairs.p_idx[sl], pairs.t_idx[sl])
        yield X, pairs.y[sl], sl


def materialize(
    bundle: FeatureBundle, pairs: PairSet, max_gb: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Fully materialise a pair set, refusing if it would blow up memory."""
    est_gb = len(pairs) * bundle.dim * 4 / 1e9
    if est_gb > max_gb:
        raise MemoryError(
            f"Materialising {len(pairs):,} pairs x {bundle.dim} features needs "
            f"~{est_gb:.1f} GB (> {max_gb} GB). Use iter_pair_chunks, or "
            f"subsample with subsample_pairs()."
        )
    X = pair_features(bundle, pairs.p_idx, pairs.t_idx)
    return X, pairs.y.astype(np.int32)


def subsample_pairs(pairs: PairSet, max_pairs: int, seed: int = 0) -> PairSet:
    """Class-stratified subsample, used for tuning only -- never for the test set."""
    if len(pairs) <= max_pairs:
        return pairs
    rng = np.random.default_rng(seed)
    pos = np.where(pairs.y == 1)[0]
    neg = np.where(pairs.y == 0)[0]
    keep_pos = pos if len(pos) <= max_pairs // 2 else rng.choice(pos, max_pairs // 2, replace=False)
    n_neg = max_pairs - len(keep_pos)
    keep_neg = neg if len(neg) <= n_neg else rng.choice(neg, n_neg, replace=False)
    keep = np.sort(np.concatenate([keep_pos, keep_neg]))
    return PairSet(
        pairs.p_idx[keep], pairs.t_idx[keep], pairs.y[keep], name=pairs.name + "[sub]"
    )
