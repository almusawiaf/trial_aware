"""
tests/test_invariants.py
========================
These are not smoke tests. Each one guards a specific way this benchmark could
silently produce a wrong number, and each maps to a claim the paper would make.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import BenchmarkConfig
from src.data import make_synthetic_dataset, normalize_code
from src.features import FeatureBuilder, FeatureRegime, pair_features
from src.labeling import RuleLabeler, audit_leakage
from src.models.gnn import _assert_no_patient_trial_edge, build_graph
from src.models.base import TrainContext
from src.splits import make_splits


@pytest.fixture(scope="module")
def cfg():
    c = BenchmarkConfig()
    c.synthetic_n_patients = 240
    c.synthetic_n_trials = 60
    c.eval.n_bootstrap = 0
    c.n_tuning_trials = 0
    c.verbose = False
    return c


@pytest.fixture(scope="module")
def prepared(cfg):
    ds = make_synthetic_dataset(cfg.synthetic_n_patients, cfg.synthetic_n_trials, seed=0)
    labels = RuleLabeler().fit(ds.patients).build(ds)
    return ds, labels


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def test_label_is_reproducible(prepared):
    """The rule must be deterministic, or seed-to-seed variance is meaningless."""
    ds, labels = prepared
    again = RuleLabeler().fit(ds.patients).build(ds)
    assert np.array_equal(labels.y, again.y)


def test_rule_inputs_determine_the_label(prepared):
    """The premise of the whole ORACLE/HONEST separation.

    If this ever fails, the label has stopped being a closed-form function of
    M_inc/M_exc and the leakage argument in the README no longer applies.
    """
    _, labels = prepared
    audit = audit_leakage(labels)
    assert audit["oracle_score_roc_auc"] > 0.99


def test_no_degenerate_trial_columns(prepared):
    """Constant columns break per-trial AUC and inflate pooled prevalence."""
    _, labels = prepared
    col_sums = labels.y.sum(axis=0)
    assert (col_sums > 0).all()
    assert (col_sums < labels.y.shape[0]).all()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_splits_are_doubly_disjoint(cfg, prepared):
    _, labels = prepared
    s = make_splits(labels, cfg, seed=0)
    for parts in (s.patients, s.trials):
        assert not set(parts["train"]) & set(parts["val"])
        assert not set(parts["train"]) & set(parts["test"])
        assert not set(parts["val"]) & set(parts["test"])


def test_no_pair_appears_in_both_train_and_test(cfg, prepared):
    _, labels = prepared
    s = make_splits(labels, cfg, seed=0)
    train = set(zip(s.train_full.p_idx.tolist(), s.train_full.t_idx.tolist()))
    test = set(zip(s.test.p_idx.tolist(), s.test.t_idx.tolist()))
    assert not (train & test)


def test_eval_quadrants_are_not_negative_subsampled(cfg, prepared):
    """Test prevalence must match the underlying matrix, or PR-AUC is fiction."""
    _, labels = prepared
    s = make_splits(labels, cfg, seed=0)
    expected = labels.y[np.ix_(s.patients["test"], s.trials["test"])].mean()
    assert s.test.prevalence == pytest.approx(expected, abs=1e-9)
    # ...whereas training is subsampled and should differ.
    assert s.train.prevalence > s.train_full.prevalence


def test_split_is_seed_dependent(cfg, prepared):
    _, labels = prepared
    a = make_splits(labels, cfg, seed=0)
    b = make_splits(labels, cfg, seed=1)
    assert not np.array_equal(a.patients["test"], b.patients["test"])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _ctx(cfg, ds, labels, regime, seed=0):
    split = make_splits(labels, cfg, seed)
    builder = FeatureBuilder(cfg, regime=regime).fit(ds, split)
    return TrainContext(ds, labels, split, builder.build(ds), cfg, seed)


def test_honest_regime_excludes_overlap_features(cfg, prepared):
    """The core anti-leakage claim: no HONEST feature reconstructs M_inc.

    Correlating every HONEST feature against the true inclusion index must not
    turn up anything near-perfect. A |r| above 0.95 means some feature is the
    rule in disguise.
    """
    ds, labels = prepared
    ctx = _ctx(cfg, ds, labels, FeatureRegime.HONEST)
    pairs = ctx.split.test
    X = pair_features(ctx.bundle, pairs.p_idx, pairs.t_idx)
    m_inc = labels.m_inc[pairs.p_idx, pairs.t_idx]

    for j in range(X.shape[1]):
        col = X[:, j]
        if col.std() < 1e-8:
            continue
        r = abs(np.corrcoef(col, m_inc)[0, 1])
        assert r < 0.95, f"HONEST feature {j} correlates {r:.3f} with M_inc"


def test_oracle_regime_does_expose_the_rule(cfg, prepared):
    """Sanity: the ORACLE regime should be leaky, else the audit proves nothing."""
    ds, labels = prepared
    ctx = _ctx(cfg, ds, labels, FeatureRegime.ORACLE)
    pairs = ctx.split.test
    X = pair_features(ctx.bundle, pairs.p_idx, pairs.t_idx)
    m_inc = labels.m_inc[pairs.p_idx, pairs.t_idx]
    best = max(
        abs(np.corrcoef(X[:, j], m_inc)[0, 1])
        for j in range(X.shape[1])
        if X[:, j].std() > 1e-8
    )
    assert best > 0.9


def test_feature_transforms_fit_on_train_only(cfg, prepared):
    """Vocabulary must come from training entities; otherwise it is transductive."""
    ds, labels = prepared
    split = make_splits(labels, cfg, seed=0)
    builder = FeatureBuilder(cfg, regime=FeatureRegime.HONEST).fit(ds, split)

    train_codes = set()
    for i in split.patients["train"]:
        train_codes |= ds.patients[i].diagnosis_codes
    assert set(builder._dx_vocab).issubset(train_codes)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def test_graph_has_no_patient_trial_edges(cfg, prepared):
    """The single most important invariant for the GNN results."""
    ds, labels = prepared
    ctx = _ctx(cfg, ds, labels, FeatureRegime.HONEST)
    g = build_graph(ctx)
    _assert_no_patient_trial_edge(g)   # raises on violation

    p_hi, t_lo = g.n_patients, g.trial_offset
    for e in g.edges.values():
        if e.shape[1]:
            assert not (((e[0] < p_hi) & (e[1] >= t_lo)).any())


def test_graph_leakage_guard_actually_fires(cfg, prepared):
    """A guard that cannot fail is not a guard. Inject a bad edge and check."""
    ds, labels = prepared
    ctx = _ctx(cfg, ds, labels, FeatureRegime.HONEST)
    g = build_graph(ctx)
    g.edges["patient_has_code"] = np.array([[0], [g.trial_offset]], dtype=np.int64)
    with pytest.raises(AssertionError):
        _assert_no_patient_trial_edge(g)


def test_code_vocabulary_is_inductive(cfg, prepared):
    """Held-out entities may cite codes the graph has never seen. That is the
    deployment reality, and the builder must not quietly include them."""
    ds, labels = prepared
    ctx = _ctx(cfg, ds, labels, FeatureRegime.HONEST)
    g = build_graph(ctx, code_vocab_from_train_only=True)
    g_all = build_graph(ctx, code_vocab_from_train_only=False)
    assert g.n_codes <= g_all.n_codes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_pr_auc_of_random_scores_equals_prevalence():
    from src.metrics import safe_pr_auc

    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.02).astype(int)
    s = rng.random(50_000)
    assert safe_pr_auc(y, s) == pytest.approx(y.mean(), abs=0.01)


def test_pooled_auc_can_hide_a_useless_within_trial_ranking():
    """Justifies reporting grouped metrics alongside pooled ones.

    Two trials with different base rates; within each trial the scores are pure
    noise. Pooled AUC looks respectable, per-trial AUC is ~0.5.
    """
    from src.metrics import grouped_metrics, safe_roc_auc

    rng = np.random.default_rng(0)
    n = 4000
    groups = np.repeat([0, 1], n // 2)
    y = np.concatenate([
        (rng.random(n // 2) < 0.40).astype(int),
        (rng.random(n // 2) < 0.02).astype(int),
    ])
    scores = np.where(groups == 0, 1.0, 0.0) + 0.01 * rng.random(n)

    pooled = safe_roc_auc(y, scores)
    per_trial = grouped_metrics(y, scores, groups, ks=(10,))["roc_auc"]
    assert pooled > 0.65
    assert per_trial == pytest.approx(0.5, abs=0.06)


def test_cluster_bootstrap_is_wider_than_naive_pair_bootstrap():
    """Pairs sharing a trial are dependent; ignoring that understates the CI."""
    from src.metrics import cluster_bootstrap_ci

    rng = np.random.default_rng(0)
    n_trials, per_trial = 20, 200
    groups = np.repeat(np.arange(n_trials), per_trial)
    offs = rng.normal(0, 1.5, n_trials)[groups]
    y = (rng.random(groups.size) < 0.05).astype(int)
    s = offs + y * 0.5 + rng.normal(0, 1, groups.size)

    clustered = cluster_bootstrap_ci(y, s, groups, n_boot=200, seed=0)["roc_auc"]
    naive = cluster_bootstrap_ci(
        y, s, np.arange(groups.size), n_boot=200, seed=0
    )["roc_auc"]
    assert (clustered[1] - clustered[0]) > (naive[1] - naive[0])


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
def test_code_normalisation_matches_upstream_conventions():
    assert normalize_code("diagnosis", "I50.9") == "I509"
    assert normalize_code("medication", "12345.0") == "00000012345"
    assert normalize_code("lab", "50912") == "50912"
