"""
report.py
=========
Turns raw per-seed rows into the tables a paper needs, and runs the comparisons
against a reference model.

Reporting conventions used here
-------------------------------
* Every number is mean +/- standard deviation across seeds, never a single run.
  A one-seed result on a fold this imbalanced is not reproducible and should not
  be published.
* The primary table is the both-cold test quadrant. The single-cold quadrants
  appear in a separate diagnostic table, because mixing them into one table
  invites the reader to compare numbers computed on different populations.
* Comparisons against the reference model report a mean difference and a
  Wilcoxon p across seeds, with the minimum attainable p stated. With five
  seeds that floor is 0.0625, so no seed-level comparison can be significant at
  0.05 -- which is exactly why the per-split paired bootstrap in `metrics.py`
  is the primary evidence and this table is a consistency check.
* ORACLE-regime rows are printed under a loud header. They are a leakage audit,
  not a leaderboard.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import across_seed_test

log = logging.getLogger(__name__)

HEADLINE_METRICS = [
    "pooled_pr_auc",
    "pooled_roc_auc",
    "trial_pr_auc",
    "trial_roc_auc",
    "trial_recall@50",
    "trial_ndcg@10",
    "patient_roc_auc",
]


def aggregate(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Mean and std across seeds for each (model, regime) on one split."""
    sub = df[df["split"] == split].copy()
    if "error" in sub.columns:
        sub = sub[sub["error"].isna()] if sub["error"].notna().any() else sub
    if sub.empty:
        return pd.DataFrame()

    metrics = [m for m in HEADLINE_METRICS if m in sub.columns]
    grouped = sub.groupby(["regime", "model"])[metrics + ["fit_seconds"]]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def format_table(
    agg: pd.DataFrame,
    metrics: Sequence[str] = ("pooled_pr_auc", "pooled_roc_auc", "trial_pr_auc", "trial_roc_auc", "trial_recall@50"),
    sort_by: str = "pooled_pr_auc",
    fmt: str = "markdown",
) -> str:
    """Render a mean +/- std table."""
    if agg.empty:
        return "_(no results)_"

    rows = []
    for _, r in agg.iterrows():
        row = {"regime": r["regime"], "model": r["model"]}
        for m in metrics:
            mu, sd = r.get(f"{m}_mean", np.nan), r.get(f"{m}_std", np.nan)
            row[m] = "n/a" if not np.isfinite(mu) else (
                f"{mu:.4f} ± {sd:.4f}" if np.isfinite(sd) else f"{mu:.4f}"
            )
        row["n_seeds"] = int(r.get(f"{metrics[0]}_count", 0))
        row["_sort"] = r.get(f"{sort_by}_mean", -np.inf)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(["regime", "_sort"], ascending=[True, False])
    table = table.drop(columns=["_sort"])

    if fmt == "latex":
        return table.to_latex(index=False, escape=True)
    try:
        return table.to_markdown(index=False)
    except Exception:
        return table.to_string(index=False)


def compare_to_reference(
    df: pd.DataFrame,
    reference_model: str,
    metric: str = "pooled_pr_auc",
    split: str = "test",
    regime: Optional[str] = None,
) -> pd.DataFrame:
    """Per-model seed-level comparison against a reference."""
    sub = df[df["split"] == split]
    if regime is not None:
        sub = sub[sub["regime"] == regime]
    if metric not in sub.columns or sub.empty:
        return pd.DataFrame()

    ref = sub[sub["model"] == reference_model].set_index("seed")[metric]
    if ref.empty:
        log.warning("Reference model '%s' not present in results.", reference_model)
        return pd.DataFrame()

    rows = []
    for model, grp in sub.groupby("model"):
        if model == reference_model:
            continue
        g = grp.set_index("seed")[metric]
        common = ref.index.intersection(g.index)
        if len(common) < 2:
            continue
        test = across_seed_test(g.loc[common].values, ref.loc[common].values)
        rows.append(
            {
                "model": model,
                "reference": reference_model,
                "metric": metric,
                "model_mean": float(g.loc[common].mean()),
                "reference_mean": float(ref.loc[common].mean()),
                "mean_diff": test["mean_diff"],
                "wilcoxon_p": test.get("p_value", np.nan),
                "n_seeds": int(len(common)),
                "min_attainable_p": test.get("min_attainable_p", np.nan),
            }
        )
    if not rows:
        log.info(
            "No seed-level comparison possible against '%s' (need >= 2 shared seeds).",
            reference_model,
        )
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def sanity_checks(df: pd.DataFrame) -> List[str]:
    """Flag result patterns that usually indicate a methodological problem."""
    issues: List[str] = []
    test = df[df["split"] == "test"]
    if test.empty:
        return ["No test-split rows found."]

    honest = test[test["regime"] != "oracle"]
    if not honest.empty and "pooled_roc_auc" in honest.columns:
        too_good = honest[honest["pooled_roc_auc"] > 0.98]
        for _, r in too_good.iterrows():
            issues.append(
                f"{r['model']} (regime={r['regime']}, seed={r['seed']}) reaches "
                f"ROC-AUC {r['pooled_roc_auc']:.4f} on a non-oracle regime. "
                "Check for a feature that reconstructs the label rule."
            )

    if {"pooled_roc_auc", "trial_roc_auc"} <= set(test.columns):
        for _, r in test.iterrows():
            p, t = r.get("pooled_roc_auc"), r.get("trial_roc_auc")
            if np.isfinite(p) and np.isfinite(t) and (p - t) > 0.15:
                issues.append(
                    f"{r['model']} (seed={r['seed']}): pooled ROC-AUC {p:.3f} far "
                    f"exceeds per-trial {t:.3f}. The model is largely ranking "
                    "trials by permissiveness, not patients within a trial."
                )

    for model, grp in honest.groupby("model"):
        if "pooled_pr_auc" in grp.columns and grp["pooled_pr_auc"].notna().sum() >= 3:
            sd = grp["pooled_pr_auc"].std()
            mu = grp["pooled_pr_auc"].mean()
            if np.isfinite(sd) and np.isfinite(mu) and mu > 0 and sd > 0.5 * mu:
                issues.append(
                    f"{model}: PR-AUC std ({sd:.4f}) exceeds half its mean ({mu:.4f}) "
                    "across seeds. Too few positives for a stable estimate."
                )
    return issues


def write_report(
    df: pd.DataFrame,
    output_dir: str,
    reference_model: str = "GraphSAGE",
    label_audit: Optional[Dict[str, float]] = None,
    dataset_note: str = "",
) -> str:
    """Write results.md plus the aggregated CSVs. Returns the markdown path."""
    os.makedirs(output_dir, exist_ok=True)

    agg_test = aggregate(df, "test")
    agg_val = aggregate(df, "val")
    agg_test.to_csv(os.path.join(output_dir, "aggregate_test.csv"), index=False)

    parts: List[str] = ["# Patient–Trial Matching: Baseline Benchmark\n"]
    if dataset_note:
        parts += [f"> {dataset_note}\n"]

    if label_audit:
        parts += [
            "## Label audit (read this first)\n",
            "The ground-truth label is a deterministic rule over the same inputs the "
            "models see. Scoring pairs directly by the rule's own statistics gives:\n",
            f"- ROC-AUC **{label_audit['oracle_score_roc_auc']:.4f}**",
            f"- PR-AUC **{label_audit['oracle_score_pr_auc']:.4f}**",
            f"- prevalence **{label_audit['prevalence']:.4%}**\n",
            "Any model given overlap features can reach that ceiling. It measures the "
            "labeller, not clinical matching ability. Results below should be read "
            "as *rule-recovery under generalisation*, not as eligibility prediction.\n",
        ]

    honest = agg_test[agg_test["regime"] != "oracle"] if not agg_test.empty else agg_test
    oracle = agg_test[agg_test["regime"] == "oracle"] if not agg_test.empty else agg_test

    parts += [
        "## Primary results — both-cold test quadrant (unseen patients × unseen trials)\n",
        "Mean ± std across seeds.\n",
        format_table(honest),
        "",
    ]

    if not oracle.empty:
        parts += [
            "## Leakage-audit regime (ORACLE) — NOT a performance result\n",
            "These runs receive exact criterion-overlap features, i.e. the label rule "
            "itself. Near-perfect scores here confirm the audit; they are reported so "
            "the gap to the honest regime is explicit.\n",
            format_table(oracle),
            "",
        ]

    diag = df[df["split"].isin(["cold_patient_only", "cold_trial_only"])]
    if not diag.empty:
        rows = []
        for split_name in ("cold_patient_only", "cold_trial_only"):
            a = aggregate(diag, split_name)
            if not a.empty:
                a = a.assign(split=split_name)
                rows.append(a)
        if rows:
            parts += [
                "## Generalisation diagnostics\n",
                "Which axis a model fails on: unseen patients only vs unseen trials only.\n",
            ]
            for a in rows:
                parts += [f"### {a['split'].iloc[0]}\n", format_table(a.drop(columns=['split'])), ""]

    cmp_df = compare_to_reference(df, reference_model)
    if not cmp_df.empty:
        parts += [
            f"## Comparison against `{reference_model}` (seed-level Wilcoxon)\n",
            "With five seeds the smallest attainable two-sided p is 0.0625, so treat "
            "this as a consistency check and use the per-split paired bootstrap "
            "(`metrics.paired_bootstrap_test`) for the primary evidence.\n",
            cmp_df.to_markdown(index=False),
            "",
        ]
        cmp_df.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)

    issues = sanity_checks(df)
    parts += ["## Automated sanity checks\n"]
    parts += ["\n".join(f"- {i}" for i in issues) if issues else "- No issues flagged."]
    parts += [""]

    if not agg_val.empty:
        parts += ["## Validation quadrant (model selection only)\n", format_table(agg_val), ""]

    path = os.path.join(output_dir, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    log.info("Report written to %s", path)
    return path
