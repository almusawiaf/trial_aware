"""
retrieval_metrics.py

Precision@k, NDCG@k, and Screen Failure Rate Reduction (Delta-SFR) for the
patient-trial ranking task: given a trial, rank all patients by score and
check how many of the top-k are genuinely eligible.

IMPORTANT: all three metrics here reuse the SAME y_true eligibility matrix
that ROC-AUC/PR-AUC already use (built from inc_threshold/exc_threshold in
build_evaluation_matrices), rather than introducing a second, separate
gamma_elig threshold as the paper's Eq. 34/36 originally specified. This
fixes an inconsistency flagged during review: the paper previously defined
two different eligibility thresholds (theta_inc=0.15/theta_exc<0.8 for
ROC/PR vs. gamma_elig for P@k/Delta-SFR) applied to the same underlying
M_inc/M_exc quantities, which a reviewer could reasonably read as picking
whichever threshold flattered a given metric. Using one eligibility
definition everywhere removes that ambiguity; update the paper's notation
table (gamma_elig) to point back to inc_threshold/exc_threshold instead of
listing it as an independent constant.

All three are computed PER TRIAL (ranking patients for that one trial),
then averaged across trials -- this matches how the pipeline would
actually be used (a coordinator picks a trial, wants the top candidates
for it), not averaged over patients.

Trials with ZERO eligible patients in the population are excluded from the
average (standard IR practice for queries with no relevant documents --
P@k for such a trial is trivially 0 regardless of model quality, and
including it would penalize the model for a property of the trial's
eligibility criteria being extremely narrow, not for ranking failure).
The number of excluded trials is returned so this is never silently
hidden from the reported numbers.
"""
import numpy as np
from sklearn.metrics import ndcg_score


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, int]:
    """
    Mean Precision@k across trials with at least one eligible patient.

    Args:
        y_true: (num_patients, num_trials) binary eligibility matrix.
        scores: (num_patients, num_trials) model scores (higher = more likely eligible).
        k: number of top-ranked patients to check per trial.

    Returns:
        (mean_precision_at_k, num_trials_used)
    """
    num_patients, num_trials = y_true.shape
    k_eff = min(k, num_patients)
    precisions = []
    for j in range(num_trials):
        if y_true[:, j].sum() == 0:
            continue
        top_k_idx = np.argsort(-scores[:, j])[:k_eff]
        precisions.append(y_true[top_k_idx, j].mean())
    if not precisions:
        return float('nan'), 0
    return float(np.mean(precisions)), len(precisions)


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, int]:
    """
    Mean NDCG@k across trials with at least one eligible patient.
    Uses binary relevance (y_true). A natural future extension is graded
    relevance using the raw M_inc score in place of the binary label, so
    "very clearly eligible" outranks "borderline eligible" in the ideal
    ranking too -- not implemented here to keep this metric directly
    comparable to the binary ROC-AUC/PR-AUC numbers already reported.

    Returns:
        (mean_ndcg_at_k, num_trials_used)
    """
    num_patients, num_trials = y_true.shape
    k_eff = min(k, num_patients)
    ndcgs = []
    for j in range(num_trials):
        true_relevance = y_true[:, j]
        if true_relevance.sum() == 0:
            continue
        ndcgs.append(
            ndcg_score(true_relevance.reshape(1, -1), scores[:, j].reshape(1, -1), k=k_eff)
        )
    if not ndcgs:
        return float('nan'), 0
    return float(np.mean(ndcgs)), len(ndcgs)


def delta_sfr_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, int]:
    """
    Mean Screen Failure Rate Reduction at k, across trials with at least
    one eligible patient.

    SFR_base(trial) = fraction of the WHOLE population who are NOT
    eligible for that trial (i.e. the screen-failure rate you'd expect if
    you screened patients in no particular order / at random).
    SFR_topk(trial)  = fraction of the top-k RANKED patients who are NOT
    eligible.
    Delta-SFR(trial) = SFR_base(trial) - SFR_topk(trial)
                     = P@k(trial) - population_eligibility_rate(trial)

    A positive Delta-SFR means the ranking concentrates real screen
    failures out of the top-k relative to screening in no particular
    order -- directly interpretable as "how much wasted coordinator time
    does this ranking save."

    Returns:
        (mean_delta_sfr_at_k, num_trials_used)
    """
    num_patients, num_trials = y_true.shape
    k_eff = min(k, num_patients)
    deltas = []
    for j in range(num_trials):
        pop_eligibility = y_true[:, j].mean()
        if y_true[:, j].sum() == 0:
            continue
        top_k_idx = np.argsort(-scores[:, j])[:k_eff]
        topk_precision = y_true[top_k_idx, j].mean()
        deltas.append(topk_precision - pop_eligibility)
    if not deltas:
        return float('nan'), 0
    return float(np.mean(deltas)), len(deltas)


def compute_all_retrieval_metrics(y_true: np.ndarray, scores: np.ndarray, k_values=(10, 20, 50)) -> dict:
    """Convenience wrapper: computes P@k, NDCG@k, Delta-SFR@k for every k in k_values."""
    out = {}
    for k in k_values:
        p, n_p = precision_at_k(y_true, scores, k)
        n, n_n = ndcg_at_k(y_true, scores, k)
        d, n_d = delta_sfr_at_k(y_true, scores, k)
        out[f'precision_at_{k}'] = p
        out[f'ndcg_at_{k}'] = n
        out[f'delta_sfr_at_{k}'] = d
        # sanity: these should always match (same skip condition in all three)
        out[f'n_trials_used_at_{k}'] = n_p
    return out