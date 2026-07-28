# """
# evaluate.py
# -----------
# Evaluation script for Trial-Aware Patient Representation Learning.
# Compares Stage A (Trial-Agnostic GCL) vs. Stage B (Trial-Aware Alignment)
# across ranked patient retrieval metrics:
# - PR-AUC & AUROC
# - Precision@K & Recall@K (K = 5, 10)
# - Exclusion Safety Rate (% of excluded patients wrongly in Top-K)
# """

# import os
# import logging
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import numpy as np
# from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

# from config import Config
# from trial_graph import PatientClinicalState, TrialStore, compute_matching_indices
# from mock_data import generate_mock_trials

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# def load_data(cfg: Config):
#     """Loads saved patient embeddings and trial embeddings from disk."""
#     logging.info("Loading baseline and trial-aware embeddings...")
    
#     # Path saved at end of Stage A
#     baseline_path = cfg.PATIENT_EMBED_PATH.replace(".pt", "_baseline.pt")
#     if not os.path.exists(baseline_path):
#         raise FileNotFoundError(f"Missing baseline file: {baseline_path}")
        
#     h_stage_a = torch.load(baseline_path, map_location="cpu", weights_only=False)
    
#     # Path saved at end of Stage B
#     if not os.path.exists(cfg.PATIENT_EMBED_PATH):
#         raise FileNotFoundError(f"Missing trial-aware file: {cfg.PATIENT_EMBED_PATH}")
        
#     h_stage_b = torch.load(cfg.PATIENT_EMBED_PATH, map_location="cpu", weights_only=False)
    
#     # Saved Trial embeddings dict {trial_id: (z_inc, z_exc)}
#     if not os.path.exists(cfg.TRIAL_EMBED_PATH):
#         raise FileNotFoundError(f"Missing trial embeddings file: {cfg.TRIAL_EMBED_PATH}")
        
#     trial_data = torch.load(cfg.TRIAL_EMBED_PATH, map_location="cpu", weights_only=False)
    
#     return h_stage_a, h_stage_b, trial_data

# def build_evaluation_matrices(cfg: Config):
#     """Builds ground-truth criteria matching matrices dynamically from clinical states."""
#     diag_path = os.path.join(cfg.OUTPUT_DIR, "diagnoses_clean.parquet")
#     rx_path = os.path.join(cfg.OUTPUT_DIR, "prescriptions_clean.parquet")
#     labs_path = os.path.join(cfg.OUTPUT_DIR, "labs_clean.parquet")
    
#     diag_df = pd.read_parquet(diag_path)
#     rx_df = pd.read_parquet(rx_path)
#     labs_df = pd.read_parquet(labs_path)
    
#     subjects = sorted(diag_df['SUBJECT_ID'].unique(), key=int)
#     patient_states = {
#         int(sid): PatientClinicalState.build_from_tables(int(sid), diag_df, rx_df, labs_df)
#         for sid in subjects
#     }
    
#     trial_store = TrialStore.from_records(generate_mock_trials())
#     trial_ids = list(trial_store.trials.keys())
    
#     num_patients = len(subjects)
#     num_trials = len(trial_ids)
    
#     M_inc = torch.zeros((num_patients, num_trials), dtype=torch.float32)
#     M_exc = torch.zeros((num_patients, num_trials), dtype=torch.float32)
    
#     for p_idx, sid in enumerate(subjects):
#         state = patient_states[int(sid)]
#         for t_idx, tid in enumerate(trial_ids):
#             trial = trial_store[tid]
#             inc_score, exc_score = compute_matching_indices(state, trial)
#             M_inc[p_idx, t_idx] = inc_score
#             M_exc[p_idx, t_idx] = exc_score
            
#     return M_inc, M_exc, trial_ids

# def evaluate_retrieval(scores: torch.Tensor, y_true: torch.Tensor, is_excluded: torch.Tensor, top_k_list=[3, 5, 10]):
#     """Computes ranking and retrieval metrics for a single trial."""
#     scores_np = scores.numpy()
#     y_true_np = y_true.numpy()
#     is_excluded_np = is_excluded.numpy()
    
#     if len(np.unique(y_true_np)) < 2:
#         return None
        
#     try:
#         auroc = roc_auc_score(y_true_np, scores_np)
#         precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, scores_np)
#         pr_auc = auc(recall_curve, precision_curve)
#     except Exception:
#         return None

#     ranked_indices = np.argsort(-scores_np)
    
#     metrics = {
#         "AUROC": auroc,
#         "PR-AUC": pr_auc
#     }
    
#     total_positives = np.sum(y_true_np == 1)
    
#     for k in top_k_list:
#         effective_k = min(k, len(ranked_indices))
#         top_k_idx = ranked_indices[:effective_k]
#         top_k_true = y_true_np[top_k_idx]
#         top_k_excluded = is_excluded_np[top_k_idx]
        
#         precision_at_k = np.mean(top_k_true) if effective_k > 0 else 0.0
#         recall_at_k = (np.sum(top_k_true) / max(total_positives, 1)) if total_positives > 0 else 0.0
#         exclusion_rate_at_k = np.mean(top_k_excluded) if effective_k > 0 else 0.0
        
#         metrics[f"P@{k}"] = precision_at_k
#         metrics[f"R@{k}"] = recall_at_k
#         metrics[f"ExclusionRate@{k}"] = exclusion_rate_at_k
        
#     return metrics

# def main():
#     cfg = Config()
    
#     try:
#         h_stage_a, h_stage_b, trial_data = load_data(cfg)
#     except FileNotFoundError as e:
#         logging.error(f"{e}")
#         return

#     logging.info("Constructing trial evaluation ground truth...")
#     M_inc_mat, M_exc_mat, trial_ids = build_evaluation_matrices(cfg)
    
#     h_stage_a = F.normalize(h_stage_a, dim=-1)
#     h_stage_b = F.normalize(h_stage_b, dim=-1)
    
#     results_a = []
#     results_b = []
    
#     for t_idx, tid in enumerate(trial_ids):
#         if tid not in trial_data:
#             continue
            
#         z_inc, z_exc = trial_data[tid]
        
#         # Trial query vector: Inclusion representation minus Exclusion penalty
#         t_query = F.normalize(z_inc - cfg.ETA_EXCLUSION_PENALTY * z_exc, dim=-1)
        
#         m_inc = M_inc_mat[:, t_idx]
#         m_exc = M_exc_mat[:, t_idx]
        
#         # Strict Ground Truth: Inclusion >= 0.5 AND Exclusion == 0.0
#         y_true = ((m_inc >= 0.5) & (m_exc == 0.0)).long()
#         is_excluded = (m_exc > 0.0).long()
        
#         if y_true.sum() == 0:
#             continue
            
#         # Cosine Similarity Scores across all patients
#         scores_a = torch.matmul(h_stage_a, t_query.unsqueeze(-1)).squeeze(-1)
#         scores_b = torch.matmul(h_stage_b, t_query.unsqueeze(-1)).squeeze(-1)
        
#         res_a = evaluate_retrieval(scores_a, y_true, is_excluded)
#         res_b = evaluate_retrieval(scores_b, y_true, is_excluded)
        
#         if res_a and res_b:
#             results_a.append(res_a)
#             results_b.append(res_b)

#     if not results_a:
#         logging.warning("No valid trial evaluations produced.")
#         return

#     avg_a = {k: np.mean([r[k] for r in results_a]) for k in results_a[0]}
#     avg_b = {k: np.mean([r[k] for r in results_b]) for k in results_b[0]}

#     print("\n" + "="*80)
#     print("      EXPERIMENTAL RESULTS: STAGE A VS. STAGE B PATIENT RETRIEVAL")
#     print("="*80)
#     print(f"{'Metric':<20} | {'Stage A (Agnostic GCL)':<22} | {'Stage B (Trial-Aware)':<22} | {'Relative Diff':<12}")
#     print("-" * 80)
    
#     for metric in ["PR-AUC", "AUROC", "P@3", "P@5", "P@10", "ExclusionRate@3", "ExclusionRate@5"]:
#         if metric not in avg_a:
#             continue
#         val_a = avg_a[metric]
#         val_b = avg_b[metric]
        
#         if "ExclusionRate" in metric:
#             diff = ((val_b - val_a) / (val_a + 1e-8)) * 100.0
#             print(f"{metric:<20} | {val_a*100.0:20.2f}% | {val_b*100.0:20.2f}% | {diff:+11.1f}% (Lower = Better)")
#         else:
#             diff = ((val_b - val_a) / (val_a + 1e-8)) * 100.0
#             print(f"{metric:<20} | {val_a:22.4f} | {val_b:22.4f} | {diff:+11.1f}%")
            
#     print("="*80 + "\n")

# if __name__ == "__main__":
#     main()

# evaluate.py
import json
import logging
import os
import numpy as np
import torch
import pandas as pd

from config import Config
from trial_graph import PatientClinicalState, TrialStore, compute_matching_indices
from alignment import similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_evaluation_matrices(cfg: Config):
    # Load real clinical trials evaluated in Stage B
    with open("structured_clinical_trials.json", "r") as f:
        real_trials_data = json.load(f)
    trial_store = TrialStore.from_records(real_trials_data)

    # Load patient parquet tables
    diag_df = pd.read_parquet(os.path.join(cfg.OUTPUT_DIR, "diagnoses_clean.parquet"))
    rx_df = pd.read_parquet(os.path.join(cfg.OUTPUT_DIR, "prescriptions_clean.parquet"))
    labs_df = pd.read_parquet(os.path.join(cfg.OUTPUT_DIR, "labs_clean.parquet"))

    subject_ids = sorted(diag_df['SUBJECT_ID'].unique(), key=int)
    
    # Calls the updated build_from_tables method from trial_graph.py
    patient_states = {
        sid: PatientClinicalState.build_from_tables(sid, diag_df, rx_df, labs_df)
        for sid in subject_ids
    }

    # Load baseline and trial-aware patient embeddings
    baseline_path = cfg.PATIENT_EMBED_PATH.replace(".pt", "_baseline.pt")
    h_baseline = torch.load(baseline_path, map_location='cpu')
    h_full = torch.load(cfg.PATIENT_EMBED_PATH, map_location='cpu')
    trial_embeds = torch.load(cfg.TRIAL_EMBED_PATH, map_location='cpu')
    
    for tid, (z_inc, z_exc) in trial_embeds.items():
        print(f"Trial {tid}:")
        print(f"  z_inc norm: {z_inc.norm().item():.4f}")
        print(f"  z_exc norm: {z_exc.norm().item():.4f}")
        print(f"  Cosine similarity: {F.cosine_similarity(z_inc, z_exc, dim=0).item():.4f}")

    trial_ids = [tid for tid in trial_store.trials.keys() if tid in trial_embeds]

    logging.info(f"Evaluating {len(subject_ids)} patients against {len(trial_ids)} real clinical trials...")

    # Calculate ground truth eligibility matrices
    y_true = np.zeros((len(subject_ids), len(trial_ids)))
    for p_idx, sid in enumerate(subject_ids):
        state = patient_states[sid]
        for t_idx, tid in enumerate(trial_ids):
            trial = trial_store[tid]
            m_inc, m_exc = compute_matching_indices(state, trial)
            y_true[p_idx, t_idx] = 1.0 if (m_inc >= cfg.HARD_NEG_INC_THRESHOLD and m_exc < cfg.HARD_NEG_EXC_THRESHOLD) else 0.0

    # Calculate predicted similarities
    scores_baseline = np.zeros_like(y_true)
    scores_full = np.zeros_like(y_true)

    for t_idx, tid in enumerate(trial_ids):
        z_inc, z_exc = trial_embeds[tid]
        for p_idx in range(len(subject_ids)):
            scores_baseline[p_idx, t_idx] = similarity(h_baseline[p_idx], z_inc, z_exc, eta=cfg.ETA_EXCLUSION_PENALTY).item()
            scores_full[p_idx, t_idx] = similarity(h_full[p_idx], z_inc, z_exc, eta=cfg.ETA_EXCLUSION_PENALTY).item()

    return y_true, scores_baseline, scores_full


def evaluate_retrieval():
    cfg = Config()
    y_true, scores_base, scores_full = build_evaluation_matrices(cfg)

    from sklearn.metrics import precision_score, roc_auc_score

    auc_base = roc_auc_score(y_true.ravel(), scores_base.ravel())
    auc_full = roc_auc_score(y_true.ravel(), scores_full.ravel())

    logging.info(f"[Results] Stage A (Baseline) ROC-AUC: {auc_base:.4f}")
    logging.info(f"[Results] Stage B (Full Model) ROC-AUC: {auc_full:.4f}")


if __name__ == "__main__":
    evaluate_retrieval()