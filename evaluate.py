# evaluate.py
import json
import logging
import os
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from config import Config
from trial_graph import PatientClinicalState, TrialStore, compute_matching_indices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_evaluation_matrices(cfg: Config):
    """
    Build ground truth and prediction matrices for evaluation.
    Returns:
        y_true: (num_patients, num_trials) binary matrix of true eligibility
        scores_baseline: (num_patients, num_trials) similarity scores from Stage A
        scores_full: (num_patients, num_trials) similarity scores from Stage B
    """
    
    # 1. Load REAL clinical trials (NOT mock data!)
    trial_json_path = "structured_clinical_trials.json"
    if not os.path.exists(trial_json_path):
        logging.error(f"Trial JSON not found at {trial_json_path}")
        logging.info("Falling back to mock trials for evaluation...")
        from mock_data import generate_mock_trials
        real_trials_data = generate_mock_trials()
    else:
        with open(trial_json_path, "r") as f:
            real_trials_data = json.load(f)
        logging.info(f"Loaded {len(real_trials_data)} trials from {trial_json_path}")
    
    trial_store = TrialStore.from_records(real_trials_data)

    # 2. Load patient data tables
    diag_path = os.path.join(cfg.OUTPUT_DIR, "diagnoses_clean.parquet")
    rx_path = os.path.join(cfg.OUTPUT_DIR, "prescriptions_clean.parquet")
    labs_path = os.path.join(cfg.OUTPUT_DIR, "labs_clean.parquet")
    
    if not all(os.path.exists(p) for p in [diag_path, rx_path, labs_path]):
        logging.error(f"Missing patient data files in {cfg.OUTPUT_DIR}")
        return None, None, None
    
    diag_df = pd.read_parquet(diag_path)
    rx_df = pd.read_parquet(rx_path)
    labs_df = pd.read_parquet(labs_path)

    # 3. Get sorted subject IDs and build patient clinical states
    subject_ids = sorted(diag_df['SUBJECT_ID'].unique(), key=int)
    logging.info(f"Building clinical states for {len(subject_ids)} patients...")
    
    patient_states = {
        sid: PatientClinicalState.build_from_tables(sid, diag_df, rx_df, labs_df)
        for sid in subject_ids
    }

    # 4. Load patient and trial embeddings
    baseline_path = cfg.PATIENT_EMBED_PATH.replace(".pt", "_baseline.pt")
    
    if not os.path.exists(baseline_path):
        logging.error(f"Baseline embeddings not found at {baseline_path}")
        return None, None, None
    
    if not os.path.exists(cfg.PATIENT_EMBED_PATH):
        logging.error(f"Patient embeddings not found at {cfg.PATIENT_EMBED_PATH}")
        return None, None, None
    
    if not os.path.exists(cfg.TRIAL_EMBED_PATH):
        logging.error(f"Trial embeddings not found at {cfg.TRIAL_EMBED_PATH}")
        return None, None, None
    
    h_baseline = torch.load(baseline_path, map_location='cpu')
    h_full = torch.load(cfg.PATIENT_EMBED_PATH, map_location='cpu')
    trial_embeds = torch.load(cfg.TRIAL_EMBED_PATH, map_location='cpu')
    
    # 5. Debug: Print trial embedding information
    for tid, (z_inc, z_exc) in trial_embeds.items():
        logging.info(f"Trial {tid}:")
        logging.info(f"  z_inc norm: {z_inc.norm().item():.4f}")
        logging.info(f"  z_exc norm: {z_exc.norm().item():.4f}")
        logging.info(f"  z_inc-z_exc cosine sim: {F.cosine_similarity(z_inc, z_exc, dim=0).item():.4f}")

    # 6. Filter to trials that have embeddings
    trial_ids = [tid for tid in trial_store.trials.keys() if tid in trial_embeds]
    
    if not trial_ids:
        logging.error("No matching trial IDs found between trial store and embeddings")
        return None, None, None
    
    logging.info(f"Evaluating {len(subject_ids)} patients against {len(trial_ids)} clinical trials...")

    # 7. Build ground truth eligibility matrix (y_true)
    num_patients = len(subject_ids)
    num_trials = len(trial_ids)
    
    y_true = np.zeros((num_patients, num_trials))
    m_inc_matrix = np.zeros((num_patients, num_trials))
    m_exc_matrix = np.zeros((num_patients, num_trials))
    
    inc_threshold = getattr(cfg, 'HARD_NEG_INC_THRESHOLD', 0.3)
    exc_threshold = getattr(cfg, 'HARD_NEG_EXC_THRESHOLD', 0.3)
    
    logging.info(f"Using thresholds: M_inc >= {inc_threshold}, M_exc < {exc_threshold}")
    
    for p_idx, sid in enumerate(subject_ids):
        state = patient_states[sid]
        for t_idx, tid in enumerate(trial_ids):
            trial = trial_store[tid]
            m_inc, m_exc = compute_matching_indices(state, trial)
            
            m_inc_matrix[p_idx, t_idx] = m_inc
            m_exc_matrix[p_idx, t_idx] = m_exc
            
            # Ground truth: eligible if inclusion score is high AND exclusion score is low
            y_true[p_idx, t_idx] = 1.0 if (m_inc >= inc_threshold and m_exc < exc_threshold) else 0.0
    
    # Log ground truth statistics
    total_eligible = y_true.sum()
    logging.info(f"Total eligible (patient, trial) pairs: {total_eligible} ({total_eligible/(num_patients*num_trials)*100:.2f}%)")
    logging.info(f"Trials with at least one eligible patient: {(y_true.sum(axis=0) > 0).sum()}")

    # 8. Calculate similarity scores for both models
    scores_baseline = np.zeros((num_patients, num_trials))
    scores_full = np.zeros((num_patients, num_trials))
    
    eta = getattr(cfg, 'ETA_EXCLUSION_PENALTY', 1.0)
    logging.info(f"Using exclusion penalty eta = {eta}")
    
    for t_idx, tid in enumerate(trial_ids):
        z_inc, z_exc = trial_embeds[tid]
        
        # Ensure tensors are on CPU and properly shaped
        z_inc = z_inc.squeeze().cpu()
        z_exc = z_exc.squeeze().cpu()
        
        for p_idx in range(num_patients):
            # Stage A (Baseline) score
            z_patient_base = h_baseline[p_idx].squeeze().cpu()
            inc_sim_base = F.cosine_similarity(z_patient_base.unsqueeze(0), z_inc.unsqueeze(0)).item()
            exc_sim_base = F.cosine_similarity(z_patient_base.unsqueeze(0), z_exc.unsqueeze(0)).item()
            scores_baseline[p_idx, t_idx] = inc_sim_base - eta * exc_sim_base
            
            # Stage B (Full Model) score
            z_patient_full = h_full[p_idx].squeeze().cpu()
            inc_sim_full = F.cosine_similarity(z_patient_full.unsqueeze(0), z_inc.unsqueeze(0)).item()
            exc_sim_full = F.cosine_similarity(z_patient_full.unsqueeze(0), z_exc.unsqueeze(0)).item()
            scores_full[p_idx, t_idx] = inc_sim_full - eta * exc_sim_full

    # 9. Debug: Print score statistics
    logging.info(f"Scores baseline - mean: {scores_baseline.mean():.4f}, std: {scores_baseline.std():.4f}")
    logging.info(f"Scores full - mean: {scores_full.mean():.4f}, std: {scores_full.std():.4f}")
    logging.info(f"y_true - mean: {y_true.mean():.4f} ({y_true.sum()} positive pairs)")

    return y_true, scores_baseline, scores_full


def evaluate_retrieval():
    """Main evaluation function."""
    cfg = Config()
    
    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Build evaluation matrices
    result = build_evaluation_matrices(cfg)
    if result[0] is None:
        logging.error("Evaluation failed - missing embeddings or data")
        return
    
    y_true, scores_base, scores_full = result
    
    # Calculate metrics
    try:
        auc_base = roc_auc_score(y_true.ravel(), scores_base.ravel())
        auc_full = roc_auc_score(y_true.ravel(), scores_full.ravel())
        
        pr_base = average_precision_score(y_true.ravel(), scores_base.ravel())
        pr_full = average_precision_score(y_true.ravel(), scores_full.ravel())
    except Exception as e:
        logging.error(f"Error computing metrics: {e}")
        return

    # Log results
    logging.info("=" * 60)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"[Results] Stage A (Baseline) ROC-AUC: {auc_base:.4f}")
    logging.info(f"[Results] Stage B (Full Model) ROC-AUC: {auc_full:.4f}")
    logging.info(f"[Results] Stage A (Baseline) PR-AUC: {pr_base:.4f}")
    logging.info(f"[Results] Stage B (Full Model) PR-AUC: {pr_full:.4f}")
    
    # Calculate improvement
    auc_improvement = ((auc_full - auc_base) / max(auc_base, 0.001)) * 100
    pr_improvement = ((pr_full - pr_base) / max(pr_base, 0.001)) * 100
    
    logging.info(f"ROC-AUC Improvement: {auc_improvement:+.2f}%")
    logging.info(f"PR-AUC Improvement: {pr_improvement:+.2f}%")
    logging.info("=" * 60)
    
    # Save results
    results = {
        'stage_a_roc_auc': float(auc_base),
        'stage_b_roc_auc': float(auc_full),
        'stage_a_pr_auc': float(pr_base),
        'stage_b_pr_auc': float(pr_full),
        'roc_improvement_pct': float(auc_improvement),
        'pr_improvement_pct': float(pr_improvement),
        'num_patients': y_true.shape[0],
        'num_trials': y_true.shape[1],
        'num_positive_pairs': int(y_true.sum()),
    }
    
    results_path = os.path.join(cfg.OUTPUT_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {results_path}")
    return results


if __name__ == "__main__":
    evaluate_retrieval()