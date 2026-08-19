# evaluate.py
import json
import logging
import os
import sys
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# This file lives in models/claude_active/evaluate/, but config.py and
# trial_graph.py live one directory up in models/claude_active/. No local
# copies of those two exist in this folder, so append (not insert(0, ...))
# is safe here -- but append is also just the safer default in general,
# since it never lets a parent-directory file shadow a same-named file
# that might later be added locally.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from trial_graph import PatientClinicalState, TrialStore, compute_matching_indices
from trial_embedding import CriterionEncoder, TrialEncoder, encode_all_trials
# build_naive_baseline_trial_embeds is defined in train.py rather than
# trial_embedding.py -- importing it here is safe because train.py's
# top-level code is only imports/def statements; its actual work happens
# inside main(), guarded by `if __name__ == "__main__":`, which does not
# run on import.
from train import build_naive_baseline_trial_embeds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_evaluation_matrices(cfg: Config):
    """
    Build ground truth and prediction matrices for evaluation.
    Returns:
        y_true: (num_patients, num_trials) binary matrix of true eligibility
        scores_baseline: (num_patients, num_trials) similarity scores from Stage A
        scores_full: (num_patients, num_trials) similarity scores from Stage B
    """
    
    # ============================================================
    # 1. Load HELD-OUT trials for evaluation (never seen during Stage B
    #    training / weak-supervision derivation). Loading TRAIN_TRIALS_PATH
    #    here was a real bug: it silently evaluated on the same trials the
    #    model trained on, producing in-sample numbers that look far
    #    stronger than genuine held-out generalization performance.
    # ============================================================
    
    eval_trials_path = cfg.EVAL_TRIALS_PATH
    
    if os.path.exists(eval_trials_path):
        logging.info(f"Loading held-out trials from: {eval_trials_path}")
        with open(eval_trials_path, "r") as f:
            eval_trials_data = json.load(f)
        logging.info(f"Loaded {len(eval_trials_data)} held-out trials for evaluation")
        trial_store = TrialStore.from_records(eval_trials_data)
    else:
        logging.error(f"Held-out eval trials not found at {eval_trials_path}. "
                       f"Run data_pipeline/run.py (or generate_trial_json.py) first "
                       f"to produce the train/eval split.")
        return None, None, None

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
    if "_seed" in cfg.PATIENT_EMBED_PATH:
        baseline_path = cfg.PATIENT_EMBED_PATH.replace("patient_embeddings_", "patient_embeddings_baseline_")
    else:
        baseline_path = cfg.PATIENT_EMBED_PATH.replace(".pt", "_baseline.pt")    


    if not os.path.exists(baseline_path):
        logging.error(f"Baseline embeddings not found at {baseline_path}")
        return None, None, None
    
    if not os.path.exists(cfg.PATIENT_EMBED_PATH):
        logging.error(f"Patient embeddings not found at {cfg.PATIENT_EMBED_PATH}")
        return None, None, None
    
    if not os.path.exists(cfg.PRE_ALIGN_POST_GNN_PATH):
        logging.error(
            f"Pre-alignment GNN entity embeddings not found at {cfg.PRE_ALIGN_POST_GNN_PATH}. "
            "Re-run train.py (it now saves this file) before evaluating."
        )
        return None, None, None

    if not os.path.exists(cfg.POST_ALIGN_POST_GNN_PATH) or \
       not os.path.exists(cfg.CRITERION_ENCODER_STATE_PATH) or \
       not os.path.exists(cfg.TRIAL_ENCODER_STATE_PATH):
        logging.error(
            "Post-alignment GNN entity embeddings / CriterionEncoder / TrialEncoder "
            "weights not found. Re-run train.py (it now saves these) before evaluating."
        )
        return None, None, None

    h_baseline = torch.load(baseline_path, map_location='cpu')
    h_full = torch.load(cfg.PATIENT_EMBED_PATH, map_location='cpu')

    # --- Re-encode the HELD-OUT trials fresh, rather than loading  ---------
    # cfg.TRIAL_EMBED_PATH / TRIAL_EMBED_BASELINE_PATH -- those only ever
    # contain embeddings for the TRAINING trials (computed inside train.py
    # from TRAIN_TRIALS_PATH). Loading them here was the original bug: it
    # silently evaluated on the same trials the model trained on.
    #
    # entity_maps is reconstructed identically to train.py's construction --
    # it's a deterministic function of the (fixed) processed data tables,
    # not something learned, so recomputing it here is safe and exact.
    d_map = {str(c): i for i, c in enumerate(sorted(diag_df['ICD10_CODE'].unique()))}
    m_map = {str(c): i for i, c in enumerate(sorted(rx_df['NDC'].unique()))}
    l_map = {str(c): i for i, c in enumerate(sorted(labs_df['ITEMID'].unique()))}
    entity_maps = {'diagnosis': d_map, 'medication': m_map, 'lab': l_map, 'procedure': {}}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_align_post_gnn = {
        k: v.to(device) for k, v in torch.load(cfg.PRE_ALIGN_POST_GNN_PATH, map_location='cpu').items()
    }
    post_align_post_gnn = {
        k: v.to(device) for k, v in torch.load(cfg.POST_ALIGN_POST_GNN_PATH, map_location='cpu').items()
    }

    criterion_encoder = CriterionEncoder(cfg.OUT_CHANNELS).to(device)
    criterion_encoder.load_state_dict(torch.load(cfg.CRITERION_ENCODER_STATE_PATH, map_location=device))
    criterion_encoder.eval()

    trial_encoder = TrialEncoder(cfg.OUT_CHANNELS).to(device)
    trial_encoder.load_state_dict(torch.load(cfg.TRIAL_ENCODER_STATE_PATH, map_location=device))
    trial_encoder.eval()

    held_out_trials = list(trial_store)
    logging.info(f"Encoding {len(held_out_trials)} held-out trials fresh "
                 f"(Stage A baseline representation)...")
    with torch.no_grad():
        trial_embeds_baseline = build_naive_baseline_trial_embeds(
            trial_store, entity_maps, pre_align_post_gnn, cfg.OUT_CHANNELS, device,
        )
        logging.info(f"Encoding {len(held_out_trials)} held-out trials fresh "
                     f"(Stage B full-model representation)...")
        trial_embeds = encode_all_trials(
            held_out_trials, entity_maps, post_align_post_gnn,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )
    
    # 5. Debug: Print trial embedding information (baseline vs full model)
    for tid, (z_inc, z_exc) in trial_embeds.items():
        z_inc_b, z_exc_b = trial_embeds_baseline.get(tid, (None, None))
        logging.info(f"Trial {tid}:")
        logging.info(f"  [Stage B] z_inc norm: {z_inc.norm().item():.4f}  z_exc norm: {z_exc.norm().item():.4f}  "
                     f"cos(z_inc,z_exc): {F.cosine_similarity(z_inc, z_exc, dim=0).item():.4f}")
        if z_inc_b is not None:
            logging.info(f"  [Baseline] z_inc norm: {z_inc_b.norm().item():.4f}  z_exc norm: {z_exc_b.norm().item():.4f}  "
                         f"cos(z_inc,z_exc): {F.cosine_similarity(z_inc_b, z_exc_b, dim=0).item():.4f}")

    # 6. Filter to trials that have BOTH baseline and full embeddings
    trial_ids = [
        tid for tid in trial_store.trials.keys()
        if tid in trial_embeds and tid in trial_embeds_baseline
    ]
    
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
        # Baseline trial vectors come from the pre-Stage-B space (matches h_baseline)
        z_inc_base, z_exc_base = trial_embeds_baseline[tid]
        z_inc_base = z_inc_base.squeeze().cpu()
        z_exc_base = z_exc_base.squeeze().cpu()

        # Full-model trial vectors come from the trained Stage B space (matches h_full)
        z_inc_full, z_exc_full = trial_embeds[tid]
        z_inc_full = z_inc_full.squeeze().cpu()
        z_exc_full = z_exc_full.squeeze().cpu()

        for p_idx in range(num_patients):
            # Stage A (Baseline) score -- both sides from the pre-alignment space
            z_patient_base = h_baseline[p_idx].squeeze().cpu()
            inc_sim_base = F.cosine_similarity(z_patient_base.unsqueeze(0), z_inc_base.unsqueeze(0)).item()
            exc_sim_base = F.cosine_similarity(z_patient_base.unsqueeze(0), z_exc_base.unsqueeze(0)).item()
            scores_baseline[p_idx, t_idx] = inc_sim_base - eta * exc_sim_base

            # Stage B (Full Model) score -- both sides from the trained space
            z_patient_full = h_full[p_idx].squeeze().cpu()
            inc_sim_full = F.cosine_similarity(z_patient_full.unsqueeze(0), z_inc_full.unsqueeze(0)).item()
            exc_sim_full = F.cosine_similarity(z_patient_full.unsqueeze(0), z_exc_full.unsqueeze(0)).item()
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
    
    results_path = os.path.join(cfg.OUTPUT_DIR, f'evaluation_results_seed{cfg.SEED}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {results_path}")
    return results


if __name__ == "__main__":
    evaluate_retrieval()