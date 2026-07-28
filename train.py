# train.py
import gc
import json
import logging
import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from alignment import AlignmentLoss, similarity
from config import Config
from gcl_framework import HeteroGNNEncoder
from trial_embedding import CriterionEncoder, TrialEncoder, encode_all_trials
from trial_graph import (
    PatientClinicalState,
    TrialStore,
    compute_matching_indices,
    derive_weak_positive_pairs,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def pretrain_contrastive(cfg: Config, base_graph, device, encoder):
    """Stage A: Self-supervised contrastive pretraining."""
    from gcl_framework import GCL  # Import here to avoid circular import
    
    base_graph = base_graph.to(device)
    gcl = GCL(encoder, cfg.OUT_CHANNELS, cfg.TEMPERATURE)
    optimizer = torch.optim.Adam(gcl.parameters(), lr=cfg.GCL_LR)
    
    logging.info("[Stage A] Starting contrastive pretraining...")
    for epoch in range(1, cfg.EPOCHS_GCL + 1):
        gcl.train()
        optimizer.zero_grad()
        
        # Create augmented views
        aug_graph1 = gcl.augment(base_graph)
        aug_graph2 = gcl.augment(base_graph)
        
        z1 = gcl(aug_graph1)['patient']
        z2 = gcl(aug_graph2)['patient']
        
        loss = gcl.ntxent_loss(z1, z2)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logging.info(f"[Stage A] Epoch {epoch:03d} | Loss: {loss.item():.4f}")
    
    return encoder


def align_with_trials(cfg: Config, base_graph, encoder: HeteroGNNEncoder,
                       trial_store: TrialStore, patient_states, p_map, entity_maps, device):
    """Stage B: Trial-aware alignment fine-tuning."""
    
    # Initialize trial encoders
    criterion_encoder = CriterionEncoder(cfg.OUT_CHANNELS).to(device)
    trial_encoder = TrialEncoder(cfg.OUT_CHANNELS).to(device)
    align_loss_fn = AlignmentLoss(
        lambda_1=cfg.LAMBDA_1, lambda_2=cfg.LAMBDA_2,
        margin_hard=cfg.MARGIN_HARD, margin_rand=cfg.MARGIN_RAND,
    )

    # Move graph to device
    base_graph_dev = base_graph.to(device)

    # Store initial embeddings for anchor loss (DO NOT detach - we want gradients)
    # We'll compute anchor embeddings on-the-fly instead of detaching
    encoder.eval()
    with torch.no_grad():
        # Store the initial patient embeddings as reference (these are frozen)
        h_a_anchor = encoder.encode(
            base_graph_dev.x_dict, 
            base_graph_dev.edge_index_dict
        )['patient'].detach().clone()  # Keep this detached as the target

    # Optimizer with different learning rates
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': cfg.ALIGN_LR * 0.1},
        {'params': criterion_encoder.parameters(), 'lr': cfg.ALIGN_LR},
        {'params': trial_encoder.parameters(), 'lr': cfg.ALIGN_LR},
    ])

    # Derive weak positive pairs
    weak_pairs = derive_weak_positive_pairs(
        patient_states, trial_store,
        inc_threshold=cfg.HARD_NEG_INC_THRESHOLD,
    )

    logging.info(f"First 10 weak pairs: {weak_pairs[:10]}")
    logging.info(f"Number of unique patients: {len(set(p[0] for p in weak_pairs))}")
    logging.info(f"Number of unique trials: {len(set(p[1] for p in weak_pairs))}")

    if not weak_pairs:
        logging.warning("[Stage B] No weak positive pairs derived -- skipping trial-alignment stage.")
        return encoder, criterion_encoder, trial_encoder, {}

    trial_ids = list(trial_store.trials.keys())
    idx_to_subject = {v: k for k, v in p_map.items()}

    # --- PRECOMPUTE MATCHING INDICES CACHE ---
    logging.info("[Stage B] Precomputing static patient-trial matching indices...")
    match_cache = {}
    for pid_idx, trial_id in weak_pairs:
        subject_id = idx_to_subject.get(pid_idx)
        state = patient_states.get(int(subject_id)) if subject_id is not None else None
        if state is None:
            continue
        for cand_id in trial_ids:
            if (pid_idx, cand_id) not in match_cache:
                match_cache[(pid_idx, cand_id)] = compute_matching_indices(state, trial_store[cand_id])

    logging.info(f"[Stage B] Trial-alignment fine-tuning for {cfg.EPOCHS_ALIGN} epochs...")

    # Store loss history for monitoring
    loss_history = []

    for epoch in range(1, cfg.EPOCHS_ALIGN + 1):
        encoder.train()
        criterion_encoder.train()
        trial_encoder.train()
        optimizer.zero_grad()

        # Forward pass - get current embeddings
        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {
            nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab') 
            if nt in h_dict
        }

        # Encode all trials
        trial_embeds = encode_all_trials(
            list(trial_store), entity_maps, post_gnn_embeddings,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )

        # Compute alignment loss
        total_align_loss = torch.zeros((), device=device)
        n_terms = 0

        for pid_idx, trial_id in weak_pairs:
            subject_id = idx_to_subject.get(pid_idx)
            state = patient_states.get(int(subject_id)) if subject_id is not None else None
            if state is None:
                continue

            z_patient = h_dict['patient'][pid_idx]
            m_inc_target, _ = match_cache[(pid_idx, trial_id)]
            z_target_inc, _ = trial_embeds[trial_id]

            # Hard and random negatives
            hard_negs, rand_negs = [], []
            for cand_id in trial_ids:
                if cand_id == trial_id:
                    continue
                m_inc_c, m_exc_c = match_cache[(pid_idx, cand_id)]
                z_inc_c, z_exc_c = trial_embeds[cand_id]
                if m_inc_c >= cfg.HARD_NEG_INC_THRESHOLD and m_exc_c >= cfg.HARD_NEG_EXC_THRESHOLD:
                    hard_negs.append((z_exc_c, m_exc_c))
                else:
                    rand_negs.append(z_inc_c)

            # Sample random negatives
            if len(rand_negs) > cfg.N_RANDOM_NEGATIVES:
                perm = torch.randperm(len(rand_negs))[:cfg.N_RANDOM_NEGATIVES]
                rand_negs = [rand_negs[i] for i in perm]

            loss = align_loss_fn(z_patient, m_inc_target, z_target_inc, hard_negs, rand_negs)
            total_align_loss = total_align_loss + loss
            n_terms += 1

        if n_terms == 0:
            continue

        # Compute average alignment loss
        avg_align_loss = total_align_loss / n_terms
        
        # Compute anchor loss - THIS IS THE FIX
        # Use MSE between current patient embeddings and initial reference embeddings
        # The current embeddings are from the trainable encoder, so gradients will flow
        anchor_loss = F.mse_loss(h_dict['patient'], h_a_anchor)
        
        # Get anchor weight from config with default
        anchor_weight = getattr(cfg, 'LAMBDA_ANCHOR', 0.5)
        composite_loss = avg_align_loss + anchor_weight * anchor_loss

        # Backward pass
        composite_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(criterion_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(trial_encoder.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Log progress
        logging.info(
            f"[Stage B] Epoch {epoch:02d} | Align Loss: {avg_align_loss.item():.4f} | "
            f"Anchor Loss: {anchor_loss.item():.4f} | Total Loss: {composite_loss.item():.4f}"
        )
        
        # Store loss history
        loss_history.append({
            'epoch': epoch,
            'align_loss': avg_align_loss.item(),
            'anchor_loss': anchor_loss.item(),
            'total_loss': composite_loss.item()
        })
        
        # Early stopping if loss converges
        if epoch > 5:
            recent_losses = [l['total_loss'] for l in loss_history[-5:]]
            if all(abs(recent_losses[i] - recent_losses[i-1]) < 1e-4 for i in range(1, len(recent_losses))):
                logging.info(f"[Stage B] Early stopping at epoch {epoch} - loss converged.")
                break

    # Save final trial embeddings
    with torch.no_grad():
        encoder.eval()
        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {
            nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab') 
            if nt in h_dict
        }
        final_trial_embeds = encode_all_trials(
            list(trial_store), entity_maps, post_gnn_embeddings,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )
        
        # Save loss history
        loss_df = pd.DataFrame(loss_history)
        loss_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'training_loss_history.csv'), index=False)

    return encoder, criterion_encoder, trial_encoder, final_trial_embeds


def load_processed_tables(cfg: Config):
    """Load preprocessed tables."""
    paths = {
        'diag': os.path.join(cfg.OUTPUT_DIR, "diagnoses_clean.parquet"),
        'rx': os.path.join(cfg.OUTPUT_DIR, "prescriptions_clean.parquet"),
        'labs': os.path.join(cfg.OUTPUT_DIR, "labs_clean.parquet"),
    }
    return (pd.read_parquet(paths['diag']), 
            pd.read_parquet(paths['rx']), 
            pd.read_parquet(paths['labs']))


def build_patient_states(diag_df, rx_df, labs_df):
    """Build patient clinical states."""
    states = {}
    # Get all unique subject IDs from all tables
    all_sids = set(diag_df['SUBJECT_ID'].unique())
    all_sids.update(rx_df['SUBJECT_ID'].unique())
    all_sids.update(labs_df['SUBJECT_ID'].unique())
    
    for sid in all_sids:
        states[int(sid)] = PatientClinicalState.build_from_tables(
            int(sid), diag_df, rx_df, labs_df
        )
    return states


def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using execution device: {device}")

    # Check if graph exists
    if not os.path.exists(cfg.GRAPH_PATH):
        raise FileNotFoundError(f"Compiled graph not found at {cfg.GRAPH_PATH}.")

    # Load graph
    base_graph = torch.load(cfg.GRAPH_PATH, map_location='cpu', weights_only=False)
    base_graph = T.ToUndirected()(base_graph)

    # Initialize encoder
    num_nodes_dict = {nt: base_graph[nt].num_nodes for nt in base_graph.node_types}
    encoder = HeteroGNNEncoder(
        metadata=base_graph.metadata(),
        num_nodes_dict=num_nodes_dict,
        patient_feat_dim=cfg.PATIENT_FEAT_DIM,
        entity_embed_dim=cfg.ENTITY_EMBED_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
    ).to(device)

    # ---------------- Stage A: Contrastive Pretraining ----------------
    baseline_embed_path = cfg.PATIENT_EMBED_PATH.replace(".pt", "_baseline.pt")
    if os.path.exists(baseline_embed_path):
        logging.info(f"[Stage A] Found existing baseline patient embeddings. Skipping pretraining!")
    else:
        logging.info("[Stage A] Starting contrastive pretraining...")
        encoder = pretrain_contrastive(cfg, base_graph, device, encoder)
        encoder.eval()
        with torch.no_grad():
            graph_dev = base_graph.to(device)
            h_dict = encoder.encode(graph_dev.x_dict, graph_dev.edge_index_dict)
        torch.save(h_dict['patient'].cpu(), baseline_embed_path)
        logging.info(f"[Stage A] Saved baseline patient embeddings to {baseline_embed_path}")

    # ---------------- Stage B: Trial-aware Alignment ----------------
    logging.info("[Stage B] Loading data for trial alignment...")
    
    # Load processed tables
    diag_df, rx_df, labs_df = load_processed_tables(cfg)
    
    # Build entity maps
    p_map = {str(sid): i for i, sid in enumerate(sorted(diag_df['SUBJECT_ID'].unique(), key=int))}
    d_map = {str(c): i for i, c in enumerate(sorted(diag_df['ICD10_CODE'].unique()))}
    m_map = {str(c): i for i, c in enumerate(sorted(rx_df['NDC'].unique()))}
    l_map = {str(c): i for i, c in enumerate(sorted(labs_df['ITEMID'].unique()))}
    entity_maps = {'diagnosis': d_map, 'medication': m_map, 'lab': l_map, 'procedure': {}}

    # Build patient states
    patient_states = build_patient_states(diag_df, rx_df, labs_df)
    logging.info(f"Built clinical states for {len(patient_states)} patients")

    # Load trials
    trial_json_path = "structured_clinical_trials.json"
    if not os.path.exists(trial_json_path):
        logging.warning(f"Trial JSON not found at {trial_json_path}. Using mock trials instead.")
        from mock_data import generate_mock_trials
        real_trials_data = generate_mock_trials()
    else:
        with open(trial_json_path, "r") as f:
            real_trials_data = json.load(f)

    trial_store = TrialStore.from_records(real_trials_data)
    logging.info(f"Loaded {len(trial_store.trials)} REAL clinical trials into Stage B!")

    # Run Stage B alignment
    encoder, criterion_encoder, trial_encoder, trial_embeds = align_with_trials(
        cfg, base_graph, encoder, trial_store, patient_states, p_map, entity_maps, device
    )

    # Save trial embeddings
    if trial_embeds:
        torch.save(
            {tid: (z_inc.detach().cpu(), z_exc.detach().cpu()) 
             for tid, (z_inc, z_exc) in trial_embeds.items()},
            cfg.TRIAL_EMBED_PATH,
        )
        logging.info(f"[Stage B] Saved trial embeddings to {cfg.TRIAL_EMBED_PATH}")

    # Save final patient embeddings
    encoder.eval()
    with torch.no_grad():
        graph_dev = base_graph.to(device)
        h_dict = encoder.encode(graph_dev.x_dict, graph_dev.edge_index_dict)
    torch.save(h_dict['patient'].cpu(), cfg.PATIENT_EMBED_PATH)
    logging.info(f"[Stage B] Saved trial-aware patient embeddings to {cfg.PATIENT_EMBED_PATH}")


if __name__ == "__main__":
    main()