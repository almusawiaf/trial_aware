"""
G/train.py
Parallel Precomputed Trial-Aware Graph Neural Network Pipeline
Optimized Workload Assembly and Clean PyTorch Autograd Graph Execution
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from alignment import AlignmentLoss
from config import Config
from gcl_framework import HeteroGNNEncoder, GraphAugmentor, InfoNCELoss
from trial_embedding import CriterionEncoder, TrialEncoder, encode_all_trials, get_concept_embedding, is_resolvable_code

from matching_engine import (
    ICD10Hierarchy,
    compute_matching_indices,
    PatientState as MatchingPatientState,
    Criterion as MatchingCriterion,
    Trial as MatchingTrial,
)

from trial_graph import (
    PatientClinicalState,
    TrialStore,
    derive_weak_positive_pairs,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global process variables initialized ONCE per CPU worker to avoid IPC overhead
_GLOBAL_HIERARCHY: ICD10Hierarchy | None = None
_GLOBAL_TRIALS: Dict[str, MatchingTrial] | None = None


def _init_worker(hierarchy_file: str | None, matching_trials: Dict[str, MatchingTrial]) -> None:
    """Initializer function run once per worker process when spawned."""
    global _GLOBAL_HIERARCHY, _GLOBAL_TRIALS
    if hierarchy_file:
        _GLOBAL_HIERARCHY = ICD10Hierarchy(hierarchy_file, log_duplicates=False)
    else:
        _GLOBAL_HIERARCHY = None
    _GLOBAL_TRIALS = matching_trials


def _worker_compute_chunk(chunk_data: List[Tuple[int, str, MatchingPatientState]]) -> List[Tuple[Tuple[int, str], Tuple[float, float]]]:
    """Processes a large chunk of (pid_idx, cand_id, patient_state) in a single CPU process."""
    global _GLOBAL_HIERARCHY, _GLOBAL_TRIALS
    results = []
    for pid_idx, cand_id, state in chunk_data:
        matching_cand = _GLOBAL_TRIALS[cand_id]
        m_inc, m_exc = compute_matching_indices(state, matching_cand, _GLOBAL_HIERARCHY)
        results.append(((pid_idx, cand_id), (m_inc, m_exc)))
    return results


def precompute_matching_cache_parallel(unique_pairs, matching_states, trial_store, idx_to_subject, hierarchy_path):
    """Chunked Batch Precomputation with Instant Vectorized Workload Assembly."""
    trial_ids = list(trial_store.trials.keys())
    
    logging.info(f"[Stage B] Pre-building trial criterion models for {len(trial_ids)} trials...")
    matching_trials = {}
    for cand_id in trial_ids:
        cand_trial = trial_store[cand_id]
        matching_trials[cand_id] = MatchingTrial(
            nct_id=cand_trial.trial_id,
            inclusion_criteria=[
                MatchingCriterion(
                    raw_entity=getattr(c, 'raw_entity', ''),
                    entity_type=c.entity_type,
                    entity_code=c.entity_code,
                    operator=c.operator.value if hasattr(c.operator, 'value') else str(c.operator),
                    value=c.value,
                    is_inclusion=c.is_inclusion,
                    severity_weight=c.severity_weight,
                    confidence=getattr(c, 'confidence', 1.0)
                ) for c in cand_trial.inclusion_criteria
            ],
            exclusion_criteria=[
                MatchingCriterion(
                    raw_entity=getattr(c, 'raw_entity', ''),
                    entity_type=c.entity_type,
                    entity_code=c.entity_code,
                    operator=c.operator.value if hasattr(c.operator, 'value') else str(c.operator),
                    value=c.value,
                    is_inclusion=c.is_inclusion,
                    severity_weight=c.severity_weight,
                    confidence=getattr(c, 'confidence', 1.0)
                ) for c in cand_trial.exclusion_criteria
            ]
        )

    logging.info("[Stage B] Assembling unique matching calculation workload...")
    
    # Accelerated Workload Assembly (Instant Set Math)
    valid_pids = {
        pid_idx for pid_idx, _ in unique_pairs
        if (subject_id := idx_to_subject.get(pid_idx)) is not None and int(subject_id) in matching_states
    }
    
    tasks = [
        (pid_idx, cand_id, matching_states[int(idx_to_subject[pid_idx])])
        for pid_idx in valid_pids
        for cand_id in trial_ids
    ]

    total_tasks = len(tasks)
    num_workers = max(1, min(cpu_count(), 126))
    
    chunk_size = max(1000, math.ceil(total_tasks / (num_workers * 4)))
    chunks = [tasks[i:i + chunk_size] for i in range(0, total_tasks, chunk_size)]

    logging.info(f"[Stage B] Parallelizing {total_tasks} tasks into {len(chunks)} chunks across {num_workers} CPU workers...")

    match_cache = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(hierarchy_path, matching_trials)
    ) as executor:
        for chunk_result in executor.map(_worker_compute_chunk, chunks):
            match_cache.update(chunk_result)

    logging.info(f"[Stage B] Successfully precomputed match cache with {len(match_cache)} unique key pairs.")
    return match_cache


def build_naive_baseline_trial_embeds(trial_store: TrialStore, entity_maps, post_gnn_embeddings, embed_dim, device):
    """
    Builds a 'before training' trial embedding for every trial using plain
    averaging of concept embeddings -- no learned pooling weights at all.

    Why: evaluate.py needs a fair Stage A comparison point. It cannot reuse
    the Stage B trial embeddings (those only exist *after* the criterion/trial
    encoders are trained), and comparing pre-training patient vectors against
    post-training trial vectors is comparing two unrelated spaces. This gives
    an honest, zero-parameter baseline that lives in the same "before" space
    as h_a_anchor.
    """
    baseline_embeds = {}
    for t in trial_store:
        inc_embs = [
            get_concept_embedding(c.entity_type, c.entity_code, entity_maps, post_gnn_embeddings, embed_dim, device)
            for c in t.inclusion_criteria
            if is_resolvable_code(c.entity_type, c.entity_code, entity_maps)
        ]
        exc_embs = [
            get_concept_embedding(c.entity_type, c.entity_code, entity_maps, post_gnn_embeddings, embed_dim, device)
            for c in t.exclusion_criteria
            if is_resolvable_code(c.entity_type, c.entity_code, entity_maps)
        ]
        z_inc = torch.stack(inc_embs).mean(dim=0) if inc_embs else torch.zeros(embed_dim, device=device)
        z_exc = torch.stack(exc_embs).mean(dim=0) if exc_embs else torch.zeros(embed_dim, device=device)
        baseline_embeds[t.trial_id] = (z_inc.detach().cpu(), z_exc.detach().cpu())
    return baseline_embeds


def align_with_trials(cfg: Config, base_graph, encoder: HeteroGNNEncoder, trial_store: TrialStore, patient_states, p_map, entity_maps, device):
    """Stage B: Accelerated Trial-aware alignment fine-tuning with Gradient Accumulation (No Graph Inplace Conflicts)."""
    criterion_encoder = CriterionEncoder(cfg.OUT_CHANNELS).to(device)
    trial_encoder = TrialEncoder(cfg.OUT_CHANNELS).to(device)
    align_loss_fn = AlignmentLoss(
        lambda_1=cfg.LAMBDA_1, lambda_2=cfg.LAMBDA_2,
        margin_hard=cfg.MARGIN_HARD, margin_rand=cfg.MARGIN_RAND,
    )

    base_graph_dev = base_graph.to(device)

    encoder.eval()
    with torch.no_grad():
        pre_align_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        h_a_anchor = pre_align_dict['patient'].detach().clone()

        # --- NEW: build and save an honest "before Stage B" pair ---------
        pre_align_post_gnn = {nt: pre_align_dict[nt] for nt in ('diagnosis', 'medication', 'lab') if nt in pre_align_dict}
        baseline_trial_embeds = build_naive_baseline_trial_embeds(
            trial_store, entity_maps, pre_align_post_gnn, cfg.OUT_CHANNELS, device,
        )
        torch.save(h_a_anchor.detach().cpu(), cfg.BASELINE_EMBED_PATH)
        torch.save(baseline_trial_embeds, cfg.TRIAL_EMBED_BASELINE_PATH)
        # NEW: persist the pre-fine-tuning diag/med/lab embeddings themselves,
        # so evaluate.py can encode held-out trials against the SAME Stage A
        # representation, not just look up trial embeddings that only exist
        # for the training trial set.
        torch.save({k: v.detach().cpu() for k, v in pre_align_post_gnn.items()}, cfg.PRE_ALIGN_POST_GNN_PATH)
        logging.info(f"[Stage A baseline] Saved pre-alignment patient embeddings to {cfg.BASELINE_EMBED_PATH}")
        logging.info(f"[Stage A baseline] Saved pre-alignment trial embeddings to {cfg.TRIAL_EMBED_BASELINE_PATH}")
        # -------------------------------------------------------------------

    hierarchy_path = "data/icd10_hierarchy.csv" if os.path.exists("data/icd10_hierarchy.csv") else None

    matching_states = {
        sid: MatchingPatientState(
            patient_id=str(sid),
            diagnosis_codes=state.diagnosis_codes,
            medication_codes=state.medication_codes,
            lab_values=state.lab_last_values
        ) for sid, state in patient_states.items()
    }

    # NEW: FIX for a verified bug -- when FREEZE_BACKBONE is off, the
    # encoder's parameters were being fine-tuned via a raw, unnormalized
    # MSE anchor loss (see the fix at anchor_loss_tensor below) alongside
    # bounded, cosine-similarity-based losses -- a scale mismatch that
    # caused catastrophic collapse, worse at HIGHER anchor strength, once
    # Stage A became a genuinely good GCL-pretrained baseline (see
    # session notes / paper Section on GCL pretraining). FREEZE_BACKBONE
    # is the stronger, more direct fix: if the encoder never updates,
    # Stage A's representation is preserved exactly, not just
    # approximately via a regularizer -- making the anchor loss moot by
    # construction rather than needing to be perfectly scaled.
    if cfg.FREEZE_BACKBONE:
        logging.info("[Stage B] FREEZE_BACKBONE=True -- GNN encoder weights will NOT be "
                     "updated; only CriterionEncoder/TrialEncoder train on top of the "
                     "frozen (GCL-pretrained) patient/entity representations.")
        for p in encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam([
            {'params': criterion_encoder.parameters(), 'lr': cfg.ALIGN_LR},
            {'params': trial_encoder.parameters(), 'lr': cfg.ALIGN_LR},
        ])
    else:
        optimizer = torch.optim.Adam([
            {'params': encoder.parameters(), 'lr': cfg.ALIGN_LR * 0.5},
            {'params': criterion_encoder.parameters(), 'lr': cfg.ALIGN_LR},
            {'params': trial_encoder.parameters(), 'lr': cfg.ALIGN_LR},
        ])

    weak_pairs = derive_weak_positive_pairs(
        patient_states, trial_store,
        inc_threshold=cfg.HARD_NEG_INC_THRESHOLD,
    )

    if not weak_pairs:
        logging.warning("[Stage B] No weak positive pairs derived -- skipping trial-alignment stage.")
        return encoder, criterion_encoder, trial_encoder, {}

    idx_to_subject = {v: k for k, v in p_map.items()}

    # Precomputation on CPU workers
    match_cache = precompute_matching_cache_parallel(weak_pairs, matching_states, trial_store, idx_to_subject, hierarchy_path)

    trial_ids = list(trial_store.trials.keys())
    trial_to_idx = {tid: i for i, tid in enumerate(trial_ids)}
    
    valid_weak_pairs = [(pid_idx, tid) for pid_idx, tid in weak_pairs if pid_idx < base_graph['patient'].num_nodes]
    
    logging.info("[Stage B] Vectorizing indexing structures for GPU Mini-batching...")
    
    p_indices = torch.tensor([p[0] for p in valid_weak_pairs], dtype=torch.long, device=device)
    t_target_indices = torch.tensor([trial_to_idx[p[1]] for p in valid_weak_pairs], dtype=torch.long, device=device)
    m_inc_targets = torch.tensor([match_cache.get((p[0], p[1]), (1.0, 0.0))[0] for p in valid_weak_pairs], dtype=torch.float32, device=device)
    # NEW: exclusion targets -- these were computed by the matching engine all
    # along (match_cache stores (m_inc, m_exc) tuples) but m_exc was silently
    # dropped on the floor before. Without this, the exclusion pooling head
    # never sees a training signal at all.
    m_exc_targets = torch.tensor([match_cache.get((p[0], p[1]), (1.0, 0.0))[1] for p in valid_weak_pairs], dtype=torch.float32, device=device)
    
    rand_neg_matrix = []
    for pid_idx, trial_id in valid_weak_pairs:
        candidates = [trial_to_idx[cid] for cid in trial_ids if cid != trial_id]
        if len(candidates) >= cfg.N_RANDOM_NEGATIVES:
            # FIX: was candidates[:N] -- same negatives every time, every epoch.
            # random.sample gives a genuinely different set of negative trials
            # for each patient, which is what "random negatives" is supposed to mean.
            sampled = random.sample(candidates, cfg.N_RANDOM_NEGATIVES)
        else:
            sampled = candidates + [candidates[0]] * (cfg.N_RANDOM_NEGATIVES - len(candidates))
        rand_neg_matrix.append(sampled)
        
    rand_neg_tensor = torch.tensor(rand_neg_matrix, dtype=torch.long, device=device)

    logging.info(f"[Stage B] Starting trial-alignment fine-tuning for {cfg.EPOCHS_ALIGN} epochs...")
    loss_history = []
    
    batch_size = 16384
    num_pairs = len(valid_weak_pairs)

    for epoch in range(1, cfg.EPOCHS_ALIGN + 1):
        encoder.train()
        criterion_encoder.train()
        trial_encoder.train()
        # Zero gradients once per epoch
        optimizer.zero_grad()

        # Step 1: Single GNN + Trial Encoder pass
        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab') if nt in h_dict}

        trial_embeds_dict = encode_all_trials(
            list(trial_store), entity_maps, post_gnn_embeddings,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )

        z_inc_all = torch.stack([trial_embeds_dict[tid][0] for tid in trial_ids])
        # NEW: z_exc_all -- was computed by encode_all_trials all along but
        # never used anywhere in the loss. Without this, the exclusion
        # pooling head (trial_encoder.W_pool_exc) never learns anything.
        z_exc_all = torch.stack([trial_embeds_dict[tid][1] for tid in trial_ids])

        # NEW: real anchor loss (kept as a tensor, not .item()'d away) so it
        # actually participates in backward() instead of only being printed.
        # This is what stops the encoder from drifting away from the
        # perfectly-fine Stage-A representations you already measured at 0.77 AUC.
        # FIX: previously raw, unnormalized MSE -- unbounded magnitude,
        # dependent on embedding dimensionality and whatever scale the
        # encoder's raw output happens to have. Every OTHER loss term in
        # this training loop (pos_loss/exc_loss/neg_loss below) operates
        # on cosine similarities, bounded in [-1, 1]. That scale mismatch
        # meant increasing LAMBDA_ANCHOR amplified an already-oversized,
        # poorly-scaled gradient rather than gently pulling embeddings
        # back toward Stage A -- consistent with higher LAMBDA_ANCHOR
        # producing MORE catastrophic collapse, not less, once Stage A
        # became a genuinely good GCL-pretrained baseline (random-init
        # Stage A had little worth preserving, so this scale problem was
        # invisible before). Normalizing both sides bounds this term to
        # the same [0, 4] range MSE-of-unit-vectors always has, matching
        # the scale of the rest of the loss.
        h_current_norm = F.normalize(h_dict['patient'], dim=-1)
        h_anchor_norm = F.normalize(h_a_anchor, dim=-1)
        anchor_loss_tensor = F.mse_loss(h_current_norm, h_anchor_norm)

        epoch_align_loss = 0.0
        n_batches = 0
        
        perm = torch.randperm(num_pairs, device=device)
        num_minibatches = math.ceil(num_pairs / batch_size)

        # Step 2: Chunked Loss Accumulation across mini-batches
        for batch_i, b_start in enumerate(range(0, num_pairs, batch_size)):
            b_perm = perm[b_start:b_start + batch_size]
            
            b_z_patients = h_dict['patient'][p_indices[b_perm]]
            b_z_targets = z_inc_all[t_target_indices[b_perm]]
            b_m_inc = m_inc_targets[b_perm]
            b_z_rand_negs = z_inc_all[rand_neg_tensor[b_perm]]

            # Vectorized Cosine Similarities
            pos_sim = F.cosine_similarity(b_z_patients, b_z_targets, dim=-1)
            pos_loss = F.mse_loss(pos_sim, b_m_inc)

            # NEW: exclusion term -- trains z_exc against the real m_exc score,
            # using LAMBDA_1 (previously configured but never actually used).
            b_z_exc_targets = z_exc_all[t_target_indices[b_perm]]
            b_m_exc = m_exc_targets[b_perm]
            exc_sim = F.cosine_similarity(b_z_patients, b_z_exc_targets, dim=-1)
            exc_loss = F.mse_loss(exc_sim, b_m_exc)

            b_z_patients_expanded = b_z_patients.unsqueeze(1)
            neg_sims = F.cosine_similarity(b_z_patients_expanded, b_z_rand_negs, dim=-1)
            neg_loss = torch.mean(F.relu(neg_sims - cfg.MARGIN_RAND))

            batch_loss = pos_loss + cfg.LAMBDA_1 * exc_loss + cfg.LAMBDA_2 * neg_loss

            # Scale loss by total batch count so gradients accumulate correctly
            scaled_batch_loss = batch_loss / num_minibatches

            # NEW: add the anchor term to exactly one mini-batch's backward
            # pass (not all of them) so it contributes its full weight once
            # per epoch, instead of zero times as before.
            if batch_i == 0:
                scaled_batch_loss = scaled_batch_loss + cfg.LAMBDA_ANCHOR * anchor_loss_tensor

            # Retain graph for inner backward passes without stepping optimizer
            scaled_batch_loss.backward(retain_graph=True)

            epoch_align_loss += batch_loss.item()
            n_batches += 1

        # Step 3: Single Optimizer Step per Epoch (After full gradient accumulation)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(criterion_encoder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(trial_encoder.parameters(), max_norm=0.5)

        optimizer.step()

        avg_align_loss = epoch_align_loss / max(1, n_batches)
        
        # anchor_loss_tensor was already used in backward() above; .item() here is only for logging
        anchor_loss = anchor_loss_tensor.item()
        composite_loss = avg_align_loss + cfg.LAMBDA_ANCHOR * anchor_loss

        logging.info(
            f"[Stage B] Epoch {epoch:02d}/{cfg.EPOCHS_ALIGN} | Align Loss: {avg_align_loss:.4f} | "
            f"Anchor Loss: {anchor_loss:.4f} | Total Loss: {composite_loss:.4f} | "
            f"z_inc spread(std): {z_inc_all.std(dim=0).mean().item():.4f} | "
            f"z_exc spread(std): {z_exc_all.std(dim=0).mean().item():.4f}"
        )

        loss_history.append({
            'epoch': epoch,
            'align_loss': avg_align_loss,
            'anchor_loss': anchor_loss,
            'total_loss': composite_loss
        })

    with torch.no_grad():
        encoder.eval()
        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab') if nt in h_dict}
        final_trial_embeds = encode_all_trials(
            list(trial_store), entity_maps, post_gnn_embeddings,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )

    # NEW: persist everything needed to encode a DIFFERENT (held-out) trial
    # set later in evaluate.py -- the post-fine-tuning diag/med/lab
    # embeddings, and the trained CriterionEncoder/TrialEncoder weights
    # themselves (encode_all_trials calls their forward methods on new
    # criterion data, so the trained weights generalize; only the specific
    # trial_embeds tensor computed here does not, since it's tied to
    # whichever trial_store was passed in -- the training trials).
    torch.save({k: v.detach().cpu() for k, v in post_gnn_embeddings.items()}, cfg.POST_ALIGN_POST_GNN_PATH)
    torch.save(criterion_encoder.state_dict(), cfg.CRITERION_ENCODER_STATE_PATH)
    torch.save(trial_encoder.state_dict(), cfg.TRIAL_ENCODER_STATE_PATH)
    logging.info(f"[Stage B] Saved post-alignment GNN entity embeddings, "
                 f"CriterionEncoder, and TrialEncoder weights for held-out evaluation.")

    return encoder, criterion_encoder, trial_encoder, final_trial_embeds


def gcl_pretrain(cfg: Config, base_graph, encoder: HeteroGNNEncoder, device) -> HeteroGNNEncoder:
    """
    Self-supervised graph contrastive pretraining (GCL), previously imported
    (GraphAugmentor, InfoNCELoss) but never actually called anywhere in this
    file -- meaning "Stage A" was, until now, a randomly-initialized encoder
    doing a single untrained forward pass, not a pretrained baseline. This
    fixes that: two augmented views of the graph are contrasted via InfoNCE
    on the patient node's projected representation, following standard GCL
    practice (Zhu et al., GRACE; You et al., GraphCL).

    Gated by cfg.ENABLE_GCL_PRETRAIN so both configurations (with and
    without this pretraining step) can be run and reported side by side --
    see the paper's Held-Out Cross-Validation Protocol section for why
    reporting both is preferable to silently swapping one for the other.
    """
    if not cfg.ENABLE_GCL_PRETRAIN:
        logging.info("[GCL] cfg.ENABLE_GCL_PRETRAIN is False -- skipping pretraining, "
                      "Stage A will be a randomly-initialized encoder (legacy behavior).")
        return encoder

    logging.info(f"[GCL] Starting self-supervised contrastive pretraining "
                 f"for {cfg.EPOCHS_GCL} epochs (lr={cfg.GCL_LR})...")
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.GCL_LR)
    info_nce = InfoNCELoss(temperature=0.1)
    base_graph_dev = base_graph.to(device)

    for epoch in range(cfg.EPOCHS_GCL):
        optimizer.zero_grad()

        # Two independently-augmented views of the same graph -- the model
        # must learn to agree on a patient's representation despite each
        # view seeing a different random 20% of edges dropped.
        view1 = GraphAugmentor.drop_edges(base_graph_dev, drop_rate=cfg.DROP_RATE_V1)
        view2 = GraphAugmentor.drop_edges(base_graph_dev, drop_rate=cfg.DROP_RATE_V2)

        h1 = encoder.encode(view1.x_dict, view1.edge_index_dict)
        h2 = encoder.encode(view2.x_dict, view2.edge_index_dict)

        z1 = encoder.project(h1['patient'])
        z2 = encoder.project(h2['patient'])

        loss = info_nce(z1, z2)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"[GCL] Epoch {epoch + 1:03d}/{cfg.EPOCHS_GCL} | InfoNCE Loss: {loss.item():.4f}")

    encoder.eval()
    logging.info("[GCL] Pretraining complete.")
    return encoder


def load_processed_tables(cfg: Config):
    paths = {
        'diag': os.path.join(cfg.OUTPUT_DIR, "diagnoses_clean.parquet"),
        'rx': os.path.join(cfg.OUTPUT_DIR, "prescriptions_clean.parquet"),
        'labs': os.path.join(cfg.OUTPUT_DIR, "labs_clean.parquet"),
    }
    return (pd.read_parquet(paths['diag']), pd.read_parquet(paths['rx']), pd.read_parquet(paths['labs']))


def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using execution device: {device}")

    base_graph = torch.load(cfg.GRAPH_PATH, map_location='cpu', weights_only=False)
    base_graph = T.ToUndirected()(base_graph)

    num_nodes_dict = {nt: base_graph[nt].num_nodes for nt in base_graph.node_types}
    encoder = HeteroGNNEncoder(
        metadata=base_graph.metadata(),
        num_nodes_dict=num_nodes_dict,
        patient_feat_dim=cfg.PATIENT_FEAT_DIM,
        entity_embed_dim=cfg.ENTITY_EMBED_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
    ).to(device)

    # NEW: run GCL pretraining before anything else touches this encoder --
    # this is what makes "Stage A" a genuine self-supervised baseline
    # instead of a random projection. See gcl_pretrain()'s docstring.
    encoder = gcl_pretrain(cfg, base_graph, encoder, device)

    diag_df, rx_df, labs_df = load_processed_tables(cfg)
    p_map = {str(sid): i for i, sid in enumerate(sorted(diag_df['SUBJECT_ID'].unique(), key=int))}
    d_map = {str(c): i for i, c in enumerate(sorted(diag_df['ICD10_CODE'].unique()))}
    m_map = {str(c): i for i, c in enumerate(sorted(rx_df['NDC'].unique()))}
    l_map = {str(c): i for i, c in enumerate(sorted(labs_df['ITEMID'].unique()))}
    entity_maps = {'diagnosis': d_map, 'medication': m_map, 'lab': l_map, 'procedure': {}}

    # Activate hierarchical diagnosis alignment so trial diagnosis codes
    # resolve to clinically-equivalent patient codes in the same ICD-10
    # category, recovering criteria previously discarded as OOV. Must be
    # called before any encode_all_trials() so the resolution logic sees it.
    from trial_embedding import init_diagnosis_aligner
    init_diagnosis_aligner(entity_maps)

    patient_states = {
        int(sid): PatientClinicalState.build_from_tables(int(sid), diag_df, rx_df, labs_df)
        for sid in set(diag_df['SUBJECT_ID'].unique()).union(rx_df['SUBJECT_ID'].unique()).union(labs_df['SUBJECT_ID'].unique())
    }

    train_trials_path = cfg.TRAIN_TRIALS_PATH
    with open(train_trials_path, "r") as f:
        train_trials_data = json.load(f)
    trial_store = TrialStore.from_records(train_trials_data)

    encoder, criterion_encoder, trial_encoder, trial_embeds = align_with_trials(
        cfg, base_graph, encoder, trial_store, patient_states, p_map, entity_maps, device
    )

    if trial_embeds:
        torch.save(
            {tid: (z_inc.detach().cpu(), z_exc.detach().cpu()) for tid, (z_inc, z_exc) in trial_embeds.items()},
            cfg.TRIAL_EMBED_PATH,
        )

    encoder.eval()
    with torch.no_grad():
        graph_dev = base_graph.to(device)
        h_dict = encoder.encode(graph_dev.x_dict, graph_dev.edge_index_dict)
    torch.save(h_dict['patient'].cpu(), cfg.PATIENT_EMBED_PATH)
    logging.info("[Stage B] Precomputation complete and saved embeddings.")


if __name__ == "__main__":
    main()