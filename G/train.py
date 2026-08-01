"""
G/train.py
Parallel Precomputed Trial-Aware Graph Neural Network Pipeline
Optimized with Large-Batch Worker Memory Initializers and Low-Overhead IPC
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from alignment import AlignmentLoss
from config import Config
from gcl_framework import HeteroGNNEncoder
from trial_embedding import CriterionEncoder, TrialEncoder, encode_all_trials

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
    """Chunked Batch Precomputation using shared worker initialization to eliminate serialization overhead."""
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
    tasks = []
    seen = set()
    for pid_idx, trial_id in unique_pairs:
        subject_id = idx_to_subject.get(pid_idx)
        if subject_id is None or int(subject_id) not in matching_states:
            continue
        state = matching_states[int(subject_id)]
        
        for cand_id in trial_ids:
            if (pid_idx, cand_id) not in seen:
                seen.add((pid_idx, cand_id))
                tasks.append((pid_idx, cand_id, state))

    total_tasks = len(tasks)
    num_workers = max(1, min(cpu_count(), 126))
    
    # Split total workloads into large balanced chunks per CPU core
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


def align_with_trials(cfg: Config, base_graph, encoder: HeteroGNNEncoder, trial_store: TrialStore, patient_states, p_map, entity_maps, device):
    """Stage B: Accelerated Trial-aware alignment fine-tuning with Robust Vectorized Batch Loss."""
    criterion_encoder = CriterionEncoder(cfg.OUT_CHANNELS).to(device)
    trial_encoder = TrialEncoder(cfg.OUT_CHANNELS).to(device)
    align_loss_fn = AlignmentLoss(
        lambda_1=cfg.LAMBDA_1, lambda_2=cfg.LAMBDA_2,
        margin_hard=cfg.MARGIN_HARD, margin_rand=cfg.MARGIN_RAND,
    )

    base_graph_dev = base_graph.to(device)

    encoder.eval()
    with torch.no_grad():
        h_a_anchor = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)['patient'].detach().clone()

    hierarchy_path = "icd10_hierarchy.csv" if os.path.exists("icd10_hierarchy.csv") else None

    matching_states = {
        sid: MatchingPatientState(
            patient_id=str(sid),
            diagnosis_codes=state.diagnosis_codes,
            medication_codes=state.medication_codes,
            lab_values=state.lab_last_values
        ) for sid, state in patient_states.items()
    }

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

    # Compute matches using low-overhead IPC worker pool
    match_cache = precompute_matching_cache_parallel(weak_pairs, matching_states, trial_store, idx_to_subject, hierarchy_path)

    trial_ids = list(trial_store.trials.keys())
    
    # Pre-index weak pairs
    valid_weak_pairs = [(pid_idx, tid) for pid_idx, tid in weak_pairs if pid_idx < base_graph['patient'].num_nodes]
    patient_indices = torch.tensor([p[0] for p in valid_weak_pairs], dtype=torch.long, device=device)
    
    # Pre-structure Hard/Random Negative lookups per pair
    logging.info("[Stage B] Indexing negative sampling lookup structures...")
    pair_negatives_map = []
    for pid_idx, trial_id in valid_weak_pairs:
        m_inc_target, _ = match_cache.get((pid_idx, trial_id), (1.0, 0.0))
        hard_negs_info = []
        rand_negs_ids = []
        
        for cand_id in trial_ids:
            if cand_id == trial_id:
                continue
            m_inc_c, m_exc_c = match_cache.get((pid_idx, cand_id), (0.0, 0.0))
            if m_inc_c >= cfg.HARD_NEG_INC_THRESHOLD and m_exc_c >= cfg.HARD_NEG_EXC_THRESHOLD:
                hard_negs_info.append((cand_id, m_exc_c))
            else:
                rand_negs_ids.append(cand_id)
                
        pair_negatives_map.append({
            'trial_id': trial_id,
            'm_inc_target': m_inc_target,
            'hard_negs': hard_negs_info,
            'rand_negs': rand_negs_ids
        })

    logging.info(f"[Stage B] Starting trial-alignment fine-tuning for {cfg.EPOCHS_ALIGN} epochs...")
    loss_history = []
    
    batch_size = getattr(cfg, 'ALIGN_BATCH_SIZE', 4096)
    num_pairs = len(valid_weak_pairs)

    for epoch in range(1, cfg.EPOCHS_ALIGN + 1):
        encoder.train()
        criterion_encoder.train()
        trial_encoder.train()

        # Forward pass for GNN
        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab') if nt in h_dict}

        # Encode all trials on GPU
        trial_embeds = encode_all_trials(
            list(trial_store), entity_maps, post_gnn_embeddings,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )

        total_epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(num_pairs)
        for b_start in range(0, num_pairs, batch_size):
            optimizer.zero_grad()
            batch_perm = perm[b_start:b_start + batch_size]
            
            batch_patient_idx = patient_indices[batch_perm]
            z_patients = h_dict['patient'][batch_patient_idx]

            batch_losses = []
            for idx_in_batch, original_idx in enumerate(batch_perm.tolist()):
                info = pair_negatives_map[original_idx]
                z_patient = z_patients[idx_in_batch]
                trial_id = info['trial_id']
                m_inc_target = info['m_inc_target']
                z_target_inc, _ = trial_embeds[trial_id]

                hard_negs = [(trial_embeds[cid][1], m_exc) for cid, m_exc in info['hard_negs']]
                
                rand_negs_ids = info['rand_negs']
                if len(rand_negs_ids) > cfg.N_RANDOM_NEGATIVES:
                    sampled_ids = [rand_negs_ids[i] for i in torch.randint(0, len(rand_negs_ids), (cfg.N_RANDOM_NEGATIVES,))]
                else:
                    sampled_ids = rand_negs_ids
                    
                rand_negs = [trial_embeds[cid][0] for cid in sampled_ids]

                loss = align_loss_fn(z_patient, m_inc_target, z_target_inc, hard_negs, rand_negs)
                batch_losses.append(loss)

            if batch_losses:
                # Accumulate batch loss cleanly into single tensor
                avg_batch_loss = torch.stack(batch_losses).mean()
                
                # Single clean backprop without retain_graph
                avg_batch_loss.backward(retain_graph=True if (b_start + batch_size < num_pairs) else False)
                
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(criterion_encoder.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(trial_encoder.parameters(), max_norm=0.5)

                optimizer.step()
                total_epoch_loss += avg_batch_loss.item()
                n_batches += 1

        avg_align_loss = total_epoch_loss / max(1, n_batches)
        
        # Anchor Loss
        anchor_loss = F.mse_loss(h_dict['patient'], h_a_anchor)
        anchor_weight = getattr(cfg, 'LAMBDA_ANCHOR', 0.1)
        composite_loss = avg_align_loss + anchor_weight * anchor_loss

        logging.info(
            f"[Stage B] Epoch {epoch:02d}/{cfg.EPOCHS_ALIGN} | Align Loss: {avg_align_loss:.4f} | "
            f"Anchor Loss: {anchor_loss.item():.4f} | Composite Loss: {composite_loss.item():.4f}"
        )

        loss_history.append({
            'epoch': epoch,
            'align_loss': avg_align_loss,
            'anchor_loss': anchor_loss.item(),
            'total_loss': composite_loss.item()
        })

    with torch.no_grad():
        encoder.eval()
        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab') if nt in h_dict}
        final_trial_embeds = encode_all_trials(
            list(trial_store), entity_maps, post_gnn_embeddings,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )

    return encoder, criterion_encoder, trial_encoder, final_trial_embeds


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

    diag_df, rx_df, labs_df = load_processed_tables(cfg)
    p_map = {str(sid): i for i, sid in enumerate(sorted(diag_df['SUBJECT_ID'].unique(), key=int))}
    d_map = {str(c): i for i, c in enumerate(sorted(diag_df['ICD10_CODE'].unique()))}
    m_map = {str(c): i for i, c in enumerate(sorted(rx_df['NDC'].unique()))}
    l_map = {str(c): i for i, c in enumerate(sorted(labs_df['ITEMID'].unique()))}
    entity_maps = {'diagnosis': d_map, 'medication': m_map, 'lab': l_map, 'procedure': {}}

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