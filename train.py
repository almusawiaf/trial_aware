# train.py
import logging
import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from alignment import AlignmentLoss, similarity
from config import Config
from gcl_framework import GraphAugmentor, HeteroGNNEncoder, NTXentLoss
from trial_embedding import CriterionEncoder, TrialEncoder, encode_all_trials
from trial_graph import PatientClinicalState, TrialStore, compute_matching_indices, derive_weak_positive_pairs
import json

    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PurePyTorchGraphLoader:
    """
    Extension-free, memory-efficient mini-batch loader for large heterogeneous
    clinical graphs. Uses pure PyTorch index slicing to isolate 1-hop
    neighborhoods without pyg-lib/torch-sparse.

    Note: since Comorbidity edges (patient<->patient) now exist in the graph
    (see graph_constructor.py), this loader's 1-hop expansion also pulls in
    OTHER patients who share diagnoses with the seed batch -- partially
    addressing the "patients can't see each other" limitation flagged in the
    review, without needing a full k-hop sampler.
    """

    def __init__(self, data, batch_size=256, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.patient_nodes = torch.arange(data['patient'].num_nodes)

    def __iter__(self):
        if self.shuffle:
            self.patient_nodes = self.patient_nodes[torch.randperm(len(self.patient_nodes))]
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.patient_nodes):
            raise StopIteration

        batch_patients = self.patient_nodes[self.current_idx: self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        node_idx_dict = {'patient': batch_patients}
        for edge_type in self.data.edge_types:
            src, rel, dst = edge_type
            if src == 'patient':
                edge_index = self.data[edge_type].edge_index
                mask = torch.isin(edge_index[0], batch_patients)
                neighbors = edge_index[1, mask].unique()
                if dst in node_idx_dict:
                    node_idx_dict[dst] = torch.cat([node_idx_dict[dst], neighbors]).unique()
                else:
                    node_idx_dict[dst] = neighbors

        for n_type in self.data.node_types:
            if n_type not in node_idx_dict:
                node_idx_dict[n_type] = torch.tensor([], dtype=torch.long)

        return self.data.subgraph(node_idx_dict)


# =====================================================================
# STAGE A: Self-supervised contrastive pretraining (generic, trial-agnostic)
# =====================================================================
# train.py

def pretrain_contrastive(cfg: Config, base_graph, device) -> HeteroGNNEncoder:
    import torch
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    num_nodes_dict = {nt: base_graph[nt].num_nodes for nt in base_graph.node_types}
    encoder = HeteroGNNEncoder(
        metadata=base_graph.metadata(),
        num_nodes_dict=num_nodes_dict,
        patient_feat_dim=cfg.PATIENT_FEAT_DIM,
        entity_embed_dim=cfg.ENTITY_EMBED_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
    ).to(device)
    
    criterion = NTXentLoss(temperature=cfg.TEMPERATURE, max_batch_size=1024) # Cap max loss computation size
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    loader = PurePyTorchGraphLoader(base_graph, batch_size=cfg.BATCH_SIZE, shuffle=True)

    logging.info(f"[Stage A] Contrastive pretraining for {cfg.EPOCHS_CONTRASTIVE} epochs...")
    for epoch in range(1, cfg.EPOCHS_CONTRASTIVE + 1):
        encoder.train()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            if batch['patient'].num_nodes <= 1:
                continue

            view_1 = GraphAugmentor.drop_edges(batch, drop_rate=cfg.DROP_RATE_V1)
            view_2 = GraphAugmentor.drop_edges(batch, drop_rate=cfg.DROP_RATE_V2)

            optimizer.zero_grad()
            h1 = encoder.encode(view_1.x_dict, view_1.edge_index_dict)
            h2 = encoder.encode(view_2.x_dict, view_2.edge_index_dict)
            z1 = encoder.project(h1['patient'])
            z2 = encoder.project(h2['patient'])

            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Free CUDA memory used by static graph batch references
            del batch, view_1, view_2, h1, h2, z1, z2, loss

        avg_loss = total_loss / n_batches if n_batches else 0.0
        if epoch % 10 == 0 or epoch == 1:
            logging.info(f"[Stage A] Epoch {epoch:03d} | Avg NT-Xent Loss: {avg_loss:.4f}")
        
        torch.cuda.empty_cache()

    return encoder


# =====================================================================
# STAGE B: Trial-aware alignment fine-tuning (the previously-missing piece)
# =====================================================================
def align_with_trials(cfg: Config, base_graph, encoder: HeteroGNNEncoder,
                       trial_store: TrialStore, patient_states, p_map, entity_maps, device):
    criterion_encoder = CriterionEncoder(cfg.OUT_CHANNELS).to(device)
    trial_encoder = TrialEncoder(cfg.OUT_CHANNELS).to(device)
    align_loss_fn = AlignmentLoss(
        lambda_1=cfg.LAMBDA_1, lambda_2=cfg.LAMBDA_2,
        margin_hard=cfg.MARGIN_HARD, margin_rand=cfg.MARGIN_RAND,
    )

    # 1. Store Stage A baseline patient embeddings as static reference anchors
    encoder.eval()
    with torch.no_grad():
        base_graph_dev = base_graph.to(device)
        h_a_anchor = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)['patient'].detach().clone()

    # Use smaller learning rate for GNN encoder to prevent distortion
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': cfg.ALIGN_LR * 0.1},
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

    trial_ids = list(trial_store.trials.keys())
    idx_to_subject = {v: k for k, v in p_map.items()}

    logging.info(f"[Stage B] Trial-alignment fine-tuning with Anchor Regularization for {cfg.EPOCHS_ALIGN} epochs...")

    for epoch in range(1, cfg.EPOCHS_ALIGN + 1):
        encoder.train()
        criterion_encoder.train()
        trial_encoder.train()
        optimizer.zero_grad()

        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab')}

        trial_embeds = encode_all_trials(
            list(trial_store), entity_maps, post_gnn_embeddings,
            criterion_encoder, trial_encoder, cfg.OUT_CHANNELS, device,
        )

        total_align_loss = torch.zeros((), device=device)
        n_terms = 0

        for pid_idx, trial_id in weak_pairs:
            subject_id = idx_to_subject.get(pid_idx)
            state = patient_states.get(int(subject_id)) if subject_id is not None else None
            if state is None:
                continue

            z_patient = h_dict['patient'][pid_idx]
            trial = trial_store[trial_id]
            m_inc_target, _ = compute_matching_indices(state, trial)
            z_target_inc, _ = trial_embeds[trial_id]

            hard_negs, rand_negs = [], []
            other_trials = [t for t in trial_ids if t != trial_id]
            for cand_id in other_trials:
                cand_trial = trial_store[cand_id]
                m_inc_c, m_exc_c = compute_matching_indices(state, cand_trial)
                z_inc_c, z_exc_c = trial_embeds[cand_id]
                if m_inc_c >= cfg.HARD_NEG_INC_THRESHOLD and m_exc_c >= cfg.HARD_NEG_EXC_THRESHOLD:
                    hard_negs.append((z_exc_c, m_exc_c))
                else:
                    rand_negs.append(z_inc_c)

            if len(rand_negs) > cfg.N_RANDOM_NEGATIVES:
                perm = torch.randperm(len(rand_negs))[:cfg.N_RANDOM_NEGATIVES]
                rand_negs = [rand_negs[i] for i in perm]

            loss = align_loss_fn(z_patient, m_inc_target, z_target_inc, hard_negs, rand_negs)
            total_align_loss = total_align_loss + loss
            n_terms += 1

        if n_terms == 0:
            continue

        # Compute average alignment loss + Anchor Regularization penalty
        avg_align_loss = total_align_loss / n_terms
        anchor_loss = F.mse_loss(h_dict['patient'], h_a_anchor)
        
        composite_loss = avg_align_loss + getattr(cfg, 'LAMBDA_ANCHOR', 0.5) * anchor_loss
        
        composite_loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            logging.info(
                f"[Stage B] Epoch {epoch:02d} | Align Loss: {avg_align_loss.item():.4f} | "
                f"Anchor Loss: {anchor_loss.item():.4f} | Total Loss: {composite_loss.item():.4f}"
            )

    with torch.no_grad():
        encoder.eval()
        h_dict = encoder.encode(base_graph_dev.x_dict, base_graph_dev.edge_index_dict)
        post_gnn_embeddings = {nt: h_dict[nt] for nt in ('diagnosis', 'medication', 'lab')}
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


def build_patient_states(diag_df, rx_df, labs_df):
    states = {}
    for sid in diag_df['SUBJECT_ID'].unique():
        states[int(sid)] = PatientClinicalState.build_from_tables(int(sid), diag_df, rx_df, labs_df)
    return states


def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using execution device: {device}")

    if not os.path.exists(cfg.GRAPH_PATH):
        raise FileNotFoundError(f"Compiled graph not found at {cfg.GRAPH_PATH}. Run run.py first.")

    base_graph = torch.load(cfg.GRAPH_PATH, map_location='cpu', weights_only=False)
    base_graph = T.ToUndirected()(base_graph)

    # ---------------- Stage A ----------------
    encoder = pretrain_contrastive(cfg, base_graph, device)

    # Save Stage-A (trial-agnostic) embeddings as the "Baseline-GCL" arm.
    encoder.eval()
    with torch.no_grad():
        h_dict = encoder.encode(base_graph.to(device).x_dict, base_graph.to(device).edge_index_dict)
    torch.save(h_dict['patient'].cpu(), cfg.PATIENT_EMBED_PATH.replace(".pt", "_baseline.pt"))
    logging.info(f"[Stage A] Saved baseline (pre-projection, trial-agnostic) patient embeddings.")

    # ---------------- Stage B ----------------
    from graph_constructor import MIMICGraphConstructor

    # entity maps are needed to align trial criteria codes with graph node
    # indices; rebuild the constructor's maps from the saved graph node
    # ordering (in a full pipeline these maps should be persisted directly
    # by graph_constructor.py during Phase 2 rather than recomputed here).
    diag_df, rx_df, labs_df = load_processed_tables(cfg)
    p_map = {str(sid): i for i, sid in enumerate(sorted(diag_df['SUBJECT_ID'].unique(), key=int))}
    d_map = {str(c): i for i, c in enumerate(sorted(diag_df['ICD10_CODE'].unique()))}
    m_map = {str(c): i for i, c in enumerate(sorted(rx_df['NDC'].unique()))}
    l_map = {str(c): i for i, c in enumerate(sorted(labs_df['ITEMID'].unique()))}
    entity_maps = {'diagnosis': d_map, 'medication': m_map, 'lab': l_map}

    patient_states = build_patient_states(diag_df, rx_df, labs_df)

    with open("structured_clinical_trials.json", "r") as f:
        real_trials_data = json.load(f)

    trial_store = TrialStore.from_records(real_trials_data)
    logging.info(f"Loaded {len(trial_store.trials)} REAL clinical trials into Stage B!")

    encoder, criterion_encoder, trial_encoder, trial_embeds = align_with_trials(
        cfg, base_graph, encoder, trial_store, patient_states, p_map, entity_maps, device
    )

    if trial_embeds:
        torch.save(
            {tid: (z_inc.detach().cpu(), z_exc.detach().cpu()) for tid, (z_inc, z_exc) in trial_embeds.items()},
            cfg.TRIAL_EMBED_PATH,
        )
        logging.info(f"[Stage B] Saved trial embeddings (z_inc, z_exc) to {cfg.TRIAL_EMBED_PATH}")

    # Final trial-aware patient embeddings ("Full Model" arm).
    encoder.eval()
    with torch.no_grad():
        h_dict = encoder.encode(base_graph.to(device).x_dict, base_graph.to(device).edge_index_dict)
    torch.save(h_dict['patient'].cpu(), cfg.PATIENT_EMBED_PATH)
    logging.info(f"[Stage B] Saved trial-aware patient embeddings to {cfg.PATIENT_EMBED_PATH}")

    # Example ranking using the composite Similarity score.
    if trial_embeds:
        example_trial_id = next(iter(trial_embeds))
        z_inc, z_exc = trial_embeds[example_trial_id]
        scores = [
            (idx, similarity(h_dict['patient'][idx].cpu(), z_inc.detach().cpu(),
                              z_exc.detach().cpu(), eta=cfg.ETA_EXCLUSION_PENALTY).item())
            for idx in range(min(10, h_dict['patient'].size(0)))
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"[Demo] Top-ranked (patient_idx, score) for trial {example_trial_id}: {scores[:5]}")


if __name__ == "__main__":
    main()
