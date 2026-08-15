"""
build_pyg_graph.py
-------------------
Converts structured_clinical_trials.json into PyTorch Geometric HeteroData.
Compatible with HeteroGNNEncoder (nn.Embedding index lookup).
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON dataset not found at path: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hetero_graph(trials_data: List[Dict[str, Any]]) -> HeteroData:
    data = HeteroData()

    # ID maps: convert string identifiers to 0-indexed integer IDs
    trial_id_map: Dict[str, int] = {}
    concept_id_maps: Dict[str, Dict[str, int]] = {
        "diagnosis": {},
        "lab": {},
        "procedure": {},
        "prescription": {},
        "administrative": {},
    }

    # Edges lookup table: (concept_type, is_inclusion) -> [[src_trials], [dst_concepts]]
    edge_stores: Dict[Tuple[str, bool], Tuple[List[int], List[int]]] = {}

    def get_concept_idx(c_type: str, c_code: str) -> Tuple[str, int]:
        c_type = c_type.lower()
        if c_type not in concept_id_maps:
            c_type = "diagnosis"  # Fallback type

        mapping = concept_id_maps[c_type]
        if c_code not in mapping:
            mapping[c_code] = len(mapping)
        return c_type, mapping[c_code]

    # 1. Parse Trial Criteria into Node Indices and Relational Edges
    for trial_idx, trial in enumerate(trials_data):
        nct_id = trial.get("nct_id", f"TRIAL_{trial_idx}")
        if nct_id not in trial_id_map:
            trial_id_map[nct_id] = len(trial_id_map)
        
        t_idx = trial_id_map[nct_id]

        for criterion in trial.get("criteria", []):
            c_type_raw = criterion.get("entity_type", "diagnosis")
            c_code = criterion.get("entity_code", "UNKNOWN_CODE")
            is_inc = criterion.get("is_inclusion", True)

            c_type, concept_idx = get_concept_idx(c_type_raw, c_code)
            edge_key = (c_type, is_inc)

            if edge_key not in edge_stores:
                edge_stores[edge_key] = ([], [])

            edge_stores[edge_key][0].append(t_idx)
            edge_stores[edge_key][1].append(concept_idx)

    # 2. Register Nodes and Build Integer Tensor Feature IDs
    data["trial"].num_nodes = len(trial_id_map)
    data["trial"].x = torch.arange(len(trial_id_map), dtype=torch.long).unsqueeze(-1)

    for c_type, mapping in concept_id_maps.items():
        if len(mapping) > 0:
            data[c_type].num_nodes = len(mapping)
            # Assign integer tensor indices matching GNN nn.Embedding expectations
            data[c_type].x = torch.arange(len(mapping), dtype=torch.long).unsqueeze(-1)

    # 3. Construct Edges
    for (c_type, is_inc), (src_list, dst_list) in edge_stores.items():
        if len(src_list) > 0:
            rel_name = "requires_inclusion" if is_inc else "requires_exclusion"
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            data["trial", rel_name, c_type].edge_index = edge_index

    # 4. Make Graph Bi-directional for GNN Message Passing
    data = T.ToUndirected()(data)

    return data


if __name__ == "__main__":
    json_path = "structured_clinical_trials.json"
    logging.info(f"Loading trial extraction dataset from: {json_path}")
    trials = load_json_data(json_path)

    logging.info("Building PyTorch Geometric HeteroData Graph Object...")
    graph = build_hetero_graph(trials)

    print("\n" + "=" * 55)
    print("✅ HeteroData Graph Successfully Constructed!")
    print("=" * 55)
    print(graph)