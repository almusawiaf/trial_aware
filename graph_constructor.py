import itertools
import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from config import Config


class MIMICGraphConstructor:
    """
    Transforms cleaned clinical dataframes into a PyTorch Geometric HeteroData object.

    Changes vs. the original version (aligning the code with the finalized LaTeX
    methodology and fixing issues raised in review):

      1. Medication edges are now diagnosis-mediated (diagnosis -> medication),
         matching the paper's explicit design choice and its stated rationale,
         instead of a direct patient -> medication edge. The diagnosis and
         prescription tables are bridged via shared HADM_ID (same admission).
      2. Patient-patient `Comorbidity` edges are added (E_Comorbidity in the
         paper's edge schema), connecting patients who share >= K diagnosis
         codes. This also mitigates the "1-hop loader can't see other patients"
         limitation noted in the review, since patients are now *directly*
         linked to each other rather than only reachable through shared
         entity nodes several hops away.
      3. Diagnosis / medication / lab node features are stored as integer
         indices (not one-hot vectors). The encoder (see gcl_framework.py)
         looks these up through nn.Embedding tables, which is dramatically
         cheaper for real vocabularies (tens of thousands of NDC codes)
         than a dense one-hot + linear layer.
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.maps: Dict[str, Dict[str, int]] = {
            'patient': {}, 'diagnosis': {}, 'medication': {}, 'lab': {}
        }

    def _create_mapping(self, unique_elements: list, node_type: str) -> Dict[str, int]:
        mapping = {str(item): idx for idx, item in enumerate(sorted(unique_elements))}
        self.maps[node_type] = mapping
        return mapping

    def construct_graph(self,
                         adm_df: pd.DataFrame,
                         diag_df: pd.DataFrame,
                         rx_df: pd.DataFrame,
                         labs_df: pd.DataFrame) -> HeteroData:
        logging.info("Starting PyTorch Geometric Heterogeneous Graph Construction...")
        data = HeteroData()

        # 1. Node ID mappings ------------------------------------------------
        patient_ids = adm_df['SUBJECT_ID'].unique()
        diag_ids = diag_df['ICD10_CODE'].unique()
        med_ids = rx_df['NDC'].unique()
        lab_ids = labs_df['ITEMID'].unique()

        p_map = self._create_mapping(patient_ids, 'patient')
        d_map = self._create_mapping(diag_ids, 'diagnosis')
        m_map = self._create_mapping(med_ids, 'medication')
        l_map = self._create_mapping(lab_ids, 'lab')

        # 2. Node counts + index-based features ------------------------------
        # Patient features stay as a *learnable* embedding table indexed by id
        # (looked up inside the encoder); diagnosis/medication/lab likewise.
        data['patient'].num_nodes = len(p_map)
        data['diagnosis'].num_nodes = len(d_map)
        data['medication'].num_nodes = len(m_map)
        data['lab'].num_nodes = len(l_map)

        data['patient'].x = torch.arange(len(p_map), dtype=torch.long).unsqueeze(-1)
        data['diagnosis'].x = torch.arange(len(d_map), dtype=torch.long).unsqueeze(-1)
        data['medication'].x = torch.arange(len(m_map), dtype=torch.long).unsqueeze(-1)
        data['lab'].x = torch.arange(len(l_map), dtype=torch.long).unsqueeze(-1)

        # 3. Patient -> Diagnosis edges --------------------------------------
        logging.info("[Edges] Mapping Patient-Diagnosis links...")
        p_idx = diag_df['SUBJECT_ID'].astype(str).map(p_map).values
        d_idx = diag_df['ICD10_CODE'].astype(str).map(d_map).values
        data['patient', 'exhibits', 'diagnosis'].edge_index = torch.tensor(
            np.array([p_idx, d_idx]), dtype=torch.long
        )

        # 4. Diagnosis -> Medication edges (diagnosis-mediated, per the paper) -
        # Bridge diagnoses and prescriptions that occurred within the same
        # admission (HADM_ID) -- there is no direct diagnosis<->drug link in
        # the raw MIMIC tables, so co-occurrence within an admission is used
        # as the linking signal.
        logging.info("[Edges] Mapping Diagnosis-Medication links via shared admission...")
        dx_by_hadm = diag_df[['HADM_ID', 'ICD10_CODE']].drop_duplicates()
        rx_by_hadm = rx_df[['HADM_ID', 'NDC']].drop_duplicates()
        bridged = dx_by_hadm.merge(rx_by_hadm, on='HADM_ID', how='inner')

        dx_idx = bridged['ICD10_CODE'].astype(str).map(d_map).values
        rx_idx = bridged['NDC'].astype(str).map(m_map).values
        valid_mask = pd.notna(dx_idx) & pd.notna(rx_idx)
        data['diagnosis', 'prescribed_for', 'medication'].edge_index = torch.tensor(
            np.array([dx_idx[valid_mask], rx_idx[valid_mask]]), dtype=torch.long
        )

        # 5. Patient -> Lab edges (with temporal-decayed attributes) ---------
        logging.info("[Edges] Mapping Patient-Lab links with temporal attributes...")
        p_idx_lab = labs_df['SUBJECT_ID'].astype(str).map(p_map).values
        l_idx_lab = labs_df['ITEMID'].astype(str).map(l_map).values
        data['patient', 'undergoes', 'lab'].edge_index = torch.tensor(
            np.array([p_idx_lab, l_idx_lab]), dtype=torch.long
        )
        lab_features = labs_df[['IMPUTED_VALUE_DECAYED', 'DAYS_SINCE_MEASURED']].values
        data['patient', 'undergoes', 'lab'].edge_attr = torch.tensor(lab_features, dtype=torch.float)

        # 6. Patient <-> Patient Comorbidity edges ---------------------------
        logging.info("[Edges] Deriving patient-patient Comorbidity links...")
        data['patient', 'comorbid_with', 'patient'].edge_index = self._build_comorbidity_edges(diag_df, p_map)

        logging.info("Graph Compilation Complete!")
        self._log_summary(data)
        return data

    def _build_comorbidity_edges(self, diag_df: pd.DataFrame, p_map: Dict[str, int]) -> torch.Tensor:
        """
        Connects two patients if they share at least COMORBIDITY_MIN_SHARED_DX
        diagnosis codes. Building this via all pairwise patient comparisons is
        O(P^2); instead we group patients by diagnosis code and only form pairs
        within each diagnosis group (capped in size to bound combinatorics),
        then keep pairs that recur across >= MIN_SHARED_DX distinct diagnoses.
        """
        pair_counts: Dict[tuple, int] = {}
        grouped = diag_df.groupby('ICD10_CODE')['SUBJECT_ID'].unique()

        for _, patients in grouped.items():
            if len(patients) < 2:
                continue
            if len(patients) > self.cfg.COMORBIDITY_MAX_GROUP_SIZE:
                # extremely common diagnoses (e.g. hypertension) are not
                # informative for comorbidity linkage and are skipped to
                # avoid an O(n^2) blowup on a group of thousands of patients
                continue
            for a, b in itertools.combinations(sorted(patients), 2):
                key = (a, b)
                pair_counts[key] = pair_counts.get(key, 0) + 1

        src, dst = [], []
        for (a, b), count in pair_counts.items():
            if count >= self.cfg.COMORBIDITY_MIN_SHARED_DX:
                a_idx, b_idx = p_map.get(str(a)), p_map.get(str(b))
                if a_idx is None or b_idx is None:
                    continue
                # undirected -> add both directions explicitly
                src.extend([a_idx, b_idx])
                dst.extend([b_idx, a_idx])

        if not src:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(np.array([src, dst]), dtype=torch.long)

    @staticmethod
    def _log_summary(data: HeteroData):
        for ntype in data.node_types:
            logging.info(f"  Node '{ntype}': {data[ntype].num_nodes} nodes")
        for etype in data.edge_types:
            logging.info(f"  Edge {etype}: {data[etype].edge_index.shape[1]} edges")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from preprocessor import MIMICDataPreprocessor
    from run import load_mimic_raw_data

    cfg = Config()
    preprocessor = MIMICDataPreprocessor(cfg)
    adm_r, diag_r, rx_r, labs_r = load_mimic_raw_data(Config.DATA_DIR)

    a_c = preprocessor.filter_cohort(adm_r)
    d_c = preprocessor.process_diagnoses(diag_r)
    m_c = preprocessor.process_prescriptions(rx_r)
    l_c = preprocessor.process_labs(labs_r)

    constructor = MIMICGraphConstructor(cfg)
    hetero_graph = constructor.construct_graph(a_c, d_c, m_c, l_c)
    print(hetero_graph)
