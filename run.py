# run.py
import os
import logging
import pandas as pd
import torch

from config import Config
from preprocessor import MIMICDataPreprocessor
from graph_constructor import MIMICGraphConstructor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_mimic_raw_data(data_dir: str):
    logging.info(f"Loading raw MIMIC-III files from {data_dir}...")
    try:
        admissions = pd.read_csv(os.path.join(data_dir, 'ADMISSIONS.csv'))
        diagnoses = pd.read_csv(os.path.join(data_dir, 'DIAGNOSES_ICD.csv'))
        prescriptions = pd.read_csv(os.path.join(data_dir, 'PRESCRIPTIONS.csv'))
        labevents = pd.read_csv(os.path.join(data_dir, 'LABEVENTS.csv'))
        logging.info("Raw CSV data read completed successfully!")
        return admissions, diagnoses, prescriptions, labevents
    except FileNotFoundError as e:
        logging.error(f"Failed to find raw CSV files: {e}")
        raise e


def save_processed_data(adm, diag, rx, labs, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving processed datasets to: {output_dir}")
    try:
        adm.to_parquet(os.path.join(output_dir, "admissions_clean.parquet"), index=False)
        diag.to_parquet(os.path.join(output_dir, "diagnoses_clean.parquet"), index=False)
        rx.to_parquet(os.path.join(output_dir, "prescriptions_clean.parquet"), index=False)
        labs.to_parquet(os.path.join(output_dir, "labs_clean.parquet"), index=False)
        logging.info("Parquet files successfully saved!")
    except ImportError:
        logging.warning("Parquet engines not found. Falling back to CSVs (slower)...")
        adm.to_csv(os.path.join(output_dir, "admissions_clean.csv"), index=False)
        diag.to_csv(os.path.join(output_dir, "diagnoses_clean.csv"), index=False)
        rx.to_csv(os.path.join(output_dir, "prescriptions_clean.csv"), index=False)
        labs.to_csv(os.path.join(output_dir, "labs_clean.csv"), index=False)


if __name__ == "__main__":
    config = Config()
    preprocessor = MIMICDataPreprocessor(config)
    graph_constructor = MIMICGraphConstructor(config)

    adm_raw, diag_raw, rx_raw, labs_raw = load_mimic_raw_data(config.DATA_DIR)

    logging.info("\n" + "=" * 50 + "\nRUNNING PHASE 1: PREPROCESSING PIPELINE\n" + "=" * 50)
    adm_clean = preprocessor.filter_cohort(adm_raw)
    diag_clean = preprocessor.process_diagnoses(diag_raw)
    rx_clean = preprocessor.process_prescriptions(rx_raw)
    labs_clean = preprocessor.process_labs(labs_raw)
    logging.info("PHASE 1 COMPLETED SUCCESSFULLY")

    save_processed_data(adm_clean, diag_clean, rx_clean, labs_clean, config.OUTPUT_DIR)

    logging.info("\n" + "=" * 50 + "\nRUNNING PHASE 2: GRAPH CONSTRUCTION PIPELINE\n" + "=" * 50)
    hetero_graph = graph_constructor.construct_graph(adm_clean, diag_clean, rx_clean, labs_clean)

    graph_save_path = config.GRAPH_PATH
    logging.info(f"Saving compiled heterogeneous graph to: {graph_save_path}")
    torch.save(hetero_graph, graph_save_path)
    logging.info("PHASE 2 COMPLETED SUCCESSFULLY")

    print("\n[Graph Details] Total Nodes of Each Group:")
    for node_type in hetero_graph.node_types:
        print(f"  - Node Type '{node_type}': {hetero_graph[node_type].num_nodes} nodes mapped.")

    print("\n[Graph Details] Total Edges and Tensor Shapes:")
    for edge_type in hetero_graph.edge_types:
        edge_index_shape = hetero_graph[edge_type].edge_index.shape
        print(f"  - Relation {edge_type}: Edge list matrix shape is {list(edge_index_shape)}")
        if 'edge_attr' in hetero_graph[edge_type]:
            attr_shape = hetero_graph[edge_type].edge_attr.shape
            print(f"    * edge_attr shape: {list(attr_shape)}")

    print("\nNext step: run `python train.py` for Stage A (contrastive pretraining) "
          "+ Stage B (trial-aware alignment).")
