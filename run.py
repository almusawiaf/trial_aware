# run.py
import os
import json
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


def check_and_prepare_trials(cfg: Config):
    """
    Check for trials in the new location and copy them to the current directory
    for backward compatibility.
    """
    train_trials_path = cfg.TRAIN_TRIALS_PATH
    eval_trials_path = cfg.EVAL_TRIALS_PATH
    
    if os.path.exists(train_trials_path) and os.path.exists(eval_trials_path):
        logging.info(f"[Trials] Using existing trials from: {cfg.TRIALS_DATA_DIR}")
        
        with open(train_trials_path, "r") as f:
            train_trials = json.load(f)
        with open(eval_trials_path, "r") as f:
            eval_trials = json.load(f)
        
        logging.info(f"[Trials] Loaded {len(train_trials)} training trials")
        logging.info(f"[Trials] Loaded {len(eval_trials)} evaluation trials")
        
        # Copy to current directory for backward compatibility
        with open("structured_clinical_trials.json", "w") as f:
            json.dump(train_trials, f, indent=2)
        with open("structured_clinical_trials_eval.json", "w") as f:
            json.dump(eval_trials, f, indent=2)
        logging.info("[Trials] Copied trials to current directory for compatibility")
        
        return True
    else:
        logging.warning(f"[Trials] Trials not found at {cfg.TRIALS_DATA_DIR}")
        logging.warning("[Trials] Will generate mock trials for testing...")
        return False


if __name__ == "__main__":
    config = Config()
    config.ensure_directories()
    
    # ============================================================
    # NEW: Check for trials in the new location
    # ============================================================
    logging.info("=" * 50)
    logging.info("CHECKING FOR TRIAL DATA")
    logging.info("=" * 50)
    
    trials_available = check_and_prepare_trials(config)
    
    if not trials_available:
        logging.info("[Trials] No real trials found. Mock trials will be used.")
        # Generate mock trials for testing
        try:
            from mock_data import generate_mock_trials
            mock_trials = generate_mock_trials()
            with open("structured_clinical_trials.json", "w") as f:
                json.dump(mock_trials, f, indent=2)
            logging.info("[Trials] Generated mock trials")
        except ImportError:
            logging.warning("[Trials] Could not import mock_data. Continuing without trials.")
    
    # ============================================================
    # Run preprocessing and graph construction
    # ============================================================
    
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