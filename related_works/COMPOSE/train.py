"""
train.py -- trains the COMPOSE-adaptation baseline.

Usage:
    cd related_works/COMPOSE
    python train.py

Reads:
    <repo_root>/data/diagnoses_clean.parquet
    <repo_root>/data/prescriptions_clean.parquet
    <repo_root>/data/labs_clean.parquet
    <repo_root>/data/10000_trials/structured_clinical_trials.json

Writes:
    related_works/COMPOSE/checkpoints/compose_seed{SEED}.pt   (model weights + vocabs)
    related_works/COMPOSE/logs/train_seed{SEED}.log
"""
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from vocab import build_vocabs
from data_utils import (
    load_patient_tables, load_trial_store, get_subject_ids,
    build_patient_states, build_visit_sequences, build_training_triples, split_patients,
)
from dataset import ComposeDataset, collate_fn
from model import ComposeModel, get_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_ce, total_sim, n_batches = 0.0, 0.0, 0.0, 0
    correct, n_seen = 0, 0

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        with torch.set_grad_enabled(train_mode):
            output, response, query, attention = model(batch, batch["entity_type"])
            total, ce, sim, pred = get_loss(output, response, query, batch["label"], device)

            if train_mode:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

        total_loss += total.item()
        total_ce += ce.item()
        total_sim += sim.item()
        n_batches += 1

        pred_label = pred.argmax(dim=-1)
        correct += (pred_label == batch["label"]).sum().item()
        n_seen += batch["label"].size(0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "ce": total_ce / max(n_batches, 1),
        "sim": total_sim / max(n_batches, 1),
        "acc": correct / max(n_seen, 1),
    }


def main():
    config.ensure_directories()
    set_seed(config.SEED)

    log_path = os.path.join(config.LOG_DIR, f"train_seed{config.SEED}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Device: {config.DEVICE}")

    # 1. Load shared data ------------------------------------------------
    diag_df, rx_df, labs_df = load_patient_tables()
    trial_store = load_trial_store()
    subject_ids = get_subject_ids(diag_df)
    logging.info(f"Loaded {len(subject_ids)} patients and {len(trial_store)} trials.")

    # 2. Vocab + visit sequences -----------------------------------------
    vocabs = build_vocabs(diag_df, rx_df, labs_df, trial_store)
    torch.save(vocabs, config.VOCAB_PATH)

    visit_sequences = build_visit_sequences(subject_ids, diag_df, rx_df, labs_df)
    patient_states = build_patient_states(subject_ids, diag_df, rx_df, labs_df)

    # 3. Split patients, build training triples ---------------------------
    train_ids, val_ids, test_ids = split_patients(subject_ids, config.SEED)
    with open(os.path.join(config.OUT_DIR, f"patient_split_seed{config.SEED}.txt"), "w") as f:
        f.write("train:" + ",".join(map(str, train_ids)) + "\n")
        f.write("val:" + ",".join(map(str, val_ids)) + "\n")
        f.write("test:" + ",".join(map(str, test_ids)) + "\n")

    rng = random.Random(config.SEED)
    train_triples = build_training_triples(train_ids, patient_states, trial_store, rng)
    val_triples = build_training_triples(val_ids, patient_states, trial_store, rng)

    train_ds = ComposeDataset(train_triples, visit_sequences, vocabs)
    val_ds = ComposeDataset(val_triples, visit_sequences, vocabs)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Model + optimizer --------------------------------------------------
    model = ComposeModel(vocabs).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, config.EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, config.DEVICE, optimizer)
        val_stats = run_epoch(model, val_loader, config.DEVICE, optimizer=None)

        logging.info(
            f"[Epoch {epoch:03d}] train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )

        if val_stats["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_stats["loss"]
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "vocabs": vocabs,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, config.CKPT_PATH)
            logging.info(f"  -> saved new best checkpoint to {config.CKPT_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                logging.info(f"Early stopping at epoch {epoch} (no val improvement for "
                             f"{config.EARLY_STOP_PATIENCE} epochs).")
                break

    logging.info("Training complete.")


if __name__ == "__main__":
    main()
