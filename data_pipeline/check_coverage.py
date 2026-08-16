"""
check_coverage.py

Simple diagnostic: for every criterion in every trial, check whether its
entity_code (diagnosis / medication / lab) exists in the vocabulary built
from your MIMIC data. If coverage is low, that alone explains collapsed
z_inc/z_exc embeddings and near-chance AUC -- no amount of hyperparameter
tuning fixes a missing-vocabulary problem.

Run this BEFORE re-running the full pipeline:
    python check_coverage.py
"""

import json
import logging
import re
from collections import Counter

import pandas as pd

from config import Config
from models.claude_active.trial_graph import TrialStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    cfg = Config()

    # Build the SAME entity_maps that train.py builds
    diag_df = pd.read_parquet(f"{cfg.OUTPUT_DIR}/diagnoses_clean.parquet")
    rx_df = pd.read_parquet(f"{cfg.OUTPUT_DIR}/prescriptions_clean.parquet")
    labs_df = pd.read_parquet(f"{cfg.OUTPUT_DIR}/labs_clean.parquet")

    entity_maps = {
        'diagnosis': {str(c): i for i, c in enumerate(sorted(diag_df['ICD10_CODE'].unique()))},
        'medication': {str(c): i for i, c in enumerate(sorted(rx_df['NDC'].unique()))},
        'lab': {str(c): i for i, c in enumerate(sorted(labs_df['ITEMID'].unique()))},
    }

    with open(cfg.TRAIN_TRIALS_PATH, "r") as f:
        trials_data = json.load(f)
    trial_store = TrialStore.from_records(trials_data)

    hits = Counter()
    total = Counter()
    placeholders = Counter()
    misses_sample = {et: [] for et in entity_maps}

    def normalize(entity_type, code):
        c = str(code).strip()
        if entity_type == 'diagnosis':
            c = c.replace('.', '').upper()  # matches the crosswalk's dot-stripping
        elif entity_type == 'medication':
            c = re.sub(r'\.0+$', '', c)
            c = re.sub(r'[^0-9]', '', c)
            if c.isdigit() and len(c) < 11:
                c = c.zfill(11)
        return c

    for t in trial_store:
        for c in list(t.inclusion_criteria) + list(t.exclusion_criteria):
            et, raw_code = c.entity_type, str(c.entity_code)
            if et not in entity_maps:
                continue  # e.g. "administrative" criteria -- not a vocab code at all
            total[et] += 1
            if raw_code.startswith('UNMATCHED_') or raw_code.strip() == '' or raw_code.lower() == 'none':
                placeholders[et] += 1
                continue
            code = normalize(et, raw_code)
            if code in entity_maps[et]:
                hits[et] += 1
            elif len(misses_sample[et]) < 10:
                misses_sample[et].append(raw_code)

    logging.info("=" * 60)
    logging.info("CONCEPT COVERAGE REPORT (after dot-normalization fix)")
    logging.info("=" * 60)
    grand_hits, grand_total, grand_placeholder = 0, 0, 0
    for et in entity_maps:
        h, tt, ph = hits[et], total[et], placeholders[et]
        grand_hits += h
        grand_total += tt
        grand_placeholder += ph
        real_total = tt - ph  # codes that were at least a real attempt, not a placeholder
        pct = (h / real_total * 100) if real_total else 0.0
        ph_pct = (ph / tt * 100) if tt else 0.0
        logging.info(f"  {et:12s}: {h:6d}/{real_total:6d} of real codes matched ({pct:5.1f}%)  "
                     f"| {ph:6d}/{tt:6d} were placeholders/unresolved upstream ({ph_pct:5.1f}%)")
        if misses_sample[et]:
            logging.info(f"    example UNMATCHED real codes: {misses_sample[et]}")
            logging.info(f"    example codes THAT DO EXIST in your vocab: "
                         f"{list(entity_maps[et].keys())[:10]}")

    real_grand_total = grand_total - grand_placeholder
    overall_pct = (grand_hits / real_grand_total * 100) if real_grand_total else 0.0
    ph_overall_pct = (grand_placeholder / grand_total * 100) if grand_total else 0.0
    logging.info("-" * 60)
    logging.info(f"  OVERALL (of codes that were real attempts): {grand_hits}/{real_grand_total} matched ({overall_pct:.1f}%)")
    logging.info(f"  OVERALL placeholders/unresolved upstream: {grand_placeholder}/{grand_total} ({ph_overall_pct:.1f}%)")
    logging.info("=" * 60)

    if overall_pct < 50:
        logging.warning(
            "Coverage of REAL codes is still under 50%. Compare the 'UNMATCHED' vs "
            "'exists in your vocab' code samples above for a remaining formatting difference."
        )
    else:
        logging.info("Coverage of real codes looks reasonable now.")
    if ph_overall_pct > 20:
        logging.warning(
            f"{ph_overall_pct:.1f}% of all criteria are placeholders (never had a real code at "
            "all) -- this needs to be fixed in whatever script produced structured_clinical_trials.json, "
            "not here. The pipeline now excludes these from both training and evaluation labels "
            "rather than silently scoring them as failures."
        )


if __name__ == "__main__":
    main()