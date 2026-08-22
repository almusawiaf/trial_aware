"""
diagnose_entity_resolution.py

Quantifies the "resolved vs UNMATCHED rate" that the paper's own
methodology section (dataset characterization) says needs reporting, and
was never actually measured. Motivated by a specific finding: 45.7% of
held-out trials had zero-norm inclusion embeddings and 35.5% had zero-norm
exclusion embeddings (see session notes) -- meaning a majority of trials
have criteria that never resolve to any usable code at all, which would
sabotage any fix to the alignment/training mechanism regardless of how
well-tuned it is, since the underlying data being trained/evaluated on is
partly empty.

Reuses the EXACT production functions (is_resolvable_code,
normalize_entity_code from trial_embedding.py) rather than reimplementing
similar logic, so this diagnostic's numbers are guaranteed to match what
train.py/evaluate.py actually do -- not an approximation.

For every criterion in every trial (both train and eval trial sets), this
classifies it into exactly one of:
  - RESOLVED             : has a real code, found in entity_maps
  - UNMATCHED_PLACEHOLDER: upstream extraction (criteria_parser.py) never
                            found a code at all -- these are not failed
                            matches, they're missing data from an earlier
                            pipeline stage
  - OOV                   : has a real-looking code, but it's not in
                            entity_maps for this entity_type -- either the
                            code genuinely doesn't appear in this patient
                            population, or normalize_entity_code()'s
                            cleanup heuristics didn't recover the match

Usage:
    python diagnose_entity_resolution.py
    python diagnose_entity_resolution.py --trials-file eval   # or "train", or "both" (default)
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # this script lives in models/claude_active/ itself

from config import Config
from trial_graph import TrialStore
from trial_embedding import is_resolvable_code, normalize_entity_code
from entity_alignment import DiagnosisAligner, synonym_to_icd, icd_category

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_processed_tables(cfg: Config):
    paths = {
        'diag': os.path.join(cfg.OUTPUT_DIR, "diagnoses_clean.parquet"),
        'rx': os.path.join(cfg.OUTPUT_DIR, "prescriptions_clean.parquet"),
        'labs': os.path.join(cfg.OUTPUT_DIR, "labs_clean.parquet"),
    }
    return (pd.read_parquet(paths['diag']), pd.read_parquet(paths['rx']), pd.read_parquet(paths['labs']))


def classify_criterion(entity_type: str, entity_code: str, entity_maps: dict) -> str:
    code = str(entity_code)
    if code.startswith('UNMATCHED_') or code.strip() == '' or code.lower() == 'none':
        return 'UNMATCHED_PLACEHOLDER'
    if is_resolvable_code(entity_type, entity_code, entity_maps):
        return 'RESOLVED'
    return 'OOV'


def classify_with_alignment(entity_type: str, entity_code: str, entity_maps: dict,
                            dx_aligner: DiagnosisAligner) -> str:
    """
    Same as classify_criterion, but applies the entity_alignment fixes to see
    how many currently-OOV criteria WOULD resolve with hierarchical diagnosis
    matching. Medications need the NDC->ingredient crosswalk (not available in
    this diagnostic without the upstream map), so this measures the diagnosis
    recovery specifically -- the largest category.
    """
    code = str(entity_code)
    if code.startswith('UNMATCHED_') or code.strip() == '' or code.lower() == 'none':
        return 'UNMATCHED_PLACEHOLDER'
    if is_resolvable_code(entity_type, entity_code, entity_maps):
        return 'RESOLVED'
    # currently OOV -- would the aligner recover it?
    if entity_type == 'diagnosis' and dx_aligner.resolve(entity_code) is not None:
        return 'RESOLVED_VIA_ALIGNMENT'
    return 'OOV'


def diagnose(trial_store: TrialStore, entity_maps: dict, label: str):
    trials = list(trial_store)
    logging.info(f"\n{'=' * 70}\nDIAGNOSTIC: {label} ({len(trials)} trials)\n{'=' * 70}")

    # Build the hierarchical diagnosis aligner from the patient vocabulary,
    # so we can measure how many OOV diagnosis criteria it would recover.
    patient_dx_codes = set(entity_maps.get('diagnosis', {}).keys())
    dx_aligner = DiagnosisAligner(patient_dx_codes)

    # Per-criterion breakdown, split by entity_type and inclusion/exclusion
    counts = defaultdict(Counter)  # counts[(entity_type, inc_or_exc)][status] = n
    oov_examples = defaultdict(list)  # (entity_type) -> list of example OOV codes, capped

    for t in trials:
        for c in t.inclusion_criteria:
            status = classify_criterion(c.entity_type, c.entity_code, entity_maps)
            counts[(c.entity_type, 'inclusion')][status] += 1
            if status == 'OOV' and len(oov_examples[c.entity_type]) < 15:
                oov_examples[c.entity_type].append(
                    f"{c.entity_code} -> normalized: {normalize_entity_code(c.entity_type, c.entity_code)}"
                )
        for c in t.exclusion_criteria:
            status = classify_criterion(c.entity_type, c.entity_code, entity_maps)
            counts[(c.entity_type, 'exclusion')][status] += 1
            if status == 'OOV' and len(oov_examples[c.entity_type]) < 15:
                oov_examples[c.entity_type].append(
                    f"{c.entity_code} -> normalized: {normalize_entity_code(c.entity_type, c.entity_code)}"
                )

    logging.info(f"\n{'Entity type':<12} {'Side':<10} {'RESOLVED':>10} {'OOV':>8} {'UNMATCHED':>10} {'Total':>8} {'Resolved %':>11}")
    grand_resolved = grand_total = 0
    for (etype, side), c in sorted(counts.items()):
        total = sum(c.values())
        resolved = c['RESOLVED']
        grand_resolved += resolved
        grand_total += total
        pct = 100 * resolved / total if total else float('nan')
        logging.info(f"{etype:<12} {side:<10} {resolved:>10} {c['OOV']:>8} {c['UNMATCHED_PLACEHOLDER']:>10} {total:>8} {pct:>10.1f}%")

    overall_pct = 100 * grand_resolved / grand_total if grand_total else float('nan')
    logging.info(f"\n{'OVERALL':<23} {'':>10} {'':>8} {'':>10} {grand_total:>8} {overall_pct:>10.1f}%  <-- headline resolved-vs-UNMATCHED rate")

    # Per-trial: how many trials end up with EMPTY inclusion or exclusion sets
    zero_inc = zero_exc = both_zero = 0
    for t in trials:
        inc_resolved = sum(1 for c in t.inclusion_criteria
                            if classify_criterion(c.entity_type, c.entity_code, entity_maps) == 'RESOLVED')
        exc_resolved = sum(1 for c in t.exclusion_criteria
                            if classify_criterion(c.entity_type, c.entity_code, entity_maps) == 'RESOLVED')
        if inc_resolved == 0:
            zero_inc += 1
        if exc_resolved == 0:
            zero_exc += 1
        if inc_resolved == 0 and exc_resolved == 0:
            both_zero += 1

    n = len(trials)
    logging.info(f"\nTrials with ZERO resolved inclusion criteria: {zero_inc}/{n} ({zero_inc/n:.1%})")
    logging.info(f"Trials with ZERO resolved exclusion criteria: {zero_exc}/{n} ({zero_exc/n:.1%})")
    logging.info(f"Trials with BOTH zero (contribute no signal at all): {both_zero}/{n} ({both_zero/n:.1%})")

    # --- Measure improvement from hierarchical diagnosis alignment ----------
    dx_total = dx_resolved_now = dx_resolved_aligned = 0
    zero_inc_aligned = zero_exc_aligned = both_zero_aligned = 0
    for t in trials:
        inc_ok = exc_ok = 0
        for c in t.inclusion_criteria:
            if c.entity_type == 'diagnosis':
                dx_total += 1
                s = classify_with_alignment(c.entity_type, c.entity_code, entity_maps, dx_aligner)
                if s == 'RESOLVED':
                    dx_resolved_now += 1; dx_resolved_aligned += 1
                elif s == 'RESOLVED_VIA_ALIGNMENT':
                    dx_resolved_aligned += 1
            # count resolvability under alignment for the zero-trial rollup
            s_any = classify_with_alignment(c.entity_type, c.entity_code, entity_maps, dx_aligner)
            if s_any in ('RESOLVED', 'RESOLVED_VIA_ALIGNMENT'):
                inc_ok += 1
        for c in t.exclusion_criteria:
            if c.entity_type == 'diagnosis':
                dx_total += 1
                s = classify_with_alignment(c.entity_type, c.entity_code, entity_maps, dx_aligner)
                if s == 'RESOLVED':
                    dx_resolved_now += 1; dx_resolved_aligned += 1
                elif s == 'RESOLVED_VIA_ALIGNMENT':
                    dx_resolved_aligned += 1
            s_any = classify_with_alignment(c.entity_type, c.entity_code, entity_maps, dx_aligner)
            if s_any in ('RESOLVED', 'RESOLVED_VIA_ALIGNMENT'):
                exc_ok += 1
        if inc_ok == 0:
            zero_inc_aligned += 1
        if exc_ok == 0:
            zero_exc_aligned += 1
        if inc_ok == 0 and exc_ok == 0:
            both_zero_aligned += 1

    logging.info(f"\n{'-'*70}\nWITH HIERARCHICAL DIAGNOSIS ALIGNMENT (entity_alignment.py):")
    if dx_total:
        logging.info(f"  Diagnosis criteria resolved: {dx_resolved_now}/{dx_total} ({dx_resolved_now/dx_total:.1%}) "
                     f"-> {dx_resolved_aligned}/{dx_total} ({dx_resolved_aligned/dx_total:.1%})")
    logging.info(f"  Trials with ZERO inclusion:  {zero_inc}/{n} ({zero_inc/n:.1%}) -> "
                 f"{zero_inc_aligned}/{n} ({zero_inc_aligned/n:.1%})")
    logging.info(f"  Trials with ZERO exclusion:  {zero_exc}/{n} ({zero_exc/n:.1%}) -> "
                 f"{zero_exc_aligned}/{n} ({zero_exc_aligned/n:.1%})")
    logging.info(f"  Trials with BOTH zero:       {both_zero}/{n} ({both_zero/n:.1%}) -> "
                 f"{both_zero_aligned}/{n} ({both_zero_aligned/n:.1%})")
    logging.info(f"  (Medication ingredient-level recovery not shown here -- needs the "
                 f"NDC->ingredient crosswalk; measured separately after that map is built.)")

    logging.info(f"\nExample OOV codes (real-looking, but not found in entity_maps after normalization):")
    for etype, examples in oov_examples.items():
        logging.info(f"  {etype}:")
        for ex in examples:
            logging.info(f"    {ex}")

    return {
        'label': label, 'n_trials': n,
        'overall_resolved_pct': overall_pct,
        'zero_inc_trials': zero_inc, 'zero_exc_trials': zero_exc, 'both_zero_trials': both_zero,
        'counts': {f"{k[0]}_{k[1]}": dict(v) for k, v in counts.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-file", choices=["train", "eval", "both"], default="both")
    args = parser.parse_args()

    cfg = Config()
    diag_df, rx_df, labs_df = load_processed_tables(cfg)

    # Same construction as evaluate.py -- deterministic, matches production exactly.
    d_map = {str(c): i for i, c in enumerate(sorted(diag_df['ICD10_CODE'].unique()))}
    m_map = {str(c): i for i, c in enumerate(sorted(rx_df['NDC'].unique()))}
    l_map = {str(c): i for i, c in enumerate(sorted(labs_df['ITEMID'].unique()))}
    entity_maps = {'diagnosis': d_map, 'medication': m_map, 'lab': l_map, 'procedure': {}}

    logging.info(f"Vocabulary sizes -- diagnosis: {len(d_map)}, medication: {len(m_map)}, lab: {len(l_map)}")

    results = []
    if args.trials_file in ("train", "both"):
        with open(cfg.TRAIN_TRIALS_PATH) as f:
            train_store = TrialStore.from_records(json.load(f))
        results.append(diagnose(train_store, entity_maps, "TRAIN trials"))

    if args.trials_file in ("eval", "both"):
        with open(cfg.EVAL_TRIALS_PATH) as f:
            eval_store = TrialStore.from_records(json.load(f))
        results.append(diagnose(eval_store, entity_maps, "EVAL (held-out) trials"))

    out_path = os.path.join(cfg.OUTPUT_DIR, "entity_resolution_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nFull diagnostic saved to {out_path}")


if __name__ == "__main__":
    main()