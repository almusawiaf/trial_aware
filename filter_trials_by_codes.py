# filter_trials_by_codes.py
"""
Filter trials to only include those with entity codes present in your MIMIC data.
"""

import json
import pandas as pd
from collections import Counter

def load_mimic_codes(cfg):
    """Load all diagnosis, medication, and lab codes from MIMIC data."""
    diag_path = f"{cfg.OUTPUT_DIR}/diagnoses_clean.parquet"
    rx_path = f"{cfg.OUTPUT_DIR}/prescriptions_clean.parquet"
    labs_path = f"{cfg.OUTPUT_DIR}/labs_clean.parquet"
    
    diag_df = pd.read_parquet(diag_path)
    rx_df = pd.read_parquet(rx_path)
    labs_df = pd.read_parquet(labs_path)
    
    diagnosis_codes = set(diag_df['ICD10_CODE'].astype(str).unique())
    medication_codes = set(rx_df['NDC'].astype(str).unique())
    lab_codes = set(labs_df['ITEMID'].astype(str).unique())
    
    print(f"📊 MIMIC Codes:")
    print(f"   Diagnosis: {len(diagnosis_codes):,}")
    print(f"   Medication: {len(medication_codes):,}")
    print(f"   Lab: {len(lab_codes):,}")
    
    return diagnosis_codes, medication_codes, lab_codes

def filter_trials_by_codes(trials, diagnosis_codes, medication_codes, lab_codes):
    """
    Filter trials to only those where at least one criterion matches existing codes.
    """
    all_codes = diagnosis_codes | medication_codes | lab_codes
    
    filtered_trials = []
    kept_count = 0
    
    for trial in trials:
        criteria = trial.get('criteria', [])
        
        # Check if any criterion has a code that exists in MIMIC
        has_match = False
        for c in criteria:
            code = c.get('entity_code', '')
            if code in all_codes:
                has_match = True
                break
        
        if has_match:
            filtered_trials.append(trial)
            kept_count += 1
    
    print(f"✅ Kept {kept_count} out of {len(trials)} trials with matching codes")
    return filtered_trials

def main():
    from config import Config
    cfg = Config()
    
    # Load MIMIC codes
    diagnosis_codes, medication_codes, lab_codes = load_mimic_codes(cfg)
    all_codes = diagnosis_codes | medication_codes | lab_codes
    print(f"   Total unique codes: {len(all_codes):,}")
    
    # Load trials
    with open('processed_data/1000_trials/structured_clinical_trials.json', 'r') as f:
        trials = json.load(f)
    print(f"\n📂 Loaded {len(trials)} trials")
    
    # Filter
    filtered = filter_trials_by_codes(trials, diagnosis_codes, medication_codes, lab_codes)
    
    # Save filtered trials
    with open('processed_data/1000_trials/structured_clinical_trials_filtered.json', 'w') as f:
        json.dump(filtered, f, indent=2)
    print(f"✅ Saved {len(filtered)} filtered trials")
    
    # Also update the main file
    split_idx = int(len(filtered) * 0.8)
    train = filtered[:split_idx]
    eval_trials = filtered[split_idx:]
    
    with open('processed_data/1000_trials/structured_clinical_trials.json', 'w') as f:
        json.dump(train, f, indent=2)
    with open('processed_data/1000_trials/structured_clinical_trials_eval.json', 'w') as f:
        json.dump(eval_trials, f, indent=2)
    
    print(f"✅ Updated training: {len(train)}, eval: {len(eval_trials)}")

if __name__ == "__main__":
    main()