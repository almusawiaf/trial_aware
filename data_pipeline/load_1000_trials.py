"""
load_1000_trials.py
Load trials from ctg-studies_10000.json, filter for target conditions, and save to
data/10000_trials/.

FIX vs. original: parse_criteria() previously never touched the ontology mapper --
entity_code was `f"CODE_{hash(line) % 10000:04d}"`, a hash of the raw sentence, so no
trial criterion could ever line up with a patient's real ICD-10/NDC code regardless of
how good the ontology tables were. This version imports DynamicOntologyMapper, loads
the MIMIC ontology + manual mapping once in main(), and passes it down so each
criterion is resolved to a real code when possible. Only truly unmatched criteria
(mostly administrative/eligibility boilerplate, e.g. "willing to provide consent")
fall back to a hash-based placeholder, and that placeholder is now clearly namespaced
as UNMATCHED_xxxx instead of looking like a plausible code.
"""

import json
import os
import re
from typing import List, Dict, Set, Optional

from ontology_loader import DynamicOntologyMapper

# ------------------------------------------------------------------
# ICD-9 -> ICD-10 crosswalk. Without this, any diagnosis code resolved
# through the MIMIC ontology table (as opposed to the manual mapping)
# comes back as a raw ICD-9 code, which will NEVER match your ICD-10
# patient data downstream. Try to reuse the same crosswalk
# generate_trial_json.py already builds via MIMICDataPreprocessor.
# ------------------------------------------------------------------
def load_icd9_to_icd10_crosswalk() -> Dict[str, str]:
    """
    Load the NBER GEMs ICD-9 -> ICD-10 crosswalk directly from CSV.

    Previously this imported preprocessor.MIMICDataPreprocessor + config.Config,
    which fails when this script is run from data_pipeline/ (config lives under
    models/claude_active/ and pulls in the whole training-config machinery just
    to read one CSV). This self-contained version reads the same
    icd9toicd10cmgem.csv the preprocessor uses, with the SAME cleaning logic
    (lowercase cols, drop no_map==1, strip dots/whitespace, keep first mapping),
    so trial diagnosis codes resolved via the MIMIC ontology table come out as
    ICD-10 and can actually match the ICD-10 patient vocabulary.

    Path resolution order (first hit wins):
      1. ICD9_TO_ICD10_CSV environment variable, if set
      2. MIMIC_DATA_DIR/icd9toicd10cmgem.csv  (the module-level constant below)
    """
    import pandas as pd

    candidates = []
    env_path = os.environ.get("ICD9_TO_ICD10_CSV")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(MIMIC_DATA_DIR, "icd9toicd10cmgem.csv"))

    csv_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if csv_path is None:
        print("⚠️  ICD9->ICD10 crosswalk CSV not found. Looked in:")
        for p in candidates:
            print(f"     - {p}")
        print("   Diagnosis codes from the MIMIC ontology table will remain as raw ICD-9 "
              "and won't match ICD-10 patient data. Set the ICD9_TO_ICD10_CSV env var to "
              "the icd9toicd10cmgem.csv path, or place it in MIMIC_DATA_DIR, then re-run.")
        return {}

    try:
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.lower()
        if 'no_map' in df.columns:
            df = df[df['no_map'] != '1']
        df['icd9cm'] = df['icd9cm'].str.strip().str.replace('.', '', regex=False).str.upper()
        df['icd10cm'] = df['icd10cm'].str.strip().str.replace('.', '', regex=False).str.upper()
        df = df.drop_duplicates(subset=['icd9cm'], keep='first')
        crosswalk = dict(zip(df['icd9cm'], df['icd10cm']))
        print(f"✅ Loaded ICD-9->ICD-10 crosswalk: {len(crosswalk)} mappings from {csv_path}")
        return crosswalk
    except Exception as e:
        print(f"⚠️  Failed to parse ICD9->ICD10 crosswalk at {csv_path}: {e}")
        print("   Proceeding without it -- MIMIC-ontology-resolved diagnosis codes will "
              "stay ICD-9 and won't match patient ICD-10 data.")
        return {}

# ============================================================
# CONFIGURATION
# ============================================================

# Resolve data paths relative to the REPO ROOT (this file lives in
# data_pipeline/, so the repo root is one directory up), not relative to the
# current working directory. This makes the script runnable from anywhere --
# `python data_pipeline/load_1000_trials.py` from the repo root, or
# `python load_1000_trials.py` from inside data_pipeline/ -- without the
# input/output paths silently pointing at the wrong place.
# Override either with the CTG_INPUT_FILE / TRIALS_OUTPUT_DIR env vars.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.environ.get(
    "CTG_INPUT_FILE",
    os.path.join(_REPO_ROOT, "data", "ctg-studies_10000.json"),
)
OUTPUT_DIR = os.environ.get(
    "TRIALS_OUTPUT_DIR",
    os.path.join(_REPO_ROOT, "data", "10000_trials"),
)

# Point this at your MIMIC-III CSV directory (D_ICD_DIAGNOSES.csv, PRESCRIPTIONS.csv,
# D_LABITEMS.csv). Adjust as needed for your environment.
MIMIC_DATA_DIR = "/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources"

TARGET_CONDITIONS = [
    "heart failure",
    "myocardial infarction",
    "diabetes",
    "pneumonia",
    "sepsis",
    "atrial fibrillation",
    "hypertension",
    "chronic kidney disease",
]

CONDITION_SYNONYMS = {
    "heart failure": ["cardiac failure", "hf", "heart insufficiency", "congestive heart failure"],
    "myocardial infarction": ["heart attack", "mi", "acute coronary syndrome", "coronary thrombosis"],
    "diabetes": ["diabetes mellitus", "dm", "type 2 diabetes", "t2dm", "insulin-dependent"],
    "pneumonia": ["lung infection", "respiratory infection", "pneumonitis"],
    "sepsis": ["septicemia", "blood infection", "septic shock", "bacteremia"],
    "atrial fibrillation": ["afib", "af", "auricular fibrillation", "a-fib"],
    "hypertension": ["high blood pressure", "htn", "essential hypertension", "hypertensive"],
    "chronic kidney disease": ["ckd", "chronic renal failure", "renal insufficiency", "kidney disease"],
}

# ============================================================
# BUILD CONDITION SET
# ============================================================

def build_condition_set() -> Set[str]:
    terms = set()
    for c in TARGET_CONDITIONS:
        terms.add(c.lower())
        for syn in CONDITION_SYNONYMS.get(c, []):
            terms.add(syn.lower())
    return terms

# ============================================================
# LOAD TRIALS
# ============================================================

def load_trials() -> List[Dict]:
    print(f"📂 Loading trials from: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Files here: {os.listdir('.')}")
        return []

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"✅ Loaded {len(data)} trials from {INPUT_FILE}")
            return data
        else:
            print(f"✅ Loaded 1 trial from {INPUT_FILE}")
            return [data]

    except Exception as e:
        print(f"❌ Error loading {INPUT_FILE}: {e}")
        return []

# ============================================================
# FILTER TRIALS
# ============================================================

def extract_conditions(trial: Dict) -> List[str]:
    conditions = []
    protocol = trial.get('protocolSection', {})
    if protocol:
        conditions_module = protocol.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', [])

    if not conditions:
        conditions = trial.get('Conditions', [])

    if not conditions:
        study_fields = trial.get('StudyFields', {})
        conditions = study_fields.get('Conditions', [])

    if isinstance(conditions, str):
        conditions = [conditions]

    cleaned = []
    for c in conditions:
        if c and isinstance(c, str):
            for part in re.split(r'[|;,]', c):
                part = part.strip()
                if part and len(part) > 2:
                    cleaned.append(part.lower())

    return cleaned

def matches_conditions(conditions: List[str], target_terms: Set[str]) -> tuple:
    """
    FIXED: word-boundary matching instead of raw substring containment.
    The old version matched short acronyms like "mi" or "af" against ANY
    condition string containing those letters in sequence (e.g. "mi" inside
    "leukemia", "af" inside "staff"/"safety"), which silently pulled thousands
    of unrelated trials into the dataset. Now each term must appear as a whole
    word/phrase.
    """
    if not conditions:
        return False, None

    for condition in conditions:
        if not condition:
            continue
        condition_lower = condition.lower()

        for term in target_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', condition_lower):
                return True, term

    return False, None

def filter_trials(trials: List[Dict], target_terms: Set[str]) -> List[Dict]:
    print(f"\n🔍 Filtering {len(trials)} trials...")

    filtered = []
    matched_conditions = {}

    for idx, trial in enumerate(trials):
        conditions = extract_conditions(trial)
        matches, matched_term = matches_conditions(conditions, target_terms)

        if matches:
            filtered.append(trial)
            if matched_term:
                matched_conditions[matched_term] = matched_conditions.get(matched_term, 0) + 1

        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1:,} trials, found {len(filtered)} matches")

    print(f"\n✅ Found {len(filtered)} matching trials")

    if matched_conditions:
        print("\n📊 Conditions matched:")
        sorted_conditions = sorted(matched_conditions.items(), key=lambda x: x[1], reverse=True)
        for cond, count in sorted_conditions[:10]:
            print(f"   {cond}: {count}")

    return filtered

# ============================================================
# CONVERT TO PIPELINE FORMAT (FIXED: now uses the ontology mapper)
# ============================================================

def parse_criteria(criteria_text: str, mapper: Optional[DynamicOntologyMapper]) -> List[Dict]:
    """Parse eligibility criteria into structured format, resolving each criterion
    to a real ontology code via `mapper` when possible."""
    if not criteria_text or len(criteria_text.strip()) < 10:
        return []

    structured = []
    lines = [l.strip() for l in criteria_text.split('\n') if l.strip()]
    is_inclusion = True

    unmatched_count = 0
    matched_count = 0
    admin_count = 0

    for line in lines:
        lower = line.lower()

        if 'inclusion' in lower and 'criteria' in lower:
            is_inclusion = True
            continue
        elif 'exclusion' in lower and 'criteria' in lower:
            is_inclusion = False
            continue

        if len(line) < 10 or line.endswith(':'):
            continue

        # --- Resolve real entity code via mapper first ---
        entity_type = 'diagnosis'
        entity_code = None
        if mapper is not None:
            matched_phrase, matched_type, matched_code = mapper.match_entity(line)
            if matched_code:
                entity_type = matched_type
                entity_code = matched_code
                if matched_type == 'administrative':
                    admin_count += 1
                else:
                    matched_count += 1

        if entity_code is None:
            # Fall back to keyword-based type guess (unchanged heuristic) and a
            # clearly-namespaced placeholder so it's never mistaken for a real code.
            if any(word in lower for word in ['medication', 'drug', 'treatment', 'therapy']):
                entity_type = 'medication'
            elif any(word in lower for word in ['creatinine', 'glucose', 'blood', 'pressure']):
                entity_type = 'lab'
            elif any(word in lower for word in ['surgery', 'procedure', 'transplant']):
                entity_type = 'procedure'
            entity_code = f"UNMATCHED_{abs(hash(line)) % 100000:05d}"
            unmatched_count += 1

        operator = 'EXISTS'
        value = None

        numbers = re.findall(r'(\d+\.?\d*)', line)
        if numbers:
            if '>' in line or 'greater than' in lower:
                operator = 'GT'
                value = float(numbers[0])
            elif '<' in line or 'less than' in lower:
                operator = 'LT'
                value = float(numbers[0])
            elif '=' in line or 'equal' in lower:
                operator = 'EQ'
                value = float(numbers[0])
            elif 'between' in lower and len(numbers) >= 2:
                operator = 'BETWEEN'
                value = float(numbers[0])

        structured.append({
            'raw_entity': line[:200],
            'entity_type': entity_type,
            'entity_code': entity_code,
            'operator': operator,
            'value': value,
            'max_value': None,
            'is_inclusion': is_inclusion,
            'severity_weight': 1.0
        })

    return structured, matched_count, unmatched_count, admin_count

def convert_to_pipeline_format(trials: List[Dict], mapper: Optional[DynamicOntologyMapper]) -> List[Dict]:
    print("\n🔄 Converting to pipeline format...")

    structured = []
    total_matched = 0
    total_unmatched = 0
    total_admin = 0

    for idx, trial in enumerate(trials):
        protocol = trial.get('protocolSection', {})

        identification = protocol.get('identificationModule', {})
        status = protocol.get('statusModule', {})
        design = protocol.get('designModule', {})
        conditions_module = protocol.get('conditionsModule', {})
        eligibility = protocol.get('eligibilityModule', {})

        nct_id = identification.get('nctId', 'Unknown')
        title = identification.get('briefTitle', 'Unknown')
        phase = design.get('phases', ['NA'])[0] if design.get('phases') else 'NA'
        sample_size = design.get('enrollmentInfo', {}).get('count', 100)

        criteria_text = eligibility.get('eligibilityCriteria', '')
        criteria, matched, unmatched, admin = parse_criteria(criteria_text, mapper)
        total_matched += matched
        total_unmatched += unmatched
        total_admin += admin

        structured.append({
            'nct_id': nct_id,
            'title': title,
            'conditions': conditions_module.get('conditions', []),
            'phase': phase,
            'sample_size': sample_size,
            'overall_status': status.get('overallStatus', 'Unknown'),
            'criteria': criteria
        })

        if (idx + 1) % 100 == 0:
            print(f"   Converted {idx + 1} trials...")

    print(f"✅ Converted {len(structured)} trials")
    total_criteria = total_matched + total_unmatched + total_admin
    if total_criteria:
        print(f"📈 Resolved to real clinical codes: {total_matched}/{total_criteria} "
              f"({100 * total_matched / total_criteria:.1f}%)")
        print(f"📋 Correctly excluded as administrative: {total_admin}/{total_criteria} "
              f"({100 * total_admin / total_criteria:.1f}%)")
        print(f"❓ Still unmatched (fallback code): {total_unmatched}/{total_criteria} "
              f"({100 * total_unmatched / total_criteria:.1f}%)")
    return structured

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  LOAD AND FILTER TRIALS (WITH REAL ONTOLOGY MATCHING)")
    print("=" * 60)
    print(f"📁 Input file: {INPUT_FILE}")
    print(f"📁 Output directory: {OUTPUT_DIR}")

    target_terms = build_condition_set()
    print(f"🎯 Target conditions: {len(TARGET_CONDITIONS)}")
    print(f"📋 Terms to match: {len(target_terms)}")
    print("=" * 60)

    # --- Build the ontology mapper ONCE (this was previously never done here) ---
    icd9_to_icd10_map = load_icd9_to_icd10_crosswalk()
    mapper = DynamicOntologyMapper(icd9_to_icd10_map=icd9_to_icd10_map)
    abs_dir = os.path.abspath(MIMIC_DATA_DIR)
    print(f"🔎 Looking for MIMIC CSVs in: {abs_dir}")
    if os.path.exists(MIMIC_DATA_DIR):
        found = os.listdir(MIMIC_DATA_DIR)
        needed = ["D_ICD_DIAGNOSES.csv", "PRESCRIPTIONS.csv", "D_LABITEMS.csv"]
        present = [f for f in needed if f in found]
        missing = [f for f in needed if f not in found]
        print(f"   Found: {present if present else 'none'}")
        if missing:
            print(f"   ⚠️  Missing: {missing} -- these rows won't be in the ontology table.")
        mapper.load_icd9_and_patient_tables(data_dir=MIMIC_DATA_DIR)
        if len(mapper.concept_table) == 0:
            print("   ❌ 0 concepts loaded -- MIMIC_DATA_DIR is pointing at the wrong folder, "
                  "or the CSVs use different filenames/casing than expected. Manual mapping "
                  "(48 conditions) is the ONLY source of real codes right now.")
    else:
        print(f"⚠️  MIMIC_DATA_DIR '{abs_dir}' does not exist -- proceeding with "
              f"manual mapping only (still resolves the common trial conditions).")

    # 1. Load trials
    trials = load_trials()
    if not trials:
        print(f"❌ No trials loaded. Make sure {INPUT_FILE} is in the current directory.")
        return

    # 2. Filter
    filtered_trials = filter_trials(trials, target_terms)

    if not filtered_trials:
        print("❌ No trials matched the target conditions.")
        return

    # 3. Convert (now with real code resolution)
    structured_trials = convert_to_pipeline_format(filtered_trials, mapper)

    # 4. Split
    split_idx = int(len(structured_trials) * 0.8)
    train_trials = structured_trials[:split_idx]
    eval_trials = structured_trials[split_idx:]

    # 5. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "structured_clinical_trials.json")
    eval_path = os.path.join(OUTPUT_DIR, "structured_clinical_trials_eval.json")

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_trials, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(train_trials)} training trials to {train_path}")

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_trials, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(eval_trials)} evaluation trials to {eval_path}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"Total trials loaded: {len(trials):,}")
    print(f"Trials matching conditions: {len(filtered_trials):,}")
    print(f"Match rate: {len(filtered_trials)/len(trials)*100:.2f}%")
    print(f"Training trials: {len(train_trials)}")
    print(f"Evaluation trials: {len(eval_trials)}")
    print("=" * 60)

    if train_trials:
        print("\n📋 Sample trial:")
        sample = train_trials[0]
        print(f"   NCT ID: {sample.get('nct_id')}")
        print(f"   Title: {sample.get('title', '')[:80]}...")
        print(f"   Conditions: {sample.get('conditions', [])[:3]}")
        print(f"   Criteria: {len(sample.get('criteria', []))} items")
        for c in sample.get('criteria', [])[:5]:
            print(f"      - [{c['entity_type']}] {c['entity_code']}: {c['raw_entity'][:60]}...")

if __name__ == "__main__":
    main()