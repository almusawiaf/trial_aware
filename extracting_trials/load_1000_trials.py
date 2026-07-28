"""
load_1000_trials.py
Load 1000 trials from ctg-studies_1000.json in the current directory.
Filter for target conditions and save to processed_data/1000_trials/
"""

import json
import os
import re
from typing import List, Dict, Set

# ============================================================
# CONFIGURATION
# ============================================================

# Input file (in the same directory as this script)
INPUT_FILE = "ctg-studies_1000.json"

# Output directory (relative to current directory)
OUTPUT_DIR = "../processed_data/1000_trials/"

# Target conditions
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

# Synonyms for matching
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
    """Build a set of all terms to match."""
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
    """
    Load trials from ctg-studies_1000.json in the current directory.
    """
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
    """Extract conditions from trial data."""
    conditions = []
    
    # Try protocolSection structure
    protocol = trial.get('protocolSection', {})
    if protocol:
        conditions_module = protocol.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', [])
    
    # Try flat structure
    if not conditions:
        conditions = trial.get('Conditions', [])
    
    # Try StudyFields structure
    if not conditions:
        study_fields = trial.get('StudyFields', {})
        conditions = study_fields.get('Conditions', [])
    
    # If conditions is a string, split it
    if isinstance(conditions, str):
        conditions = [conditions]
    
    # Clean and return
    cleaned = []
    for c in conditions:
        if c and isinstance(c, str):
            for part in re.split(r'[|;,]', c):
                part = part.strip()
                if part and len(part) > 2:
                    cleaned.append(part.lower())
    
    return cleaned

def matches_conditions(conditions: List[str], target_terms: Set[str]) -> tuple:
    """Check if conditions match target terms."""
    if not conditions:
        return False, None
    
    for condition in conditions:
        if not condition:
            continue
        condition_lower = condition.lower()
        
        for term in target_terms:
            if term in condition_lower or condition_lower in term:
                return True, term
    
    return False, None

def filter_trials(trials: List[Dict], target_terms: Set[str]) -> List[Dict]:
    """
    Filter trials to those matching target conditions.
    """
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
# CONVERT TO PIPELINE FORMAT
# ============================================================

def parse_criteria(criteria_text: str) -> List[Dict]:
    """Parse eligibility criteria into structured format."""
    if not criteria_text or len(criteria_text.strip()) < 10:
        return []
    
    structured = []
    lines = [l.strip() for l in criteria_text.split('\n') if l.strip()]
    is_inclusion = True
    
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
        
        entity_type = 'diagnosis'
        if any(word in lower for word in ['medication', 'drug', 'treatment', 'therapy']):
            entity_type = 'medication'
        elif any(word in lower for word in ['creatinine', 'glucose', 'blood', 'pressure']):
            entity_type = 'lab'
        elif any(word in lower for word in ['surgery', 'procedure', 'transplant']):
            entity_type = 'procedure'
        
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
            'entity_code': f"CODE_{hash(line) % 10000:04d}",
            'operator': operator,
            'value': value,
            'max_value': None,
            'is_inclusion': is_inclusion,
            'severity_weight': 1.0
        })
    
    return structured

def convert_to_pipeline_format(trials: List[Dict]) -> List[Dict]:
    """Convert to your pipeline's format."""
    print("\n🔄 Converting to pipeline format...")
    
    structured = []
    
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
        criteria = parse_criteria(criteria_text)
        
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
    return structured

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  LOAD AND FILTER 1000 TRIALS")
    print("=" * 60)
    print(f"📁 Input file: {INPUT_FILE}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    target_terms = build_condition_set()
    print(f"🎯 Target conditions: {len(TARGET_CONDITIONS)}")
    print(f"📋 Terms to match: {len(target_terms)}")
    print("=" * 60)
    
    # 1. Load trials
    trials = load_trials()
    if not trials:
        print("❌ No trials loaded. Make sure ctg-studies_1000.json is in the current directory.")
        return
    
    # 2. Filter
    filtered_trials = filter_trials(trials, target_terms)
    
    if not filtered_trials:
        print("❌ No trials matched the target conditions.")
        print("   Try adjusting the condition list.")
        return
    
    # 3. Convert
    structured_trials = convert_to_pipeline_format(filtered_trials)
    
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
    
    # Show sample
    if train_trials:
        print("\n📋 Sample trial:")
        sample = train_trials[0]
        print(f"   NCT ID: {sample.get('nct_id')}")
        print(f"   Title: {sample.get('title', '')[:80]}...")
        print(f"   Conditions: {sample.get('conditions', [])[:3]}")
        print(f"   Criteria: {len(sample.get('criteria', []))} items")

if __name__ == "__main__":
    main()