import json
from medcat.cat import CAT

cat = CAT.load_model_pack('/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources/v2_Snomed_2024-10-5a7aee8d9d163a5b.zip')

def fix_trial_codes(trials):
    for trial in trials:
        for criterion in trial.get('criteria', []):
            # Only process fake codes
            if criterion.get('entity_code', '').startswith('EXTRACTED_'):
                raw_entity = criterion.get('raw_entity', '')
                if raw_entity:
                    # Extract real ICD-10 codes using MedCAT
                    doc = cat.get_entities(raw_entity)
                    for cui, entity_info in doc.items():
                        # This assumes your model is configured to map to ICD-10
                        icd10_code = entity_info.get('icd10')
                        if icd10_code:
                            criterion['entity_code'] = icd10_code
                            criterion['entity_type'] = 'diagnosis'
                        break # Use the first result
    return trials

with open('../processed_data/10000_trials/structured_clinical_trials.json', 'r') as f:
    trials = json.load(f)

fixed_trials = fix_trial_codes(trials)

with open('../processed_data/10000_trials/structured_clinical_trials_fixed.json', 'w') as f:
    json.dump(fixed_trials, f, indent=2)