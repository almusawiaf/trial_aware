'''test.py'''

import json, re

with open("../processed_data/10000_trials/structured_clinical_trials.json") as f:
    trials = json.load(f)

icd9_shaped = re.compile(r'^\d')
count_icd9_shaped = 0
count_total_diagnosis = 0
examples = []

for t in trials:
    for c in t["criteria"]:
        if c["entity_type"] == "diagnosis":
            count_total_diagnosis += 1
            code = c["entity_code"]
            if icd9_shaped.match(code):
                count_icd9_shaped += 1
                if len(examples) < 10:
                    examples.append((code, c["raw_entity"][:60]))

print(f"Diagnosis codes that look like ICD-9 (not ICD-10): {count_icd9_shaped}/{count_total_diagnosis}")
for code, text in examples:
    print(f"  {code}: {text}")