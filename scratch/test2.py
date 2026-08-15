import json, random
with open("../data/10000_trials/structured_clinical_trials.json") as f:
    trials = json.load(f)
for t in random.sample(trials, 8):
    print(t["nct_id"], t["title"][:60])
    for c in t["criteria"]:
        print(f"   [{c['entity_type']}] {c['entity_code']}: {c['raw_entity'][:70]}")
    print()