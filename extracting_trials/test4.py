import json
with open("../data/10000_trials/structured_clinical_trials.json") as f:
    trials = json.load(f)
for t in trials:
    for c in t["criteria"]:
        if c["entity_code"] == "I219" and "kidney" in c["raw_entity"].lower():
            print(c["raw_entity"])