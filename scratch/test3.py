import json
from collections import defaultdict

with open("../data/10000_trials/structured_clinical_trials.json") as f:
    trials = json.load(f)

code_to_texts = defaultdict(set)
for t in trials:
    for c in t["criteria"]:
        if c["entity_type"] in ("diagnosis", "medication", "lab"):
            code_to_texts[c["entity_code"]].add(c["raw_entity"][:50])

# A code that's legitimately common (e.g. hypertension) will appear across many
# criteria that are topically similar. A code that's a false-positive magnet will
# appear across criteria that have nothing in common with each other.
suspicious = [(code, texts) for code, texts in code_to_texts.items() if len(texts) > 15]
suspicious.sort(key=lambda x: -len(x[1]))

print(f"Codes appearing in >15 distinct criteria texts: {len(suspicious)}")
for code, texts in suspicious[:5]:
    print(f"\n{code} ({len(texts)} distinct texts):")
    for t in list(texts)[:6]:
        print(f"   {t}")