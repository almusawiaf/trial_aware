import json
from collections import defaultdict
from typing import Dict, List

try:
    from criteria_parser import STOP_ENTITIES
except ImportError:
    STOP_ENTITIES = {
        "who", "that", "them", "their", "patient", "patients",
        "participant", "participants", "subject", "subjects",
        "group", "groups", "use", "assessment", "score", "history",
    }


def validate_trial_json(json_path: str):
    print(f"🔍 Validating dataset: {json_path}\n" + "=" * 50)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            trials = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: File '{json_path}' not found. Run your extractor first.")
        return

    if not isinstance(trials, list) or len(trials) == 0:
        print("❌ ERROR: JSON is empty or not formatted as a list of trials.")
        return

    total_trials = len(trials)
    total_criteria = 0
    mapped_codes = 0
    numeric_operators = 0
    inclusion_count = 0
    exclusion_count = 0

    stopword_leak_count = 0
    unmatched_paren_count = 0
    duplicate_entity_conflicting_ops = 0

    required_keys = {
        "raw_entity",
        "entity_type",
        "entity_code",
        "operator",
        "value",
        "is_inclusion",
        "severity_weight",
    }

    for idx, trial in enumerate(trials):
        nct_id = trial.get("nct_id", f"UNKNOWN_INDEX_{idx}")
        criteria = trial.get("criteria", [])

        if not isinstance(criteria, list):
            print(f"❌ [NCT: {nct_id}] 'criteria' field is not a list!")
            continue

        total_criteria += len(criteria)
        entity_value_map = defaultdict(set)

        for c_idx, c in enumerate(criteria):
            missing_keys = required_keys - set(c.keys())
            if missing_keys:
                print(f"❌ [NCT: {nct_id} | Rule {c_idx}] Missing keys: {missing_keys}")

            raw_entity = str(c.get("raw_entity", ""))

            # Check 1: Stopword Leakage
            if raw_entity.strip().lower() in STOP_ENTITIES:
                stopword_leak_count += 1
                print(
                    f"⚠️ [NCT: {nct_id} | Rule {c_idx}] Stopword leaked as raw_entity: '{raw_entity}'"
                )

            # Check 2: Unsanitized Characters
            if any(ch in raw_entity for ch in "()[]\\"):
                unmatched_paren_count += 1
                print(
                    f"⚠️ [NCT: {nct_id} | Rule {c_idx}] Unsanitized symbols in raw_entity: '{raw_entity}'"
                )

            # Check 3: Numeric Operator Conflicts
            op = c.get("operator")
            val = c.get("value")
            if op not in ("EXISTS",) and val is not None:
                entity_value_map[raw_entity.strip().lower()].add((op, val))

            if c.get("entity_code") != "UNKNOWN_CODE":
                mapped_codes += 1

            if c.get("operator") in ["LTE", "GTE", "LT", "GT", "EQ"]:
                if c.get("value") is not None:
                    numeric_operators += 1
                else:
                    print(
                        f"⚠️ [NCT: {nct_id}] Operator '{c.get('operator')}' found but value is None!"
                    )

            if c.get("is_inclusion") is True:
                inclusion_count += 1
            else:
                exclusion_count += 1

        for entity_name, op_val_set in entity_value_map.items():
            if len(op_val_set) > 1:
                duplicate_entity_conflicting_ops += 1
                print(
                    f"🚨 [NCT: {nct_id}] Entity '{entity_name}' assigned conflicting bounds: {op_val_set}"
                )

    # --- Summary Metrics ---
    print(f"✅ Total Trials Parsed        : {total_trials}")
    print(f"✅ Total Criteria Extracted   : {total_criteria}")
    if total_criteria > 0:
        print(
            f"📊 Mapped Ontology Codes     : {mapped_codes}/{total_criteria} ({mapped_codes/total_criteria:.1%})"
        )
        print(
            f"📊 Numeric Bounds Detected   : {numeric_operators}/{total_criteria} ({numeric_operators/total_criteria:.1%})"
        )
        print(
            f"📊 Criteria Split (Inc / Exc): {inclusion_count} Inc / {exclusion_count} Exc"
        )
        print("-" * 50)
        print("📊 Semantic Quality Checks (Extraction Cleanliness):")
        print(
            f"   • Leaked Stopwords                   : {stopword_leak_count} "
            f"({stopword_leak_count/total_criteria:.1%})"
        )
        print(
            f"   • Unsanitized Symbols / Brackets     : {unmatched_paren_count} "
            f"({unmatched_paren_count/total_criteria:.1%})"
        )
        print(
            f"   • Conflicting Numeric Value Bounds   : {duplicate_entity_conflicting_ops}"
        )
        real_quality = mapped_codes - stopword_leak_count
        print(
            f"   • Estimated True Ontology Coverage   : "
            f"{max(real_quality, 0)}/{total_criteria} "
            f"({max(real_quality, 0)/total_criteria:.1%})"
        )
    else:
        print("⚠️ WARNING: 0 criteria extracted! Check parser implementation.")

    print("=" * 50)

    if total_criteria > 0:
        sample = trials[0]["criteria"][0]
        print("\n📋 Sample Extracted Triplet (First Trial, First Rule):")
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    validate_trial_json("structured_clinical_trials.json")