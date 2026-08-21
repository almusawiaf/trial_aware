import json
from typing import Dict, List


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

        for c_idx, c in enumerate(criteria):
            # Check schema keys
            missing_keys = required_keys - set(c.keys())
            if missing_keys:
                print(
                    f"❌ [NCT: {nct_id} | Rule {c_idx}] Missing keys: {missing_keys}"
                )

            # Check mapping status
            if c.get("entity_code") != "UNKNOWN_CODE":
                mapped_codes += 1

            # Check numeric operators
            if c.get("operator") in ["LTE", "GTE", "LT", "GT", "EQ"]:
                if c.get("value") is not None:
                    numeric_operators += 1
                else:
                    print(
                        f"⚠️ [NCT: {nct_id}] Operator '{c.get('operator')}' found but value is None!"
                    )

            # Check logic split
            if c.get("is_inclusion") is True:
                inclusion_count += 1
            else:
                exclusion_count += 1

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
    else:
        print(
            "⚠️ WARNING: 0 criteria extracted! Your NLP parser failed to match entities."
        )

    print("=" * 50)

    # Print a sample extracted triplet for visual inspection
    if total_criteria > 0:
        sample = trials[0]["criteria"][0]
        print("\n📋 Sample Extracted Triplet (First Trial, First Rule):")
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    validate_trial_json("structured_clinical_trials.json")