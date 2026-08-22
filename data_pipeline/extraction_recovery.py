"""
extraction_recovery.py
-----------------------
Second-chance concept extraction for trial criteria that the primary
whole-line ontology matcher fails to resolve.

WHY THIS EXISTS
The diagnostic (diagnose_entity_resolution.py) showed that ~73% of
diagnosis criteria end up as UNMATCHED_PLACEHOLDER -- meaning no code was
ever extracted, NOT that a code was extracted and failed to align. The
root cause is in load_1000_trials.py's parse_criteria(): it calls
mapper.match_entity(line) on the ENTIRE raw criterion sentence, which only
succeeds if a dictionary key appears verbatim somewhere in that sentence.
Real criteria are full clinical sentences ("Documented history of
myocardial infarction within the prior 6 months"), so verbatim
dictionary hits are rare.

This module adds a recovery layer that runs AFTER the primary matcher
fails but BEFORE falling back to UNMATCHED. It tries, in order:

  1. SYNONYM SCAN: look for any known clinical synonym phrase
     (CLINICAL_SYNONYMS from entity_alignment.py -- "afib", "prior mi",
     "t2dm", etc.) anywhere in the line. These are the surface forms trials
     actually use, which rarely match official ICD titles.

  2. MEDICATION KEYWORD SCAN: pull candidate drug tokens out of the line
     and fuzzy-match them against the patient medication vocabulary via the
     mapper's existing _fuzzy_match_medication (already implemented, just
     never reached because it only fires inside match_entity's automaton
     path).

  3. LAB/VALUE HEURISTIC: lines expressing a numeric lab threshold
     ("creatinine > 2.0", "eGFR < 30") name a lab even when the specific
     ITEMID isn't extractable -- these are currently thrown away; we at
     least tag them as lab-typed so downstream numeric logic can use them.

Every recovery is CONSERVATIVE: it only fires on a real, specific match,
and returns None otherwise so the caller still falls back to UNMATCHED.
The point is to convert the recoverable slice of that 73%, not to
manufacture false matches (which would be worse than UNMATCHED -- a wrong
code actively mis-trains the model, whereas UNMATCHED is just dropped).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# NOTE: this synonym table is intentionally duplicated from
# models/claude_active/entity_alignment.py rather than imported, because
# data_pipeline/ and models/claude_active/ are separate components that
# shouldn't import each other's internals (the pipeline runs standalone to
# BUILD the trial JSON; the model consumes it later). If you extend one
# table, extend both -- they serve the same clinical purpose (trial surface
# form -> ICD-10 category) at two different pipeline stages: this copy at
# EXTRACTION time (recovering a code from raw text), the other at
# RESOLUTION time (aligning an already-extracted code to patient vocab).
CLINICAL_SYNONYMS = {
    "mi": "I21", "heart attack": "I21", "myocardial infarction": "I21",
    "prior mi": "I21", "history of mi": "I21", "stemi": "I21", "nstemi": "I21",
    "afib": "I48", "a-fib": "I48", "atrial fibrillation": "I48",
    "atrial flutter": "I48", "chf": "I50", "heart failure": "I50",
    "congestive heart failure": "I50", "hfref": "I50", "hfpef": "I50",
    "cad": "I25", "coronary artery disease": "I25", "coronary disease": "I25",
    "htn": "I10", "hypertension": "I10", "high blood pressure": "I10",
    "stroke": "I63", "cva": "I63", "cerebrovascular accident": "I63",
    "prior stroke": "I63", "ischemic stroke": "I63", "cerebral infarction": "I63",
    "tia": "G45", "transient ischemic attack": "G45",
    "dm": "E11", "diabetes": "E11", "diabetes mellitus": "E11",
    "type 2 diabetes": "E11", "t2dm": "E11", "type ii diabetes": "E11",
    "type 1 diabetes": "E10", "t1dm": "E10",
    "ckd": "N18", "chronic kidney disease": "N18", "renal failure": "N18",
    "esrd": "N18", "end stage renal disease": "N18",
    "aki": "N17", "acute kidney injury": "N17",
    "copd": "J44", "chronic obstructive pulmonary disease": "J44",
    "asthma": "J45", "pneumonia": "J18", "ards": "J80",
    "sepsis": "A41", "septic shock": "A41", "bacteremia": "A41",
    "tb": "A15", "tuberculosis": "A15",
    "cancer": "C80", "malignancy": "C80", "carcinoma": "C80", "tumor": "C80",
    "cirrhosis": "K74", "liver failure": "K72", "hepatic failure": "K72",
    "depression": "F32", "anxiety": "F41", "dementia": "F03",
    "obesity": "E66", "anemia": "D64",
}


# Lab-name hints -> whether a line is expressing a lab threshold. Used only
# to TYPE a criterion as 'lab' when a specific code can't be pulled; the
# numeric bound itself is parsed separately by the caller.
_LAB_HINT_WORDS = {
    "creatinine", "egfr", "gfr", "hemoglobin", "hgb", "hba1c", "a1c",
    "platelet", "bilirubin", "albumin", "sodium", "potassium", "glucose",
    "inr", "ast", "alt", "wbc", "neutrophil", "ejection fraction", "lvef",
    "bnp", "troponin", "ldl", "hdl", "triglyceride", "tsh", "psa",
}

# Common criterion boilerplate that should NOT trigger a diagnosis match
# even if a synonym substring appears (e.g. "no history of cancer" is an
# exclusion the caller's negation logic already handles -- we just avoid
# double-processing here).
_NON_ENTITY_LEADERS = re.compile(
    r"^\s*(willing|able|must|should|provides?|signed?|written|informed"
    r"|life expectancy|performance status|ecog|karnofsky)\b",
    re.IGNORECASE,
)


def recover_concept(line: str, mapper=None) -> Optional[Tuple[str, str, str]]:
    """
    Attempt to recover (matched_phrase, entity_type, entity_code) from a
    criterion line that the primary matcher missed. Returns None if nothing
    specific can be recovered (caller then falls back to UNMATCHED).
    """
    if not line or len(line.strip()) < 4:
        return None

    text = line.lower().strip()

    # Skip pure administrative/eligibility boilerplate outright.
    if _NON_ENTITY_LEADERS.match(text):
        return None

    # --- Strategy 1: clinical synonym scan (diagnoses) ---------------------
    # Longest key first so "type 2 diabetes" beats "diabetes".
    for key in sorted(CLINICAL_SYNONYMS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", text):
            return (key, "diagnosis", CLINICAL_SYNONYMS[key])

    # --- Strategy 2: medication fuzzy recovery -----------------------------
    # Reuse the mapper's existing fuzzy medication matcher, which is already
    # implemented but only reachable from inside match_entity's automaton
    # branch. Feeding it the raw line lets it pull a drug token out.
    if mapper is not None and hasattr(mapper, "_fuzzy_match_medication"):
        try:
            fuzzy = mapper._fuzzy_match_medication(text, original_text=line)
            if fuzzy is not None:
                # fuzzy is (matched_key, entity_type, entity_code)
                return fuzzy
        except Exception:
            pass  # never let a recovery attempt crash extraction

    # --- Strategy 3: lab threshold typing ----------------------------------
    # If the line names a lab and expresses a numeric bound, type it as lab.
    # We can't always get the exact ITEMID, so we return a namespaced but
    # TYPED placeholder -- distinct from UNMATCHED because downstream numeric
    # matching can still use the operator/value even without an exact code.
    has_number = bool(re.search(r"\d", text))
    if has_number:
        for hint in _LAB_HINT_WORDS:
            if hint in text:
                # Represent as a lab-typed, hint-named soft code. Kept clearly
                # namespaced so it's never confused with a real ITEMID, but
                # TYPED so it's not lumped in with fully-unmatched criteria.
                soft_code = "LABHINT_" + re.sub(r"[^a-z0-9]", "", hint)
                return (hint, "lab", soft_code)

    return None