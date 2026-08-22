"""
entity_alignment.py
--------------------
Comprehensive fixes for the trial-criteria -> patient-vocabulary resolution
gap measured in diagnose_entity_resolution.py (only ~22% of clinical
criteria resolved; 45% of held-out trials had zero resolvable inclusion
criteria).

The diagnosis was that this is a DATA-ALIGNMENT problem, not a model
problem: trial criteria and patient records frequently refer to the SAME
clinical concept but end up with non-matching codes. This module provides
three alignment layers, each targeting one measured failure mode. All three
are pure functions of code strings / small crosswalks, so they're cheap,
deterministic, and testable without touching the model.

Failure mode 1 -- ICD-10 GRANULARITY MISMATCH (biggest diagnosis issue).
  Trial criterion resolves to e.g. I219 (MI, unspecified) but the patient
  cohort, crosswalked from MIMIC-III ICD-9, carries I21 or I2109 for the
  same event. Exact string match fails. FIX: hierarchical ICD matching --
  a trial code resolves against the patient vocabulary if the patient has
  ANY code sharing its 3-character category (I21), with the full code
  preferred when present. This mirrors how clinical eligibility actually
  works ("history of MI" doesn't care about the 5th digit).

Failure mode 2 -- NDC GRANULARITY MISMATCH (nearly all medication failures).
  Trial says "metformin"; both sides carry NDCs, but NDCs encode
  manufacturer + package, so the same drug almost never shares an NDC
  across sources. FIX: collapse NDCs to their labeler-agnostic ingredient
  identity via an NDC->ingredient map, and match at the ingredient level.

Failure mode 3 -- SYNONYM / SURFACE-FORM GAPS (diagnosis extraction misses).
  Trial text ("prior stroke", "afib") never overlaps the official ICD
  title ("cerebral infarction, unspecified"). FIX: an expanded
  clinical-synonym table, applied before falling back to dictionary match.

This module does NOT replace ontology_loader.py's matcher; it augments the
resolution check (is_resolvable_code) and the vocabulary lookup so that
codes already produced by the existing parser stop being discarded as OOV
when a clinically-equivalent patient code exists.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Set


# ---------------------------------------------------------------------------
# Failure mode 3: expanded clinical synonym table (trial surface form -> ICD-10).
# Deliberately kept as data, not code -- extend freely. Codes are 3-4 char
# ICD-10 categories on purpose, so they align at the category level (see
# failure mode 1). This SUPPLEMENTS ontology_loader.MANUAL_MAPPING rather
# than replacing it; overlap is harmless (same code) and intentional.
# ---------------------------------------------------------------------------
CLINICAL_SYNONYMS: Dict[str, str] = {
    # cardiovascular
    "mi": "I21", "heart attack": "I21", "myocardial infarction": "I21",
    "prior mi": "I21", "history of mi": "I21", "stemi": "I21", "nstemi": "I21",
    "afib": "I48", "a-fib": "I48", "atrial fibrillation": "I48",
    "atrial flutter": "I48", "chf": "I50", "heart failure": "I50",
    "congestive heart failure": "I50", "hfref": "I50", "hfpef": "I50",
    "cad": "I25", "coronary artery disease": "I25", "coronary disease": "I25",
    "htn": "I10", "hypertension": "I10", "high blood pressure": "I10",
    # cerebrovascular
    "stroke": "I63", "cva": "I63", "cerebrovascular accident": "I63",
    "prior stroke": "I63", "ischemic stroke": "I63", "cerebral infarction": "I63",
    "tia": "G45", "transient ischemic attack": "G45",
    # metabolic / endocrine
    "dm": "E11", "diabetes": "E11", "diabetes mellitus": "E11",
    "type 2 diabetes": "E11", "t2dm": "E11", "type ii diabetes": "E11",
    "type 1 diabetes": "E10", "t1dm": "E10",
    "ckd": "N18", "chronic kidney disease": "N18", "renal failure": "N18",
    "esrd": "N18", "end stage renal disease": "N18",
    "aki": "N17", "acute kidney injury": "N17",
    # respiratory
    "copd": "J44", "chronic obstructive pulmonary disease": "J44",
    "asthma": "J45", "pneumonia": "J18", "ards": "J80",
    # infection / other
    "sepsis": "A41", "septic shock": "A41", "bacteremia": "A41",
    "tb": "A15", "tuberculosis": "A15",
    "cancer": "C80", "malignancy": "C80", "carcinoma": "C80", "tumor": "C80",
    "cirrhosis": "K74", "liver failure": "K72", "hepatic failure": "K72",
    "depression": "F32", "anxiety": "F41", "dementia": "F03",
    "obesity": "E66", "anemia": "D64",
}


def _clean_icd(code: str) -> str:
    """Uppercase, strip dots/whitespace -- matches ontology_loader's normalize."""
    return str(code).replace(".", "").replace(" ", "").upper().strip()


def icd_category(code: str) -> str:
    """First 3 characters of an ICD-10 code = its category (e.g. I2109 -> I21)."""
    return _clean_icd(code)[:3]


# ---------------------------------------------------------------------------
# Failure mode 1: hierarchical diagnosis resolution.
# ---------------------------------------------------------------------------
class DiagnosisAligner:
    """
    Wraps the patient diagnosis vocabulary with category-level indexing so a
    trial code resolves whenever the cohort contains a clinically-equivalent
    code, not only on exact string identity.
    """

    def __init__(self, patient_diagnosis_codes: Set[str]):
        self.exact: Set[str] = {_clean_icd(c) for c in patient_diagnosis_codes}
        # category -> set of full patient codes in that category
        self.by_category: Dict[str, Set[str]] = {}
        for c in self.exact:
            self.by_category.setdefault(icd_category(c), set()).add(c)

    def resolve(self, trial_code: str) -> Optional[str]:
        """
        Return the patient-vocabulary code this trial code should map to, or
        None if the cohort genuinely has nothing in this diagnostic category.
        Preference order: exact match > any code in the same 3-char category.
        """
        code = _clean_icd(trial_code)
        if code in self.exact:
            return code
        cat = icd_category(code)
        if cat in self.by_category:
            # Deterministic pick: shortest then lexicographically first, so the
            # most general available code in the category represents it.
            return sorted(self.by_category[cat], key=lambda x: (len(x), x))[0]
        return None


# ---------------------------------------------------------------------------
# Failure mode 2: ingredient-level medication resolution.
# ---------------------------------------------------------------------------
class MedicationAligner:
    """
    Resolves trial medication references to the patient vocabulary at the
    ingredient level instead of exact NDC.

    ndc_to_ingredient: maps an NDC (any format) to a canonical ingredient
      string (e.g. all metformin NDCs -> "metformin"). Built upstream from
      an NDC->RxNorm-ingredient crosswalk; if unavailable, falls back to a
      drug-name map so at least name-level matching still works.
    patient_ndcs: the cohort's medication vocabulary (NDCs).
    """

    def __init__(self, patient_ndcs: Set[str], ndc_to_ingredient: Optional[Dict[str, str]] = None):
        self.ndc_to_ingredient = ndc_to_ingredient or {}
        # ingredient -> a representative patient NDC carrying that ingredient
        self.ingredient_to_patient_ndc: Dict[str, str] = {}
        self.patient_ndcs = {self._clean_ndc(n) for n in patient_ndcs}
        for ndc in self.patient_ndcs:
            ing = self.ndc_to_ingredient.get(ndc)
            if ing:
                self.ingredient_to_patient_ndc.setdefault(ing.lower(), ndc)

    @staticmethod
    def _clean_ndc(code: str) -> str:
        code = re.sub(r"\.0+$", "", str(code).strip())
        code = re.sub(r"[^0-9]", "", code)
        return code.zfill(11) if code.isdigit() and len(code) < 11 else code

    def resolve(self, trial_code: str, trial_ingredient: Optional[str] = None) -> Optional[str]:
        """
        Resolve a trial medication to a patient NDC. Tries, in order:
          1. exact NDC match (rare but possible)
          2. same ingredient as some patient NDC (the real workhorse)
        trial_ingredient, if provided, short-circuits the NDC->ingredient step.
        """
        ndc = self._clean_ndc(trial_code)
        if ndc in self.patient_ndcs:
            return ndc
        ing = (trial_ingredient or self.ndc_to_ingredient.get(ndc) or "").lower()
        if ing and ing in self.ingredient_to_patient_ndc:
            return self.ingredient_to_patient_ndc[ing]
        return None


def synonym_to_icd(text: str) -> Optional[str]:
    """
    Failure mode 3: map a trial-criterion surface form to an ICD-10 category
    via the expanded synonym table. Longest-key-first so 'type 2 diabetes'
    wins over 'diabetes'. Returns a 3-4 char category code (aligns at the
    category level with DiagnosisAligner).
    """
    t = text.lower().strip()
    for key in sorted(CLINICAL_SYNONYMS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", t):
            return CLINICAL_SYNONYMS[key]
    return None