"""
matching_engine.py
Trial-Aware Patient-Trial Matching: Core Matching Pipeline
============================================================

This module implements the M_inc / M_exc computation pipeline discussed
in the design doc: ICD-10 hierarchy matching, negation-aware criteria,
and soft (rather than hard-max) exclusion scoring.

FIX APPLIED IN THIS VERSION
----------------------------
compute_matching_indices() now skips criteria with entity_type ==
"administrative" entirely, instead of scoring them through
match_with_hierarchy_single() (which falls through to 0.0 for any
entity_type it doesn't recognize, treating them as FAILED criteria).
Administrative/logistics text (informed consent, caregiver availability,
etc.) appears in nearly every trial's inclusion criteria, so scoring it
as a failure was systematically depressing M_inc across the entire
dataset -- verified concretely: a patient who perfectly matches every
real clinical criterion in a trial that also has one administrative
line scored M_inc=0.5 instead of 1.0 before this fix.

WHAT THIS FILE DOES NOT SOLVE
-------------------------------
- Real entity extraction + linking from trial free text. `extract_criteria_stub`
  below is a hand-coded placeholder standing in for MedCAT + a UMLS-licensed
  CDB. Getting real (entity_type, entity_code) triplets out of arbitrary
  ClinicalTrials.gov text requires:
    1. A UMLS license (application process, not instant)
    2. A MedCAT model pack (or scispaCy + QuickUMLS + a UMLS install)
    3. Measuring precision/recall of that pipeline specifically on trial
       eligibility text, since MedCAT's off-the-shelf tuning is usually on
       clinical notes, a different register
  This file proves the *matching math* is correct once you have real
  structured criteria and real patient codes -- it does not prove that
  MedCAT-on-trial-text will produce enough real code overlap with MIMIC
  ICD-10 codes to give you a useful training signal. That is a separate,
  empirical, unresolved question.
- "procedure"-type criteria are also unhandled by _match_single() (falls
  through to 0.0, same as administrative used to). Unlike administrative
  text, this isn't a case where skipping is obviously correct -- procedure
  criteria (e.g. "history of cardiac surgery") ARE clinically meaningful
  and arguably SHOULD be matched, but PatientState has no procedure_codes
  field and nothing in your preprocessing pipeline (MIMICDataPreprocessor)
  extracts patient procedure codes from PROCEDURES_ICD.csv. Right now any
  inclusion criterion tagged "procedure" will ALWAYS score 0 (fail) for
  every patient, regardless of whether they actually had that procedure --
  this is a missing feature, not a deliberate design choice. Decide
  explicitly whether to (a) add real procedure_codes support to
  PatientState + a matching branch, using MIMIC's PROCEDURES_ICD.csv, or
  (b) skip "procedure" criteria the same way "administrative" is now
  skipped, accepting that procedure-based eligibility signal is lost.
  Silently leaving it as "always fails" is the one option NOT reviewed
  or endorsed here.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Criterion:
    raw_entity: str
    entity_type: str          # "diagnosis" | "medication" | "lab"
    entity_code: str
    operator: str              # "EXISTS" | "GT" | "LT" | "EQ"
    value: Optional[float]
    is_inclusion: bool
    severity_weight: float = 1.0
    confidence: float = 1.0


@dataclass
class Trial:
    nct_id: str
    inclusion_criteria: List[Criterion] = field(default_factory=list)
    exclusion_criteria: List[Criterion] = field(default_factory=list)


@dataclass
class PatientState:
    patient_id: str
    diagnosis_codes: Set[str] = field(default_factory=set)
    medication_codes: Set[str] = field(default_factory=set)
    lab_values: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ICD-10 hierarchy
# ---------------------------------------------------------------------------

class ICD10Hierarchy:
    def __init__(self, hierarchy_file: Optional[str] = None,
                 log_duplicates: bool = True, has_header: bool = True):
        self.parent_to_children: Dict[str, List[str]] = {}
        self.child_to_parent: Dict[str, str] = {}
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self.duplicate_count = 0
        self.conflicting_count = 0

        if hierarchy_file:
            self.load_from_file(hierarchy_file, log_duplicates=log_duplicates,
                                 has_header=has_header)

    def load_from_file(self, file_path: str, log_duplicates: bool = True,
                        has_header: bool = True) -> None:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if has_header and i == 0:
                    continue
                if not row or len(row) < 2:
                    continue
                parent, child = row[0].strip(), row[1].strip()
                if not parent or not child:
                    continue

                children = self.parent_to_children.setdefault(parent, [])
                if child not in children:
                    children.append(child)

                if child in self.child_to_parent:
                    self.duplicate_count += 1
                    existing_parent = self.child_to_parent[child]
                    if existing_parent != parent:
                        self.conflicting_count += 1
                        if log_duplicates:
                            print(f"WARNING conflicting parent for {child}: "
                                  f"existing='{existing_parent}' new='{parent}' "
                                  f"(keeping existing)")
                else:
                    self.child_to_parent[child] = parent

        if log_duplicates and self.duplicate_count:
            print(f"Loaded {len(self.parent_to_children)} parents, "
                  f"{len(self.child_to_parent)} children, "
                  f"{self.duplicate_count} duplicates, "
                  f"{self.conflicting_count} conflicts")

    def get_ancestors(self, code: str) -> Set[str]:
        if code in self._ancestor_cache:
            return self._ancestor_cache[code]
        ancestors: Set[str] = set()
        current = code
        while current in self.child_to_parent:
            parent = self.child_to_parent[current]
            ancestors.add(parent)
            current = parent
        self._ancestor_cache[code] = ancestors
        return ancestors

    def is_ancestor(self, ancestor_code: str, descendant_code: str) -> bool:
        if ancestor_code == descendant_code:
            return True
        return ancestor_code in self.get_ancestors(descendant_code)

    def get_match_score(self, code1: str, code2: str) -> float:
        if code1 == code2:
            return 1.0
        if self.is_ancestor(code1, code2):
            return 0.8
        if self.is_ancestor(code2, code1):
            return 0.6
        return 0.0


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

NEGATION_TRIGGERS = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bexcept\b", r"\bexcluding\b",
    r"\bdenies\b", r"\bnever\b", r"\bnone\b", r"\babsence\b",
    r"\bfree of\b", r"\block of\b", r"\bwithout evidence of\b",
]


def detect_negation(full_text: str, span_start: int, span_end: Optional[int] = None,
                     context_window: int = 40) -> bool:
    """Check for a negation trigger in the text BEFORE the span, clipped to
    the current sentence so negation doesn't leak across sentence boundaries."""
    if span_end is None:
        span_end = span_start
    start = max(0, span_start - context_window)
    sentence_start = max(
        full_text.rfind(".", 0, span_start) + 1,
        full_text.rfind("?", 0, span_start) + 1,
        full_text.rfind("!", 0, span_start) + 1,
    )
    window_start = max(start, sentence_start)
    window = full_text[window_start:span_start].lower()
    return any(re.search(p, window) for p in NEGATION_TRIGGERS)


# ---------------------------------------------------------------------------
# Soft exclusion
# ---------------------------------------------------------------------------

def softmax_exclusion(matches: List[float], temperature: float = 0.5) -> float:
    arr = np.asarray(matches, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    weights = np.exp(arr / temperature)
    weights = weights / weights.sum()
    return float(np.sum(weights * arr))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _sigmoid(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-k * x))


def _match_single(state: PatientState, criterion: Criterion) -> float:
    if criterion.entity_type == "diagnosis":
        return 1.0 if criterion.entity_code in state.diagnosis_codes else 0.0
    if criterion.entity_type == "medication":
        return 1.0 if criterion.entity_code in state.medication_codes else 0.0
    if criterion.entity_type == "lab":
        if criterion.entity_code not in state.lab_values:
            return 0.0
        v = state.lab_values[criterion.entity_code]
        if criterion.operator == "EXISTS":
            return 1.0
        if criterion.operator == "GT":
            return _sigmoid(v - criterion.value)
        if criterion.operator == "LT":
            return _sigmoid(criterion.value - v)
        if criterion.operator == "EQ":
            return _sigmoid(1.0 - abs(v - criterion.value))
    return 0.0


def match_with_hierarchy_single(state: PatientState, criterion: Criterion,
                                 hierarchy: Optional[ICD10Hierarchy] = None) -> float:
    direct = _match_single(state, criterion)
    if direct > 0:
        return direct
    if hierarchy is not None and criterion.entity_type == "diagnosis":
        best = 0.0
        for patient_code in state.diagnosis_codes:
            score = hierarchy.get_match_score(patient_code, criterion.entity_code)
            if score > best:
                best = score
                if best == 1.0:
                    return 1.0
        return best
    return 0.0


def compute_matching_indices(state: PatientState, trial: Trial,
                              hierarchy: Optional[ICD10Hierarchy] = None,
                              temperature: float = 0.5) -> (float, float):
    # FIX: administrative/logistics criteria (informed consent, caregiver
    # availability, etc.) are not clinical facts and cannot be "matched"
    # against patient data. _match_single() falls through its if/elif chain
    # for any entity_type it doesn't recognize -- including "administrative"
    # -- and returns 0.0, which was being scored as a FAILED criterion rather
    # than skipped. Since administrative text (informed consent especially)
    # appears in nearly every trial's inclusion criteria, this systematically
    # depressed M_inc across the whole dataset: a patient who perfectly
    # matches every real clinical criterion in a trial that also has one
    # administrative line would score M_inc=0.5 instead of 1.0. Verified with
    # a concrete before/after test. Filter these out before scoring.
    inc_scores, inc_weights = [], []
    for c in trial.inclusion_criteria:
        if c.entity_type == "administrative":
            continue
        score = match_with_hierarchy_single(state, c, hierarchy)
        inc_scores.append(score * c.severity_weight)
        inc_weights.append(c.severity_weight)
    m_inc = (sum(inc_scores) / sum(inc_weights)) if inc_weights and sum(inc_weights) > 0 else 1.0

    exc_scores = [match_with_hierarchy_single(state, c, hierarchy)
                  for c in trial.exclusion_criteria if c.entity_type != "administrative"]
    m_exc = softmax_exclusion(exc_scores, temperature) if exc_scores else 0.0

    return m_inc, m_exc


# ---------------------------------------------------------------------------
# STUB: entity extraction (placeholder for MedCAT + UMLS)
# ---------------------------------------------------------------------------

def extract_criteria_stub(inclusion_text: str, exclusion_text: str) -> Trial:
    """
    PLACEHOLDER ONLY. Real trial parsing needs MedCAT (NER + UMLS linking +
    negation) plus a UMLS MRCONSO crosswalk to ICD-10/LOINC/RxNorm. This
    function hand-codes a couple of example criteria so the matching math
    below can be demonstrated end-to-end. Do not use this in production --
    swap it for a real MedCAT pipeline once the UMLS license and model pack
    are in place, and validate precision/recall on a labeled sample of real
    trial text before trusting its output in the loss.
    """
    criteria = []
    if "heart failure" in inclusion_text.lower():
        negated = detect_negation(inclusion_text, inclusion_text.lower().find("heart failure"))
        criteria.append(Criterion(
            raw_entity="heart failure", entity_type="diagnosis",
            entity_code="I509", operator="EXISTS", value=None,
            is_inclusion=not negated, severity_weight=1.0,
        ))
    if "lvef" in inclusion_text.lower():
        criteria.append(Criterion(
            raw_entity="LVEF", entity_type="lab", entity_code="LVEF",
            operator="LT", value=40.0, is_inclusion=True, severity_weight=0.8,
        ))
    exclusion_criteria = []
    if "creatinine" in exclusion_text.lower():
        exclusion_criteria.append(Criterion(
            raw_entity="creatinine", entity_type="lab", entity_code="50912",
            operator="GT", value=2.0, is_inclusion=False, severity_weight=1.0,
        ))
    return Trial(nct_id="DEMO", inclusion_criteria=criteria,
                 exclusion_criteria=exclusion_criteria)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _write_temp_csv(rows: List[List[str]], header: Optional[List[str]] = None) -> str:
    import tempfile
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    writer = csv.writer(fd)
    if header:
        writer.writerow(header)
    writer.writerows(rows)
    fd.close()
    return fd.name


def test_hierarchy_from_file():
    import os
    path = _write_temp_csv(
        [["I50", "I509"], ["I50", "I501"], ["I509", "I5090"], ["I509", "I5099"]],
        header=["parent", "child"],
    )
    try:
        h = ICD10Hierarchy(path)
        assert h.parent_to_children["I50"] == ["I509", "I501"]
        assert h.child_to_parent["I509"] == "I50"
        assert h.child_to_parent["I5090"] == "I509"
        assert h.is_ancestor("I50", "I509")
        assert h.is_ancestor("I50", "I5090")
        assert h.is_ancestor("I509", "I5090")
        assert not h.is_ancestor("I5090", "I509")
        assert not h.is_ancestor("I50", "E119")
        assert h.get_match_score("I50", "I509") == 0.8
        assert h.get_match_score("I509", "I50") == 0.6
        assert h.get_match_score("I509", "I509") == 1.0
        assert h.get_match_score("I50", "E119") == 0.0
        assert "parent" not in h.child_to_parent
        print("PASS test_hierarchy_from_file")
    finally:
        os.unlink(path)


def test_hierarchy_duplicate_handling():
    import os
    path = _write_temp_csv([["I50", "I509"], ["I51", "I509"], ["I50", "I509"]])
    try:
        h = ICD10Hierarchy(path, log_duplicates=False, has_header=False)
        assert h.child_to_parent["I509"] == "I50"
        assert h.parent_to_children["I50"].count("I509") == 1
        assert h.duplicate_count == 2
        assert h.conflicting_count == 1
        print("PASS test_hierarchy_duplicate_handling")
    finally:
        os.unlink(path)


def test_negation():
    text = "Patients with no history of heart failure"
    idx = text.lower().find("heart failure")
    assert detect_negation(text, idx) is True

    text2 = "Patients with heart failure"
    idx2 = text2.lower().find("heart failure")
    assert detect_negation(text2, idx2) is False

    text3 = "Patient denies chest pain. History of heart failure noted."
    idx3 = text3.lower().find("heart failure")
    assert detect_negation(text3, idx3) is False

    text4 = "Patients with abnormal LVEF are eligible"
    idx4 = text4.lower().find("lvef")
    assert detect_negation(text4, idx4) is False

    print("PASS test_negation")


def test_softmax_exclusion():
    matches = [0.0, 0.0, 0.0, 0.0, 0.9]
    hard_max = max(matches)
    soft_mid = softmax_exclusion(matches, temperature=0.5)
    soft_low_temp = softmax_exclusion(matches, temperature=0.05)
    soft_high_temp = softmax_exclusion(matches, temperature=5.0)

    assert soft_mid < hard_max, "soft exclusion must be less sensitive than hard max"
    assert soft_low_temp > soft_mid > soft_high_temp, "temperature must control sharpness"
    assert softmax_exclusion([], 0.5) == 0.0
    assert softmax_exclusion([0.0, 0.0], 0.5) == 0.0
    print(f"PASS test_softmax_exclusion (hard_max={hard_max:.3f}, "
          f"soft(T=0.05)={soft_low_temp:.3f}, soft(T=0.5)={soft_mid:.3f}, "
          f"soft(T=5.0)={soft_high_temp:.3f})")


def test_matching_indices_end_to_end():
    import os
    hpath = _write_temp_csv([["I50", "I509"]], header=["parent", "child"])
    try:
        hierarchy = ICD10Hierarchy(hpath)

        trial = extract_criteria_stub(
            inclusion_text="Patients with heart failure and LVEF < 40%",
            exclusion_text="Patients with severe renal impairment (creatinine > 2.0 mg/dL)",
        )

        patient_a = PatientState(
            patient_id="A", diagnosis_codes={"I509"},
            lab_values={"LVEF": 30.0, "50912": 2.5},
        )
        m_inc_a, m_exc_a = compute_matching_indices(patient_a, trial, hierarchy)
        assert m_inc_a > 0.5, f"expected strong inclusion match, got {m_inc_a}"
        assert m_exc_a > 0.5, f"expected strong exclusion violation, got {m_exc_a}"

        patient_b = PatientState(
            patient_id="B", diagnosis_codes={"I50"},
            lab_values={"LVEF": 35.0, "50912": 1.0},
        )
        m_inc_b, m_exc_b = compute_matching_indices(patient_b, trial, hierarchy)
        assert 0.0 < m_inc_b < 1.0, f"expected partial hierarchy match, got {m_inc_b}"
        assert m_exc_b < 0.3, f"expected low exclusion score, got {m_exc_b}"

        patient_c = PatientState(
            patient_id="C", diagnosis_codes={"E119"},
            lab_values={"LVEF": 55.0, "50912": 0.9},
        )
        m_inc_c, m_exc_c = compute_matching_indices(patient_c, trial, hierarchy)
        assert m_inc_c < m_inc_b, "unrelated patient should match worse than partial-hierarchy patient"

        print(f"PASS test_matching_indices_end_to_end "
              f"(A: inc={m_inc_a:.2f} exc={m_exc_a:.2f} | "
              f"B: inc={m_inc_b:.2f} exc={m_exc_b:.2f} | "
              f"C: inc={m_inc_c:.2f} exc={m_exc_c:.2f})")
    finally:
        os.unlink(hpath)


def test_administrative_criteria_skipped():
    """Regression test for the fix: administrative criteria must not be
    scored as failed criteria -- a patient who perfectly matches every real
    clinical criterion should get M_inc=1.0 even if the trial also has an
    administrative inclusion line like informed consent."""
    patient = PatientState(patient_id="P1", diagnosis_codes={"I509"})
    trial = Trial(
        nct_id="DEMO",
        inclusion_criteria=[
            Criterion(raw_entity="heart failure", entity_type="diagnosis",
                      entity_code="I509", operator="EXISTS", value=None,
                      is_inclusion=True, severity_weight=1.0),
            Criterion(raw_entity="informed consent", entity_type="administrative",
                      entity_code="NON_CLINICAL", operator="EXISTS", value=None,
                      is_inclusion=True, severity_weight=1.0),
        ],
    )
    m_inc, m_exc = compute_matching_indices(patient, trial)
    assert m_inc == 1.0, f"administrative criterion should be skipped, not scored as a failure; got m_inc={m_inc}"
    print(f"PASS test_administrative_criteria_skipped (m_inc={m_inc:.2f})")


if __name__ == "__main__":
    test_hierarchy_from_file()
    test_hierarchy_duplicate_handling()
    test_negation()
    test_softmax_exclusion()
    test_matching_indices_end_to_end()
    test_administrative_criteria_skipped()
    print("\nAll tests passed.")