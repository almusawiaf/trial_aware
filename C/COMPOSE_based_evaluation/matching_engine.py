"""
matching_engine.py
Optimized Matching Engine with Hierarchy Caching and Direct Set Intersections
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


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


class ICD10Hierarchy:
    def __init__(self, hierarchy_file: Optional[str] = None, log_duplicates: bool = True, has_header: bool = True):
        self.parent_to_children: Dict[str, List[str]] = {}
        self.child_to_parent: Dict[str, str] = {}
        self.duplicate_count = 0
        self.conflicting_count = 0

        if hierarchy_file:
            self.load_from_file(hierarchy_file, log_duplicates=log_duplicates, has_header=has_header)

    def load_from_file(self, file_path: str, log_duplicates: bool = True, has_header: bool = True) -> None:
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
                else:
                    self.child_to_parent[child] = parent

    @lru_cache(maxsize=65536)
    def get_ancestors(self, code: str) -> Set[str]:
        ancestors: Set[str] = set()
        current = code
        while current in self.child_to_parent:
            parent = self.child_to_parent[current]
            ancestors.add(parent)
            current = parent
        return ancestors

    def is_ancestor(self, ancestor_code: str, descendant_code: str) -> bool:
        if ancestor_code == descendant_code:
            return True
        return ancestor_code in self.get_ancestors(descendant_code)

    @lru_cache(maxsize=131072)
    def get_match_score(self, code1: str, code2: str) -> float:
        if code1 == code2:
            return 1.0
        if self.is_ancestor(code1, code2):
            return 0.8
        if self.is_ancestor(code2, code1):
            return 0.6
        return 0.0


def softmax_exclusion(matches: List[float], temperature: float = 0.5) -> float:
    if not matches:
        return 0.0
    arr = np.asarray(matches, dtype=np.float32)
    if np.all(arr == 0):
        return 0.0
    exp_arr = np.exp(arr / temperature)
    weights = exp_arr / np.sum(exp_arr)
    return float(np.sum(weights * arr))


def _sigmoid(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-k * x))


def _normalize_code(entity_type: str, code: str) -> str:
    """Same fix as trial_embedding.py: diagnosis dots stripped to match the
    crosswalk, and medication float-artifact '.0' + lost leading zeros
    recovered heuristically (see trial_embedding.normalize_entity_code)."""
    c = str(code).strip()
    if entity_type == "diagnosis":
        c = c.replace(".", "").upper()
    elif entity_type == "medication":
        c = re.sub(r'\.0+$', '', c)
        c = re.sub(r'[^0-9]', '', c)
        if c.isdigit() and len(c) < 11:
            c = c.zfill(11)
    return c


def _is_placeholder_code(code: str) -> bool:
    """True for criteria the upstream extractor never resolved to a real
    code (e.g. 'UNMATCHED_47327'). These should be excluded from scoring
    entirely -- they are not failed matches, they were never real codes."""
    c = str(code)
    return c.startswith("UNMATCHED_") or c.strip() == "" or c.lower() == "none"


def match_with_hierarchy_single(state: PatientState, criterion: Criterion, hierarchy: Optional[ICD10Hierarchy] = None) -> Optional[float]:
    """Returns None (meaning: exclude from scoring) for criteria we cannot
    actually assess, instead of silently scoring them as failed."""
    if _is_placeholder_code(criterion.entity_code):
        return None

    c_type = criterion.entity_type
    c_code = _normalize_code(c_type, criterion.entity_code)

    if c_type == "diagnosis":
        patient_codes = {_normalize_code(c_type, d) for d in state.diagnosis_codes}
        if c_code in patient_codes:
            return 1.0
        if hierarchy is not None:
            best = 0.0
            for p_code in patient_codes:
                score = hierarchy.get_match_score(p_code, c_code)
                if score > best:
                    best = score
                    if best == 1.0:
                        return 1.0
            return best
        return 0.0

    elif c_type == "medication":
        return 1.0 if c_code in state.medication_codes else 0.0

    elif c_type == "lab":
        if c_code not in state.lab_values:
            return 0.0
        v = state.lab_values[c_code]
        op = criterion.operator
        if op == "EXISTS":
            return 1.0
        if op == "GT":
            return _sigmoid(v - criterion.value)
        if op == "LT":
            return _sigmoid(criterion.value - v)
        if op == "EQ":
            return _sigmoid(1.0 - abs(v - criterion.value))

    return 0.0


def compute_matching_indices(state: PatientState, trial: Trial, hierarchy: Optional[ICD10Hierarchy] = None, temperature: float = 0.5) -> Tuple[float, float]:
    inc_scores, inc_weights = [], []
    for c in trial.inclusion_criteria:
        if c.entity_type == "administrative":
            continue
        score = match_with_hierarchy_single(state, c, hierarchy)
        if score is None:
            # Unresolvable criterion (e.g. placeholder code) -- exclude
            # entirely rather than silently counting it as a failed match.
            continue
        inc_scores.append(score * c.severity_weight)
        inc_weights.append(c.severity_weight)
    
    total_w = sum(inc_weights)
    m_inc = (sum(inc_scores) / total_w) if total_w > 0 else 1.0

    exc_scores = [
        score for c in trial.exclusion_criteria
        if c.entity_type != "administrative"
        and (score := match_with_hierarchy_single(state, c, hierarchy)) is not None
    ]
    m_exc = softmax_exclusion(exc_scores, temperature) if exc_scores else 0.0

    return m_inc, m_exc


def compute_strict_trial_match(
    state: PatientState,
    trial: Trial,
    hierarchy: Optional[ICD10Hierarchy] = None,
    match_threshold: float = 0.5,
) -> Optional[bool]:
    """
    Implements COMPOSE's exact trial-level accuracy definition:

        "a patient matches a trial only when the patient matches ALL
        inclusion criteria and mismatches ALL exclusion criteria"

    This is an all-or-nothing AND over every individual criterion --
    completely different from compute_matching_indices() above, which
    returns a soft, averaged score (m_inc/m_exc). Here, ONE failed
    inclusion criterion or ONE triggered exclusion criterion fails the
    WHOLE trial, no partial credit.

    match_threshold: a per-criterion score >= this value counts as
    "matched". 0.5 is a reasonable midpoint default given our scoring
    functions (exact match = 1.0, hierarchy partial match = 0.6/0.8,
    lab sigmoid centered at the threshold value = 0.5) -- but this is a
    real hyperparameter, not a fact of nature. If you tune it, tune it on
    a held-out validation split, never on the same data you report
    final numbers on -- otherwise you're threshold-shopping against your
    own test set, which invalidates the comparison.

    Returns:
        True  -- patient satisfies every assessable inclusion criterion
                 AND fails every assessable exclusion criterion.
        False -- fails at least one assessable inclusion criterion, OR
                 triggers at least one assessable exclusion criterion.
        None  -- trial has ZERO assessable criteria at all (every single
                 one was administrative or unresolvable/placeholder) --
                 there's nothing to judge, so returning a boolean here
                 would be fabricating a label from no information. Filter
                 these out before computing accuracy rather than silently
                 counting them as a match OR a mismatch.

    NOTE on unresolved/placeholder criteria: same principle as
    compute_matching_indices() -- a criterion we can't actually assess
    (administrative, or a code that never resolved to something real) is
    EXCLUDED from the requirement entirely, not counted as an automatic
    pass or fail. This differs from a literal reading of "ALL criteria"
    in the quoted definition, which implicitly assumes every criterion
    IS assessable (COMPOSE's data doesn't have our upstream
    code-extraction placeholder problem). Document this deviation
    explicitly if you report this metric.
    """
    any_assessable = False

    for c in trial.inclusion_criteria:
        if c.entity_type == "administrative":
            continue
        score = match_with_hierarchy_single(state, c, hierarchy)
        if score is None:
            continue  # unresolvable -- excluded from the requirement, see docstring
        any_assessable = True
        if score < match_threshold:
            return False  # one failed inclusion criterion fails the whole trial

    for c in trial.exclusion_criteria:
        if c.entity_type == "administrative":
            continue
        score = match_with_hierarchy_single(state, c, hierarchy)
        if score is None:
            continue
        any_assessable = True
        if score >= match_threshold:
            return False  # one triggered exclusion criterion fails the whole trial

    if not any_assessable:
        return None  # nothing to judge -- see docstring

    return True