"""
criteria_parser.py
-------------------
Single Source of Truth for Clinical Trial Criteria Parsing.
Contains common cleaning, acronym resolution, and numerical bound logic.
"""

import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class CriteriaTriplet(BaseModel):
    raw_entity: str
    entity_type: str
    entity_code: str
    operator: str = "EXISTS"
    value: Optional[float] = None
    max_value: Optional[float] = Field(
        default=None,
        description="Upper bound for range operators like BETWEEN",
    )
    is_inclusion: bool = True
    severity_weight: float = 1.0


ACRONYM_MAP = {
    "mci": ("diagnosis", "33183"),
    "vad": ("diagnosis", "29040"),
    "dlb": ("diagnosis", "33182"),
    "inph": ("diagnosis", "3315"),
    "mmse": ("lab", "MMSE_SCORE"),
    "cdr": ("diagnosis", "CDR_SCORE"),
    "ad": ("diagnosis", "3310"),
}

STOP_ENTITIES = {
    "who", "that", "them", "their", "this", "these", "those",
    "patient", "patients", "participant", "participants",
    "subject", "subjects", "group", "groups",
    "use", "assessment", "score", "history",
    "the written icf", "icf", "legally acceptable representative",
    "a legal custodian", "legal custodian",
}

NEGATION_WORDS = {"no", "not", "without", "except", "exception", "excluding"}


def sanitize_entity_text(text: str) -> str:
    """
    Cleans entity text from:
    - Escaped characters and backslashes
    - Parentheses, brackets, and slashes (converting 'MCI/dementia' -> 'MCI dementia')
    - Leading operator/markdown symbols
    - Duplicate whitespace
    """
    cleaned = text
    cleaned = cleaned.replace("\\", " ")
    cleaned = re.sub(r"[\(\)\[\]\/]", " ", cleaned)
    cleaned = re.sub(r"[<>=]+", " ", cleaned)
    cleaned = re.sub(r"^[\s\-\*•\.]+", "", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned


def is_stop_entity(entity_text: str) -> bool:
    """Checks if the candidate entity is a generic word/stopword."""
    normalized = entity_text.strip().lower()
    
    # Allow 2-letter acronyms if explicitly mapped in ACRONYM_MAP (e.g., 'ad')
    if len(normalized) < 3 and normalized not in ACRONYM_MAP:
        return True
        
    if normalized in STOP_ENTITIES:
        return True
        
    return False


def extract_numeric_bounds(
    text: str,
) -> Tuple[str, Optional[float], Optional[float], Optional[Tuple[int, int]]]:
    """Extracts numeric operators, values, ranges, and positional character spans."""
    match_between = re.search(
        r"\b(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\b",
        text,
        re.IGNORECASE,
    )
    if match_between:
        val1 = float(match_between.group(1))
        val2 = float(match_between.group(2))
        return "BETWEEN", min(val1, val2), max(val1, val2), match_between.span()

    match_op = re.search(r"(?:\\>|\\<|>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)", text)
    if match_op:
        raw_op = match_op.group(0)
        val = float(match_op.group(1))
        if ">" in raw_op:
            op = "GTE" if "=" in raw_op else "GT"
        elif "<" in raw_op:
            op = "LTE" if "=" in raw_op else "LT"
        else:
            op = "EQ"
        return op, val, None, match_op.span()

    match_desc_gt = re.search(
        r"\b(?:at least|greater than|more than|above)\s+(\d+(?:\.\d+)?)\b",
        text,
        re.IGNORECASE,
    )
    if match_desc_gt:
        return "GTE", float(match_desc_gt.group(1)), None, match_desc_gt.span()

    match_desc_lt = re.search(
        r"\b(?:at most|less than|fewer than|under|below)\s+(\d+(?:\.\d+)?)\b",
        text,
        re.IGNORECASE,
    )
    if match_desc_lt:
        return "LTE", float(match_desc_lt.group(1)), None, match_desc_lt.span()

    return "EXISTS", None, None, None


def find_nearest_entity(
    text: str, candidates: List[str], number_span: Optional[Tuple[int, int]]
) -> Optional[str]:
    """Identifies candidate entity closest in character distance to the numeric span."""
    if number_span is None or not candidates:
        return None

    num_center = (number_span[0] + number_span[1]) / 2
    best_entity = None
    best_distance = float("inf")

    for cand in candidates:
        idx = text.lower().find(cand.lower())
        if idx == -1:
            continue
        cand_center = idx + len(cand) / 2
        distance = abs(cand_center - num_center)
        if distance < best_distance:
            best_distance = distance
            best_entity = cand

    return best_entity


def is_negated_near(text: str, entity_text: str, window_chars: int = 25) -> bool:
    """Checks for negation keywords strictly within a local character window around entity."""
    idx = text.lower().find(entity_text.lower())
    if idx == -1:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return any(t in NEGATION_WORDS for t in tokens)

    start = max(0, idx - window_chars)
    end = min(len(text), idx + len(entity_text) + window_chars)
    local_window = text[start:end].lower()
    local_tokens = re.findall(r"\b\w+\b", local_window)
    return any(t in NEGATION_WORDS for t in local_tokens)


class DynamicCriteriaParser:
    ACRONYM_MAP = ACRONYM_MAP
    STOP_ENTITIES = STOP_ENTITIES

    def __init__(self, ontology_mapper):
        self.mapper = ontology_mapper

    def _sanitize_text(self, text: str) -> str:
        return sanitize_entity_text(text)

    def extract_numeric_bounds(
        self, text: str
    ) -> Tuple[str, Optional[float], Optional[float]]:
        op, val, max_val, _span = extract_numeric_bounds(text)
        return op, val, max_val

    def parse_criterion_line(
        self, line: str, is_inclusion: bool
    ) -> Optional[CriteriaTriplet]:
        raw_clean = line.strip("-*• ").strip()
        if not raw_clean:
            return None

        sanitized_entity = self._sanitize_text(raw_clean)
        entity_lower = sanitized_entity.lower()

        if is_stop_entity(entity_lower):
            return None

        if re.search(
            r"\b(age|years old|male|female|sex|gender)\b",
            sanitized_entity,
            re.IGNORECASE,
        ):
            return None

        if self.mapper.is_administrative(sanitized_entity):
            return CriteriaTriplet(
                raw_entity=sanitized_entity,
                entity_type="administrative",
                entity_code="NON_CLINICAL",
                operator="EXISTS",
                value=None,
                max_value=None,
                is_inclusion=is_inclusion,
                severity_weight=1.0,
            )

        if entity_lower in self.ACRONYM_MAP:
            entity_type, entity_code = self.ACRONYM_MAP[entity_lower]
        else:
            _, entity_type, entity_code = self.mapper.match_term(sanitized_entity)

        operator, value, max_value = self.extract_numeric_bounds(raw_clean)

        if is_negated_near(raw_clean, sanitized_entity):
            is_inclusion = not is_inclusion

        return CriteriaTriplet(
            raw_entity=sanitized_entity,
            entity_type=entity_type,
            entity_code=entity_code,
            operator=operator,
            value=value,
            max_value=max_value,
            is_inclusion=is_inclusion,
            severity_weight=1.0,
        )