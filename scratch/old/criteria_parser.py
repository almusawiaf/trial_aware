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


class DynamicCriteriaParser:

    # Direct mapping for non-standard abbreviations common in trial criteria
    ACRONYM_MAP = {
        "mci": ("diagnosis", "33183"),  # Mild Cognitive Impairment
        "vad": ("diagnosis", "29040"),  # Vascular Dementia
        "dlb": ("diagnosis", "33182"),  # Dementia with Lewy Bodies
        "inph": (
            "diagnosis",
            "3315",
        ),  # Idiopathic Normal Pressure Hydrocephalus
        "mmse": ("lab", "MMSE_SCORE"),  # Mini-Mental State Examination
        "cdr": ("diagnosis", "CDR_SCORE"),  # Clinical Dementia Rating
        "ad": ("diagnosis", "3310"),  # Alzheimer's Disease
    }

    # Stopwords and generic pronouns to ignore completely
    STOP_ENTITIES = {
        "who",
        "that",
        "them",
        "their",
        "patient",
        "patients",
        "participant",
        "participants",
        "subject",
        "subjects",
        "group",
        "groups",
        "use",
        "assessment",
        "score",
    }

    def __init__(self, ontology_mapper):
        self.mapper = ontology_mapper

    def _sanitize_text(self, text: str) -> str:
        """Cleans raw text of markdown formatting, unmatched parentheses, and leading operators."""
        # 1. Strip markdown / regex escape artifacts like (\>10 -> 10
        cleaned = re.sub(r"[\(\)\[\]\\/]", " ", text)
        # 2. Strip leading operational symbols
        cleaned = re.sub(r"^[\s>\=<]+", "", cleaned)
        # 3. Collapse whitespace
        return " ".join(cleaned.split()).strip()

    def extract_numeric_bounds(
        self, text: str
    ) -> Tuple[str, Optional[float], Optional[float]]:
        """Extracts operators, lower bounds, and optional upper bounds from clinical text."""
        # 1. Range: "between X and Y" or "X to Y" or "X-Y"
        match_between = re.search(
            r"\b(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\b",
            text,
            re.IGNORECASE,
        )
        if match_between:
            val1 = float(match_between.group(1))
            val2 = float(match_between.group(2))
            return "BETWEEN", min(val1, val2), max(val1, val2)

        # 2. Relational operators (>=, <=, >, <, =) including escaped backslashes
        match_op = re.search(r"(?:\\>|\\<|>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)", text)
        if match_op:
            raw_op = match_op.group(0)
            val = float(match_op.group(1))
            if ">" in raw_op:
                return "GT" if ">" in raw_op and "=" not in raw_op else "GTE", val, None
            if "<" in raw_op:
                return "LT" if "<" in raw_op and "=" not in raw_op else "LTE", val, None
            return "EQ", val, None

        # 3. Textual operators (e.g., "at least 18", "greater than 10")
        match_desc_gt = re.search(
            r"\b(?:at least|greater than|more than|above)\s+(\d+(?:\.\d+)?)\b",
            text,
            re.IGNORECASE,
        )
        if match_desc_gt:
            return "GTE", float(match_desc_gt.group(1)), None

        match_desc_lt = re.search(
            r"\b(?:at most|less than|fewer than|under|below)\s+(\d+(?:\.\d+)?)\b",
            text,
            re.IGNORECASE,
        )
        if match_desc_lt:
            return "LTE", float(match_desc_lt.group(1)), None

        return "EXISTS", None, None

    def parse_criterion_line(
        self, line: str, is_inclusion: bool
    ) -> Optional[CriteriaTriplet]:
        """Parses a single text line into a structured CriteriaTriplet."""
        raw_clean = line.strip("-*• ").strip()
        if not raw_clean:
            return None

        # Sanitize entity text
        sanitized_entity = self._sanitize_text(raw_clean)
        entity_lower = sanitized_entity.lower()

        # 1. Filter out simple generic words/pronouns
        if entity_lower in self.STOP_ENTITIES or len(entity_lower) < 2:
            return None

        # 2. Guard Clause: Skip Age/Demographic metadata
        if re.search(
            r"\b(age|years old|male|female|sex|gender)\b",
            sanitized_entity,
            re.IGNORECASE,
        ):
            return None

        # 3. Guard Clause: Skip Administrative/Operational terms
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

        # 4. Check Direct Acronym Lookup Table
        if entity_lower in self.ACRONYM_MAP:
            entity_type, entity_code = self.ACRONYM_MAP[entity_lower]
        else:
            # Match concept against MIMIC-III dynamic ontology index
            _, entity_type, entity_code = self.mapper.match_term(
                sanitized_entity
            )

        # 5. Extract numeric operators and values
        operator, value, max_value = self.extract_numeric_bounds(raw_clean)

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