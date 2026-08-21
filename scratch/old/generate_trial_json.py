import json
import re
from typing import Any, Dict, List, Optional, Tuple
from config import Config
from ontology_loader import DynamicOntologyMapper
from pydantic import BaseModel, Field
import requests
import spacy

# Load SpaCy model
nlp = spacy.load("en_ner_bc5cdr_md")

# Initialize global configuration and dynamic mapper
config = Config()
mapper = DynamicOntologyMapper()
mapper.load_icd9_and_patient_tables(data_dir=config.DATA_DIR)

# Operator mapping
OPERATOR_MAP = {
    "<=": "LTE",
    "=<": "LTE",
    "≤": "LTE",
    "less than or equal to": "LTE",
    "<": "LT",
    "less than": "LT",
    ">=": "GTE",
    "=>": "GTE",
    "≥": "GTE",
    "greater than or equal to": "GTE",
    "at least": "GTE",
    ">": "GT",
    "greater than": "GT",
    "more than": "GT",
    "=": "EQ",
    "equal to": "EQ",
}


# -------------------------------------------------------------------
# 1. Schema Definitions
# -------------------------------------------------------------------
class CriterionTriplet(BaseModel):
    raw_entity: str
    entity_type: str
    entity_code: str
    operator: str = "EXISTS"
    value: Optional[float] = None
    is_inclusion: bool
    severity_weight: float = 1.0


class StructuredTrialRecord(BaseModel):
    nct_id: str
    title: str
    conditions: List[str]
    phase: str
    sample_size: Optional[int] = None
    criteria: List[CriterionTriplet] = Field(default_factory=list)


# -------------------------------------------------------------------
# 2. Dynamic Criteria Parser
# -------------------------------------------------------------------
class DynamicCriteriaParser:

    def clean_rule_text(self, text: str) -> str:
        return re.sub(r"^\s*[\*\-\•\d+\.]+\s*", "", text).strip()

    def _extract_operator_and_value(
        self, text: str
    ) -> Tuple[Optional[str], Optional[float]]:
        regex_match = re.search(
            r"(<=|=<|≤|<|>=|=>|≥|>|=)\s*([0-9]+(?:\.[0-9]+)?)", text
        )
        if regex_match:
            symbol, val_str = regex_match.groups()
            return OPERATOR_MAP.get(symbol, "EQ"), float(val_str)

        for symbol, op_code in OPERATOR_MAP.items():
            if symbol in text.lower():
                val_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
                if val_match:
                    return op_code, float(val_match.group(1))

        return None, None

    def _extract_entities(
        self, doc: spacy.tokens.Doc
    ) -> List[Tuple[str, str, str]]:
        found_entities = []

        # Target both recognized Entities and Noun Chunks
        candidates = [ent.text for ent in doc.ents] + [
            chunk.text for chunk in doc.noun_chunks
        ]

        for cand in candidates:
            cand_clean = cand.strip()
            if len(cand_clean) < 3:
                continue

            # Query dynamic MIMIC mapper
            _, e_type, e_code = mapper.match_term(cand_clean, threshold=72.0)
            found_entities.append((cand_clean, e_type, e_code))

        # Deduplicate results
        deduped = list({item[0]: item for item in found_entities}.values())
        return deduped

    def parse_criteria_line(
        self, text: str, is_inclusion_context: bool
    ) -> List[CriterionTriplet]:
        cleaned_text = self.clean_rule_text(text)
        if len(cleaned_text) < 4:
            return []

        # 🚨 Guard clause: Ignore demographic/age lines
        if re.search(
            r"\b(years? old|age|aged|male|female|gender|sex|patients? aged)\b",
            cleaned_text,
            re.IGNORECASE,
        ):
            return []

        doc = nlp(cleaned_text)
        op_type, val = self._extract_operator_and_value(cleaned_text)
        entities = self._extract_entities(doc)

        has_negation = any(
            token.lower_ in ["no", "not", "without", "except", "exception"]
            for token in doc
        )
        effective_is_inclusion = (
            is_inclusion_context if not has_negation else not is_inclusion_context
        )

        extracted_triplets = []
        for entity_name, entity_type, entity_code in entities:
            triplet = CriterionTriplet(
                raw_entity=entity_name,
                entity_type=entity_type,
                entity_code=entity_code,
                operator=op_type if op_type else "EXISTS",
                value=val,
                is_inclusion=effective_is_inclusion,
                severity_weight=1.0,
            )
            extracted_triplets.append(triplet)

        return extracted_triplets

    def parse_full_eligibility_text(
        self, raw_criteria: str
    ) -> List[CriterionTriplet]:
        all_triplets = []
        is_inclusion_context = True
        lines = raw_criteria.split("\n")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if re.search(r"inclusion criteria", line_str, re.IGNORECASE):
                is_inclusion_context = True
                continue
            elif re.search(r"exclusion criteria", line_str, re.IGNORECASE):
                is_inclusion_context = False
                continue

            triplets = self.parse_criteria_line(
                line_str, is_inclusion_context=is_inclusion_context
            )
            all_triplets.extend(triplets)

        return all_triplets


# -------------------------------------------------------------------
# 3. Trial Fetcher & JSON Generator
# -------------------------------------------------------------------
class ClinicalTrialJsonExtractor:

    def __init__(self):
        self.parser = DynamicCriteriaParser()
        self.api_url = "https://clinicaltrials.gov/api/v2/studies"

    def fetch_raw_trials(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        params = {
            "query.cond": query,
            "pageSize": max_results,
            "fields": "NCTId,BriefTitle,ConditionsModule,DesignModule,EligibilityModule",
        }
        response = requests.get(self.api_url, params=params, timeout=12)
        if response.status_code == 200:
            return response.json().get("studies", [])
        return []

    def generate_json_records(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        raw_studies = self.fetch_raw_trials(query, max_results)
        structured_records = []

        for study in raw_studies:
            protocol = study.get("protocolSection", {})

            nct_id = protocol.get("identificationModule", {}).get(
                "nctId", "UNKNOWN"
            )
            title = protocol.get("identificationModule", {}).get(
                "briefTitle", "No Title"
            )
            conditions = (
                protocol.get("conditionsModule", {}).get("conditions", [])
            )

            design = protocol.get("designModule", {})
            phases = design.get("phases", ["Not specified"])
            phase_str = ", ".join(phases)
            sample_size = design.get("enrollmentInfo", {}).get("count", None)

            eligibility = protocol.get("eligibilityModule", {})
            raw_criteria = eligibility.get("eligibilityCriteria", "")
            parsed_triplets = self.parser.parse_full_eligibility_text(
                raw_criteria
            )

            record = StructuredTrialRecord(
                nct_id=nct_id,
                title=title,
                conditions=conditions,
                phase=phase_str,
                sample_size=sample_size,
                criteria=parsed_triplets,
            )
            structured_records.append(record.model_dump())

        return structured_records


if __name__ == "__main__":

    extractor = ClinicalTrialJsonExtractor()
    query_term = "Alzheimer"
    output_filename = "structured_clinical_trials.json"

    print(f"Fetching and processing trials for: '{query_term}'...")
    json_data = extractor.generate_json_records(
        query=query_term, max_results=5
    )

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Processed {len(json_data)} trials successfully.")
    print(f"[OUTPUT] Dataset saved to: {output_filename}")