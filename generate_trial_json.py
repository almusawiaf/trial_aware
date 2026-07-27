import json
import re
from typing import Any, Dict, List, Optional, Tuple
from config import Config
from ontology_loader import DynamicOntologyMapper
from pydantic import BaseModel, Field
import requests
import spacy

from criteria_parser import (
    ACRONYM_MAP,
    extract_numeric_bounds,
    find_nearest_entity,
    is_negated_near,
    is_stop_entity,
    sanitize_entity_text,
)

nlp = spacy.load("en_ner_bc5cdr_md")

config = Config()
mapper = DynamicOntologyMapper()
mapper.load_icd9_and_patient_tables(data_dir=config.DATA_DIR)

FUZZY_MATCH_THRESHOLD = 85.0


class CriterionTriplet(BaseModel):
    raw_entity: str
    entity_type: str
    entity_code: str
    operator: str = "EXISTS"
    value: Optional[float] = None
    max_value: Optional[float] = None
    is_inclusion: bool
    severity_weight: float = 1.0


class StructuredTrialRecord(BaseModel):
    nct_id: str
    title: str
    conditions: List[str]
    phase: str
    sample_size: Optional[int] = None
    criteria: List[CriterionTriplet] = Field(default_factory=list)


class DynamicCriteriaParser:

    def clean_rule_text(self, text: str) -> str:
        return re.sub(r"^\s*[\*\-\•\d+\.]+\s*", "", text).strip()

    def _extract_entities(self, doc: "spacy.tokens.Doc") -> List[str]:
        candidates = [ent.text for ent in doc.ents] + [
            chunk.text for chunk in doc.noun_chunks
        ]

        cleaned_candidates = []
        seen = set()

        for cand in candidates:
            # 1. Split compound entities separated by slashes or 'or'/'and'
            sub_terms = re.split(r"\s+(?:or|and)\s+|[\/]", cand, flags=re.IGNORECASE)

            for term in sub_terms:
                cand_clean = sanitize_entity_text(term)
                
                if is_stop_entity(cand_clean):
                    continue
                    
                key = cand_clean.lower()
                if key in seen:
                    continue
                    
                seen.add(key)
                cleaned_candidates.append(cand_clean)

        return cleaned_candidates

    def _resolve_entity_type_code(self, entity_text: str) -> Tuple[str, str]:
        entity_lower = entity_text.lower()
        if entity_lower in ACRONYM_MAP:
            return ACRONYM_MAP[entity_lower]
        _, entity_type, entity_code = mapper.match_term(
            entity_text, threshold=FUZZY_MATCH_THRESHOLD
        )
        return entity_type, entity_code

    def parse_criteria_line(
        self, text: str, is_inclusion_context: bool
    ) -> List[CriterionTriplet]:
        cleaned_text = self.clean_rule_text(text)
        if len(cleaned_text) < 4:
            return []

        if re.search(
            r"\b(years? old|age|aged|male|female|gender|sex|patients? aged)\b",
            cleaned_text,
            re.IGNORECASE,
        ):
            return []

        doc = nlp(cleaned_text)
        op_type, val, max_val, number_span = extract_numeric_bounds(cleaned_text)
        entities = self._extract_entities(doc)

        if not entities:
            return []

        nearest = None
        # 🚨 شبكة الأمان: إذا وجدنا رقمًا حقيقيًا بالسطر
        if op_type != "EXISTS":
            nearest = find_nearest_entity(cleaned_text, entities, number_span)
            
            # إذا فشلت المطابقة الحرفية (Silent Drop Hazard)، تفادَ الضياع وأسند للكيان الأول كـ Fallback
            if nearest is None and len(entities) > 0:
                nearest = entities[0]
                # print(f"⚠️ [FALLBACK APPLIED] Numeric bound {op_type}:{val} assigned to fallback entity '{nearest}' in line: '{text}'")

        extracted_triplets = []
        for entity_name in entities:
            entity_type, entity_code = self._resolve_entity_type_code(entity_name)
            negated = is_negated_near(cleaned_text, entity_name)
            effective_is_inclusion = (
                is_inclusion_context if not negated else not is_inclusion_context
            )

            # الاقتران بالرقم فقط إذا كان هو الكيان المحدد أو الكيان الاحتياطي (Fallback)
            if entity_name == nearest:
                triplet = CriterionTriplet(
                    raw_entity=entity_name,
                    entity_type=entity_type,
                    entity_code=entity_code,
                    operator=op_type,
                    value=val,
                    max_value=max_val,
                    is_inclusion=effective_is_inclusion,
                    severity_weight=1.0,
                )
            else:
                triplet = CriterionTriplet(
                    raw_entity=entity_name,
                    entity_type=entity_type,
                    entity_code=entity_code,
                    operator="EXISTS",
                    value=None,
                    max_value=None,
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

            nct_id = protocol.get("identificationModule", {}).get("nctId", "UNKNOWN")
            title = protocol.get("identificationModule", {}).get(
                "briefTitle", "No Title"
            )
            conditions = protocol.get("conditionsModule", {}).get("conditions", [])

            design = protocol.get("designModule", {})
            phases = design.get("phases", ["Not specified"])
            phase_str = ", ".join(phases)
            sample_size = design.get("enrollmentInfo", {}).get("count", None)

            eligibility = protocol.get("eligibilityModule", {})
            raw_criteria = eligibility.get("eligibilityCriteria", "")
            parsed_triplets = self.parser.parse_full_eligibility_text(raw_criteria)

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
    json_data = extractor.generate_json_records(query=query_term, max_results=5)

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Processed {len(json_data)} trials successfully.")
    print(f"[OUTPUT] Dataset saved to: {output_filename}")