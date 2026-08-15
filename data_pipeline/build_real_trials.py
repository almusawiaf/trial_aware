import os
import json
import logging
import requests
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize OpenAI Client
client = OpenAI()

# =====================================================================
# 1. PYDANTIC SCHEMAS (Matches Downstream Graph Requirements)
# =====================================================================

class ExtractedCriterion(BaseModel):
    """Schema for a single extracted clinical criterion from LLM."""
    raw_concept: str = Field(
        description="The exact clinical condition, lab test, or medication mentioned (e.g., 'Type 2 Diabetes', 'HbA1c', 'Metformin')"
    )
    entity_type: Literal["diagnosis", "lab", "prescription"] = Field(
        description="Categorize whether this is a medical diagnosis/condition, a lab test/vital, or a prescription medication."
    )
    operator: Literal["EXISTS", "GT", "LT", "EQ", "GTE", "LTE", "NOT_EXISTS"] = Field(
        description="The logical condition. GT (>), LT (<), GTE (>=), LTE (<=), EQ (==), EXISTS (present), NOT_EXISTS (forbidden)."
    )
    value: Optional[float] = Field(
        default=None, 
        description="Numeric value or threshold associated with the condition (e.g., 7.0 for HbA1c > 7.0%). Null if binary presence."
    )
    is_inclusion: bool = Field(
        description="True if this criterion is an INCLUSION rule (must be met). False if EXCLUSION rule (must NOT be met)."
    )
    severity_weight: float = Field(
        default=1.0, 
        description="Importance weight of the criterion from 0.0 to 1.0 (default 1.0)."
    )


class TrialExtractionResult(BaseModel):
    """Master output schema for a trial's extracted eligibility criteria."""
    trial_id: str
    criteria: List[ExtractedCriterion]


# Schema after standardizing raw entity names to standard ontology codes
class FinalCriterion(BaseModel):
    entity_type: str
    entity_code: str  # E.g., ICD-10 code, MIMIC ITEMID, or RxNorm NDC
    operator: str
    value: Optional[float]
    is_inclusion: bool
    severity_weight: float


class FinalTrialRecord(BaseModel):
    trial_id: str
    criteria: List[FinalCriterion]


# =====================================================================
# 2. STEP 1: FETCH RAW TRIALS FROM CLINICALTRIALS.GOV REST API
# =====================================================================

def fetch_trial_criteria_from_api(nct_id: str) -> str:
    """Fetches raw eligibility criteria text using the ClinicalTrials.gov REST API v2."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    logging.info(f"Fetching protocol for {nct_id} from ClinicalTrials.gov API...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse nested protocol section
        eligibility_module = (
            data.get("protocolSection", {})
                .get("eligibilityModule", {})
        )
        criteria_text = eligibility_module.get("eligibilityCriteria", "")
        
        if not criteria_text:
            logging.warning(f"No eligibility text found for {nct_id}")
            
        return criteria_text
    
    except Exception as e:
        logging.error(f"Failed to fetch data for {nct_id}: {e}")
        return ""


# =====================================================================
# 3. STEP 2: STRUCTURED EXTRACTION VIA OPENAI LLM
# =====================================================================

def extract_structured_criteria(nct_id: str, raw_text: str) -> TrialExtractionResult:
    """Uses OpenAI Structured Outputs API to extract typed medical criteria directly into Pydantic models."""
    logging.info(f"Parsing criteria for {nct_id} via OpenAI GPT Structured Output...")

    system_prompt = (
        "You are an expert Clinical Trial Eligibility Parser. "
        "Analyze the provided eligibility criteria text from a clinical trial and extract structured rules. "
        "Carefully distinguish between Inclusion Criteria (is_inclusion=True) and Exclusion Criteria (is_inclusion=False). "
        "Extract specific diagnosis names, lab test values, and medications."
    )

    user_prompt = f"Trial ID: {nct_id}\n\nEligibility Criteria Text:\n{raw_text}"

    # Use native structured completions via .beta.chat.completions.parse
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=TrialExtractionResult,
        temperature=0.0  # Zero temperature for deterministic extraction
    )

    return completion.choices[0].message.parsed


# =====================================================================
# 4. STEP 3: CONCEPT STANDARDIZATION & MAPPER
# =====================================================================

class ConceptStandardizer:
    """Maps raw extracted strings (e.g., 'Type 2 Diabetes') to standard dataset codes (ICD-10, LOINC/ITEMIDs)."""
    
    def __init__(self):
        # Local mapping table for exact concept resolution
        self.diagnosis_map = {
            "type 2 diabetes": "E119",
            "type 2 diabetes mellitus": "E119",
            "acute kidney failure": "N179",
            "hypertension": "I10",
            "heart failure": "I509"
        }
        
        self.lab_map = {
            "hba1c": "50852",          # MIMIC-III HbA1c ITEMID
            "hemoglobin a1c": "50852",
            "serum creatinine": "50912",# MIMIC-III Creatinine ITEMID
            "creatinine": "50912",
            "glucose": "50931"
        }
        
        self.prescription_map = {
            "metformin": "00093104801", # Sample NDC / RxNorm identifier
            "insulin": "00002821501",
            "lisinopril": "00006020715"
        }

    def normalize(self, extracted: ExtractedCriterion) -> FinalCriterion:
        concept_clean = extracted.raw_concept.strip().lower()
        entity_code = "UNKNOWN"

        if extracted.entity_type == "diagnosis":
            entity_code = self.diagnosis_map.get(concept_clean, f"ICD_{concept_clean.upper().replace(' ', '_')}")
        elif extracted.entity_type == "lab":
            entity_code = self.lab_map.get(concept_clean, f"LAB_{concept_clean.upper().replace(' ', '_')}")
        elif extracted.entity_type == "prescription":
            entity_code = self.prescription_map.get(concept_clean, f"NDC_{concept_clean.upper().replace(' ', '_')}")

        return FinalCriterion(
            entity_type=extracted.entity_type,
            entity_code=entity_code,
            operator=extracted.operator,
            value=extracted.value,
            is_inclusion=extracted.is_inclusion,
            severity_weight=extracted.severity_weight
        )


# =====================================================================
# 5. STEP 4: FULL PIPELINE EXECUTION & SAVING
# =====================================================================

def run_extraction_pipeline(nct_ids: List[str], output_path: str = "data/real_trials.json"):
    standardizer = ConceptStandardizer()
    final_trials_dataset = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for nct_id in nct_ids:
        # 1. Ingest
        raw_text = fetch_trial_criteria_from_api(nct_id)
        if not raw_text:
            continue

        # 2. Extract Structured JSON
        extracted_data = extract_structured_criteria(nct_id, raw_text)

        # 3. Standardize and Map Entities
        processed_criteria = []
        for crit in extracted_data.criteria:
            mapped_crit = standardizer.normalize(crit)
            processed_criteria.append(mapped_crit)

        final_record = FinalTrialRecord(
            trial_id=nct_id,
            criteria=processed_criteria
        )
        
        # Convert Pydantic object to dictionary
        final_trials_dataset.append(final_record.model_dump())

    # 4. Save to target file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_trials_dataset, f, indent=2)

    logging.info(f"Pipeline finished! Saved {len(final_trials_dataset)} processed trials to '{output_path}'.")


if __name__ == "__main__":
    # Test on target clinical trials from ClinicalTrials.gov
    target_nct_ids = [
        "NCT00685828",  # Study on Type 2 Diabetes and Cardiovascular Risk
        "NCT02506828"   # Renal Disease Intervention Trial
    ]
    
    run_extraction_pipeline(target_nct_ids, output_path="data/real_trials.json")