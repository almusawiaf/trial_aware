# generate_trial_json.py
import json
import logging
import os
import requests
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import Config
from ontology_loader import DynamicOntologyMapper
from preprocessor import MIMICDataPreprocessor

from clinical_trials_api import ClinicalTrialsFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TrialCriteriaParser:
    """Parse raw eligibility criteria into structured format."""
    
    def __init__(self, mapper: DynamicOntologyMapper):
        self.mapper = mapper
        
        # Common medical entities and their patterns
        self.entity_patterns = {
            'diagnosis': ['diagnosis of', 'diagnosed with', 'history of', 'has', 'with'],
            'medication': ['treated with', 'taking', 'on', 'using', 'medication'],
            'lab': ['creatinine', 'hemoglobin', 'glucose', 'blood', 'pressure', 'LVEF'],
            'procedure': ['surgery', 'procedure', 'transplant', 'bypass', 'stent'],
        }
        
        # Operator mapping from text
        self.operator_map = {
            '>': 'GT', 'greater than': 'GT', 'above': 'GT', 'higher than': 'GT',
            '>=': 'GTE', 'greater than or equal': 'GTE', 'at least': 'GTE',
            '<': 'LT', 'less than': 'LT', 'below': 'LT', 'lower than': 'LT',
            '<=': 'LTE', 'less than or equal': 'LTE', 'at most': 'LTE',
            '=': 'EQ', 'equal to': 'EQ', 'exactly': 'EQ',
            'between': 'BETWEEN',
            'exists': 'EXISTS',
            'not exists': 'NOT_EXISTS',
        }
        
        # Normal ranges for lab values (for normalization)
        self.lab_ranges = {
            'creatinine': (0.4, 1.5),
            'hemoglobin': (10, 18),
            'glucose': (70, 140),
            'potassium': (3.5, 5.5),
            'sodium': (135, 145),
            'LVEF': (0, 100),
        }
    
    def parse_criteria_text(self, criteria_text: str) -> List[Dict]:
        """
        Parse raw eligibility criteria text into structured triplets.
        This is a simplified stub - you'd want a proper NER+RE system here.
        """
        structured = []
        
        # Split into lines/sentences
        lines = [l.strip() for l in criteria_text.split('\n') if l.strip()]
        
        is_inclusion_section = True
        
        for line in lines:
            # Detect inclusion/exclusion sections
            lower_line = line.lower()
            if 'inclusion' in lower_line:
                is_inclusion_section = True
                continue
            elif 'exclusion' in lower_line:
                is_inclusion_section = False
                continue
            
            # Skip short lines or headings
            if len(line) < 10 or line.endswith(':'):
                continue
            
            # Try to extract entity, operator, value
            criterion = self._extract_single_criterion(line, is_inclusion_section)
            if criterion:
                structured.append(criterion)
        
        return structured
    
    def _extract_single_criterion(self, text: str, is_inclusion: bool) -> Optional[Dict]:
        """Extract a single criterion from text."""
        # Simplified extraction - you'd want more sophisticated parsing
        
        # Try to identify operator
        operator = 'EXISTS'
        value = None
        max_value = None
        
        for op_text, op_code in self.operator_map.items():
            if op_text in text.lower():
                operator = op_code
                # Try to extract numeric value
                import re
                numbers = re.findall(r'(\d+\.?\d*)', text)
                if numbers and op_code in ['GT', 'GTE', 'LT', 'LTE', 'EQ']:
                    value = float(numbers[0])
                elif numbers and op_code == 'BETWEEN' and len(numbers) >= 2:
                    value = float(numbers[0])
                    max_value = float(numbers[1])
                break
        
        # Identify entity type
        entity_type = 'diagnosis'
        entity_code = None
        
        for etype, keywords in self.entity_patterns.items():
            if any(keyword in text.lower() for keyword in keywords):
                entity_type = etype
                break
        
        # Try to map to ontology
        matched_entity = self.mapper.match_entity(text)
        if matched_entity:
            _, entity_type, entity_code = matched_entity
        else:
            # Fallback: create a hash-based code
            import hashlib
            entity_code = f"UNK_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        
        return {
            'raw_entity': text,
            'entity_type': entity_type,
            'entity_code': entity_code or 'UNKNOWN_CODE',
            'operator': operator,
            'value': value,
            'max_value': max_value,
            'is_inclusion': is_inclusion,
            'severity_weight': 1.0
        }


def main():
    cfg = Config()
    
    # Initialize preprocessor & get crosswalk
    pp = MIMICDataPreprocessor(cfg)
    icd10_map = getattr(pp, 'icd9_to_icd10_map', {})
    
    # Initialize ontology mapper
    data_dir = getattr(cfg, 'DATA_DIR', getattr(cfg, 'MIMIC_DIR', './data'))
    mapper = DynamicOntologyMapper(icd9_to_icd10_map=icd10_map)
    mapper.load_icd9_and_patient_tables(data_dir=data_dir)
    
    # Initialize trial fetcher
    fetcher = ClinicalTrialsFetcher()
    parser = TrialCriteriaParser(mapper)
    
    # ============================================================
    # CONFIGURATION: Adjust these parameters to get more trials
    # ============================================================
    
    # List of conditions relevant to your MIMIC cohort
    # Expand this list to get more trials
    conditions = [
        "heart failure",           # Common in MIMIC
        "myocardial infarction",   # Heart attack
        "diabetes",                # Very common
        "pneumonia",               # Common ICU condition
        "sepsis",                  # Critical care
        "acute kidney injury",     # Common in ICU
        "chronic obstructive pulmonary disease",
        "stroke",
        "atrial fibrillation",
        "hypertension",
        "hyperlipidemia",
        "cancer",
        "breast cancer",
        "lung cancer",
        "colorectal cancer",
        "prostate cancer",
        "leukemia",
        "lymphoma",
        "multiple sclerosis",
        "rheumatoid arthritis",
        "osteoarthritis",
        "depression",
        "anxiety",
        "schizophrenia",
        "alzheimer's disease",
        "parkinson's disease",
        "epilepsy",
        "migraine",
        "asthma",
        "chronic kidney disease",
        "liver disease",
        "hepatitis",
        "hiv",
        "tuberculosis",
        "covid-19",
        "influenza",
        "urinary tract infection",
        "deep vein thrombosis",
        "pulmonary embolism",
    ]
    
    # Trial parameters
    max_trials_per_condition = 10   # Increase for more trials
    total_trials_target = 100       # Target total number of trials
    statuses = ["COMPLETED", "RECRUITING", "ACTIVE_NOT_RECRUITING"]
    phases = ["PHASE2", "PHASE3", "PHASE2/PHASE3"]
    min_study_size = 50
    
    # ============================================================
    # Fetch trials
    # ============================================================
    
    all_trials = []
    trial_ids_seen = set()
    
    for status in statuses:
        for phase in phases:
            for condition in conditions:
                if len(all_trials) >= total_trials_target:
                    break
                
                logging.info(f"Fetching trials for: {condition} | Status: {status} | Phase: {phase}")
                
                try:
                    trials = fetcher.search_trials(
                        condition=condition,
                        max_results=min(max_trials_per_condition, total_trials_target - len(all_trials)),
                        status=status,
                        phase=phase,
                        min_study_size=min_study_size,
                        years_back=10
                    )
                    
                    # Deduplicate
                    for trial in trials:
                        nct_id = trial.get('nct_id')
                        if nct_id and nct_id not in trial_ids_seen:
                            trial_ids_seen.add(nct_id)
                            all_trials.append(trial)
                            logging.info(f"  Added trial {nct_id}: {trial.get('title', '')[:50]}...")
                    
                except Exception as e:
                    logging.error(f"Error fetching trials for {condition}: {e}")
                    continue
                
                # Rate limiting
                time.sleep(0.5)
            
            if len(all_trials) >= total_trials_target:
                break
        if len(all_trials) >= total_trials_target:
            break
    
    logging.info(f"Fetched {len(all_trials)} unique trials")
    
    if not all_trials:
        logging.error("No trials fetched! Check API connection.")
        return
    
    # ============================================================
    # Parse criteria and generate structured trials
    # ============================================================
    
    structured_trials = []
    
    for trial_data in all_trials:
        criteria_text = trial_data.get('eligibility_criteria', '')
        
        # Try to parse criteria text
        parsed_criteria = parser.parse_criteria_text(criteria_text)
        
        # If parsing failed or produced no criteria, create a fallback
        if not parsed_criteria:
            logging.warning(f"No criteria parsed for {trial_data.get('nct_id')}, creating fallback criteria")
            parsed_criteria = [
                {
                    "raw_entity": criteria_text[:200] if criteria_text else "Unknown criteria",
                    "entity_type": "diagnosis",
                    "entity_code": "UNKNOWN_CODE",
                    "operator": "EXISTS",
                    "value": None,
                    "max_value": None,
                    "is_inclusion": True,
                    "severity_weight": 1.0
                }
            ]
        
        structured_trial = {
            "nct_id": trial_data.get('nct_id', 'NCT_UNKNOWN'),
            "title": trial_data.get('title', ''),
            "conditions": trial_data.get('conditions', []),
            "phase": trial_data.get('phase', 'PHASE2'),
            "sample_size": trial_data.get('sample_size', 100),
            "criteria": parsed_criteria
        }
        structured_trials.append(structured_trial)
    
    # ============================================================
    # Split into train/evaluation sets
    # ============================================================
    
    # Use 80% for training, 20% for held-out evaluation
    split_idx = int(len(structured_trials) * 0.8)
    
    train_trials = structured_trials[:split_idx]
    eval_trials = structured_trials[split_idx:]
    
    # ============================================================
    # Save outputs
    # ============================================================
    
    # Main file for training (Stage B)
    out_path = "structured_clinical_trials.json"
    with open(out_path, "w") as f:
        json.dump(train_trials, f, indent=2)
    logging.info(f"Saved {len(train_trials)} training trials to {out_path}")
    
    # Held-out evaluation file
    eval_path = "structured_clinical_trials_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_trials, f, indent=2)
    logging.info(f"Saved {len(eval_trials)} evaluation trials to {eval_path}")
    
    # Full dataset for reference
    full_path = "structured_clinical_trials_full.json"
    with open(full_path, "w") as f:
        json.dump(structured_trials, f, indent=2)
    logging.info(f"Saved {len(structured_trials)} total trials to {full_path}")
    
    # Summary statistics
    logging.info("=" * 60)
    logging.info("TRIAL DATASET SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total trials fetched: {len(all_trials)}")
    logging.info(f"Trials with parsed criteria: {len(structured_trials)}")
    logging.info(f"Training trials: {len(train_trials)}")
    logging.info(f"Evaluation trials: {len(eval_trials)}")
    
    # Count by phase
    phase_counts = {}
    for t in structured_trials:
        phase = t.get('phase', 'Unknown')
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    logging.info(f"Trials by phase: {phase_counts}")
    
    # Count by condition
    condition_counts = {}
    for t in structured_trials:
        for cond in t.get('conditions', []):
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
    top_conditions = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logging.info(f"Top 10 conditions: {top_conditions}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()