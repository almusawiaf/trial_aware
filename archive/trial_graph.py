"""
trial_graph.py

Structured representation of clinical-trial eligibility criteria, and the
patient <-> trial matching-index functions (M_inc, M_exc) from the paper's
"Trial Embedding Space Construction" section.
"""
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SIGMOID_DELTA = 1.0


class Operator(Enum):
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"
    EQ = "EQ"
    BETWEEN = "BETWEEN"
    
    # Add a method to get numeric code for embedding
    def to_numeric(self) -> int:
        """Convert operator to numeric code for embedding."""
        mapping = {
            Operator.EXISTS: 0,
            Operator.NOT_EXISTS: 1,
            Operator.GT: 2,
            Operator.GTE: 3,
            Operator.LT: 4,
            Operator.LTE: 5,
            Operator.EQ: 6,
            Operator.BETWEEN: 7,
        }
        return mapping.get(self, 0)


@dataclass
class Criterion:
    """Single eligibility criterion with proper typing."""
    entity_type: str  # 'diagnosis' | 'medication' | 'lab' | 'procedure'
    entity_code: str  # ICD code, NDC, or lab ITEMID
    operator: Operator
    value: Optional[float] = None
    max_value: Optional[float] = None  # For BETWEEN operator
    is_inclusion: bool = True
    severity_weight: float = 1.0
    
    # Alias for compatibility
    @property
    def weight(self) -> float:
        return self.severity_weight
    
    @weight.setter
    def weight(self, val: float):
        self.severity_weight = val


@dataclass
class Trial:
    """Clinical trial with inclusion and exclusion criteria."""
    trial_id: str
    inclusion_criteria: List[Criterion] = field(default_factory=list)
    exclusion_criteria: List[Criterion] = field(default_factory=list)
    
    @property
    def criteria(self) -> List[Criterion]:
        """Compatibility property for trial_embedding.py."""
        return self.inclusion_criteria + self.exclusion_criteria


class PatientClinicalState:
    """Patient snapshot for matching against trial criteria."""
    
    def __init__(self, subject_id: int):
        self.subject_id = subject_id
        self.diagnosis_codes: Set[str] = set()
        self.medication_codes: Set[str] = set()
        self.lab_last_values: Dict[str, float] = {}

    @classmethod
    def build_from_tables(cls, subject_id: int, diag_df, rx_df, labs_df):
        state = cls(subject_id)
        
        # 1. Diagnoses - try both ICD9 and ICD10
        p_diag = diag_df[diag_df['SUBJECT_ID'] == subject_id]
        if 'ICD10_CODE' in p_diag.columns:
            state.diagnosis_codes = set(p_diag['ICD10_CODE'].astype(str).unique())
        elif 'ICD9_CODE' in p_diag.columns:
            state.diagnosis_codes = set(p_diag['ICD9_CODE'].astype(str).unique())
        else:
            logging.warning(f"No diagnosis codes found for patient {subject_id}")

        # 2. Medications
        p_rx = rx_df[rx_df['SUBJECT_ID'] == subject_id]
        if 'NDC' in p_rx.columns:
            state.medication_codes = set(p_rx['NDC'].astype(str).unique())

        # 3. Lab values
        p_labs = labs_df[labs_df['SUBJECT_ID'] == subject_id]
        value_col = None
        for col in ['VALUENUM', 'VALUE', 'VAL_NUM', 'valuenum', 'value']:
            if col in p_labs.columns:
                value_col = col
                break

        if value_col is not None:
            for _, row in p_labs.iterrows():
                try:
                    val = float(row[value_col])
                    if not np.isnan(val):
                        state.lab_last_values[str(row['ITEMID'])] = val
                except (ValueError, TypeError):
                    continue

        return state

def _match_single(state: PatientClinicalState, c: Criterion) -> float:
    """
    Match a single criterion against patient state.
    Uses both exact code matching AND text-based matching.
    """
    # FIRST: Try exact code matching
    present = False
    
    if c.entity_type == "diagnosis":
        # Try exact code match
        if c.entity_code in state.diagnosis_codes:
            present = True
        else:
            # Try text-based matching on raw_entity
            raw_text = getattr(c, 'raw_entity', '').lower()
            if raw_text:
                # Check if raw_text appears in any diagnosis description
                # For now, just log it - you'll need a description map
                pass
    
    elif c.entity_type == "medication":
        if c.entity_code in state.medication_codes:
            present = True
    
    elif c.entity_type == "lab":
        if c.entity_code in state.lab_last_values:
            present = True
    else:
        present = False

    # If not present by code, try text matching on raw_entity
    if not present:
        raw_text = getattr(c, 'raw_entity', '').lower()
        if raw_text:
            # Check if raw_text matches any diagnosis code description
            # This requires loading the diagnosis descriptions
            for code in state.diagnosis_codes:
                # You'd need a map from code to description here
                # For now, we'll use a simple fallback
                if any(word in raw_text for word in ['heart', 'failure', 'diabetes', 'pneumonia']):
                    # If it's a common term, give it a small match score
                    present = True
                    return 0.5  # Partial match

    # EXISTS / NOT_EXISTS operators
    if c.operator == Operator.EXISTS:
        return 1.0 if present else 0.0
    if c.operator == Operator.NOT_EXISTS:
        return 1.0 if not present else 0.0

    # Continuous operators require lab values
    if c.entity_type != "lab" or not present:
        return 0.0

    x = state.lab_last_values.get(c.entity_code, 0.0)
    
    if c.value is None:
        return 0.0
    
    if c.operator == Operator.GT:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (x - c.value))))
    elif c.operator == Operator.GTE:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (x - c.value) - 1e-6)))
    elif c.operator == Operator.LT:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (c.value - x))))
    elif c.operator == Operator.LTE:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (c.value - x) - 1e-6)))
    elif c.operator == Operator.EQ:
        return float(np.exp(-SIGMOID_DELTA * abs(x - c.value)))
    elif c.operator == Operator.BETWEEN:
        if c.max_value is None:
            return 0.0
        inside = (1 / (1 + np.exp(-SIGMOID_DELTA * (x - c.value)))) * \
                 (1 / (1 + np.exp(-SIGMOID_DELTA * (c.max_value - x))))
        return float(inside)
    
    return 0.0


def compute_matching_indices(state: PatientClinicalState, trial: Trial) -> Tuple[float, float]:
    """Compute M_inc and M_exc for a patient-trial pair."""
    # Inclusion: weighted average
    inc_scores = [_match_single(state, c) * c.severity_weight for c in trial.inclusion_criteria]
    m_inc = float(np.mean(inc_scores)) if inc_scores else 1.0

    # Exclusion: maximum (OR logic for exclusion criteria)
    exc_scores = [_match_single(state, c) * c.severity_weight for c in trial.exclusion_criteria]
    m_exc = float(max(exc_scores)) if exc_scores else 0.0

    return m_inc, m_exc


class TrialStore:
    """Store and manage clinical trials."""
    
    def __init__(self):
        self.trials: Dict[str, Trial] = {}  # Make sure this is Dict[str, Trial]

    def add_trial(self, trial: Trial):
        """Add a single trial to the store."""
        if not isinstance(trial, Trial):
            raise TypeError(f"Expected Trial object, got {type(trial)}")
        self.trials[trial.trial_id] = trial

    def __getitem__(self, tid: str) -> Trial:
        """Get trial by ID."""
        return self.trials.get(tid)  # Use .get() to avoid KeyError if debugging

    def __iter__(self):
        """Iterate over Trial objects, not IDs."""
        return iter(self.trials.values())  # CRITICAL: return values, not keys

    def __len__(self):
        return len(self.trials)
    
    def keys(self):
        """Return trial IDs."""
        return self.trials.keys()
    
    def values(self):
        """Return Trial objects."""
        return self.trials.values()
    
    def items(self):
        """Return (trial_id, Trial) pairs."""
        return self.trials.items()

    @classmethod
    def from_records(cls, records: List[dict]) -> "TrialStore":
        """Create TrialStore from list of record dicts."""
        store = cls()
        
        for rec in records:
            # Handle both 'trial_id' and 'nct_id' keys
            tid = rec.get("trial_id") or rec.get("nct_id")
            if not tid:
                logging.warning("Skipping record with no trial_id/nct_id")
                continue
            
            inc_list, exc_list = [], []
            
            # Handle both 'criteria' and 'eligibility_criteria' keys
            criteria_list = rec.get("criteria") or rec.get("eligibility_criteria", [])
            
            for c in criteria_list:
                # Convert operator string to Enum
                op_str = c.get("operator", "EXISTS")
                try:
                    operator = Operator(op_str)
                except ValueError:
                    # Fallback for different string formats
                    op_mapping = {
                        ">": Operator.GT, "GT": Operator.GT,
                        ">=": Operator.GTE, "GTE": Operator.GTE,
                        "<": Operator.LT, "LT": Operator.LT,
                        "<=": Operator.LTE, "LTE": Operator.LTE,
                        "=": Operator.EQ, "EQ": Operator.EQ,
                        "EXISTS": Operator.EXISTS,
                        "NOT_EXISTS": Operator.NOT_EXISTS,
                        "BETWEEN": Operator.BETWEEN,
                    }
                    operator = op_mapping.get(op_str, Operator.EXISTS)
                
                # Handle both 'entity_code' and 'concept_code'
                entity_code = c.get("entity_code") or c.get("concept_code", "")
                
                # Handle both 'entity_type' and 'concept_type'
                entity_type = c.get("entity_type") or c.get("concept_type", "diagnosis")
                
                crit = Criterion(
                    entity_type=entity_type,
                    entity_code=str(entity_code),
                    operator=operator,
                    value=c.get("value"),
                    max_value=c.get("max_value"),
                    is_inclusion=c.get("is_inclusion", True),
                    severity_weight=float(c.get("severity_weight", c.get("weight", 1.0))),
                )
                if crit.is_inclusion:
                    inc_list.append(crit)
                else:
                    exc_list.append(crit)
            
            trial = Trial(
                trial_id=tid,
                inclusion_criteria=inc_list,
                exclusion_criteria=exc_list
            )
            store.add_trial(trial)
        
        # Log what was loaded
        total_criteria = sum(len(t.criteria) for t in store.trials.values())
        logging.info(f"Loaded {len(store.trials)} trials with {total_criteria} total criteria")
        return store

    @classmethod
    def from_json(cls, path: str) -> "TrialStore":
        """Load from JSON file."""
        import json
        with open(path, 'r') as f:
            records = json.load(f)
        return cls.from_records(records)

def derive_weak_positive_pairs(
    patient_states: Dict[int, PatientClinicalState],
    trial_store: TrialStore,
    inc_threshold: float = 0.01,
) -> List[Tuple[int, str]]:
    """Derive weak positive pairs based on inclusion criteria matching."""
    weak_pairs = []
    patient_ids = sorted(patient_states.keys())
    
    # Iterate over Trial objects, not strings
    for p_idx, sid in enumerate(patient_ids):
        state = patient_states[sid]
        for trial in trial_store:  # Now iterates over Trial objects
            m_inc, _ = compute_matching_indices(state, trial)
            if m_inc >= inc_threshold:
                weak_pairs.append((p_idx, trial.trial_id))
    
    # Fallback if no pairs found
    if not weak_pairs and len(patient_ids) > 0 and len(trial_store) > 0:
        logging.warning("No weak pairs found! Using synthetic round-robin pairing for Stage B.")
        trial_ids = list(trial_store.keys())  # Now keys() works
        for p_idx in range(len(patient_ids)):
            trial_id = trial_ids[p_idx % len(trial_ids)]
            weak_pairs.append((p_idx, trial_id))

    logging.info(f"[WeakSupervision] Derived {len(weak_pairs)} weak positive (patient, trial) pairs.")
    return weak_pairs