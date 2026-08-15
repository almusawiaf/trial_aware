"""
Synthetic MIMIC-III-shaped tables + a small set of structured trial-criteria
records, used to run the pipeline end-to-end without real PHI data.
"""
import numpy as np
import pandas as pd

def generate_mock_trials():
    """
    A minimal set of structured trial-criteria records, in the same schema
    `trial_graph.py` expects. In a production system, these triplets are the
    OUTPUT of a clinical NER-RE pipeline (e.g. MedCAT / scispaCy) applied to
    ClinicalTrials.gov free-text eligibility criteria -- that extraction step
    is out of scope here, so this stands in as the structured artifact the
    rest of the pipeline consumes.
    """
    return [
        {
            "trial_id": "NCT_MOCK_001",
            "criteria": [
                {"entity_type": "diagnosis", "entity_code": "E119", "operator": "EXISTS",
                 "value": None, "is_inclusion": True, "severity_weight": 1.0},
                {"entity_type": "lab", "entity_code": "50912", "operator": "GT",
                 "value": 7.0, "is_inclusion": True, "severity_weight": 0.8},
                {"entity_type": "diagnosis", "entity_code": "N179", "operator": "EXISTS",
                 "value": None, "is_inclusion": False, "severity_weight": 1.0},
            ],
        },
        {
            "trial_id": "NCT_MOCK_002",
            "criteria": [
                {"entity_type": "diagnosis", "entity_code": "I509", "operator": "EXISTS",
                 "value": None, "is_inclusion": True, "severity_weight": 1.0},
                {"entity_type": "lab", "entity_code": "50971", "operator": "LT",
                 "value": 5.0, "is_inclusion": True, "severity_weight": 0.6},
                {"entity_type": "medication", "entity_code": "00054829025", "operator": "EXISTS",
                 "value": None, "is_inclusion": False, "severity_weight": 0.9},
            ],
        },
    ]
