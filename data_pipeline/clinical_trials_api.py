"""
clinical_trials_api.py
----------------------
ClinicalTrials.gov API v1 client (more stable than v2)
"""

import logging
import time
import requests
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ClinicalTrialsFetcher:
    """Fetch clinical trials using the v1 API."""
    
    BASE_URL = "https://clinicaltrials.gov/api/query/study_fields"
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'TrialAwareResearch/1.0 (Research Project)'
        })
    
    def search_trials(
        self, 
        condition: str = "", 
        max_results: int = 10,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        min_study_size: Optional[int] = None,
        years_back: Optional[int] = None
    ) -> List[Dict]:
        """
        Search trials using v1 API.
        """
        if not condition:
            logging.warning("No condition provided")
            return []
        
        # Build the search expression
        expr_parts = [condition]  # v1 uses simple condition string
        
        # Add filters if needed (v1 has different filter syntax)
        if phase:
            expr_parts.append(f"AND phase:{phase}")
        
        # For v1, status is handled differently
        if status:
            expr_parts.append(f"AND overall_status:{status}")
        
        expr = " ".join(expr_parts)
        
        # Fields to request
        fields = [
            'NCTId', 'BriefTitle', 'OverallStatus', 'Phase', 
            'EnrollmentCount', 'StudyType', 'EligibilityCriteria',
            'Conditions', 'Interventions', 'StartDate', 'CompletionDate'
        ]
        
        all_trials = []
        min_rnk = 1
        fetched = 0
        
        while fetched < max_results:
            try:
                params = {
                    'expr': expr,
                    'fields': ','.join(fields),
                    'min_rnk': min_rnk,
                    'max_rnk': min_rnk + min(50, max_results - fetched),
                    'fmt': 'json'
                }
                
                logging.info(f"Fetching trials with expr: {expr}")
                response = self._make_request(params)
                
                if 'StudyFieldsResponse' not in response:
                    break
                
                study_fields = response['StudyFieldsResponse'].get('StudyFields', [])
                if not study_fields:
                    break
                
                for study in study_fields:
                    trial_data = self._extract_trial_data(study)
                    all_trials.append(trial_data)
                    fetched += 1
                    if fetched >= max_results:
                        break
                
                # Check if there are more results
                total_results = response['StudyFieldsResponse'].get('NStudiesFound', 0)
                if min_rnk + 50 > total_results:
                    break
                
                min_rnk += 50
                time.sleep(self.delay)
                
            except Exception as e:
                logging.error(f"Error fetching trials: {e}")
                break
        
        logging.info(f"Fetched {len(all_trials)} trials for condition: {condition}")
        return all_trials
    
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (2 ** attempt))
                else:
                    raise
        return {}
    
    def _extract_trial_data(self, study: Dict) -> Dict:
        """Extract relevant fields from study data."""
        def get_first(field_list):
            return field_list[0] if field_list and len(field_list) > 0 else ""
        
        return {
            'nct_id': get_first(study.get('NCTId', [])),
            'title': get_first(study.get('BriefTitle', [])),
            'conditions': study.get('Conditions', []),
            'phase': get_first(study.get('Phase', ['PHASE2'])),
            'sample_size': int(get_first(study.get('EnrollmentCount', [0]))),
            'overall_status': get_first(study.get('OverallStatus', [])),
            'study_type': get_first(study.get('StudyType', [])),
            'eligibility_criteria': get_first(study.get('EligibilityCriteria', [''])),
            'interventions': study.get('Interventions', []),
            'start_date': get_first(study.get('StartDate', [''])),
            'completion_date': get_first(study.get('CompletionDate', [''])),
        }


# Quick test
if __name__ == "__main__":
    fetcher = ClinicalTrialsFetcher()
    
    print("Testing v1 API...")
    trials = fetcher.search_trials(
        condition="heart failure",
        max_results=3,
        phase="PHASE2",
        status="COMPLETED"
    )
    
    print(f"Found {len(trials)} trials")
    for trial in trials:
        print(f"  - {trial.get('nct_id')}: {trial.get('title', '')[:50]}...")