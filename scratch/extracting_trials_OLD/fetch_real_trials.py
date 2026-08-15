# fetch_real_trials.py
import requests
import pandas as pd
import json
from io import StringIO

def fetch_real_trials_from_csv():
    """
    Download real clinical trial data from public CSV source.
    """
    
    # Using the ClinicalTrials.gov CSV export
    url = "https://clinicaltrials.gov/ct2/results/download?down=study_csv&field_list=NCTId,BriefTitle,Phase,EnrollmentCount,OverallStatus,StartDate,CompletionDate"
    
    print("Downloading real clinical trial data...")
    
    try:
        response = requests.get(url, timeout=60, headers={
            'User-Agent': 'Mozilla/5.0 (Research/1.0)'
        })
        
        if response.status_code == 200:
            # Parse CSV
            df = pd.read_csv(StringIO(response.text))
            print(f"Downloaded {len(df)} trials")
            
            # Filter for completed Phase 2/3 trials with enrollment > 50
            filtered = df[
                (df['OverallStatus'].str.contains('Completed', case=False, na=False)) &
                (df['Phase'].str.contains('Phase 2|Phase 3', case=False, na=False)) &
                (df['EnrollmentCount'] > 50)
            ]
            
            print(f"Filtered to {len(filtered)} relevant trials")
            
            # Save to JSON
            trials = []
            for _, row in filtered.iterrows():
                trials.append({
                    'nct_id': row.get('NCTId', ''),
                    'title': row.get('BriefTitle', ''),
                    'phase': row.get('Phase', 'PHASE2'),
                    'sample_size': int(row.get('EnrollmentCount', 100)),
                    'conditions': [],  # We'll need to fetch these separately
                })
            
            with open('structured_clinical_trials.json', 'w') as f:
                json.dump(trials, f, indent=2)
            
            print(f"✅ Saved {len(trials)} real trials to structured_clinical_trials.json")
            return trials
        else:
            print(f"HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    fetch_real_trials_from_csv()