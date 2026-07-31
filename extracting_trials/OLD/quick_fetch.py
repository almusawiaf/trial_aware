# quick_fetch.py
import requests
import pandas as pd
import json

# Download real trial data
url = "https://clinicaltrials.gov/ct2/results/download?down=study_csv&field_list=NCTId,BriefTitle,Phase,EnrollmentCount,OverallStatus"
response = requests.get(url, timeout=30)

if response.status_code == 200:
    df = pd.read_csv(pd.StringIO(response.text))
    
    # Filter for completed Phase 2/3 trials
    trials = df[
        (df['OverallStatus'] == 'Completed') &
        (df['Phase'].str.contains('Phase 2|Phase 3', na=False)) &
        (df['EnrollmentCount'] > 50)
    ][:100]  # Take first 100
    
    # Save
    with open('structured_clinical_trials.json', 'w') as f:
        json.dump(trials.to_dict('records'), f, indent=2)
    
    print(f"✅ Saved {len(trials)} real trials")