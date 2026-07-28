# fetch_trials_local.py
import requests
import json
import time
import urllib.parse

def test_api_connection():
    """Test if the API is reachable."""
    print("Testing API connection...")
    
    # Test with a simple request
    url = "https://clinicaltrials.gov/api/query/study_fields"
    params = {
        'expr': 'heart failure',
        'fields': 'NCTId,BriefTitle',
        'min_rnk': 1,
        'max_rnk': 3,
        'fmt': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("✅ API is reachable!")
            return True
        else:
            print(f"❌ API returned status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def fetch_trials_v1(condition="heart failure", max_results=20):
    """Fetch trials using v1 API."""
    # The correct endpoint for v1 API
    url = "https://clinicaltrials.gov/api/query/study_fields"
    
    all_trials = []
    min_rnk = 1
    
    # Fields to request
    fields = [
        'NCTId', 'BriefTitle', 'OverallStatus', 'Phase', 
        'EnrollmentCount', 'StudyType', 'EligibilityCriteria',
        'Conditions', 'Interventions', 'StartDate', 'CompletionDate'
    ]
    
    while len(all_trials) < max_results:
        try:
            params = {
                'expr': condition,
                'fields': ','.join(fields),
                'min_rnk': min_rnk,
                'max_rnk': min_rnk + min(50, max_results - len(all_trials)),
                'fmt': 'json'
            }
            
            print(f"  Fetching records {min_rnk} to {min_rnk + 50}...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"  ⚠️ Error: HTTP {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                break
                
            data = response.json()
            
            # Check for studies in response
            studies = data.get('StudyFieldsResponse', {}).get('StudyFields', [])
            
            if not studies:
                print("  No more studies found")
                break
                
            all_trials.extend(studies)
            min_rnk += 50
            
            # Check if we've fetched all available
            total_results = data.get('StudyFieldsResponse', {}).get('NStudiesFound', 0)
            if min_rnk > total_results:
                break
                
            time.sleep(0.5)  # Rate limiting
            
        except requests.exceptions.Timeout:
            print("  ⚠️ Timeout - retrying...")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"  ❌ Error: {e}")
            break
    
    print(f"  Fetched {len(all_trials)} trials")
    return all_trials

def parse_and_save_trials():
    """Fetch trials for all conditions and save to JSON."""
    
    # First test the connection
    if not test_api_connection():
        print("\n❌ Cannot connect to ClinicalTrials.gov API.")
        print("Check your internet connection or try using a VPN.")
        return
    
    conditions = [
        "heart failure", "myocardial infarction", "diabetes",
        "pneumonia", "sepsis", "acute kidney injury",
        "chronic obstructive pulmonary disease", "stroke",
        "atrial fibrillation", "hypertension", "hyperlipidemia",
        "cancer", "breast cancer", "lung cancer",
        "colorectal cancer", "prostate cancer", "leukemia",
        "lymphoma", "multiple sclerosis", "rheumatoid arthritis",
        "osteoarthritis", "depression", "anxiety",
        "alzheimer's disease", "parkinson's disease",
        "asthma", "chronic kidney disease", "liver disease",
        "hepatitis", "hiv", "tuberculosis", "covid-19"
    ]
    
    all_trials = []
    seen_ids = set()
    
    print(f"\n📊 Fetching trials for {len(conditions)} conditions...")
    print("=" * 60)
    
    for i, condition in enumerate(conditions, 1):
        print(f"\n[{i}/{len(conditions)}] Fetching: {condition}")
        trials = fetch_trials_v1(condition, max_results=5)  # 5 per condition
        
        for trial in trials:
            # Extract NCT ID safely
            nct_id = trial.get('NCTId', [''])[0] if trial.get('NCTId') else ''
            if nct_id and nct_id not in seen_ids:
                seen_ids.add(nct_id)
                all_trials.append(trial)
        
        print(f"  Total unique trials so far: {len(all_trials)}")
    
    print("\n" + "=" * 60)
    print(f"✅ Fetched {len(all_trials)} unique trials")
    
    if all_trials:
        # Save to JSON
        with open("structured_clinical_trials.json", "w") as f:
            json.dump(all_trials, f, indent=2)
        print(f"✅ Saved to structured_clinical_trials.json")
        
        # Also save a smaller sample for testing
        with open("structured_clinical_trials_sample.json", "w") as f:
            json.dump(all_trials[:20], f, indent=2)
        print(f"✅ Saved sample (20 trials) to structured_clinical_trials_sample.json")
    else:
        print("❌ No trials fetched. Check your network connection.")

def fetch_single_condition(condition="heart failure", max_results=50):
    """Fetch trials for a single condition (useful for testing)."""
    print(f"Fetching trials for: {condition}")
    trials = fetch_trials_v1(condition, max_results)
    
    if trials:
        with open(f"trials_{condition.replace(' ', '_')}.json", "w") as f:
            json.dump(trials, f, indent=2)
        print(f"✅ Saved {len(trials)} trials to trials_{condition.replace(' ', '_')}.json")
    else:
        print("❌ No trials fetched")
    
    return trials

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Test a single condition
            condition = sys.argv[2] if len(sys.argv) > 2 else "heart failure"
            fetch_single_condition(condition, 20)
        else:
            parse_and_save_trials()
    else:
        parse_and_save_trials()