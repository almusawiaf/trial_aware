# download_kaggle_trials.py
import json
import pandas as pd
import requests
import zipfile
import io
import os

def download_kaggle_dataset():
    """
    Download real clinical trials from Kaggle using Python.
    """
    
    # Option 1: Try downloading a CSV directly from Kaggle (without API)
    # Some Kaggle datasets are available as direct CSV downloads
    
    print("=" * 60)
    print("DOWNLOADING REAL CLINICAL TRIAL DATA")
    print("=" * 60)
    
    # Try multiple sources
    sources = [
        {
            'name': 'Clinical Trials 2019-2020 (Kaggle)',
            'url': 'https://raw.githubusercontent.com/your-repo/clinical-trials-data/main/clinical_trials.csv'
        },
        {
            'name': 'NIH Clinical Trials (Direct)',
            'url': 'https://clinicaltrials.gov/ct2/results/download?down=study_csv'
        },
        {
            'name': 'AACT Database (ClinicalTrials.gov mirror)',
            'url': 'https://aact.ctti-clinicaltrials.org/static/exported_files/aact_subset.zip'
        }
    ]
    
    all_trials = []
    
    for source in sources:
        print(f"\nTrying: {source['name']}")
        try:
            response = requests.get(source['url'], timeout=30, stream=True)
            
            if response.status_code == 200:
                print(f"  ✅ Download successful!")
                
                # Try to parse as CSV
                if '.csv' in source['url']:
                    df = pd.read_csv(io.StringIO(response.text))
                    print(f"  Found {len(df)} records")
                    all_trials.extend(df.to_dict('records'))
                    break
                elif '.zip' in source['url']:
                    # Handle zip file
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        for filename in z.namelist():
                            if filename.endswith('.csv'):
                                with z.open(filename) as f:
                                    df = pd.read_csv(f)
                                    print(f"  Found {len(df)} records in {filename}")
                                    all_trials.extend(df.to_dict('records'))
                        if all_trials:
                            break
            else:
                print(f"  HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    return all_trials

def download_from_aact():
    """
    Download real trials from the AACT database (ClinicalTrials.gov mirror).
    This is a reliable source of REAL data.
    """
    
    print("\n" + "=" * 60)
    print("DOWNLOADING FROM AACT DATABASE")
    print("=" * 60)
    
    # AACT provides a subset of trials for download
    url = "https://aact.ctti-clinicaltrials.org/static/exported_files/aact_subset.zip"
    
    try:
        print("Downloading AACT subset (real clinical trial data)...")
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Look for the studies file
                for filename in z.namelist():
                    if 'studies' in filename.lower() and filename.endswith('.csv'):
                        with z.open(filename) as f:
                            df = pd.read_csv(f)
                            print(f"✅ Found {len(df)} real trials!")
                            return df.to_dict('records')
        else:
            print(f"HTTP {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return []

def download_nih_csv():
    """
    Download real trials directly from NIH ClinicalTrials.gov.
    This is the OFFICIAL source of real data.
    """
    
    print("\n" + "=" * 60)
    print("DOWNLOADING FROM NIH CLINICALTRIALS.GOV")
    print("=" * 60)
    
    # CSV export from the official site
    url = "https://clinicaltrials.gov/ct2/results/download?down=study_csv"
    
    try:
        print("Downloading official NIH data...")
        response = requests.get(url, timeout=30, stream=True)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            print(f"✅ Found {len(df)} real trials!")
            
            # Filter for completed Phase 2/3 trials
            filtered = df[
                (df['OverallStatus'].str.contains('Completed', na=False)) &
                (df['Phase'].str.contains('Phase 2|Phase 3', na=False))
            ]
            
            print(f"   {len(filtered)} are completed Phase 2/3 trials")
            return filtered.to_dict('records')
        else:
            print(f"HTTP {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return []

def save_trials(trials, filename="structured_clinical_trials.json"):
    """
    Save trials to JSON format.
    """
    if not trials:
        print("❌ No trials to save!")
        return
    
    # Convert to your expected format
    formatted_trials = []
    for trial in trials:
        formatted = {
            'nct_id': trial.get('NCTId', trial.get('nct_id', 'NCT_UNKNOWN')),
            'title': trial.get('BriefTitle', trial.get('title', 'Unknown')),
            'conditions': trial.get('Conditions', trial.get('conditions', [])),
            'phase': trial.get('Phase', trial.get('phase', 'PHASE2')),
            'sample_size': trial.get('EnrollmentCount', trial.get('sample_size', 100)),
            'overall_status': trial.get('OverallStatus', trial.get('overall_status', '')),
            'criteria': []  # We need to fetch criteria separately
        }
        formatted_trials.append(formatted)
    
    # Split train/eval (80/20)
    split_idx = int(len(formatted_trials) * 0.8)
    train_trials = formatted_trials[:split_idx]
    eval_trials = formatted_trials[split_idx:]
    
    # Save files
    with open("structured_clinical_trials.json", "w") as f:
        json.dump(train_trials, f, indent=2)
    
    with open("structured_clinical_trials_eval.json", "w") as f:
        json.dump(eval_trials, f, indent=2)
    
    with open("structured_clinical_trials_full.json", "w") as f:
        json.dump(formatted_trials, f, indent=2)
    
    print(f"\n✅ Saved:")
    print(f"   - {len(train_trials)} training trials")
    print(f"   - {len(eval_trials)} evaluation trials")
    print(f"   - {len(formatted_trials)} total trials")

def main():
    print("=" * 60)
    print("REAL CLINICAL TRIAL DATA DOWNLOADER")
    print("=" * 60)
    print("\nThis will download REAL clinical trial data from official sources.")
    print("No mock data will be used.\n")
    
    # Try NIH first (official source)
    trials = download_nih_csv()
    
    if not trials:
        print("\nTrying AACT database...")
        trials = download_from_aact()
    
    if not trials:
        print("\nTrying Kaggle...")
        trials = download_kaggle_dataset()
    
    if trials:
        save_trials(trials)
        print("\n🎉 Successfully downloaded REAL clinical trial data!")
    else:
        print("\n❌ Could not download real data automatically.")
        print("Please try one of these alternatives:")
        print("1. Open in browser: https://clinicaltrials.gov/ct2/results/download")
        print("2. Manually download the CSV and place it here")
        print("3. Use the mock data generator as a fallback")

if __name__ == "__main__":
    main()