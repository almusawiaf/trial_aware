# fetch_trials_eu.py
import requests
import json
import time

def fetch_eu_trials(condition="heart failure", max_results=10):
    """Fetch trials from EU Clinical Trials Register."""
    
    # EU Clinical Trials Register API
    url = "https://www.clinicaltrialsregister.eu/ctr-search/api/rest/trials"
    
    all_trials = []
    page = 0
    
    while len(all_trials) < max_results:
        params = {
            'format': 'json',
            'query': condition,
            'page': page,
            'size': min(20, max_results - len(all_trials))
        }
        
        try:
            print(f"Fetching page {page}...")
            response = requests.get(url, params=params, timeout=30, headers={
                'Accept': 'application/json'
            })
            
            if response.status_code != 200:
                print(f"HTTP {response.status_code}: {response.text[:100]}")
                break
                
            data = response.json()
            trials = data.get('result', {}).get('items', [])
            
            if not trials:
                break
                
            all_trials.extend(trials)
            page += 1
            
            # Check if we've reached the end
            total = data.get('result', {}).get('total', 0)
            if len(all_trials) >= total:
                break
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"Fetched {len(all_trials)} trials")
    return all_trials

# Test it
if __name__ == "__main__":
    trials = fetch_eu_trials("heart failure", 5)
    print(f"Found {len(trials)} trials")