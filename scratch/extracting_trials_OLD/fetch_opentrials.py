# fetch_opentrials.py
import requests
import json

def fetch_opentrials(condition="heart failure", max_results=10):
    """
    Fetch trials from OpenTrials API.
    """
    
    # OpenTrials API (sometimes works when others don't)
    url = "https://api.opentrials.net/v1/search"
    
    params = {
        'q': condition,
        'limit': min(max_results, 50)
    }
    
    try:
        print(f"Searching OpenTrials for: {condition}")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"HTTP {response.status_code}")
            return []
            
        data = response.json()
        trials = data.get('data', [])
        
        print(f"Found {len(trials)} trials")
        return trials
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    trials = fetch_opentrials("heart failure", 5)
    if trials:
        print(f"Found {len(trials)} trials")
        print(json.dumps(trials[0], indent=2)[:500])