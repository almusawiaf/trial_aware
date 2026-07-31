# fetch_who_trials.py
import requests
import json
import time

def fetch_who_trials(condition="heart failure", max_results=10):
    """
    Fetch trials from WHO ICTRP API.
    This often works when other APIs are blocked.
    """
    
    # WHO ICTRP API endpoint
    url = "https://trialsearch.who.int/api/studies"
    
    all_trials = []
    page = 1
    
    while len(all_trials) < max_results:
        params = {
            'query': f'conditions:"{condition}"',
            'page': page,
            'size': min(10, max_results - len(all_trials)),
            'format': 'json'
        }
        
        try:
            print(f"Fetching page {page}...")
            response = requests.get(url, params=params, timeout=30, headers={
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (compatible; Research/1.0)'
            })
            
            if response.status_code != 200:
                print(f"HTTP {response.status_code}: {response.text[:100]}")
                break
                
            data = response.json()
            trials = data.get('data', [])
            
            if not trials:
                break
                
            all_trials.extend(trials)
            page += 1
            
            # Check if we've reached the end
            if len(all_trials) >= data.get('total', 0):
                break
                
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"Fetched {len(all_trials)} trials")
    return all_trials

# Test it
if __name__ == "__main__":
    trials = fetch_who_trials("heart failure", 5)
    print(f"Found {len(trials)} trials")
    if trials:
        print(json.dumps(trials[0], indent=2)[:500])