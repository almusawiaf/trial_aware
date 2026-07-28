# download_bulk_trials.py
import requests
import zipfile
import io
import json
import xml.etree.ElementTree as ET

def download_bulk_trials():
    """
    Download the full ClinicalTrials.gov dataset (real data!)
    This is the official data that powers the website.
    """
    
    # Download the full dataset (this is ~5GB compressed)
    url = "https://clinicaltrials.gov/ct2/results/download?down=studies"
    
    print("Downloading full clinical trials dataset...")
    print("(This may take several minutes and is ~5GB)")
    
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        # Save the zip file
        with open('clinical_trials.zip', 'wb') as f:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                if total_size > 0:
                    print(f"\rProgress: {progress:.1f}%", end='')
        
        print("\nDownload complete! Extracting...")
        
        # Extract and parse
        with zipfile.ZipFile('clinical_trials.zip', 'r') as zip_ref:
            zip_ref.extractall('clinical_trials_data')
        
        print("Extraction complete!")
        return True
    
    return False

# Alternative: Download smaller subset
def download_recent_trials():
    """Download only recent trials (much smaller)."""
    
    # Download just the last year of trials
    url = "https://clinicaltrials.gov/ct2/results/download?down=studies&recr=Completed&period=1"
    
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open('recent_trials.zip', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

if __name__ == "__main__":
    # Start with the recent trials (smaller download)
    print("Downloading recent clinical trials...")
    download_recent_trials()