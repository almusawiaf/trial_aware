# convert_downloaded_trials.py
import json
import os
import glob
import re

def convert_downloaded_trials(folder_path="."):
    """Convert downloaded ClinicalTrials.gov JSON files to pipeline format."""
    
    structured_trials = []
    
    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    for file_path in json_files:
        try:
            # Use utf-8-sig encoding to handle BOM and Unicode
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            print(f"Processing: {file_path}")
            
            # Extract trial data from the structure
            # The downloaded format might have different keys
            trial = extract_trial_data(data)
            structured_trials.append(trial)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not structured_trials:
        print("❌ No trials processed! Check the JSON format.")
        return
    
    print(f"✅ Processed {len(structured_trials)} trials")
    
    # Split train/eval (80/20)
    split_idx = int(len(structured_trials) * 0.8)
    train_trials = structured_trials[:split_idx]
    eval_trials = structured_trials[split_idx:]
    
    # Save to files
    with open('structured_clinical_trials.json', 'w', encoding='utf-8') as f:
        json.dump(train_trials, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(train_trials)} training trials")
    
    with open('structured_clinical_trials_eval.json', 'w', encoding='utf-8') as f:
        json.dump(eval_trials, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(eval_trials)} evaluation trials")
    
    # Print sample
    if structured_trials:
        print("\n📊 Sample converted trial:")
        sample = structured_trials[0]
        print(f"NCT ID: {sample.get('nct_id')}")
        print(f"Title: {sample.get('title', 'Unknown')[:100]}...")
        print(f"Conditions: {sample.get('conditions', [])}")
        print(f"Phase: {sample.get('phase')}")
        print(f"Sample size: {sample.get('sample_size')}")
        print(f"Number of criteria: {len(sample.get('criteria', []))}")

def extract_trial_data(data):
    """
    Extract trial data from JSON structure.
    Handles both single-trial and multi-trial formats.
    """
    # Check if it's a list of studies or a single study
    if isinstance(data, list):
        # It's a list of trials - take the first one
        study_data = data[0] if data else {}
    else:
        study_data = data
    
    # Try different possible structures
    protocol = study_data.get('protocolSection', {})
    if protocol:
        # This is the structure we saw earlier
        identification = protocol.get('identificationModule', {})
        status = protocol.get('statusModule', {})
        design = protocol.get('designModule', {})
        conditions_module = protocol.get('conditionsModule', {})
        eligibility = protocol.get('eligibilityModule', {})
        
        nct_id = identification.get('nctId', 'NCT_UNKNOWN')
        title = identification.get('briefTitle', 'Unknown')
        
        # Extract criteria from eligibility text
        criteria_text = eligibility.get('eligibilityCriteria', '')
        criteria = parse_criteria_text(criteria_text)
        
        return {
            'nct_id': nct_id,
            'title': title,
            'conditions': conditions_module.get('conditions', []),
            'phase': design.get('phases', ['NA'])[0] if design.get('phases') else 'NA',
            'sample_size': design.get('enrollmentInfo', {}).get('count', 100),
            'overall_status': status.get('overallStatus', 'UNKNOWN'),
            'criteria': criteria
        }
    else:
        # Try flat structure
        return {
            'nct_id': study_data.get('NCTId', study_data.get('nct_id', 'NCT_UNKNOWN')),
            'title': study_data.get('BriefTitle', study_data.get('title', 'Unknown')),
            'conditions': study_data.get('Conditions', study_data.get('conditions', [])),
            'phase': study_data.get('Phase', study_data.get('phase', 'NA')),
            'sample_size': study_data.get('EnrollmentCount', study_data.get('sample_size', 100)),
            'overall_status': study_data.get('OverallStatus', study_data.get('overall_status', 'UNKNOWN')),
            'criteria': parse_criteria_text(study_data.get('EligibilityCriteria', ''))
        }

def parse_criteria_text(criteria_text):
    """Parse eligibility criteria text into structured format."""
    if not criteria_text or len(criteria_text.strip()) < 10:
        return []
    
    structured = []
    lines = [l.strip() for l in criteria_text.split('\n') if l.strip()]
    
    is_inclusion = True
    
    for line in lines:
        lower = line.lower()
        
        # Detect section headers
        if 'inclusion' in lower and 'criteria' in lower:
            is_inclusion = True
            continue
        elif 'exclusion' in lower and 'criteria' in lower:
            is_inclusion = False
            continue
        
        # Skip empty, short, or header lines
        if len(line) < 10 or line.endswith(':'):
            continue
        
        # Try to identify entity type
        entity_type = 'diagnosis'
        if any(word in lower for word in ['medication', 'drug', 'treatment', 'therapy', 'taking']):
            entity_type = 'medication'
        elif any(word in lower for word in ['creatinine', 'glucose', 'blood', 'pressure', 'LVEF']):
            entity_type = 'lab'
        elif any(word in lower for word in ['surgery', 'procedure', 'transplant']):
            entity_type = 'procedure'
        
        # Try to identify operator and value
        operator = 'EXISTS'
        value = None
        
        numbers = re.findall(r'(\d+\.?\d*)', line)
        if numbers:
            if '>' in line or 'greater than' in lower:
                operator = 'GT'
                value = float(numbers[0])
            elif '<' in line or 'less than' in lower:
                operator = 'LT'
                value = float(numbers[0])
            elif '=' in line or 'equal' in lower:
                operator = 'EQ'
                value = float(numbers[0])
            elif 'between' in lower and len(numbers) >= 2:
                operator = 'BETWEEN'
                value = float(numbers[0])
        
        # Create a code from the text
        import hashlib
        code_hash = hashlib.md5(line.encode()).hexdigest()[:8]
        
        structured.append({
            'raw_entity': line[:200],  # Truncate to avoid huge strings
            'entity_type': entity_type,
            'entity_code': f"EXTRACTED_{code_hash}",
            'operator': operator,
            'value': value,
            'max_value': None,
            'is_inclusion': is_inclusion,
            'severity_weight': 1.0
        })
    
    return structured

def inspect_json_file(file_path):
    """Helper function to inspect the structure of the JSON file."""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    print("JSON Structure:")
    if isinstance(data, list):
        print(f"List of {len(data)} items")
        if data:
            print("First item keys:", list(data[0].keys())[:10])
    else:
        print("Dictionary with keys:", list(data.keys())[:10])

if __name__ == "__main__":
    # First, inspect the JSON to understand its structure
    print("Inspecting JSON file...")
    json_files = glob.glob("*.json")
    if json_files:
        print(f"Found JSON file: {json_files[0]}")
        inspect_json_file(json_files[0])
        print("\n" + "="*60)
        print("Converting trials...")
        print("="*60)
        convert_downloaded_trials()
    else:
        print("No JSON files found in current directory.")
        print("Please ensure your downloaded JSON file is in this folder.")