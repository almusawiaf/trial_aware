# fix_all_trials.py
import json
import glob
import re
import hashlib

def process_all_trials():
    """Process all 100 trials from the downloaded JSON file."""
    
    # Load the original downloaded file
    with open('ctg-studies.json', 'r', encoding='utf-8-sig') as f:
        all_trials = json.load(f)
    
    print(f"Loaded {len(all_trials)} trials from ctg-studies.json")
    
    structured_trials = []
    
    for study_data in all_trials:
        protocol = study_data.get('protocolSection', {})
        
        # Extract basic info
        identification = protocol.get('identificationModule', {})
        status = protocol.get('statusModule', {})
        design = protocol.get('designModule', {})
        conditions_module = protocol.get('conditionsModule', {})
        eligibility = protocol.get('eligibilityModule', {})
        
        nct_id = identification.get('nctId', 'NCT_UNKNOWN')
        title = identification.get('briefTitle', 'Unknown')
        phase = design.get('phases', ['NA'])[0] if design.get('phases') else 'NA'
        sample_size = design.get('enrollmentInfo', {}).get('count', 100)
        overall_status = status.get('overallStatus', 'UNKNOWN')
        conditions = conditions_module.get('conditions', [])
        
        # Extract criteria from eligibility text
        criteria_text = eligibility.get('eligibilityCriteria', '')
        criteria = parse_criteria(criteria_text)
        
        structured_trials.append({
            'nct_id': nct_id,
            'title': title,
            'conditions': conditions,
            'phase': phase,
            'sample_size': sample_size,
            'overall_status': overall_status,
            'criteria': criteria
        })
    
    print(f"✅ Processed {len(structured_trials)} trials")
    
    # Split 80/20
    split_idx = int(len(structured_trials) * 0.8)
    train_trials = structured_trials[:split_idx]
    eval_trials = structured_trials[split_idx:]
    
    # Save files
    with open('structured_clinical_trials.json', 'w', encoding='utf-8') as f:
        json.dump(train_trials, f, indent=2, ensure_ascii=False)
    
    with open('structured_clinical_trials_eval.json', 'w', encoding='utf-8') as f:
        json.dump(eval_trials, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(train_trials)} training trials")
    print(f"✅ Saved {len(eval_trials)} evaluation trials")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"Total trials: {len(structured_trials)}")
    print(f"Training: {len(train_trials)}")
    print(f"Evaluation: {len(eval_trials)}")
    
    # Show sample
    if train_trials:
        print("\n📋 Sample training trial:")
        sample = train_trials[0]
        print(f"  NCT ID: {sample.get('nct_id')}")
        print(f"  Title: {sample.get('title', '')[:80]}...")
        print(f"  Phase: {sample.get('phase')}")
        print(f"  Criteria: {len(sample.get('criteria', []))} items")

def parse_criteria(criteria_text):
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
        
        # Clean up bullet points and numbering
        cleaned = re.sub(r'^[\*\-\d\.]+\s*', '', line).strip()
        
        if len(cleaned) < 10 or cleaned.endswith(':'):
            continue
        
        # Identify entity type
        entity_type = 'diagnosis'
        if any(word in lower for word in ['medication', 'drug', 'treatment', 'therapy', 'taking']):
            entity_type = 'medication'
        elif any(word in lower for word in ['creatinine', 'glucose', 'blood', 'pressure', 'LVEF']):
            entity_type = 'lab'
        elif any(word in lower for word in ['surgery', 'procedure', 'transplant']):
            entity_type = 'procedure'
        
        # Identify operator and value
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
            elif '≥' in line or 'or above' in lower:
                operator = 'GTE'
                value = float(numbers[0])
            elif '≤' in line or 'or below' in lower:
                operator = 'LTE'
                value = float(numbers[0])
        
        # Create unique code
        code_hash = hashlib.md5(cleaned.encode()).hexdigest()[:8]
        
        structured.append({
            'raw_entity': cleaned[:200],
            'entity_type': entity_type,
            'entity_code': f"EXTRACTED_{code_hash}",
            'operator': operator,
            'value': value,
            'max_value': None,
            'is_inclusion': is_inclusion,
            'severity_weight': 1.0
        })
    
    return structured

if __name__ == "__main__":
    process_all_trials()