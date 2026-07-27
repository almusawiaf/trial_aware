import re
import spacy

# Load local English NLP model
nlp = spacy.load("en_core_web_sm")


def extract_trial_info(text: str) -> dict:
    """Extract clinical trial metadata using spaCy Dependency Parsing & RegEx."""
    doc = nlp(text)

    # 1. Extract Sample Size using RegEx
    sample_size = None
    sample_match = re.search(
        r"(\d[\d,]*)\s*(?:patients|subjects|participants|enrolled)",
        text,
        re.IGNORECASE,
    )
    if sample_match:
        sample_size = int(sample_match.group(1).replace(",", ""))

    # 2. Extract Phase using RegEx
    phase_match = re.search(
        r"\b(?:Phase\s*(?:[1-4]|I{1,3}|IV)|Phase\s*0)\b", text, re.IGNORECASE
    )
    phase = phase_match.group(0) if phase_match else "Not specified"

    # 3. Extract Condition using spaCy Dependency Parsing
    condition = "Unknown"

    # Strategy A: Find "with <Noun Chunk>" or "for <Noun Chunk>"
    for token in doc:
        if token.text.lower() in ["with", "for", "having"] and token.pos_ == "ADP":
            # Grab the noun chunk rooted at this preposition's child
            for child in token.children:
                # Get the full noun phrase (e.g., "Non-Small Cell Lung Cancer")
                chunk = [
                    chunk
                    for chunk in doc.noun_chunks
                    if child in list(chunk)
                ]
                if chunk:
                    condition = chunk[0].text.strip()
                    break

    # Strategy B: Fallback Regex if dependency parse didn't catch it
    if condition == "Unknown":
        cond_match = re.search(
            r"(?:patients|subjects)\s+with\s+([A-Za-z0-9\s\-]+?)(?=\.|\,|$)",
            text,
            re.IGNORECASE,
        )
        if cond_match:
            condition = cond_match.group(1).strip()

    return {
        "condition": condition,
        "sample_size": sample_size,
        "phase": phase,
    }


# --- Test Script ---
if __name__ == "__main__":
    sample_text = "A Phase 3 clinical trial evaluating drug X in 450 patients with Non-Small Cell Lung Cancer."

    print("--- Processing with Local NLP ---")
    results = extract_trial_info(sample_text)

    print(f"Condition   : {results['condition']}")
    print(f"Sample Size : {results['sample_size']}")
    print(f"Phase       : {results['phase']}")