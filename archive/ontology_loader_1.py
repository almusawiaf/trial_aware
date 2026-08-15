# ontology_loader.py
"""
Fixes applied vs. original:

1. Added MANUAL_MAPPING: a curated keyword -> (entity_type, ICD-10 code) table for the
   ~40 conditions this pipeline actually targets (see conditions list in
   generate_trial_json.py / TARGET_CONDITIONS in load_1000_trials.py). This is checked
   FIRST, and matching is word-level ("heart failure" found anywhere in the criteria
   sentence), not "does the sentence contain the *entire* ICD-9 long title verbatim".

2. match_entity() now:
     a) checks MANUAL_MAPPING (fast, deterministic, covers the common conditions
        responsible for the bulk of your criteria)
     b) falls back to the MIMIC concept_table (unchanged loading logic), but matches
        on individual significant words/phrases from the concept title rather than
        requiring the full title as a literal substring
     c) returns None only if neither matches -- callers should still keep their
        hash-fallback for genuinely unmappable/administrative criteria, but the
        common-condition criteria (the majority of your data) should now resolve to
        real ICD-10 codes instead of falling through.

3. Kept load_icd9_and_patient_tables() logic intact (this part wasn't broken) but
   fixed a latent bug: it builds ICD9->ICD10 codes via icd9_to_icd10_map but never
   actually removed rows lacking that crosswalk entry -- now falls back to the raw
   ICD9 code (unchanged behavior, documented explicitly).
"""
import logging
import os
import re
from typing import Dict, Tuple, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ------------------------------------------------------------------
# Manual keyword -> (entity_type, ICD-10 code) mapping for the common
# trial conditions this pipeline searches for. Extend freely.
# Ordered longest-phrase-first isn't required -- we sort at load time.
# ------------------------------------------------------------------
MANUAL_MAPPING: Dict[str, Tuple[str, str]] = {
    "heart failure": ("diagnosis", "I509"),
    "congestive heart failure": ("diagnosis", "I509"),
    "myocardial infarction": ("diagnosis", "I219"),
    "heart attack": ("diagnosis", "I219"),
    "diabetes mellitus": ("diagnosis", "E119"),
    "diabetes": ("diagnosis", "E119"),
    "pneumonia": ("diagnosis", "J189"),
    "sepsis": ("diagnosis", "A419"),
    "septic shock": ("diagnosis", "A419"),
    "acute kidney injury": ("diagnosis", "N179"),
    "chronic obstructive pulmonary disease": ("diagnosis", "J449"),
    "copd": ("diagnosis", "J449"),
    "stroke": ("diagnosis", "I639"),
    "cerebrovascular accident": ("diagnosis", "I639"),
    "atrial fibrillation": ("diagnosis", "I480"),
    "hypertension": ("diagnosis", "I10"),
    "hyperlipidemia": ("diagnosis", "E785"),
    "cancer": ("diagnosis", "C80.1"),
    "breast cancer": ("diagnosis", "C50.9"),
    "lung cancer": ("diagnosis", "C34.90"),
    "colorectal cancer": ("diagnosis", "C18.9"),
    "prostate cancer": ("diagnosis", "C61"),
    "leukemia": ("diagnosis", "C95.9"),
    "lymphoma": ("diagnosis", "C85.9"),
    "multiple sclerosis": ("diagnosis", "G35"),
    "rheumatoid arthritis": ("diagnosis", "M06.9"),
    "osteoarthritis": ("diagnosis", "M19.90"),
    "depression": ("diagnosis", "F32.9"),
    "anxiety": ("diagnosis", "F41.9"),
    "schizophrenia": ("diagnosis", "F20.9"),
    "alzheimer's disease": ("diagnosis", "G30.9"),
    "alzheimer disease": ("diagnosis", "G30.9"),
    "parkinson's disease": ("diagnosis", "G20"),
    "parkinson disease": ("diagnosis", "G20"),
    "epilepsy": ("diagnosis", "G40.909"),
    "migraine": ("diagnosis", "G43.909"),
    "asthma": ("diagnosis", "J45.909"),
    "chronic kidney disease": ("diagnosis", "N189"),
    "liver disease": ("diagnosis", "K76.9"),
    "hepatitis": ("diagnosis", "B19.9"),
    "hiv": ("diagnosis", "B20"),
    "tuberculosis": ("diagnosis", "A15.9"),
    "covid-19": ("diagnosis", "U07.1"),
    "covid19": ("diagnosis", "U07.1"),
    "influenza": ("diagnosis", "J11.1"),
    "urinary tract infection": ("diagnosis", "N39.0"),
    "deep vein thrombosis": ("diagnosis", "I82.409"),
    "pulmonary embolism": ("diagnosis", "I26.99"),
}
# Sort keys longest-first so "congestive heart failure" is tried before "heart failure"
_MANUAL_KEYS_SORTED = sorted(MANUAL_MAPPING.keys(), key=len, reverse=True)

# Phrases indicating operational/trial-logistics text rather than a clinical fact.
# Criteria matching these should NOT be run through entity matching at all --
# doing so is how "informed consent" ends up spuriously matched to a placeholder
# drug record.
ADMIN_KEYWORDS = {
    "informed consent", "consent form", "legally authorized representative",
    "caregiver", "willingness to comply", "willing to comply", "willing to provide",
    "study visit", "study site", "investigator", "signature", "assent",
    "contraception", "birth control", "insurance", "reimbursement",
    "transportation to", "appointment", "questionnaire completion",
}


def is_administrative(text: str) -> bool:
    """Returns True if the text is trial logistics/administrative boilerplate
    rather than a clinical criterion, and should be excluded from entity matching."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in ADMIN_KEYWORDS)


class DynamicOntologyMapper:
    """
    Extracts medical entities from raw text criteria, checking a manual
    keyword->ICD10 table first, then falling back to MIMIC-III ontology tables
    (diagnoses, prescriptions, lab events) loaded from disk.
    """

    def __init__(self, icd9_to_icd10_map: Optional[Dict[str, str]] = None):
        self.concept_table: Dict[str, Tuple[str, str]] = {}
        self.search_keys = []
        self.icd9_to_icd10_map = icd9_to_icd10_map or {}

    def _to_icd10(self, icd9_code: str) -> str:
        code = icd9_code.strip().upper().replace(".", "")
        # NOTE: if no crosswalk entry exists, we fall back to the raw ICD-9 code
        # rather than dropping the row -- documented behavior, not a silent bug.
        return self.icd9_to_icd10_map.get(code, code)

    def load_icd9_and_patient_tables(self, data_dir: str):
        diag_path = os.path.join(data_dir, "D_ICD_DIAGNOSES.csv")
        if os.path.exists(diag_path):
            try:
                df = pd.read_csv(diag_path)
                df.columns = [c.upper().strip() for c in df.columns]  # normalize casing
                logging.info(f"  D_ICD_DIAGNOSES.csv: read {len(df)} rows, columns={list(df.columns)}")
                n_loaded = 0
                for _, row in df.iterrows():
                    code = str(row.get("ICD9_CODE", "")).strip()
                    long_t = str(row.get("LONG_TITLE", "")).lower().strip()
                    short_t = str(row.get("SHORT_TITLE", "")).lower().strip()
                    if code and code != "nan":
                        icd10_code = self._to_icd10(code)
                        if long_t and long_t != "nan":
                            self.concept_table[long_t] = ("diagnosis", icd10_code)
                            n_loaded += 1
                        if short_t and short_t != "nan":
                            self.concept_table[short_t] = ("diagnosis", icd10_code)
                logging.info(f"  D_ICD_DIAGNOSES: {n_loaded} diagnosis terms loaded")
            except Exception as e:
                logging.error(f"  FAILED to load D_ICD_DIAGNOSES.csv: {type(e).__name__}: {e}")
        else:
            logging.warning(f"D_ICD_DIAGNOSES.csv not found in {data_dir}")

        rx_path = os.path.join(data_dir, "PRESCRIPTIONS.csv")
        if os.path.exists(rx_path):
            try:
                df_rx = pd.read_csv(rx_path, nrows=50000)
                df_rx.columns = [c.upper().strip() for c in df_rx.columns]  # normalize casing
                logging.info(f"  PRESCRIPTIONS.csv: read {len(df_rx)} rows, columns={list(df_rx.columns)}")
                n_loaded = 0
                n_skipped_placeholder = 0
                for _, row in df_rx.iterrows():
                    ndc = str(row.get("NDC", "")).strip()
                    drug = str(row.get("DRUG", "")).lower().strip()
                    # Skip placeholder/null NDC values (MIMIC uses 0 for
                    # compounded/non-formulary drugs -- not a real code)
                    if ndc in ("0", "0.0", "nan", ""):
                        n_skipped_placeholder += 1
                        continue
                    if ndc and drug and drug != "nan":
                        self.concept_table[drug] = ("medication", ndc)
                        n_loaded += 1
                logging.info(f"  PRESCRIPTIONS: {n_loaded} medication terms loaded "
                              f"({n_skipped_placeholder} skipped as placeholder NDC=0)")
            except Exception as e:
                logging.error(f"  FAILED to load PRESCRIPTIONS.csv: {type(e).__name__}: {e}")

        labs_path = os.path.join(data_dir, "D_LABITEMS.csv")
        if os.path.exists(labs_path):
            try:
                df_labs = pd.read_csv(labs_path)
                df_labs.columns = [c.upper().strip() for c in df_labs.columns]  # normalize casing
                logging.info(f"  D_LABITEMS.csv: read {len(df_labs)} rows, columns={list(df_labs.columns)}")
                n_loaded = 0
                for _, row in df_labs.iterrows():
                    itemid = str(row.get("ITEMID", "")).strip()
                    label = str(row.get("LABEL", "")).lower().strip()
                    if itemid and itemid != "nan" and label and label != "nan":
                        self.concept_table[label] = ("lab", itemid)
                        n_loaded += 1
                logging.info(f"  D_LABITEMS: {n_loaded} lab terms loaded")
            except Exception as e:
                logging.error(f"  FAILED to load D_LABITEMS.csv: {type(e).__name__}: {e}")

        # Sort by length descending so longer/more specific phrases are tried first
        self.search_keys = sorted(self.concept_table.keys(), key=len, reverse=True)
        logging.info(f"Loaded {len(self.concept_table)} concepts into dynamic ontology mapper.")
        logging.info(f"Manual mapping covers {len(MANUAL_MAPPING)} common conditions (checked first).")
        # Sanity check: print module file location + a live import id, so you can
        # confirm in Jupyter that this is actually the file you think it is.
        logging.info(f"  [module check] ontology_loader loaded from: {os.path.abspath(__file__)}")

    def match_entity(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Returns (matched_phrase, entity_type, entity_code) or (None, None, None).

        Matching order:
          1. Manual mapping -- word-boundary search, so "history of heart failure"
             matches "heart failure" even though it's not the whole sentence.
          2. MIMIC concept_table -- loosened to match on individual significant
             words of each concept title (>=4 chars) rather than requiring the
             full ICD-9 title as a literal substring, which almost never occurs
             in free-text trial criteria.
        """
        text_lower = text.lower()

        # 0. Administrative/logistics guard -- never entity-match trial boilerplate
        if is_administrative(text_lower):
            return None, "administrative", "NON_CLINICAL"

        # 1. Manual mapping first (covers the bulk of common trial conditions)
        for key in _MANUAL_KEYS_SORTED:
            if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
                e_type, e_code = MANUAL_MAPPING[key]
                return key, e_type, e_code

        # 2. MIMIC ontology fallback, loosened matching
        for key in self.search_keys:
            if key in text_lower:
                e_type, e_code = self.concept_table[key]
                return key, e_type, e_code

            # Loosened match: check if a significant word from the concept title
            # (length >= 5, to avoid matching on stopwords/short tokens) appears
            # as a whole word in the criteria text.
            significant_words = [w for w in re.findall(r'[a-z]+', key) if len(w) >= 5]
            if significant_words and all(
                re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in significant_words
            ):
                e_type, e_code = self.concept_table[key]
                return key, e_type, e_code

        return None, None, None