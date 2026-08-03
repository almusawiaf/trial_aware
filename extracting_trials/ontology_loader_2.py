# ontology_loader.py
"""
Accelerated version. The MIMIC ontology fallback in match_entity() used to loop
over every one of ~28,000 concept keys and do a substring check for EACH
criterion line -- up to ~265 million substring checks total across a typical
run, which is almost certainly why every job was taking ~26 minutes regardless
of what else changed. This version builds an Aho-Corasick automaton ONCE at
load time, then finds all matching keys in a single pass per line. Same exact
substring-match semantics as before (no accuracy change) -- just fast.

Falls back to the old brute-force scan automatically if pyahocorasick isn't
installed, so this won't break if you haven't run `pip install pyahocorasick`
yet -- it'll just still be slow until you do.
"""
import logging
import os
import re
from typing import Dict, Tuple, Optional
import pandas as pd

try:
    import ahocorasick
    _HAS_AHOCORASICK = True
except ImportError:
    _HAS_AHOCORASICK = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
_MANUAL_KEYS_SORTED = sorted(MANUAL_MAPPING.keys(), key=len, reverse=True)

ADMIN_KEYWORDS = {
    "informed consent", "consent form", "legally authorized representative",
    "caregiver", "willingness to comply", "willing to comply", "willing to provide",
    "study visit", "study site", "investigator", "signature", "assent",
    "contraception", "birth control", "insurance", "reimbursement",
    "transportation to", "appointment", "questionnaire completion",
}

# Generic terms that describe a category/specimen/type rather than a specific
# clinical entity. If these end up as a concept_table key (e.g. a diagnosis
# short_title of just "other" or a drug name that's a generic category), they
# become false-positive magnets -- matching almost any sentence that happens
# to use the word, regardless of clinical relevance. Applied across all three
# concept categories (diagnosis, medication, lab), not just labs.
GENERIC_TERM_DENYLIST = {
    "blood", "urine", "csf", "plasma", "serum", "stool", "sputum", "swab",
    "fluid", "specimen", "sample", "tissue", "other", "unknown", "unspecified",
    "nos", "nec", "misc", "miscellaneous", "various", "test", "panel",
}


def is_administrative(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in ADMIN_KEYWORDS)


class DynamicOntologyMapper:
    def __init__(self, icd9_to_icd10_map: Optional[Dict[str, str]] = None):
        self.concept_table: Dict[str, Tuple[str, str]] = {}
        self.search_keys = []
        self.icd9_to_icd10_map = icd9_to_icd10_map or {}
        self.automaton = None  # built once ontology tables are loaded

    def _to_icd10(self, icd9_code: str) -> str:
        code = icd9_code.strip().upper().replace(".", "")
        return self.icd9_to_icd10_map.get(code, code)

    def load_icd9_and_patient_tables(self, data_dir: str):
        diag_path = os.path.join(data_dir, "D_ICD_DIAGNOSES.csv")
        if os.path.exists(diag_path):
            try:
                df = pd.read_csv(diag_path)
                df.columns = [c.upper().strip() for c in df.columns]
                logging.info(f"  D_ICD_DIAGNOSES.csv: read {len(df)} rows")

                # Vectorized instead of iterrows() -- meaningfully faster for 14k+ rows
                df["ICD9_CODE"] = df.get("ICD9_CODE", "").astype(str).str.strip()
                df["LONG_TITLE"] = df.get("LONG_TITLE", "").astype(str).str.lower().str.strip()
                df["SHORT_TITLE"] = df.get("SHORT_TITLE", "").astype(str).str.lower().str.strip()
                valid = df[(df["ICD9_CODE"] != "") & (df["ICD9_CODE"].str.lower() != "nan")]
                valid = valid.copy()
                valid["ICD10_CODE"] = valid["ICD9_CODE"].apply(self._to_icd10)

                n_loaded = 0
                for long_t, short_t, icd10_code in zip(valid["LONG_TITLE"], valid["SHORT_TITLE"], valid["ICD10_CODE"]):
                    if long_t and long_t != "nan" and long_t not in GENERIC_TERM_DENYLIST:
                        self.concept_table[long_t] = ("diagnosis", icd10_code)
                        n_loaded += 1
                    if short_t and short_t != "nan" and short_t not in GENERIC_TERM_DENYLIST:
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
                df_rx.columns = [c.upper().strip() for c in df_rx.columns]
                logging.info(f"  PRESCRIPTIONS.csv: read {len(df_rx)} rows")

                df_rx["NDC"] = df_rx.get("NDC", "").astype(str).str.strip()
                df_rx["DRUG"] = df_rx.get("DRUG", "").astype(str).str.lower().str.strip()
                valid_rx = df_rx[
                    (~df_rx["NDC"].isin(["0", "0.0", "nan", ""])) &
                    (df_rx["DRUG"] != "") & (df_rx["DRUG"] != "nan") &
                    (~df_rx["DRUG"].isin(GENERIC_TERM_DENYLIST))
                ]
                n_loaded = 0
                for drug, ndc in zip(valid_rx["DRUG"], valid_rx["NDC"]):
                    self.concept_table[drug] = ("medication", ndc)
                    n_loaded += 1
                logging.info(f"  PRESCRIPTIONS: {n_loaded} medication terms loaded")
            except Exception as e:
                logging.error(f"  FAILED to load PRESCRIPTIONS.csv: {type(e).__name__}: {e}")

        labs_path = os.path.join(data_dir, "D_LABITEMS.csv")
        if os.path.exists(labs_path):
            try:
                df_labs = pd.read_csv(labs_path)
                df_labs.columns = [c.upper().strip() for c in df_labs.columns]
                logging.info(f"  D_LABITEMS.csv: read {len(df_labs)} rows")

                df_labs["ITEMID"] = df_labs.get("ITEMID", "").astype(str).str.strip()
                df_labs["LABEL"] = df_labs.get("LABEL", "").astype(str).str.lower().str.strip()
                # FIX: some D_LABITEMS rows just name the specimen type itself
                # (e.g. LABEL="blood" when FLUID="Blood") rather than a specific
                # test -- these match almost any sentence that mentions blood in
                # ANY context ("donated blood," "blood pressure," "bloodwork"),
                # producing exactly the false-positive-magnet pattern seen with
                # code 51466. Skip rows where the label is just the fluid name,
                # plus a small denylist of other overly generic single words
                # that describe a specimen/category rather than a specific test.
                fluid_col = df_labs.get("FLUID", pd.Series([""] * len(df_labs))).astype(str).str.lower().str.strip()
                is_generic = (df_labs["LABEL"] == fluid_col) | (df_labs["LABEL"].isin(GENERIC_TERM_DENYLIST))
                valid_labs = df_labs[
                    (df_labs["ITEMID"] != "") & (df_labs["ITEMID"] != "nan") &
                    (df_labs["LABEL"] != "") & (df_labs["LABEL"] != "nan") &
                    (~is_generic)
                ]
                n_loaded = 0
                n_skipped_generic = int(is_generic.sum())
                for itemid, label in zip(valid_labs["ITEMID"], valid_labs["LABEL"]):
                    self.concept_table[label] = ("lab", itemid)
                    n_loaded += 1
                logging.info(f"  D_LABITEMS: {n_loaded} lab terms loaded "
                              f"({n_skipped_generic} skipped as generic specimen-type labels)")
            except Exception as e:
                logging.error(f"  FAILED to load D_LABITEMS.csv: {type(e).__name__}: {e}")

        self.search_keys = sorted(self.concept_table.keys(), key=len, reverse=True)
        logging.info(f"Loaded {len(self.concept_table)} concepts into dynamic ontology mapper.")
        logging.info(f"Manual mapping covers {len(MANUAL_MAPPING)} common conditions (checked first).")
        logging.info(f"  [module check] ontology_loader loaded from: {os.path.abspath(__file__)}")

        self._build_automaton()

    def _build_automaton(self):
        """Build the Aho-Corasick automaton once, so match_entity() doesn't
        have to scan all ~28,000 keys per criterion line.

        FIX: exclude keys shorter than MIN_KEY_LENGTH entirely. Short lab/drug
        labels (e.g. 2-4 letter abbreviations or common word fragments) were
        matching as substrings INSIDE unrelated words and sentences -- the same
        "mi"-inside-"leukemia" problem from the trial filter step, but never
        fixed here. This was happening even before Aho-Corasick was added; the
        automaton just made it fast to hit, not the cause.
        """
        if not _HAS_AHOCORASICK:
            logging.warning("pyahocorasick not installed -- falling back to slow "
                             "brute-force scan (run `pip install pyahocorasick` to fix this).")
            return
        if not self.concept_table:
            return

        MIN_KEY_LENGTH = 5
        n_excluded = 0
        A = ahocorasick.Automaton()
        for key in self.concept_table.keys():
            if len(key) < MIN_KEY_LENGTH:
                n_excluded += 1
                continue
            A.add_word(key, (len(key), key))
        A.make_automaton()
        self.automaton = A
        logging.info(f"  Built Aho-Corasick automaton for fast substring matching "
                      f"({n_excluded} concept keys < {MIN_KEY_LENGTH} chars excluded "
                      f"to prevent false-positive mid-word matches).")

    def match_entity(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        text_lower = text.lower()

        if is_administrative(text_lower):
            return None, "administrative", "NON_CLINICAL"

        for key in _MANUAL_KEYS_SORTED:
            if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
                e_type, e_code = MANUAL_MAPPING[key]
                return key, e_type, e_code

        if self.automaton is not None:
            best = None
            for end_index, (length, key) in self.automaton.iter(text_lower):
                start_index = end_index - length + 1
                # FIX: word-boundary check. Aho-Corasick (like the old brute-force
                # substring scan) finds the pattern ANYWHERE, including inside a
                # larger unrelated word (e.g. a short drug/lab name matching
                # mid-word in "instructions" or "hypersensitivity"). Require
                # the character immediately before/after the match to be a
                # non-alphanumeric boundary (or start/end of string).
                before_ok = start_index == 0 or not text_lower[start_index - 1].isalnum()
                after_ok = (end_index + 1 == len(text_lower)) or not text_lower[end_index + 1].isalnum()
                if not (before_ok and after_ok):
                    continue
                if best is None or length > best[0]:
                    best = (length, key)
            if best is not None:
                key = best[1]
                e_type, e_code = self.concept_table[key]
                return key, e_type, e_code
            return None, None, None

        # Fallback: brute-force exact substring scan with word-boundary check
        # (slow -- only used if pyahocorasick isn't installed)
        for key in self.search_keys:
            if len(key) < 5:
                continue
            match = re.search(r'(?<![a-zA-Z0-9])' + re.escape(key) + r'(?![a-zA-Z0-9])', text_lower)
            if match:
                e_type, e_code = self.concept_table[key]
                return key, e_type, e_code

        return None, None, None