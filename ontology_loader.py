import os
from typing import Dict, Tuple
import pandas as pd
from rapidfuzz import fuzz, process


class DynamicOntologyMapper:

    # Keywords indicative of operational/trial metadata rather than clinical conditions
    ADMIN_KEYWORDS = {
        "partner",
        "consent",
        "caregiver",
        "visit",
        "signature",
        "investigator",
        "study site",
        "willingness",
        "compliance",
        "contraception",
        "birth control",
        "insurance",
        "reimbursement",
        "transportation",
        "appointment",
        "form",
        "assent",
        "informed consent",
    }

    def __init__(self):
        # Format: {clean_text_description: (entity_type, code)}
        self.concept_table: Dict[str, Tuple[str, str]] = {}
        self.search_keys = []

    def load_icd9_and_patient_tables(self, data_dir: str):
        """Loads MIMIC-III dictionary tables directly into memory."""
        print(f"📖 Building MIMIC-III dynamic ontology index from: {data_dir}")

        # 1. Parse Diagnoses (D_ICD_DIAGNOSES.csv)
        diag_path = self._find_file(data_dir, "D_ICD_DIAGNOSES")
        if diag_path:
            df = pd.read_csv(diag_path, low_memory=False)
            df.columns = [c.upper().strip() for c in df.columns]
            count = 0
            for _, row in df.iterrows():
                code = str(row.get("ICD9_CODE", "")).strip()
                long_t = str(row.get("LONG_TITLE", "")).lower().strip()
                short_t = str(row.get("SHORT_TITLE", "")).lower().strip()
                if code and code != "nan":
                    if long_t and long_t != "nan":
                        self.concept_table[long_t] = ("diagnosis", code)
                        count += 1
                    if short_t and short_t != "nan":
                        self.concept_table[short_t] = ("diagnosis", code)
            print(
                f"   -> Parsed Diagnoses: Loaded {count} terms from {os.path.basename(diag_path)}"
            )

        # 2. Parse Procedures (D_ICD_PROCEDURES.csv)
        proc_path = self._find_file(data_dir, "D_ICD_PROCEDURES")
        if proc_path:
            df = pd.read_csv(proc_path, low_memory=False)
            df.columns = [c.upper().strip() for c in df.columns]
            count = 0
            for _, row in df.iterrows():
                code = str(row.get("ICD9_CODE", "")).strip()
                long_t = str(row.get("LONG_TITLE", "")).lower().strip()
                short_t = str(row.get("SHORT_TITLE", "")).lower().strip()
                if code and code != "nan":
                    if long_t and long_t != "nan":
                        self.concept_table[long_t] = ("procedure", code)
                        count += 1
                    if short_t and short_t != "nan":
                        self.concept_table[short_t] = ("procedure", code)
            print(
                f"   -> Parsed Procedures: Loaded {count} terms from {os.path.basename(proc_path)}"
            )

        # 3. Parse Lab Items (D_LABITEMS.csv)
        lab_path = self._find_file(data_dir, "D_LABITEMS")
        if lab_path:
            df = pd.read_csv(lab_path, low_memory=False)
            df.columns = [c.upper().strip() for c in df.columns]
            count = 0
            for _, row in df.iterrows():
                item_id = str(row.get("ITEMID", "")).strip()
                label = str(row.get("LABEL", "")).lower().strip()
                if item_id and label and item_id != "nan" and label != "nan":
                    self.concept_table[label] = ("lab", item_id)
                    count += 1
            print(
                f"   -> Parsed Lab Items: Loaded {count} terms from {os.path.basename(lab_path)}"
            )

        # 4. Parse Prescriptions (PRESCRIPTIONS.csv)
        rx_path = self._find_file(data_dir, "PRESCRIPTIONS")
        if rx_path:
            df_rx = pd.read_csv(
                rx_path,
                usecols=lambda c: c.upper() in ["DRUG", "NDC"],
                low_memory=False,
            )
            df_rx.columns = [c.upper().strip() for c in df_rx.columns]
            df_rx = df_rx.dropna().drop_duplicates()
            count = 0
            for _, row in df_rx.iterrows():
                drug_name = str(row["DRUG"]).lower().strip()
                ndc_code = str(row["NDC"]).split(".")[0].strip()
                if drug_name and ndc_code and drug_name != "nan":
                    self.concept_table[drug_name] = ("medication", ndc_code)
                    count += 1
            print(
                f"   -> Parsed Prescriptions: Loaded {count} terms from {os.path.basename(rx_path)}"
            )

        self.search_keys = list(self.concept_table.keys())
        print(
            f"✅ [INDEX READY] Total loaded concepts in dynamic index: {len(self.concept_table)}\n"
        )

    def _find_file(self, data_dir: str, prefix: str) -> str:
        if not os.path.exists(data_dir):
            return None
        for fname in os.listdir(data_dir):
            if fname.upper().startswith(prefix.upper()) and (
                fname.endswith(".csv") or fname.endswith(".csv.gz")
            ):
                return os.path.join(data_dir, fname)
        return None

    def is_administrative(self, text: str) -> bool:
        """Returns True if the phrase contains operational or non-clinical trial boilerplate."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.ADMIN_KEYWORDS)

    def match_term(
        self, text: str, threshold: float = 72.0
    ) -> Tuple[str, str, str]:
        text_clean = text.lower().strip()

        # Step 1: Guard clause for Administrative / Operational terms
        if self.is_administrative(text_clean):
            return text_clean, "administrative", "NON_CLINICAL"

        # Step 2: Direct Exact Match
        if text_clean in self.concept_table:
            e_type, code = self.concept_table[text_clean]
            return text_clean, e_type, code

        # Step 3: Fuzzy Match using token_set_ratio with tuned threshold (72.0)
        if self.search_keys:
            match = process.extractOne(
                text_clean,
                self.search_keys,
                scorer=fuzz.token_set_ratio,
                score_cutoff=threshold,
            )
            if match:
                matched_title = match[0]
                e_type, code = self.concept_table[matched_title]
                return matched_title, e_type, code

        return text, "diagnosis", "UNKNOWN_CODE"