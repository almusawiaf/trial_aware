"""
hierarchy.py
ICD-10 hierarchy management with memoization and duplicate handling.
"""

import csv
import logging
from typing import Dict, Set, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ICD10Hierarchy:
    """
    Clean data model for ICD-10 hierarchy.
    
    Canonical representation:
    - parent_to_children: dict[str, list[str]]  # Parent → Children
    - child_to_parent: dict[str, str]          # Child → Parent (built once)
    - _ancestor_cache: dict[str, set]           # Memoized ancestors
    """
    
    def __init__(self, hierarchy_file: str = None, log_duplicates: bool = True):
        self.parent_to_children: Dict[str, list] = {}
        self.child_to_parent: Dict[str, str] = {}
        self._ancestor_cache: Dict[str, set] = {}
        self.log_duplicates = log_duplicates
        
        if hierarchy_file:
            self.load_from_file(hierarchy_file)
    
    def load_from_file(self, file_path: str):
        """
        Load ICD-10 hierarchy from CSV file.
        Expected format: parent_code, child_code
        """
        duplicate_count = 0
        conflicting_count = 0
        skipped_count = 0
        
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                
                # Try to detect header
                first_row = next(reader, None)
                if first_row and self._is_header_row(first_row):
                    # Skip header
                    start_row = 0
                else:
                    # Rewind and start from beginning
                    f.seek(0)
                    reader = csv.reader(f)
                    start_row = -1
                
                for row in reader:
                    # Skip empty rows
                    if not row or len(row) < 2:
                        skipped_count += 1
                        continue
                    
                    parent = row[0].strip()
                    child = row[1].strip()
                    
                    # Skip empty or malformed entries
                    if not parent or not child:
                        skipped_count += 1
                        continue
                    
                    # Skip if it looks like a header
                    if self._is_header_row([parent, child]):
                        continue
                    
                    # Build parent → children
                    if parent not in self.parent_to_children:
                        self.parent_to_children[parent] = []
                    self.parent_to_children[parent].append(child)
                    
                    # Build child → parent with duplicate checking
                    if child in self.child_to_parent:
                        duplicate_count += 1
                        existing_parent = self.child_to_parent[child]
                        if existing_parent != parent:
                            conflicting_count += 1
                            if self.log_duplicates:
                                logging.warning(
                                    f"Conflicting parent for {child}: "
                                    f"existing='{existing_parent}', new='{parent}' - keeping existing"
                                )
                        else:
                            if self.log_duplicates:
                                logging.debug(f"Duplicate entry for {child} -> {parent} (ignored)")
                    else:
                        self.child_to_parent[child] = parent
        
        except FileNotFoundError:
            logging.error(f"Hierarchy file not found: {file_path}")
            raise
        
        logging.info(
            f"Loaded {len(self.parent_to_children)} parents, "
            f"{len(self.child_to_parent)} children, "
            f"{duplicate_count} duplicates, {conflicting_count} conflicts, "
            f"{skipped_count} skipped rows"
        )
    
    def _is_header_row(self, row: list) -> bool:
        """Check if a row looks like a header."""
        if not row:
            return False
        header_indicators = ['parent', 'child', 'code', 'icd10', 'icd', 'description']
        for cell in row:
            if cell and cell.lower().strip() in header_indicators:
                return True
        return False
    
    def get_ancestors(self, code: str) -> set:
        """
        Get all ancestors of a code.
        Uses memoization for performance.
        """
        if code in self._ancestor_cache:
            return self._ancestor_cache[code]
        
        ancestors = set()
        current = code
        
        while current in self.child_to_parent:
            parent = self.child_to_parent[current]
            ancestors.add(parent)
            current = parent
        
        self._ancestor_cache[code] = ancestors
        return ancestors
    
    def is_ancestor(self, ancestor_code: str, descendant_code: str) -> bool:
        """
        Check if ancestor_code is an ancestor of descendant_code.
        """
        # Quick exact match
        if ancestor_code == descendant_code:
            return True
        
        # Get all ancestors of descendant and check
        ancestors = self.get_ancestors(descendant_code)
        return ancestor_code in ancestors
    
    def get_match_score(self, code1: str, code2: str) -> float:
        """
        Get match score between two codes with hierarchy support.
        Returns:
            1.0: exact match
            0.8: code1 is ancestor of code2 (broader)
            0.6: code2 is ancestor of code1 (broader)
            0.0: no match
        """
        if code1 == code2:
            return 1.0
        
        if self.is_ancestor(code1, code2):
            return 0.8
        
        if self.is_ancestor(code2, code1):
            return 0.6
        
        return 0.0
    
    def clear_cache(self):
        """Clear the ancestor cache (useful if hierarchy changes)."""
        self._ancestor_cache = {}
    
    def __len__(self):
        return len(self.child_to_parent)


# ============================================================
# End-to-End Sanity Check
# ============================================================

def run_sanity_check():
    """Run a small end-to-end test on real data to verify M_inc/M_exc > 0."""
    import json
    import pandas as pd
    from config import Config
    from models.claude_active.trial_graph import PatientClinicalState, TrialStore, compute_matching_indices
    
    cfg = Config()
    
    print("=" * 60)
    print("  END-TO-END SANITY CHECK")
    print("=" * 60)
    
    # 1. Load patient data
    print("\n📊 Loading patient data...")
    diag_path = f"{cfg.OUTPUT_DIR}/diagnoses_clean.parquet"
    if not pd.io.parquet.read_parquet(diag_path, engine='pyarrow', use_nullable_dtypes=True):
        print(f"   ❌ No diagnosis data found at {diag_path}")
        return
    
    diag_df = pd.read_parquet(diag_path)
    rx_df = pd.read_parquet(f"{cfg.OUTPUT_DIR}/prescriptions_clean.parquet")
    labs_df = pd.read_parquet(f"{cfg.OUTPUT_DIR}/labs_clean.parquet")
    
    # Take a small sample for speed
    sample_patients = diag_df['SUBJECT_ID'].unique()[:50]
    diag_df = diag_df[diag_df['SUBJECT_ID'].isin(sample_patients)]
    rx_df = rx_df[rx_df['SUBJECT_ID'].isin(sample_patients)]
    labs_df = labs_df[labs_df['SUBJECT_ID'].isin(sample_patients)]
    
    patient_states = {
        sid: PatientClinicalState.build_from_tables(sid, diag_df, rx_df, labs_df)
        for sid in sample_patients
    }
    print(f"   ✅ Loaded {len(patient_states)} patients")
    
    # 2. Load trials
    print("\n📊 Loading trial data...")
    trial_path = f"{cfg.TRIALS_DATA_DIR}/structured_clinical_trials.json"
    
    if not os.path.exists(trial_path):
        print(f"   ❌ Trials not found at {trial_path}")
        return
    
    with open(trial_path, "r") as f:
        trials_data = json.load(f)
    trials_data = trials_data[:10]  # First 10 trials
    trial_store = TrialStore.from_records(trials_data)
    print(f"   ✅ Loaded {len(trial_store.trials)} trials")
    
    # 3. Load hierarchy (if available)
    print("\n📊 Loading ICD-10 hierarchy...")
    hierarchy = ICD10Hierarchy()
    # Try common hierarchy file locations
    hierarchy_paths = [
        "icd10_hierarchy.csv",
        "../icd10_hierarchy.csv",
        cfg.DATA_DIR + "/icd10_hierarchy.csv"
    ]
    for path in hierarchy_paths:
        if os.path.exists(path):
            hierarchy.load_from_file(path)
            break
    print(f"   ✅ Loaded {len(hierarchy)} hierarchy entries")
    
    # 4. Compute M_inc/M_exc
    print("\n📊 Computing patient-trial matches...")
    nonzero_count = 0
    total_count = 0
    m_inc_values = []
    m_exc_values = []
    
    # Define a wrapper that uses hierarchy
    def compute_with_hierarchy(state, trial):
        # Use the existing compute_matching_indices but pass hierarchy through
        # Since compute_matching_indices doesn't take hierarchy, we need to
        # check if we can call it with hierarchy parameter
        try:
            m_inc, m_exc = compute_matching_indices(state, trial, hierarchy)
        except TypeError:
            # Fallback to original without hierarchy
            m_inc, m_exc = compute_matching_indices(state, trial)
        return m_inc, m_exc
    
    for pid, state in patient_states.items():
        for tid, trial in trial_store.trials.items():
            total_count += 1
            try:
                m_inc, m_exc = compute_with_hierarchy(state, trial)
                m_inc_values.append(m_inc)
                m_exc_values.append(m_exc)
                if m_inc > 0 or m_exc > 0:
                    nonzero_count += 1
            except Exception as e:
                print(f"   ⚠️ Error computing matches: {e}")
    
    # 5. Report results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"Total pairs: {total_count}")
    print(f"Nonzero M_inc/M_exc pairs: {nonzero_count}")
    print(f"Fraction: {nonzero_count/total_count*100:.2f}%")
    
    if m_inc_values:
        print(f"Mean M_inc: {sum(m_inc_values)/len(m_inc_values):.4f}")
        print(f"Mean M_exc: {sum(m_exc_values)/len(m_exc_values):.4f}")
        print(f"Max M_inc: {max(m_inc_values):.4f}")
        print(f"Max M_exc: {max(m_exc_values):.4f}")
    
    if nonzero_count == 0:
        print("\n❌ CRITICAL: No matches found. Pipeline will not learn.")
        print("   Possible causes:")
        print("   - Trial codes still don't match patient codes")
        print("   - ICD-10 hierarchy file is empty or misformatted")
        print("   - Trial criteria need to be remapped to ICD-10 codes")
    else:
        print("\n✅ Matches found! Pipeline can proceed.")
        print(f"   {nonzero_count} pairs have nonzero matches out of {total_count}")

if __name__ == "__main__":
    import os
    run_sanity_check()