import os
import sys
import subprocess
import logging
import numpy as np
import pandas as pd
from typing import Set, List
from config import Config

try:
    from tqdm import tqdm
except ImportError:
    logging.info("tqdm not found. Installing dynamically...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
    except Exception:
        def tqdm(iterable, *args, **kwargs):
            return iterable


class MIMICDataPreprocessor:
    def __init__(self, config: Config):
        self.cfg = config
        self.selected_patients: Set[int] = set()
        self.active_labs: List[int] = []
        self.lab_normalization_stats = {}
        
        # Load NBER ICD-9 to ICD-10 CM (Diagnoses) crosswalk dictionary
        csv_path = getattr(self.cfg, "ICD9_TO_ICD10_CSV", None)
        self.icd9_to_icd10_map = self._load_icd_mapping_csv(csv_path)

    def filter_cohort(self, admissions_df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting Cohort Selection...")
        df = admissions_df.copy()
        df.columns = df.columns.str.upper()

        df = df.dropna(subset=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'])
        df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(int)
        df['HADM_ID'] = df['HADM_ID'].astype(int)
        df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')
        df = df.dropna(subset=['ADMITTIME'])

        pt_stats = df.groupby('SUBJECT_ID').agg(
            encounter_count=('HADM_ID', 'nunique'),
            first_admit=('ADMITTIME', 'min'),
            last_admit=('ADMITTIME', 'max')
        ).reset_index()

        pt_stats['observation_span'] = (pt_stats['last_admit'] - pt_stats['first_admit']).dt.days

        valid_pt_mask = (
            (pt_stats['encounter_count'] >= self.cfg.MIN_ENCOUNTERS) &
            (pt_stats['observation_span'] >= self.cfg.MIN_TEMPORAL_SPACING_DAYS)
        )
        valid_patients_df = pt_stats[valid_pt_mask]

        self.selected_patients = set(valid_patients_df['SUBJECT_ID'].astype(int))
        logging.info(f"[Cohort] Selected {len(self.selected_patients)} patients out of {len(pt_stats)} total.")

        return df[df['SUBJECT_ID'].isin(self.selected_patients)]

    def _load_icd_mapping_csv(self, csv_path: str) -> dict:
        """Loads NBER GEMs crosswalk CSV and constructs an in-memory ICD9 -> ICD10 lookup dict."""
        if not csv_path or not os.path.exists(csv_path):
            logging.warning(f"ICD crosswalk CSV not found at '{csv_path}'. Will fallback to raw codes if unmapped.")
            return {}

        logging.info(f"Loading official ICD-9 to ICD-10 GEMs mapping from {csv_path}...")
        mapping_df = pd.read_csv(csv_path, dtype=str)
        
        # Standardize column names to lowercase just in case
        mapping_df.columns = mapping_df.columns.str.lower()
        
        # 1. Drop codes marked as having no valid mapping
        if 'no_map' in mapping_df.columns:
            mapping_df = mapping_df[mapping_df['no_map'] != '1']

        # 2. Clean/format codes (strip whitespaces and dots)
        mapping_df['icd9cm'] = mapping_df['icd9cm'].str.strip().str.replace('.', '', regex=False).str.upper()
        mapping_df['icd10cm'] = mapping_df['icd10cm'].str.strip().str.replace('.', '', regex=False).str.upper()

        # 3. Handle duplicates: Keep the first mapped ICD-10 code for each ICD-9 code
        mapping_df = mapping_df.drop_duplicates(subset=['icd9cm'], keep='first')

        # Convert to dictionary for O(1) fast lookup
        icd_dict = dict(zip(mapping_df['icd9cm'], mapping_df['icd10cm']))
        logging.info(f"Loaded {len(icd_dict)} unique ICD-9 -> ICD-10 diagnosis mapping pairs.")
        return icd_dict

    def process_diagnoses(self, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Processing Diagnosis Codes...")
        df = diagnoses_df.copy()
        df.columns = df.columns.str.upper()

        df = df.dropna(subset=['SUBJECT_ID', 'HADM_ID'])
        df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(int)
        df['HADM_ID'] = df['HADM_ID'].astype(int)
        df = df[df['SUBJECT_ID'].isin(self.selected_patients)]

        # Clean raw ICD-9 codes
        df['ICD9_CODE'] = df['ICD9_CODE'].astype(str).str.strip().str.replace('.', '', regex=False).str.upper()
        df = df[df['ICD9_CODE'] != 'NAN']

        # Vectorized map using preloaded NBER CSV dictionary
        logging.info("[Diagnoses] Mapping ICD-9 to ICD-10 via NBER GEMs CSV...")
        df['ICD10_CODE'] = df['ICD9_CODE'].map(self.icd9_to_icd10_map).fillna(df['ICD9_CODE'])

        # Log mapping coverage stats
        unmapped = (df['ICD10_CODE'] == df['ICD9_CODE']).sum()
        mapped_count = len(df) - unmapped
        mapping_pct = (mapped_count / len(df)) * 100 if len(df) > 0 else 0
        
        logging.info(f"[Diagnoses] Successfully mapped {mapped_count}/{len(df)} codes ({mapping_pct:.2f}% coverage).")
        if unmapped > 0:
            logging.warning(
                f"[Diagnoses] {unmapped} entries had no match in NBER crosswalk and retained their raw code."
            )

        return df[['SUBJECT_ID', 'HADM_ID', 'ICD10_CODE']]

    def process_prescriptions(self, prescriptions_df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Processing Medication Prescriptions...")
        df = prescriptions_df.copy()
        df.columns = df.columns.str.upper()

        df = df.dropna(subset=['SUBJECT_ID', 'HADM_ID'])
        df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(int)
        df['HADM_ID'] = df['HADM_ID'].astype(int)
        df = df[df['SUBJECT_ID'].isin(self.selected_patients)]

        df = df.dropna(subset=['DRUG', 'NDC'])
        df['NDC'] = df['NDC'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df = df[df['NDC'] != '0']

        logging.info(f"[Prescriptions] Processed {len(df)} standardized drug prescriptions.")
        return df[['SUBJECT_ID', 'HADM_ID', 'DRUG', 'NDC']]

    def process_labs(self, labevents_df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Processing Laboratory Events with Daily Grid Temporal Expansion...")
        df = labevents_df.copy()
        df.columns = df.columns.str.upper()

        df['VALUENUM'] = pd.to_numeric(df['VALUENUM'], errors='coerce')
        df = df.dropna(subset=['VALUENUM', 'ITEMID', 'CHARTTIME', 'SUBJECT_ID'])

        df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(int)
        df['ITEMID'] = df['ITEMID'].astype(int)
        df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'], errors='coerce').dt.normalize()
        df = df.dropna(subset=['CHARTTIME'])

        df = df[df['SUBJECT_ID'].isin(self.selected_patients)]

        patient_counts = df.groupby('ITEMID')['SUBJECT_ID'].nunique()
        total_selected = len(self.selected_patients)
        self.active_labs = patient_counts[
            (patient_counts / total_selected) >= self.cfg.MIN_LAB_FREQ_THRESHOLD
        ].index.tolist()

        df = df[df['ITEMID'].isin(self.active_labs)]
        logging.info(f"[Labs] Retained {len(self.active_labs)} active lab items after sparsity filtering.")

        processed_chunks = []
        for item_id, group in df.groupby('ITEMID'):
            group = group.copy()
            mean = group['VALUENUM'].mean()
            std = group['VALUENUM'].std()

            if std == 0 or np.isnan(std):
                continue

            lower_bound = mean - (self.cfg.OUTLIER_SIGMA_THRESHOLD * std)
            upper_bound = mean + (self.cfg.OUTLIER_SIGMA_THRESHOLD * std)

            clamped_values = np.clip(group['VALUENUM'], lower_bound, upper_bound)
            group['VALUENUM_NORM'] = (clamped_values - mean) / std

            self.lab_normalization_stats[item_id] = {'mean': mean, 'std': std}
            processed_chunks.append(group)

        df_norm = pd.concat(processed_chunks, ignore_index=True)

        logging.info("[Labs] Expanding trajectories to bounded daily timelines "
                      f"(cap={self.cfg.MAX_DAYS_SINCE_MEASURED} days)...")
        
        # Memory-efficient list extension
        all_charttimes = []
        all_sub_ids = []
        all_item_ids = []
        all_decayed = []
        all_days = []

        cap = self.cfg.MAX_DAYS_SINCE_MEASURED
        rho = self.cfg.RHO

        grouped = df_norm.groupby(['SUBJECT_ID', 'ITEMID'])
        for (sub_id, item_id), group in tqdm(grouped, desc="Processing Patient Lab Grids"):
            group = group.sort_values('CHARTTIME').drop_duplicates(subset=['CHARTTIME'])

            times = group['CHARTTIME'].values
            values = group['VALUENUM_NORM'].values

            if len(times) == 1:
                all_charttimes.append(times[0])
                all_sub_ids.append(sub_id)
                all_item_ids.append(item_id)
                all_decayed.append(values[0])
                all_days.append(0.0)
                continue

            for i in range(len(times) - 1):
                seg_start = pd.Timestamp(times[i])
                seg_end = pd.Timestamp(times[i + 1])
                horizon = min((seg_end - seg_start).days, cap)
                
                day_offsets = np.arange(0, horizon + 1)
                seg_dates = seg_start + pd.to_timedelta(day_offsets, unit='D')
                decayed_vals = values[i] * np.exp(-rho * day_offsets)

                all_charttimes.extend(seg_dates)
                all_sub_ids.extend([sub_id] * len(day_offsets))
                all_item_ids.extend([item_id] * len(day_offsets))
                all_decayed.extend(decayed_vals)
                all_days.extend(day_offsets)

            # Tail segment after the last measurement
            last_start = pd.Timestamp(times[-1])
            day_offsets = np.arange(0, cap + 1)
            seg_dates = last_start + pd.to_timedelta(day_offsets, unit='D')
            decayed_vals = values[-1] * np.exp(-rho * day_offsets)

            all_charttimes.extend(seg_dates)
            all_sub_ids.extend([sub_id] * len(day_offsets))
            all_item_ids.extend([item_id] * len(day_offsets))
            all_decayed.extend(decayed_vals)
            all_days.extend(day_offsets)

        logging.info("[Labs] Constructing final expanded DataFrame...")
        final_labs_expanded = pd.DataFrame({
            'CHARTTIME': all_charttimes,
            'SUBJECT_ID': np.array(all_sub_ids, dtype=np.int32),
            'ITEMID': np.array(all_item_ids, dtype=np.int32),
            'IMPUTED_VALUE_DECAYED': np.array(all_decayed, dtype=np.float32),
            'DAYS_SINCE_MEASURED': np.array(all_days, dtype=np.float32)
        })

        final_labs_expanded = final_labs_expanded.drop_duplicates(
            subset=['SUBJECT_ID', 'ITEMID', 'CHARTTIME'], keep='first'
        )
        logging.info("[Labs] Bounded daily trajectory expansion and continuous time decay complete.")

        return final_labs_expanded