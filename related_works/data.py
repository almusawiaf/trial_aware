"""
data.py
=======
Loads the patient cohort and the structured trial set, and provides a synthetic
fallback so the benchmark is runnable without MIMIC-III credentials.

Three input routes, tried in order:

1. `--data real`      : upstream parquet outputs + structured trial JSON.
2. `--data real_ctg`  : upstream parquet outputs + criteria parsed here directly
                        from the raw ClinicalTrials.gov dump shipped in the repo
                        (`extracting_trials/ctg-studies_1000.json`).
3. `--data synthetic` : fully synthetic MIMIC-like cohort + synthetic trials.
                        Structure (comorbidity clusters, drug-diagnosis
                        coupling, lab correlation) is realistic enough to
                        exercise every code path, but numbers produced from it
                        are smoke-test artefacts and must never be reported.

Note on a real upstream bug
---------------------------
`trial_graph.PatientClinicalState.build_from_tables` looks for a lab value
column named VALUENUM/VALUE/..., but `preprocessor.process_labs` writes the
column as IMPUTED_VALUE_DECAYED. On the real pipeline output the patient lab
dictionary therefore comes back EMPTY, which silently turns every lab criterion
into a non-match. `load_patients` below accepts either column name and warns
when it has to fall back, so the effect is visible instead of silent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

VALID_ENTITY_TYPES = ("diagnosis", "medication", "lab", "procedure")
VALID_OPERATORS = (
    "EXISTS",
    "NOT_EXISTS",
    "GT",
    "GTE",
    "LT",
    "LTE",
    "EQ",
    "BETWEEN",
)


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------
@dataclass
class PatientRecord:
    patient_id: str
    diagnosis_codes: Set[str] = field(default_factory=set)
    medication_codes: Set[str] = field(default_factory=set)
    lab_values: Dict[str, float] = field(default_factory=dict)
    n_admissions: int = 1

    def summary(self) -> Dict[str, float]:
        return {
            "n_dx": float(len(self.diagnosis_codes)),
            "n_rx": float(len(self.medication_codes)),
            "n_lab": float(len(self.lab_values)),
            "n_adm": float(self.n_admissions),
        }


@dataclass
class CriterionSpec:
    entity_type: str
    entity_code: str
    operator: str
    value: Optional[float] = None
    max_value: Optional[float] = None
    is_inclusion: bool = True
    severity_weight: float = 1.0
    raw_text: str = ""

    @property
    def is_resolved(self) -> bool:
        """False for codes the upstream extractor never mapped to a real term."""
        c = str(self.entity_code).strip()
        return not (
            c == ""
            or c.lower() in {"none", "nan", "unknown_code"}
            or c.startswith("UNMATCHED_")
            or c.startswith("UNK_")
        )


@dataclass
class TrialSpec:
    trial_id: str
    criteria: List[CriterionSpec] = field(default_factory=list)
    title: str = ""
    conditions: List[str] = field(default_factory=list)
    phase: str = ""
    sample_size: int = 0

    @property
    def inclusion(self) -> List[CriterionSpec]:
        return [c for c in self.criteria if c.is_inclusion]

    @property
    def exclusion(self) -> List[CriterionSpec]:
        return [c for c in self.criteria if not c.is_inclusion]

    def text(self) -> str:
        parts = [self.title] + list(self.conditions)
        parts += [c.raw_text for c in self.criteria if c.raw_text]
        return " ".join(p for p in parts if p)


@dataclass
class Dataset:
    patients: List[PatientRecord]
    trials: List[TrialSpec]
    source: str = "unknown"

    @property
    def patient_ids(self) -> List[str]:
        return [p.patient_id for p in self.patients]

    @property
    def trial_ids(self) -> List[str]:
        return [t.trial_id for t in self.trials]

    def describe(self) -> str:
        n_inc = np.mean([len(t.inclusion) for t in self.trials]) if self.trials else 0
        n_exc = np.mean([len(t.exclusion) for t in self.trials]) if self.trials else 0
        n_dx = np.mean([len(p.diagnosis_codes) for p in self.patients]) if self.patients else 0
        return (
            f"Dataset(source={self.source}, patients={len(self.patients)}, "
            f"trials={len(self.trials)}, mean_inc={n_inc:.1f}, mean_exc={n_exc:.1f}, "
            f"mean_dx_per_patient={n_dx:.1f})"
        )


# ---------------------------------------------------------------------------
# Code normalisation -- must agree with the upstream matcher
# ---------------------------------------------------------------------------
def normalize_code(entity_type: str, code) -> str:
    """Normalise a code the same way the upstream matching engine does.

    Diagnoses: strip dots, upper-case (ICD-10 crosswalk convention).
    Medications: strip pandas float artefacts ('.0'), keep digits only, then
    zero-pad to the 11-digit NDC convention when plausible.
    """
    c = str(code).strip()
    if entity_type == "diagnosis":
        return c.replace(".", "").upper()
    if entity_type == "medication":
        c = re.sub(r"\.0+$", "", c)
        c = re.sub(r"[^0-9]", "", c)
        if c.isdigit() and 0 < len(c) < 11:
            c = c.zfill(11)
        return c
    return c


# ---------------------------------------------------------------------------
# Loading the real cohort
# ---------------------------------------------------------------------------
def load_patients(
    diagnoses_path: str,
    prescriptions_path: str,
    labs_path: str,
    max_patients: Optional[int] = None,
) -> List[PatientRecord]:
    """Build PatientRecords from the upstream pipeline's parquet outputs."""
    diag = _read_table(diagnoses_path)
    rx = _read_table(prescriptions_path)
    labs = _read_table(labs_path)

    dx_col = _first_present(diag, ["ICD10_CODE", "ICD9_CODE"])
    if dx_col is None:
        raise ValueError(f"No diagnosis code column found in {diagnoses_path}")

    diag = diag[["SUBJECT_ID", dx_col]].dropna()
    diag["code"] = [normalize_code("diagnosis", c) for c in diag[dx_col].astype(str)]
    dx_map = diag.groupby("SUBJECT_ID")["code"].apply(set).to_dict()

    rx_col = _first_present(rx, ["NDC", "DRUG"])
    rx_map: Dict = {}
    if rx_col is not None:
        rx_small = rx[["SUBJECT_ID", rx_col]].dropna()
        rx_small["code"] = [
            normalize_code("medication", c) for c in rx_small[rx_col].astype(str)
        ]
        rx_map = rx_small.groupby("SUBJECT_ID")["code"].apply(set).to_dict()

    lab_val_col = _first_present(
        labs, ["VALUENUM", "VALUE", "IMPUTED_VALUE_DECAYED", "valuenum", "value"]
    )
    if lab_val_col == "IMPUTED_VALUE_DECAYED":
        log.warning(
            "Lab table exposes only IMPUTED_VALUE_DECAYED. Upstream "
            "PatientClinicalState.build_from_tables looks for VALUENUM and would "
            "silently produce an EMPTY lab dict here. Using the decayed column."
        )
    lab_map: Dict = {}
    if lab_val_col is not None and "ITEMID" in labs.columns:
        lab_small = labs[["SUBJECT_ID", "ITEMID", lab_val_col]].dropna()
        # Most-recent-per-item semantics: keep the last row per (patient, item).
        lab_small = lab_small.drop_duplicates(
            subset=["SUBJECT_ID", "ITEMID"], keep="last"
        )
        for sid, grp in lab_small.groupby("SUBJECT_ID"):
            lab_map[sid] = {
                str(int(i)): float(v)
                for i, v in zip(grp["ITEMID"], grp[lab_val_col])
                if np.isfinite(v)
            }

    subject_ids = sorted(dx_map.keys(), key=lambda x: int(x))
    if max_patients is not None:
        subject_ids = subject_ids[:max_patients]

    records = []
    for sid in subject_ids:
        records.append(
            PatientRecord(
                patient_id=str(sid),
                diagnosis_codes=dx_map.get(sid, set()),
                medication_codes=rx_map.get(sid, set()),
                lab_values=lab_map.get(sid, {}),
            )
        )
    log.info("Loaded %d patients from %s", len(records), diagnoses_path)
    return records


def _read_table(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    csv_path = path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither {path} nor {csv_path} exists")


def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_structured_trials(*paths: str) -> List[TrialSpec]:
    """Load one or more structured trial JSON files produced upstream."""
    trials: List[TrialSpec] = []
    seen: Set[str] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r") as f:
            records = json.load(f)
        for rec in records:
            t = _trial_from_record(rec)
            if t is not None and t.trial_id not in seen:
                seen.add(t.trial_id)
                trials.append(t)
    log.info("Loaded %d structured trials", len(trials))
    return trials


def _trial_from_record(rec: dict) -> Optional[TrialSpec]:
    tid = rec.get("trial_id") or rec.get("nct_id")
    if not tid:
        return None
    raw_criteria = rec.get("criteria") or rec.get("eligibility_criteria") or []
    criteria = []
    for c in raw_criteria:
        etype = c.get("entity_type") or c.get("concept_type") or "diagnosis"
        code = c.get("entity_code") or c.get("concept_code") or ""
        op = str(c.get("operator", "EXISTS")).upper()
        op = op if op in VALID_OPERATORS else _map_operator_symbol(op)
        criteria.append(
            CriterionSpec(
                entity_type=etype if etype in VALID_ENTITY_TYPES else "diagnosis",
                entity_code=normalize_code(etype, code),
                operator=op,
                value=_as_float(c.get("value")),
                max_value=_as_float(c.get("max_value")),
                is_inclusion=bool(c.get("is_inclusion", True)),
                severity_weight=float(c.get("severity_weight", c.get("weight", 1.0))),
                raw_text=str(c.get("raw_entity", ""))[:400],
            )
        )
    return TrialSpec(
        trial_id=str(tid),
        criteria=criteria,
        title=str(rec.get("title", "")),
        conditions=[str(x) for x in rec.get("conditions", [])],
        phase=str(rec.get("phase", "")),
        sample_size=int(rec.get("sample_size", 0) or 0),
    )


def _map_operator_symbol(op: str) -> str:
    return {
        ">": "GT",
        ">=": "GTE",
        "<": "LT",
        "<=": "LTE",
        "=": "EQ",
        "==": "EQ",
    }.get(op, "EXISTS")


def _as_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Parsing the raw ClinicalTrials.gov dump directly
# ---------------------------------------------------------------------------
_LAB_TERMS = {
    "creatinine": "50912",
    "hemoglobin": "51222",
    "glucose": "50931",
    "potassium": "50971",
    "sodium": "50983",
    "platelet": "51265",
    "bilirubin": "50885",
    "albumin": "50862",
    "wbc": "51301",
    "hematocrit": "51221",
    "inr": "51237",
    "bun": "51006",
    "lactate": "50813",
    "troponin": "51002",
    "hba1c": "50852",
}

_MED_TERMS = (
    "metformin insulin warfarin heparin aspirin statin furosemide lisinopril "
    "metoprolol amiodarone digoxin clopidogrel prednisone chemotherapy "
    "antibiotic anticoagulant beta-blocker ace inhibitor diuretic"
).split()

_OPERATOR_PATTERNS = [
    (r"\bbetween\s+([\d.]+)\s*(?:and|-|to)\s*([\d.]+)", "BETWEEN"),
    (r"(?:>=|\u2265|greater than or equal to|at least|no less than)\s*([\d.]+)", "GTE"),
    (r"(?:<=|\u2264|less than or equal to|at most|no more than|not exceed(?:ing)?)\s*([\d.]+)", "LTE"),
    (r"(?:>|greater than|above|exceeds?|higher than)\s*([\d.]+)", "GT"),
    (r"(?:<|less than|below|lower than|under)\s*([\d.]+)", "LT"),
    (r"(?:=|equal to|exactly)\s*([\d.]+)", "EQ"),
]


def parse_raw_ctg_trials(
    path: str,
    diagnosis_vocab: Optional[Set[str]] = None,
    condition_to_codes: Optional[Dict[str, List[str]]] = None,
    max_trials: Optional[int] = None,
) -> List[TrialSpec]:
    """Parse ClinicalTrials.gov study records into structured criteria.

    This is a self-contained, deterministic rule parser. It is deliberately
    conservative: a criterion line that cannot be resolved to a code in
    `diagnosis_vocab` is emitted with an UNK_ code and is later dropped from
    scoring (mirroring the upstream `_is_placeholder_code` behaviour), rather
    than being silently counted as a failed match.

    `condition_to_codes` maps a lower-cased condition phrase to the ICD-10
    prefixes that represent it in the cohort; supply one built from the actual
    patient vocabulary for best coverage.
    """
    with open(path, "r") as f:
        studies = json.load(f)
    if isinstance(studies, dict):
        studies = studies.get("studies", [])

    condition_to_codes = condition_to_codes or {}
    trials: List[TrialSpec] = []

    for study in studies:
        proto = study.get("protocolSection", study)
        ident = proto.get("identificationModule", {})
        nct = ident.get("nctId")
        if not nct:
            continue

        elig = proto.get("eligibilityModule", {})
        criteria_text = elig.get("eligibilityCriteria", "") or ""
        cond_mod = proto.get("conditionsModule", {})
        conditions = [str(c) for c in cond_mod.get("conditions", [])]
        design = proto.get("designModule", {})
        phases = design.get("phases", []) or []
        enroll = (design.get("enrollmentInfo", {}) or {}).get("count", 0) or 0

        criteria = _parse_criteria_text(
            criteria_text, conditions, diagnosis_vocab, condition_to_codes
        )
        if not criteria:
            continue

        trials.append(
            TrialSpec(
                trial_id=str(nct),
                criteria=criteria,
                title=str(ident.get("briefTitle", "")),
                conditions=conditions,
                phase=",".join(str(p) for p in phases),
                sample_size=int(enroll),
            )
        )
        if max_trials is not None and len(trials) >= max_trials:
            break

    log.info("Parsed %d trials from raw CTG dump %s", len(trials), path)
    return trials


def _parse_criteria_text(
    text: str,
    conditions: Sequence[str],
    diagnosis_vocab: Optional[Set[str]],
    condition_to_codes: Dict[str, List[str]],
) -> List[CriterionSpec]:
    lines = [ln.strip(" \t-*\u2022") for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) >= 8]

    is_inclusion = True
    out: List[CriterionSpec] = []
    for line in lines:
        low = line.lower()
        if low.startswith("inclusion") or "inclusion criteria" in low:
            is_inclusion = True
            continue
        if low.startswith("exclusion") or "exclusion criteria" in low:
            is_inclusion = False
            continue
        if line.endswith(":") and len(line) < 60:
            continue
        c = _extract_criterion(line, is_inclusion, diagnosis_vocab, condition_to_codes)
        if c is not None:
            out.append(c)

    # Attach the registered conditions as inclusion criteria when they resolve.
    for cond in conditions:
        for code in condition_to_codes.get(cond.lower().strip(), []):
            out.append(
                CriterionSpec(
                    entity_type="diagnosis",
                    entity_code=code,
                    operator="EXISTS",
                    is_inclusion=True,
                    severity_weight=1.5,   # registered condition weighs more
                    raw_text=cond,
                )
            )
    return out


def _extract_criterion(
    line: str,
    is_inclusion: bool,
    diagnosis_vocab: Optional[Set[str]],
    condition_to_codes: Dict[str, List[str]],
) -> Optional[CriterionSpec]:
    low = line.lower()

    # 1. Lab criteria carry an operator and a numeric threshold.
    for term, itemid in _LAB_TERMS.items():
        if term in low:
            op, val, maxval = _extract_operator(low)
            return CriterionSpec(
                entity_type="lab",
                entity_code=itemid,
                operator=op,
                value=val,
                max_value=maxval,
                is_inclusion=is_inclusion,
                raw_text=line[:400],
            )

    # 2. Medications.
    for term in _MED_TERMS:
        if term in low:
            return CriterionSpec(
                entity_type="medication",
                entity_code=_stable_unk_code("medication", term),
                operator="EXISTS",
                is_inclusion=is_inclusion,
                raw_text=line[:400],
            )

    # 3. Diagnoses -- resolve through the supplied condition map.
    for phrase, codes in condition_to_codes.items():
        if phrase and phrase in low and codes:
            return CriterionSpec(
                entity_type="diagnosis",
                entity_code=codes[0],
                operator="NOT_EXISTS" if not is_inclusion else "EXISTS",
                is_inclusion=is_inclusion,
                raw_text=line[:400],
            )

    # 4. Unresolvable -- emit a stable placeholder so it is reproducibly dropped.
    return CriterionSpec(
        entity_type="diagnosis",
        entity_code=_stable_unk_code("diagnosis", line),
        operator="EXISTS",
        is_inclusion=is_inclusion,
        raw_text=line[:400],
    )


def _extract_operator(text: str) -> Tuple[str, Optional[float], Optional[float]]:
    for pattern, op in _OPERATOR_PATTERNS:
        m = re.search(pattern, text)
        if m:
            if op == "BETWEEN":
                return op, _as_float(m.group(1)), _as_float(m.group(2))
            return op, _as_float(m.group(1)), None
    return "EXISTS", None, None


def _stable_unk_code(prefix: str, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"UNK_{prefix[:3].upper()}_{h}"


def build_condition_map(patients: Sequence[PatientRecord], top_n: int = 400) -> Dict[str, List[str]]:
    """Map common clinical phrases to ICD-10 prefixes present in the cohort.

    Only phrases whose prefix actually occurs in the cohort vocabulary are
    retained, so a trial criterion can never match a code no patient can have.
    """
    vocab: Set[str] = set()
    for p in patients:
        vocab |= p.diagnosis_codes

    phrase_to_prefix = {
        "heart failure": "I50",
        "congestive heart failure": "I50",
        "myocardial infarction": "I21",
        "coronary artery disease": "I25",
        "atrial fibrillation": "I48",
        "hypertension": "I10",
        "hyperlipidemia": "E78",
        "diabetes": "E11",
        "type 2 diabetes": "E11",
        "type 1 diabetes": "E10",
        "diabetes mellitus": "E11",
        "chronic kidney disease": "N18",
        "acute kidney injury": "N17",
        "renal failure": "N19",
        "pneumonia": "J18",
        "sepsis": "A41",
        "septic shock": "R65",
        "copd": "J44",
        "chronic obstructive pulmonary disease": "J44",
        "asthma": "J45",
        "respiratory failure": "J96",
        "stroke": "I63",
        "anemia": "D64",
        "atrial flutter": "I48",
        "obesity": "E66",
        "depression": "F32",
        "cirrhosis": "K74",
        "gastrointestinal bleeding": "K92",
        "pulmonary embolism": "I26",
        "deep vein thrombosis": "I82",
        "cancer": "C80",
        "breast cancer": "C50",
        "lung cancer": "C34",
        "hypothyroidism": "E03",
    }

    out: Dict[str, List[str]] = {}
    for phrase, prefix in phrase_to_prefix.items():
        matches = sorted(c for c in vocab if c.startswith(prefix))[:top_n]
        if matches:
            out[phrase] = matches
    return out


# ---------------------------------------------------------------------------
# Synthetic fallback cohort
# ---------------------------------------------------------------------------
_SYN_DISEASE_CLUSTERS = {
    "cardiac": ["I50", "I21", "I25", "I48", "I10"],
    "metabolic": ["E11", "E78", "E66", "E03"],
    "renal": ["N18", "N17", "N19"],
    "pulmonary": ["J18", "J44", "J45", "J96"],
    "infectious": ["A41", "R65", "J18"],
    "onc": ["C50", "C34", "C80", "D64"],
}


def make_synthetic_dataset(
    n_patients: int = 1500,
    n_trials: int = 300,
    seed: int = 0,
) -> Dataset:
    """Generate a MIMIC-like cohort with correlated structure.

    Deliberately *not* i.i.d. noise: patients are drawn from comorbidity
    clusters, medications are conditioned on diagnoses, and lab values shift
    with disease burden. Without that structure a graph model has nothing to
    exploit and the comparison would be vacuous.
    """
    rng = np.random.default_rng(seed)
    cluster_names = list(_SYN_DISEASE_CLUSTERS)

    # Expand each 3-char category into leaf codes.
    leaf_codes: Dict[str, List[str]] = {}
    for cl, prefixes in _SYN_DISEASE_CLUSTERS.items():
        codes = []
        for p in prefixes:
            codes += [f"{p}{i}" for i in range(10)]
        leaf_codes[cl] = codes

    all_dx = sorted({c for v in leaf_codes.values() for c in v})
    med_pool = [f"{i:011d}" for i in range(1, 121)]
    lab_items = [str(x) for x in sorted(_LAB_TERMS.values())]

    # Each medication is preferentially associated with one cluster.
    med_cluster = {m: cluster_names[i % len(cluster_names)] for i, m in enumerate(med_pool)}

    patients: List[PatientRecord] = []
    for i in range(n_patients):
        primary = cluster_names[rng.integers(len(cluster_names))]
        secondary = cluster_names[rng.integers(len(cluster_names))]
        burden = float(rng.gamma(shape=2.0, scale=1.5))

        dx: Set[str] = set()
        n_primary = 1 + int(rng.poisson(3))
        dx |= set(rng.choice(leaf_codes[primary], size=min(n_primary, len(leaf_codes[primary])), replace=False))
        if rng.random() < 0.6:
            n_sec = 1 + int(rng.poisson(1.5))
            dx |= set(rng.choice(leaf_codes[secondary], size=min(n_sec, len(leaf_codes[secondary])), replace=False))
        if rng.random() < 0.25:
            dx |= set(rng.choice(all_dx, size=int(rng.poisson(2)) + 1, replace=False))

        rx: Set[str] = set()
        for m in med_pool:
            base = 0.22 if med_cluster[m] in (primary, secondary) else 0.02
            if rng.random() < base:
                rx.add(m)

        labs: Dict[str, float] = {}
        for item in lab_items:
            if rng.random() < 0.75:
                mu = 1.0 + 0.35 * burden * (1 if hash(item) % 2 else -1)
                labs[item] = float(rng.normal(mu, 1.0))

        patients.append(
            PatientRecord(
                patient_id=str(10000 + i),
                diagnosis_codes=dx,
                medication_codes=rx,
                lab_values=labs,
                n_admissions=int(rng.integers(2, 8)),
            )
        )

    # Trials: each targets a cluster, with a few inclusion and exclusion codes.
    trials: List[TrialSpec] = []
    for j in range(n_trials):
        target = cluster_names[rng.integers(len(cluster_names))]
        other = cluster_names[rng.integers(len(cluster_names))]
        # 3-9 inclusion criteria keeps the rule selective: with a mean-based
        # M_inc and tau=0.15, matching one of eight criteria is not enough.
        # That is what produces a realistic low-single-digit prevalence rather
        # than the ~25% you get from one-or-two-criterion toy trials.
        n_inc = int(rng.integers(6, 16))
        n_exc = int(rng.integers(0, 4))

        criteria: List[CriterionSpec] = []
        inc_codes = rng.choice(leaf_codes[target], size=min(n_inc, len(leaf_codes[target])), replace=False)
        for code in inc_codes:
            criteria.append(
                CriterionSpec("diagnosis", str(code), "EXISTS", is_inclusion=True,
                              severity_weight=float(rng.choice([1.0, 1.0, 1.5])),
                              raw_text=f"Diagnosis of {code}")
            )
        if n_exc:
            exc_codes = rng.choice(leaf_codes[other], size=min(n_exc, len(leaf_codes[other])), replace=False)
            for code in exc_codes:
                criteria.append(
                    CriterionSpec("diagnosis", str(code), "EXISTS", is_inclusion=False,
                                  raw_text=f"History of {code}")
                )
        if rng.random() < 0.5:
            item = str(rng.choice(lab_items))
            op = str(rng.choice(["GT", "LT"]))
            criteria.append(
                CriterionSpec("lab", item, op, value=float(rng.normal(0.5, 1.0)),
                              is_inclusion=True, raw_text=f"Lab {item} {op}")
            )
        if rng.random() < 0.4:
            criteria.append(
                CriterionSpec("medication", str(rng.choice(med_pool)), "EXISTS",
                              is_inclusion=bool(rng.random() < 0.5),
                              raw_text="Concomitant medication")
            )
        # A share of criteria are genuinely unresolvable, as in the real parser.
        for _ in range(int(rng.poisson(1.2))):
            criteria.append(
                CriterionSpec("diagnosis", _stable_unk_code("diagnosis", f"{j}-{_}"),
                              "EXISTS", is_inclusion=bool(rng.random() < 0.7),
                              raw_text="Investigator discretion")
            )

        trials.append(
            TrialSpec(
                trial_id=f"NCT9{j:06d}",
                criteria=criteria,
                title=f"A study in {target} disease (cohort {j})",
                conditions=[target],
                phase=str(rng.choice(["PHASE2", "PHASE3"])),
                sample_size=int(rng.integers(50, 800)),
            )
        )

    return Dataset(patients=patients, trials=trials, source=f"synthetic(seed={seed})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_dataset(cfg, mode: str = "auto") -> Dataset:
    """Load the benchmark dataset according to `mode`.

    mode: 'auto' | 'real' | 'real_ctg' | 'synthetic'
    """
    paths = cfg.paths

    def _try_real_patients():
        return load_patients(paths.diagnoses, paths.prescriptions, paths.labs)

    if mode in ("auto", "real"):
        try:
            patients = _try_real_patients()
            trials = load_structured_trials(paths.train_trials, paths.eval_trials)
            if trials:
                return Dataset(patients, trials, source="real(parquet+structured)")
            log.warning("Structured trials missing; falling back to raw CTG parse.")
            mode = "real_ctg"
        except FileNotFoundError as e:
            if mode == "real":
                raise
            log.warning("Real cohort unavailable (%s); using synthetic fallback.", e)
            mode = "synthetic"

    if mode == "real_ctg":
        patients = _try_real_patients()
        cond_map = build_condition_map(patients)
        trials = parse_raw_ctg_trials(paths.raw_ctg_json, condition_to_codes=cond_map)
        return Dataset(patients, trials, source="real(parquet+raw_ctg)")

    ds = make_synthetic_dataset(
        n_patients=cfg.synthetic_n_patients,
        n_trials=cfg.synthetic_n_trials,
        seed=0,
    )
    log.warning(
        "USING SYNTHETIC DATA. Results are for smoke-testing the pipeline only "
        "and must not be reported as findings."
    )
    return ds
