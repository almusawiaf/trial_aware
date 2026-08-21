"""
config.py
=========
Single source of truth for the baseline benchmark.

Design notes
------------
The upstream Trial-Aware GCL repo derives its ground-truth eligibility label
from a *deterministic rule* over (patient state, trial criteria):

    y = 1  iff  M_inc >= HARD_NEG_INC_THRESHOLD  and  M_exc < HARD_NEG_EXC_THRESHOLD

Because the label is a closed-form function of the inputs, a supervised model
that is handed the matched-criterion statistics can reproduce it exactly. That
is *label leakage*, not learning. This benchmark therefore runs every model
under two explicitly separated feature regimes (see `features.py`):

    FeatureRegime.ORACLE   -- overlap statistics exposed. Diagnostic only.
                              Expected AUC ~ 1.0. Reported as a leakage audit.
    FeatureRegime.HONEST   -- overlap statistics withheld. Models see patient
                              and trial representations separately and must
                              learn the interaction from training pairs.
                              This is the regime results should be quoted from.

Every model shares the same splits, the same features, the same metric code and
the same tuning budget so the comparison is apples to apples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
@dataclass
class Paths:
    """Where inputs live and where outputs go.

    `mimic_processed_dir` should point at the OUTPUT_DIR of the upstream
    pipeline (`data_pipeline/run.py`), i.e. the directory holding
    diagnoses_clean.parquet / prescriptions_clean.parquet / labs_clean.parquet.
    If those files are absent the adapter falls back to a synthetic cohort so
    the code is runnable without MIMIC-III credentials.
    """

    mimic_processed_dir: str = os.environ.get("TA_MIMIC_DIR", "./data")
    trials_dir: str = os.environ.get("TA_TRIALS_DIR", "./data/10000_trials")
    raw_ctg_json: str = os.environ.get(
        "TA_RAW_CTG", "./extracting_trials/ctg-studies_1000.json"
    )
    gcl_embeddings_dir: str = os.environ.get("TA_GCL_DIR", "./data")
    output_dir: str = os.environ.get("TA_OUT", "./results")
    cache_dir: str = os.environ.get("TA_CACHE", "./cache")

    @property
    def diagnoses(self) -> str:
        return os.path.join(self.mimic_processed_dir, "diagnoses_clean.parquet")

    @property
    def prescriptions(self) -> str:
        return os.path.join(self.mimic_processed_dir, "prescriptions_clean.parquet")

    @property
    def labs(self) -> str:
        return os.path.join(self.mimic_processed_dir, "labs_clean.parquet")

    @property
    def train_trials(self) -> str:
        return os.path.join(self.trials_dir, "structured_clinical_trials.json")

    @property
    def eval_trials(self) -> str:
        return os.path.join(self.trials_dir, "structured_clinical_trials_eval.json")

    def ensure(self) -> "Paths":
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        return self


# ----------------------------------------------------------------------------
# Labelling rule -- kept numerically identical to the upstream repo
# ----------------------------------------------------------------------------
@dataclass
class LabelConfig:
    """Mirrors models/claude_active/config.py so labels are comparable.

    `inc_threshold` / `exc_threshold` are the upstream HARD_NEG_INC_THRESHOLD
    and HARD_NEG_EXC_THRESHOLD. `sigmoid_delta` matches trial_graph.py.
    """

    inc_threshold: float = 0.15
    exc_threshold: float = 0.80
    sigmoid_delta: float = 1.0
    # Upstream treats a trial with zero inclusion criteria as M_inc = 1.0.
    # That makes such trials universally "eligible" and inflates prevalence,
    # so we drop them by default and log how many were removed.
    drop_trials_without_inclusion: bool = True
    # Trials whose criteria never resolved to a real code carry no signal.
    drop_unmapped_only_trials: bool = True
    min_criteria_per_trial: int = 1


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
@dataclass
class SplitConfig:
    """Grouped, doubly-disjoint splits.

    Patients and trials are partitioned independently. The primary test set is
    the *both-cold* quadrant (unseen patients x unseen trials), which is the
    only quadrant that measures genuine generalisation for a matching model.
    The two single-cold quadrants are reported as diagnostics.
    """

    patient_train: float = 0.70
    patient_val: float = 0.15
    patient_test: float = 0.15

    trial_train: float = 0.70
    trial_val: float = 0.15
    trial_test: float = 0.15

    # Training-set negative subsampling. Positives are always kept in full.
    # Evaluation is ALWAYS on the complete dense matrix so PR-AUC reflects the
    # true operating prevalence.
    neg_per_pos_train: int = 20
    max_train_pairs: int = 2_000_000

    # Stratify the trial split by positive rate so that rare, highly selective
    # trials are not all dumped into one fold.
    stratify_trials_by_prevalence: bool = True
    n_prevalence_bins: int = 4


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    top_k_diagnoses: int = 512
    top_k_medications: int = 256
    top_k_labs: int = 64
    top_k_criterion_codes: int = 512
    # SVD compression of the sparse multi-hot blocks. Set to 0 to disable and
    # keep raw sparse blocks (tree models cope; the MLP prefers the dense form).
    svd_components_patient: int = 128
    svd_components_trial: int = 64
    use_text_svd: bool = True
    text_svd_components: int = 32
    min_df: int = 2


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
@dataclass
class MLPConfig:
    hidden: Tuple[int, ...] = (256, 128)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 8
    pos_weight_cap: float = 50.0


@dataclass
class TwoTowerConfig:
    tower_hidden: Tuple[int, ...] = (256, 128)
    embed_dim: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 8


@dataclass
class GNNConfig:
    conv: str = "sage"          # 'gcn' | 'sage' | 'gat' | 'hetero_sage'
    hidden_dim: int = 128
    out_dim: int = 64
    num_layers: int = 2
    heads: int = 4              # GAT only
    dropout: float = 0.3
    lr: float = 5e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 20
    decoder: str = "mlp"        # 'dot' | 'mlp'
    batch_size: int = 8192
    # Patient<->trial edges are NEVER inserted into the message-passing graph.
    # Trials attach only to their criterion-code nodes; patients only to their
    # own clinical-code nodes. This keeps the encoder inductive and makes
    # supervision-edge leakage structurally impossible.
    forbid_patient_trial_edges: bool = True


@dataclass
class XGBConfig:
    n_estimators: int = 2000
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    early_stopping_rounds: int = 50
    tree_method: str = "hist"


@dataclass
class RFConfig:
    n_estimators: int = 400
    max_depth: Optional[int] = 20
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    class_weight: str = "balanced_subsample"


@dataclass
class LogRegConfig:
    C: float = 1.0
    penalty: str = "l2"
    max_iter: int = 2000
    class_weight: str = "balanced"


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@dataclass
class EvalConfig:
    ks: Tuple[int, ...] = (10, 50, 100)
    n_bootstrap: int = 1000
    bootstrap_unit: str = "trial"   # resample trials, not pairs (pairs are dependent)
    ci_alpha: float = 0.05
    primary_metric: str = "pr_auc"
    compute_calibration: bool = True


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    paths: Paths = field(default_factory=Paths)
    label: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    mlp: MLPConfig = field(default_factory=MLPConfig)
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    logreg: LogRegConfig = field(default_factory=LogRegConfig)

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    n_tuning_trials: int = 20      # identical randomised-search budget for all models
    device: str = os.environ.get("TA_DEVICE", "auto")
    n_jobs: int = int(os.environ.get("TA_NJOBS", "-1"))
    verbose: bool = True

    # Synthetic-cohort fallback (smoke tests only -- never quote these numbers)
    synthetic_n_patients: int = 1500
    synthetic_n_trials: int = 300

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def to_dict(self) -> Dict:
        return asdict(self)


DEFAULT = BenchmarkConfig()
