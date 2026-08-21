# config.py
import os
import torch


class Config:
    """Hyperparameters and file paths for the full pipeline (Phases 1-5)."""

    # ------------------------------------------------------------------
    # Data / IO
    # ------------------------------------------------------------------
    DATA_DIR = "/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources"
    OUTPUT_DIR = "./data"
    
    # NEW: Location for the 1000 trials data
    TRIALS_DATA_DIR = "./data/10000_trials/"
    
    TRIALS_PATH = "./trial_criteria.json"  # structured trial-criteria file (see trial_graph.py)
    
    ICD9_TO_ICD10_CSV = os.path.join(DATA_DIR, "icd9toicd10cmgem.csv")  
    
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    # NEW: without this, every run used whatever random state PyTorch/numpy
    # happened to start in -- there was no way to tell if a change in AUC
    # was real or just run-to-run noise. Set from the RUN_SEED environment
    # variable so a multi-seed sweep can override it per-run without editing
    # this file each time (see run_multi_seed.py).
    SEED = int(os.environ.get("RUN_SEED", 42))

    # NEW: cross-validation fold selector, mirrors models/claude_active/config.py
    FOLD = os.environ.get("RUN_FOLD", None)
    FOLD = int(FOLD) if FOLD is not None else None

    # ------------------------------------------------------------------
    # Cohort Selection Rules (Phase 1)
    # ------------------------------------------------------------------
    MIN_ENCOUNTERS = 2
    MIN_TEMPORAL_SPACING_DAYS = 30

    # ------------------------------------------------------------------
    # Statistical Cleaning (Phase 1)
    # ------------------------------------------------------------------
    OUTLIER_SIGMA_THRESHOLD = 4.0
    MIN_LAB_FREQ_THRESHOLD = 0.05
    MAX_DAYS_SINCE_MEASURED = 30

    # Temporal decay parameter (rho)
    RHO = 0.1

    # ------------------------------------------------------------------
    # Graph construction (Phase 2)
    # ------------------------------------------------------------------
    PATIENT_FEAT_DIM = 16
    ENTITY_EMBED_DIM = 32
    COMORBIDITY_MIN_SHARED_DX = 2
    COMORBIDITY_MAX_GROUP_SIZE = 500

    # ------------------------------------------------------------------
    # Encoder (Phase 3)
    # ------------------------------------------------------------------
    HIDDEN_CHANNELS = 16
    OUT_CHANNELS = 32

    # ------------------------------------------------------------------
    # Contrastive pretraining (Phase 3)
    # ------------------------------------------------------------------
    TEMPERATURE = 0.1
    GCL_LR = 1e-3
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS_GCL = 100
    EPOCHS_CONTRASTIVE = 100
    DROP_RATE_V1 = 0.15
    DROP_RATE_V2 = 0.20
    BATCH_SIZE = 256

    # ------------------------------------------------------------------
    # Trial-aware alignment (Phase 4)
    # ------------------------------------------------------------------
    EPOCHS_ALIGN = 60
    # NEW: overridable via env vars, same pattern as SEED. This lets
    # run_anchor_sweep.py test several values without editing this file
    # for every combination.
    ALIGN_LR = float(os.environ.get("RUN_ALIGN_LR", 1e-4))
    LAMBDA_1 = 1.0
    LAMBDA_2 = 2.5
    LAMBDA_ANCHOR = float(os.environ.get("RUN_LAMBDA_ANCHOR", 0.5))
    MARGIN_HARD = 0.4
    MARGIN_RAND = 0.2
    # Reverted from 0.3 -- that cut usable trials (>=1 eligible patient) from
    # 193 down to 59 without fixing the collapse, so it wasn't the real cause.
    # Splitting the difference from the original 0.05 while we investigate
    # concept coverage as the more likely root cause.
    HARD_NEG_INC_THRESHOLD = 0.15
    HARD_NEG_EXC_THRESHOLD = 0.8
    N_RANDOM_NEGATIVES = 5
    ELIGIBILITY_THRESHOLD_GAMMA = 0.8

    # ------------------------------------------------------------------
    # Inference-time scoring (Phase 5)
    # ------------------------------------------------------------------
    ETA_EXCLUSION_PENALTY = 1.0

    # ------------------------------------------------------------------
    # Strict/COMPOSE-style trial-level matching (evaluate.py)
    # ------------------------------------------------------------------
    # Per-criterion score >= this counts as "matched" when computing the
    # strict all-or-nothing trial-level match (see
    # matching_engine.compute_strict_trial_match). This is a real
    # hyperparameter -- if you tune it, do so on a held-out validation
    # split, not on the data you report final numbers on.
    STRICT_MATCH_THRESHOLD = 0.5

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------
    @property
    def _tag(self):
        """Suffix used on every seed-specific artifact path. Includes the
        fold number when RUN_FOLD is set, so different folds never
        overwrite each other's checkpoints."""
        if self.FOLD is not None:
            return f"seed{self.SEED}_fold{self.FOLD}"
        return f"seed{self.SEED}"

    @property
    def GRAPH_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "hetero_graph.pt")  # graph itself doesn't depend on seed

    @property
    def PATIENT_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, f"patient_embeddings_{self._tag}.pt")

    @property
    def TRIAL_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, f"trial_embeddings_{self._tag}.pt")
    
    @property
    def BASELINE_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, f"patient_embeddings_baseline_{self._tag}.pt")

    @property
    def TRIAL_EMBED_BASELINE_PATH(self):
        """Pre-Stage-B trial embeddings, saved so Stage A vs Stage B is a fair comparison."""
        return os.path.join(self.OUTPUT_DIR, f"trial_embeddings_baseline_{self._tag}.pt")

    # NEW: same four artifacts added to models/claude_active/config.py --
    # needed to re-encode held-out trials at eval time. See the comment
    # there for the full explanation.
    @property
    def PRE_ALIGN_POST_GNN_PATH(self):
        return os.path.join(self.OUTPUT_DIR, f"pre_align_post_gnn_{self._tag}.pt")

    @property
    def POST_ALIGN_POST_GNN_PATH(self):
        return os.path.join(self.OUTPUT_DIR, f"post_align_post_gnn_{self._tag}.pt")

    @property
    def CRITERION_ENCODER_STATE_PATH(self):
        return os.path.join(self.OUTPUT_DIR, f"criterion_encoder_{self._tag}.pt")

    @property
    def TRIAL_ENCODER_STATE_PATH(self):
        return os.path.join(self.OUTPUT_DIR, f"trial_encoder_{self._tag}.pt")

    @property
    def LOSS_HISTORY_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "training_loss_history.csv")

    @property
    def TRAIN_TRIALS_PATH(self):
        """Path to training trials JSON. Fold-aware, see models/claude_active/config.py."""
        if self.FOLD is not None:
            return os.path.join(self.TRIALS_DATA_DIR, "folds", f"fold{self.FOLD}_train.json")
        return os.path.join(self.TRIALS_DATA_DIR, "structured_clinical_trials.json")
    
    @property
    def EVAL_TRIALS_PATH(self):
        """Path to evaluation trials JSON. Fold-aware, see models/claude_active/config.py."""
        if self.FOLD is not None:
            return os.path.join(self.TRIALS_DATA_DIR, "folds", f"fold{self.FOLD}_eval.json")
        return os.path.join(self.TRIALS_DATA_DIR, "structured_clinical_trials_eval.json")

    # ------------------------------------------------------------------
    # Device configuration
    # ------------------------------------------------------------------
    @property
    def DEVICE(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def ensure_directories(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.TRIALS_DATA_DIR, exist_ok=True)
        return self
    
    def to_dict(self):
        return {
            'DATA_DIR': self.DATA_DIR,
            'OUTPUT_DIR': self.OUTPUT_DIR,
            'TRIALS_DATA_DIR': self.TRIALS_DATA_DIR,
            'OUT_CHANNELS': self.OUT_CHANNELS,
            'HIDDEN_CHANNELS': self.HIDDEN_CHANNELS,
            'EPOCHS_ALIGN': self.EPOCHS_ALIGN,
            'EPOCHS_GCL': self.EPOCHS_GCL,
            'LAMBDA_ANCHOR': self.LAMBDA_ANCHOR,
            'LAMBDA_1': self.LAMBDA_1,
            'LAMBDA_2': self.LAMBDA_2,
            'MARGIN_HARD': self.MARGIN_HARD,
            'MARGIN_RAND': self.MARGIN_RAND,
            'TEMPERATURE': self.TEMPERATURE,
            'DEVICE': str(self.DEVICE),
        }


config = Config()