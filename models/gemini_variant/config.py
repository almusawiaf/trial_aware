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
    ALIGN_LR = 1e-4
    LAMBDA_1 = 1.0
    LAMBDA_2 = 2.5
    LAMBDA_ANCHOR = 0.5
    MARGIN_HARD = 0.4
    MARGIN_RAND = 0.2
    HARD_NEG_INC_THRESHOLD = 0.05
    HARD_NEG_EXC_THRESHOLD = 0.8
    N_RANDOM_NEGATIVES = 5
    ELIGIBILITY_THRESHOLD_GAMMA = 0.8

    # ------------------------------------------------------------------
    # Inference-time scoring (Phase 5)
    # ------------------------------------------------------------------
    ETA_EXCLUSION_PENALTY = 1.0

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------
    @property
    def GRAPH_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "hetero_graph.pt")

    @property
    def PATIENT_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "patient_embeddings.pt")

    @property
    def TRIAL_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "trial_embeddings.pt")
    
    @property
    def BASELINE_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "patient_embeddings_baseline.pt")
    
    @property
    def LOSS_HISTORY_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "training_loss_history.csv")

    @property
    def TRAIN_TRIALS_PATH(self):
        """Path to training trials JSON."""
        return os.path.join(self.TRIALS_DATA_DIR, "structured_clinical_trials.json")
    
    @property
    def EVAL_TRIALS_PATH(self):
        """Path to evaluation trials JSON."""
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