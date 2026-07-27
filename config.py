# config.py
import os


class Config:
    """Hyperparameters and file paths for the full pipeline (Phases 1-5)."""

    # ------------------------------------------------------------------
    # Data / IO
    # ------------------------------------------------------------------
    DATA_DIR = "/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources"
    OUTPUT_DIR = "./processed_data"
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
    MIN_LAB_FREQ_THRESHOLD = 0.05  # Lab must be present in >= 5% of patients
    MAX_DAYS_SINCE_MEASURED = 30   # cap on daily-grid expansion / decay window (Fix #10)

    # Temporal decay parameter (rho) -- renamed from GAMMA to avoid symbol collision
    # with the exclusion-attention coefficient alpha_c and eligibility threshold
    # gamma_elig used later in the trial-alignment stage (see review notes).
    RHO = 0.1  # Decay rate per day (half-life ~= ln(2)/0.1 ~= 7 days)

    # ------------------------------------------------------------------
    # Graph construction (Phase 2)
    # ------------------------------------------------------------------
    PATIENT_FEAT_DIM = 16          # dim of the (currently trainable-random) patient input embedding
    ENTITY_EMBED_DIM = 32          # shared embedding dim for diagnosis / medication / lab nodes
    COMORBIDITY_MIN_SHARED_DX = 2  # min shared diagnoses to draw a patient-patient Comorbidity edge
    COMORBIDITY_MAX_GROUP_SIZE = 500  # cap per-diagnosis patient group to avoid O(n^2) blowup

    # ------------------------------------------------------------------
    # Encoder (Phase 3)
    # ------------------------------------------------------------------
    HIDDEN_CHANNELS = 64
    OUT_CHANNELS = 32              # this is the shared d used by both patient and trial embeddings

    # ------------------------------------------------------------------
    # Contrastive pretraining (Phase 3)
    # ------------------------------------------------------------------
    TEMPERATURE = 0.1
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS_CONTRASTIVE = 100
    DROP_RATE_V1 = 0.15
    DROP_RATE_V2 = 0.20
    BATCH_SIZE = 256

    # ------------------------------------------------------------------
    # Trial-aware alignment (Phase 4 -- new, mirrors the LaTeX methodology)
    # ------------------------------------------------------------------
    EPOCHS_ALIGN = 5
    ALIGN_LR = 1e-4
    LAMBDA_1 = 1.0     # weight on the positive-attraction + hard-negative-repulsion terms
    LAMBDA_2 = 2.5     # weight on the random-negative-repulsion term
    LAMBDA_ANCHOR: float = 0.5     # Weight for anchor regularization relative to Stage A
    MARGIN_HARD = 0.4  # m_hard > m_rand, per the LaTeX design rationale (hard negatives need a
    MARGIN_RAND = 0.2  # bigger safety margin since they are "almost positives" that got excluded)
    HARD_NEG_INC_THRESHOLD = 0.5   # M_inc >= this AND
    HARD_NEG_EXC_THRESHOLD = 0.8   # M_exc >= this  => trial counts as a hard negative for patient
    N_RANDOM_NEGATIVES = 5
    ELIGIBILITY_THRESHOLD_GAMMA = 0.8  # gamma_elig: used only for held-out P@k evaluation,
                                        # NEVER for generating training positives (avoids the
                                        # train/eval circularity flagged in the methodology review)

    # ------------------------------------------------------------------
    # Inference-time scoring (Phase 5)
    # ------------------------------------------------------------------
    ETA_EXCLUSION_PENALTY = 1.0  # eta in Similarity(P_i, T_j) = cos(z_Pi, z_inc) - eta * cos(z_Pi, z_exc)

    @property
    def GRAPH_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "hetero_graph.pt")

    @property
    def PATIENT_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "patient_embeddings.pt")

    @property
    def TRIAL_EMBED_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "trial_embeddings.pt")
