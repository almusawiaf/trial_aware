"""
config.py -- COMPOSE baseline configuration.

This file lives at:  related_works/COMPOSE/config.py
The main pipeline lives at: <repo_root>/models/claude_active/

All paths below are resolved relative to <repo_root>, so this baseline
reads the SAME preprocessed patient tables and the SAME structured trial
JSON that models/claude_active/train.py and evaluate.py use. That is
intentional: it is the only way a COMPOSE-vs-our-model comparison means
anything. Do not point this at a separate copy of the data.
"""
import os
import torch

# related_works/COMPOSE/config.py -> repo root is two levels up
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))


class ComposeConfig:
    # ------------------------------------------------------------------
    # Data / IO -- all shared with the main pipeline, do not duplicate
    # ------------------------------------------------------------------
    MAIN_DATA_DIR = os.path.join(REPO_ROOT, "data")          # diagnoses_clean.parquet etc.
    TRIALS_DATA_DIR = os.path.join(REPO_ROOT, "data", "10000_trials")
    TRAIN_TRIALS_PATH = os.path.join(TRIALS_DATA_DIR, "structured_clinical_trials.json")

    DIAG_PATH = os.path.join(MAIN_DATA_DIR, "diagnoses_clean.parquet")
    RX_PATH = os.path.join(MAIN_DATA_DIR, "prescriptions_clean.parquet")
    LABS_PATH = os.path.join(MAIN_DATA_DIR, "labs_clean.parquet")

    # This baseline's own outputs -- never written into the main data/ dir,
    # so it can never silently overwrite anything the main pipeline needs.
    OUT_DIR = os.path.join(_THIS_DIR, "results")
    CKPT_DIR = os.path.join(_THIS_DIR, "checkpoints")
    LOG_DIR = os.path.join(_THIS_DIR, "logs")
    VOCAB_PATH = os.path.join(OUT_DIR, "vocab.pt")

    # ------------------------------------------------------------------
    # Reproducibility -- same env-var pattern as the main config.py, so
    # run_multi_seed-style sweeps work the same way for this baseline.
    # ------------------------------------------------------------------
    SEED = int(os.environ.get("RUN_SEED", 42))

    # ------------------------------------------------------------------
    # Vocabulary / embedding sizes
    # ------------------------------------------------------------------
    # NOTE ON DEVIATION FROM ORIGINAL COMPOSE:
    # The original COMPOSE encodes trial-criteria text and 4-level
    # hierarchical code descriptions with pretrained clinical BERT
    # (word_dim=768). Our processed data has neither free-text criteria
    # nor 4-level text descriptions for diagnosis/medication/lab codes --
    # everything is already reduced to discrete codes (ICD-10, NDC,
    # ITEMID) by the shared data pipeline. We therefore replace BERT
    # embeddings with END-TO-END TRAINED CODE EMBEDDINGS (nn.Embedding),
    # and keep every other architectural piece of COMPOSE (highway-conv
    # text encoder, EHR memory network, attention query network,
    # 3-way classification + cosine-embedding alignment loss) unchanged.
    # This is the standard adaptation used whenever COMPOSE is reproduced
    # on a dataset without literal free-text criteria.
    CODE_EMBED_DIM = 128        # replaces BERT's word_dim=768
    CONV_DIM = 64                # per-branch conv channels (orig used 128)
    MEM_DIM = 128                # memory network hidden size
    MLP_DIM = 128                # query network MLP hidden size
    DEMO_DIM = 3                 # [age_norm, gender_M, gender_F]

    # NOTE ON DEVIATION: original COMPOSE uses a fixed 12 memory slots
    # (3 code categories x 4 hierarchy levels). We have 3 categories
    # (diagnosis, medication, lab) and no separate hierarchy levels, so
    # the memory has 3 slots (+1 demographic slot, exactly as original).
    NUM_CODE_CATEGORIES = 3       # diagnosis, medication, lab
    MAX_VISITS = 20                # truncate/pad patient visit history
    MAX_CODES_PER_CATEGORY = 32    # truncate/pad codes within a visit/category
    MAX_CRITERION_TOKENS = 4       # [entity_type, code, operator, value] per criterion

    # ------------------------------------------------------------------
    # Training data construction
    # ------------------------------------------------------------------
    # Ground-truth per-criterion labels for training come directly from
    # the SAME rule-based scorer the main pipeline uses for evaluation
    # (trial_graph.compute_matching_indices's underlying _match_single).
    # A criterion is a MATCH if its match score >= MATCH_LABEL_THRESHOLD,
    # a MISMATCH if its score is very low, and UNKNOWN(2) otherwise --
    # mirroring COMPOSE's own semantics, where "unknown" means "we cannot
    # confidently assign a label from the available signal".
    MATCH_LABEL_THRESHOLD = 0.7
    MISMATCH_LABEL_THRESHOLD = 0.3
    # Fraction of (patient, trial, criterion) triples to additionally
    # relabel as UNKNOWN(2) at random, matching the original paper's
    # "randomly sample a criterion as unknown for each known criterion".
    UNKNOWN_RESAMPLE_RATIO = 1.0

    TRAIN_FRAC = 0.7
    VAL_FRAC = 0.15
    # remaining ~0.15 is TEST -- held out until final evaluation only.

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30
    SIMILARITY_LOSS_WEIGHT = 0.3   # weight on the CosineEmbeddingLoss term
    GRAD_CLIP = 5.0
    EARLY_STOP_PATIENCE = 5

    # ------------------------------------------------------------------
    # Evaluation -- MUST mirror models/claude_active/evaluate/compose_based
    # so the reported numbers are directly comparable.
    # ------------------------------------------------------------------
    ETA_EXCLUSION_PENALTY = 1.0
    STRICT_MATCH_THRESHOLD = 0.5
    N_BOOTSTRAP = 1000

    @property
    def DEVICE(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ensure_directories(self):
        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.CKPT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        return self

    @property
    def CKPT_PATH(self):
        return os.path.join(self.CKPT_DIR, f"compose_seed{self.SEED}.pt")

    @property
    def RESULTS_PATH(self):
        return os.path.join(self.OUT_DIR, f"compose_evaluation_results_seed{self.SEED}.json")

    @property
    def SCORES_PATH(self):
        return os.path.join(self.OUT_DIR, f"compose_scores_seed{self.SEED}.npz")


config = ComposeConfig()
