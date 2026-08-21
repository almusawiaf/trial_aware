# COMPOSE Baseline (related_works/COMPOSE/)

A trainable re-implementation of **COMPOSE** (Gao, Xiao, Glass & Sun,
*"COMPOSE: Cross-Modal Pseudo-Siamese Network for Patient Trial
Matching,"* KDD 2020), adapted to run on this project's processed data
and evaluated with the exact same protocol as Stage A / Stage B, so the
three models can be compared on one table.

Official COMPOSE code: https://github.com/v1xerunt/COMPOSE

---

## 1. Why this isn't a line-for-line copy of the original code

The original COMPOSE expects:
* free-text eligibility criteria, encoded with pretrained **clinical BERT**;
* a 4-level hierarchical text description (e.g. UNSPSC-style taxonomy) for
  every diagnosis/procedure/product code, also BERT-encoded;
* 12 fixed EHR memory slots = 3 code categories (diagnosis, procedure,
  product) × 4 hierarchy levels.

This project's shared data pipeline (`models/claude_active/`) has none of
that: trials are already parsed into **structured, code-level criteria**
(`entity_type`, `entity_code`, `operator`, `value`), and patient data is
**coded** (ICD-10, NDC, ITEMID), with no free text and no 4-level
hierarchy descriptions.

So this baseline keeps every piece of COMPOSE's *architecture* that does
not depend on free text (`ECEmbedding`'s highway-conv text encoder, the
`EHRMemoryNetwork` erase/add memory bank, the attention-based
`QueryNetwork`, the 3-way classification + cosine-embedding alignment
loss), and replaces only the *input representation*:

| Original COMPOSE | This adaptation |
|---|---|
| BERT embeddings of free-text criteria | Trained `nn.Embedding` lookups of criterion tokens (entity_type, code, operator, value) |
| BERT embeddings of 4-level code hierarchy text | Trained `nn.Embedding` lookups of raw codes, masked mean-pooled per category |
| 12 memory slots (3 categories × 4 hierarchy levels) | 3 memory slots (diagnosis, medication, lab -- this project has no procedure/product tables) |
| Manually-labeled match/mismatch/unknown criteria | Labels derived automatically from `trial_graph._match_single`, the SAME rule-based scorer the main pipeline itself uses |

Every deviation is called out with a `NOTE ON DEVIATION` comment at the
point it occurs in `config.py`, `data_utils.py`, and `model.py` -- read
those before citing numbers from this baseline in the paper.

---

## 2. Directory contents

```
related_works/COMPOSE/
├── config.py            # paths + hyperparameters (points into ../../data)
├── vocab.py              # builds code / entity-type / operator vocabularies
├── data_utils.py          # loads shared data, builds visit sequences + labels + ground truth
├── dataset.py              # PyTorch Dataset / collate_fn
├── model.py                # COMPOSE architecture (documented adaptation)
├── train.py                 # trains the model
├── evaluate.py               # evaluates + prints the comparison table
├── compare_models.py          # OPTIONAL: rigorous 3-way bootstrap (see step 6)
├── checkpoints/                 # trained weights land here
├── results/                      # metrics + score matrices land here
└── logs/                           # training/eval logs land here
```

---

## 3. Prerequisites

This baseline reuses the **same processed data** as the main pipeline.
Do not create a separate copy of it. Before running anything here, make
sure you have already run the main pipeline far enough to produce:

```
<repo_root>/data/diagnoses_clean.parquet
<repo_root>/data/prescriptions_clean.parquet
<repo_root>/data/labs_clean.parquet
<repo_root>/data/10000_trials/structured_clinical_trials.json
```

These are produced by `models/claude_active`'s own pipeline (`run.py`
and the trial-structuring step under `trial_graph.py` /
`data_pipeline/generate_trial_json.py`). If you've already trained
Stage A / Stage B, these files already exist -- you don't need to
regenerate them.

Python dependencies (same environment as the main pipeline; nothing new
beyond what it already needs):
```bash
conda activate trial_aware
pip install torch pandas numpy scikit-learn pyarrow tqdm
```

---

## 4. Step-by-step: train and evaluate COMPOSE

All commands are run from `related_works/COMPOSE/`.

### Step 1 -- sanity-check paths
```bash
cd related_works/COMPOSE
python -c "from config import config; print(config.DIAG_PATH); print(config.TRAIN_TRIALS_PATH)"
```
Both printed paths must exist on disk. If they don't, fix
`MAIN_DATA_DIR` / `TRIALS_DATA_DIR` at the top of `config.py` (e.g. if
your data lives somewhere other than `<repo_root>/data`).

### Step 2 -- train
```bash
RUN_SEED=42 python train.py
```
This will:
1. Load the shared patient tables and trial store.
2. Build code/entity-type/operator vocabularies and save them to
   `results/vocab.pt`.
3. Build per-patient visit sequences (Section 1's "3 slots" adaptation).
4. Split patients 70/15/15 into train/val/test (`results/patient_split_seed42.txt`).
5. Sample weak-positive/weak-negative trials per patient and expand every
   sampled trial's criteria into training triples (see
   `data_utils.build_training_triples`).
6. Train for up to `config.EPOCHS` epochs with early stopping on
   validation loss, saving the best checkpoint to
   `checkpoints/compose_seed42.pt`.

Expect a log like:
```
[Epoch 001] train_loss=1.10 train_acc=0.41 | val_loss=1.05 val_acc=0.44
[Epoch 002] train_loss=0.95 train_acc=0.52 | val_loss=0.98 val_acc=0.49
...
  -> saved new best checkpoint to checkpoints/compose_seed42.pt
```

Runtime and final accuracy depend entirely on your MIMIC-III cohort size
and the number of structured trials -- there are no numbers to
"reproduce" here since this is a new adaptation, not the original
paper's exact benchmark.

### Step 3 -- evaluate
```bash
RUN_SEED=42 python evaluate.py
```
This will:
1. Load the trained checkpoint and vocabularies.
2. Rebuild the ground-truth `y_true` / `y_true_strict` matrices using the
   exact same functions (`compute_matching_indices`,
   `compute_strict_trial_match`) that
   `models/claude_active/evaluate/compose_based/evaluate.py` uses for
   Stage A / Stage B -- same thresholds, same strict-match rule.
3. Encode every patient's memory once, then score every trial against
   every patient in a batched fashion.
4. Report ROC-AUC / PR-AUC on:
   * the **full population** (directly comparable to how Stage A/B are
     reported), and
   * the **held-out test patients only** (the fair, no-leakage number,
     since COMPOSE -- unlike Stage A/B's self-supervised contrastive
     objective -- is trained with supervision).
5. If `models/claude_active`'s own `evaluate.py` has already been run
   with the same `RUN_SEED` (so
   `data/evaluation_results_seed{SEED}.json` exists), print a 3-row
   comparison table:

```
======================================================================
COMPARISON TABLE (ROC-AUC / PR-AUC)
======================================================================
Model                            ROC-AUC      PR-AUC
Stage A (our, no align)           0.4709      0.0190
Stage B (our, full model)         0.5533      0.0210
COMPOSE (full population)         0.XXXX      0.XXXX
COMPOSE (test-only)               0.XXXX      0.XXXX
======================================================================
```

Results are saved to `results/compose_evaluation_results_seed42.json`
and the raw score matrices to `results/compose_scores_seed42.npz`.

### Step 4 -- run this for every seed you use for Stage A/B
For a fair comparison, use the **same seeds** you used for the main
pipeline's `run_multi_seed.py` sweep:
```bash
for s in 0 1 2 3 4; do
    RUN_SEED=$s python train.py
    RUN_SEED=$s python evaluate.py
done
```
Then average `compose_roc_auc_full_population` /
`compose_roc_auc_test_only` across
`results/compose_evaluation_results_seed*.json`, exactly how you'd
already be averaging Stage A/B's numbers across seeds.

### Step 5 -- collect the final numbers
Everything you need for the paper is in
`results/compose_evaluation_results_seed{SEED}.json`:
```json
{
  "compose_roc_auc_full_population": ...,
  "compose_pr_auc_full_population": ...,
  "compose_roc_auc_test_only": ...,
  "compose_pr_auc_test_only": ...,
  "compose_strict_acc_full": ...,
  "compose_strict_acc_test_only": ...,
  "stage_a_roc_auc": ...,
  "stage_b_roc_auc": ...,
  "stage_a_pr_auc": ...,
  "stage_b_pr_auc": ...,
  "num_patients": ...,
  "num_test_patients": ...,
  "num_trials": ...,
  "num_positive_pairs": ...,
  "seed": 42
}
```
Report **`compose_roc_auc_test_only`** / **`compose_pr_auc_test_only`**
as the headline COMPOSE baseline numbers in the paper -- that's the one
computed without any of COMPOSE's own training data in the evaluation
set. Report the `full_population` numbers only as a secondary,
explicitly-labeled comparison to Stage A/B (which don't have this
leakage concern, since they're trained self-supervised).

### Step 6 (optional) -- statistically rigorous 3-way comparison
`evaluate.py`'s comparison table uses point estimates only. If you want
bootstrap 95% CIs and p-values across all three models at once (COMPOSE
vs Stage A vs Stage B), add these two lines to
`models/claude_active/evaluate/compose_based/evaluate.py`, right before
its `return y_true, scores_baseline, scores_full, y_true_strict`
statement inside `build_evaluation_matrices()`:
```python
np.savez(os.path.join(cfg.OUTPUT_DIR, f"raw_score_matrices_seed{cfg.SEED}.npz"),
         y_true=y_true, scores_baseline=scores_baseline, scores_full=scores_full)
```
Then:
```bash
RUN_SEED=42 python models/claude_active/evaluate/compose_based/evaluate.py   # from repo root
cd related_works/COMPOSE
RUN_SEED=42 python compare_models.py
```
This writes `results/three_way_comparison_seed42.json` with 95% CIs and
p-values for Stage B vs COMPOSE on both ROC-AUC and PR-AUC.

---

## 5. Key hyperparameters (in `config.py`)

| Parameter | Default | Meaning |
|---|---:|---|
| `CODE_EMBED_DIM` | 128 | Replaces BERT's 768-dim word embeddings |
| `CONV_DIM` | 64 | Per-branch channels in the highway-conv criterion encoder |
| `MEM_DIM` | 128 | Memory network hidden size |
| `MAX_VISITS` | 20 | Max admissions (HADM_ID) kept per patient |
| `MAX_CODES_PER_CATEGORY` | 32 | Max codes per category per visit |
| `MATCH_LABEL_THRESHOLD` | 0.7 | Per-criterion score ≥ this → training label "match" |
| `MISMATCH_LABEL_THRESHOLD` | 0.3 | Per-criterion score ≤ this → training label "mismatch" |
| `TRAIN_FRAC` / `VAL_FRAC` | 0.7 / 0.15 | Patient-level split (remaining ~0.15 is test) |
| `EPOCHS` / `EARLY_STOP_PATIENCE` | 30 / 5 | Training budget |
| `ETA_EXCLUSION_PENALTY` | 1.0 | Same η as Stage A/B's `score = inc_sim - η·exc_sim` |
| `STRICT_MATCH_THRESHOLD` | 0.5 | Same threshold as `compute_strict_trial_match` in the main pipeline |

---

## 6. Known limitations (be upfront about these in the paper)

1. **No procedure/product codes.** The shared pipeline only has
   diagnosis, medication, and lab tables, so this adaptation uses 3
   memory slots instead of COMPOSE's original 12.
2. **No demographics.** Age/gender are not present in
   `diagnoses_clean.parquet` / `prescriptions_clean.parquet` /
   `labs_clean.parquet`, so the demographic vector is all-zero for every
   patient. This slightly understates what the original COMPOSE
   architecture is capable of.
3. **Labs are patient-level, not visit-level**, because
   `labs_clean.parquet` stores a temporally-decayed daily trajectory
   without `HADM_ID`. Every visit in a patient's sequence sees the same
   (static) lab profile.
4. **Visit order uses `HADM_ID` ascending as a chronology proxy**, since
   admission timestamps are not retained in the three shared tables this
   baseline (and the main pipeline's own evaluate.py) reads.
5. **Training labels are automatically derived**, not manually annotated
   as in the original COMPOSE paper, because this is a repurposed dataset
   without ground-truth trial enrollment records.

None of these change the comparison's validity -- Stage A/B are subject
to the same underlying data limitations -- but they should be stated
alongside the COMPOSE numbers wherever they're reported.
