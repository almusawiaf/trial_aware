# Patient–Trial Matching Baselines

Comparison baselines for the Trial-Aware GCL pipeline: XGBoost, MLP, two-tower,
and four GNN architectures, all evaluated on the same data, the same splits and
the same metrics as the upstream heterogeneous-graph contrastive model.

Nothing here needs `torch_geometric`, `torch_scatter` or `torch_sparse` — the
graph models are written directly against `torch.sparse`.

---

## Read this before running anything

The upstream label is a **deterministic rule** over the same inputs a model sees:

```
M_inc = mean_c ( match(patient, c) · w_c )    over inclusion criteria
M_exc = max_c  ( match(patient, c) · w_c )    over exclusion criteria
y     = 1  iff  M_inc >= 0.15  and  M_exc < 0.80
```

Scoring pairs directly by `M_inc − M_exc` gives **ROC-AUC ≈ 1.0**. Any model
handed criterion-overlap features reproduces the label exactly. That is not
performance; it is the labeller measuring itself.

Every model therefore runs under an explicit **feature regime**:

| Regime | What the model sees | Interpretation |
| --- | --- | --- |
| `oracle` | exact overlap counts, Jaccard, exclusion hits | Leakage audit. Expect ~1.0. **Never quote as a result.** |
| `honest` *(default)* | patient block + trial block + low-rank SVD interactions only | The reportable regime. The interaction must be learned. |
| `honest_noint` | plain `[patient ‖ trial]` concatenation | Strictest setting; cleanest contrast with two-tower and GNN. |

`tests/test_invariants.py::test_honest_regime_excludes_overlap_features`
asserts that no HONEST feature correlates above |r| = 0.95 with `M_inc`. If that
test fails, a leak has been introduced and the results are void.

---

## Installation

```bash
conda create -n ta_baselines python=3.11 -y
conda activate ta_baselines
pip install -r requirements.txt
pytest tests/ -v          # ~2 minutes, no MIMIC required
```

## Running

```bash
# Sanity pass on synthetic data — no credentials needed
python run_all.py --data synthetic --models all --seeds 0 --no-tune --out results/smoke

# Real run against the upstream pipeline's parquet outputs
python run_all.py --data real \
    --mimic-dir  /path/to/Trial_Aware_2/data \
    --trials-dir /path/to/Trial_Aware_2/data/10000_trials \
    --models all --regimes honest oracle \
    --seeds 0 1 2 3 4 --tune-trials 20 --bootstrap 1000 \
    --out results/full

# On SLURM
sbatch scripts/job_baselines.sh
```

`python run_all.py --list-models` prints every model key and group.

---

## Experimental design

### Splits — doubly disjoint, both-cold primary

Patients and trials are partitioned **independently**, giving a 3 × 3 grid:

```
train      = train patients × train trials      (negatives subsampled 20:1)
validation = val patients   × val trials        both-cold; tuning + early stopping
test       = test patients  × test trials       both-cold; PRIMARY RESULT
```

plus two diagnostic quadrants, `cold_patient_only` and `cold_trial_only`, whose
gap tells you which axis a model fails to generalise along.

A uniform random split over *pairs* — the obvious thing to do — leaks both the
patient and the trial across folds, and a model scores well by memorising which
patients are broadly eligible. Disjointness is asserted at runtime, not assumed.
Evaluation quadrants are never negative-subsampled, so PR-AUC reflects the true
operating prevalence.

### Models

| Group | Keys |
| --- | --- |
| Reference | `random`, `prior_trial`, `prior_patient`, `cosine` |
| Classical | `logreg`, `rf`, `extratrees` |
| Boosting | `xgboost`, `xgboost_rank` |
| Neural | `mlp`, `two_tower` |
| Graph | `gcn`, `graphsage`, `gat`, `rgcn`, `graphsage_gcl` |

`prior_trial` and `prior_patient` score using *only* the trial or *only* the
patient. A model that beats random but not these has learned who is generally
eligible, not who matches this trial. `cosine` is unsupervised — the fair
comparator for the GCL model, since comparing a zero-shot retriever against
supervised XGBoost without it flatters the supervised side for reasons that
have nothing to do with architecture.

### The GNN graph

```
patient ──has──▶ code ◀──requires── trial      (inclusion)
                 code ◀──excludes── trial      (exclusion)
                 + reverse of each → 6 relations
```

**There is no patient–trial edge anywhere in the message-passing graph**, and
`_assert_no_patient_trial_edge` enforces it. In GNN link prediction it is
standard — and a standard source of inflated numbers — to put training edges in
the graph and predict held-out ones. Here eligibility is transitive through
shared codes, so a two-layer model would reach a test patient's label via
`patient → trial → similar patient`. Excluding those edges makes that
structurally impossible and keeps the encoder inductive: a new trial is encoded
from its criterion codes alone.

Patients and trials carry the **same SVD features the tabular models get**, so
any difference is attributable to message passing rather than to richer inputs.

### Metrics

PR-AUC is primary (ROC-AUC barely moves at 2% prevalence). Ranking metrics are
computed **per trial and then averaged**, because a pooled AUC over the
flattened matrix rewards a model that merely learns which trials are permissive.
Bootstrap resampling is over whole trials, not pairs — pairs sharing a trial are
dependent, and a pair-level bootstrap understates the interval substantially.
Calibration (Brier, ECE) is reported because training used negative subsampling,
which shifts the prior.

### Fairness controls

Identical splits, features, metric code, seeds, and an identical 20-draw random
search per model selected on the same validation quadrant. Random search rather
than Bayesian optimisation is deliberate: no internal state that could differ
between models, and trivially parallel.

---

## Output

```
results/
├── results.md          # main report: primary table, oracle audit, diagnostics
├── raw_results.csv     # one row per (model, regime, seed, split)
├── aggregate_test.csv  # mean ± std across seeds
├── comparison.csv      # seed-level Wilcoxon vs the reference model
└── config_used.json
```

`report.sanity_checks` flags result patterns that usually mean a methodological
problem: non-oracle ROC-AUC above 0.98, a pooled-vs-per-trial gap above 0.15,
or a PR-AUC standard deviation exceeding half its mean.

---

## Two bugs found in the upstream repo

**1. Patient lab values are always empty.**
`preprocessor.process_labs` writes the column `IMPUTED_VALUE_DECAYED`, but
`trial_graph.PatientClinicalState.build_from_tables` looks for
`VALUENUM`/`VALUE`/`VAL_NUM`. On real pipeline output the lab dictionary comes
back empty, so **every lab criterion silently scores 0** — it is counted as a
failed match rather than an unmeasured one. `data.load_patients` accepts either
name and warns when it falls back.

**2. The two matchers disagree.**
`trial_graph.compute_matching_indices` (used by `evaluate.py`) scores an
unresolvable `UNMATCHED_*` criterion as a *failed match* and aggregates
inclusion as `mean(w·s)`. `matching_engine.compute_matching_indices` *drops*
such criteria and aggregates as `sum(w·s)/sum(w)`. These give materially
different labels. Both are implemented here behind `drop_unresolved` and
`inclusion_aggregation`; the prevalence shift is logged.

Also worth noting: `evaluate.py` uses `HARD_NEG_INC_THRESHOLD` — the same
constant that derives Stage B's weak supervision — to define `y_true`. Training
signal and evaluation ground truth come from one function.

---

## File map

```
src/config.py      dataclass config; all thresholds mirror the upstream repo
src/data.py        loaders, CTG criteria parser, synthetic fallback cohort
src/labeling.py    vectorised rule labeller + leakage audit
src/splits.py      doubly-disjoint grouped splits, quadrants, health checks
src/features.py    regime-controlled feature builder, chunked pair assembly
src/metrics.py     PR/ROC, grouped ranking, cluster bootstrap, paired tests
src/tune.py        equal-budget randomised search
src/train_eval.py  orchestration
src/report.py      aggregation, significance, markdown/LaTeX tables
src/models/        one file per family; registry in __init__.py
tests/             invariant tests, each guarding a specific failure mode
```

Adding a model is one `@register("key")` class implementing `_fit` and `score`.
