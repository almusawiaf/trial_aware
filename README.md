# Trial-Aware GCL Pipeline — Revised Implementation

This is a rewrite of the original Phase 1–4 codebase, aligned with the
finalized LaTeX methodology and fixing the issues raised in review. It has
been run end-to-end on mock data (see `mock_data.py`) and completes without
errors.

## What changed, mapped to the review

| # | Issue | Fix / Location |
|---|---|---|
| 1 | Projection head never bypassed — saved embeddings weren't "raw" | `HeteroGNNEncoder.encode()` / `.project()` split (`gcl_framework.py`) |
| 2 | Standalone smoke test crashed without `ToUndirected()` | Fixed in `gcl_framework.py __main__` |
| 3 | 1-hop loader isolates patients from each other | Added `patient<->patient` `Comorbidity` edges (`graph_constructor.py`); the existing loader now naturally pulls in comorbid patients |
| 4 | Medication schema didn't match the paper (direct vs. diagnosis-mediated) | `('diagnosis','prescribed_for','medication')` edges, bridged via shared `HADM_ID` (`graph_constructor.py`) |
| 9 | One-hot features don't scale to real vocabularies | Replaced with per-type `nn.Embedding` tables, indexed by integer node id (`graph_constructor.py` + `gcl_framework.py`) |
| 10 | Unbounded daily lab-grid expansion | Capped at `MAX_DAYS_SINCE_MEASURED` days per gap (`preprocessor.py`) |
| — | Missing `generate_mock_mimic_data` | Added in `mock_data.py`, plus `generate_mock_trials` |
| — | **The core missing mechanism**: trial-conditioned contrastive alignment | New modules: `trial_graph.py` (structured criteria + M_inc/M_exc + weak-pair derivation), `trial_embedding.py` (bifurcated z_inc/z_exc trial embeddings via attention pooling), `alignment.py` (dual-force loss + inference-time `Similarity` score) |

## Two-stage training (`train.py`)

- **Stage A** — generic self-supervised contrastive pretraining (NT-Xent + edge-dropping), matching what was implemented before, bugs fixed. This is your **baseline arm**.
- **Stage B** — trial-aware alignment fine-tuning using the dual-force loss from the LaTeX methodology (attraction to `z_T^inc`, repulsion from hard/random negatives). This is the **full model** and the piece that was previously never implemented.

Both stages save their own patient embeddings (`patient_embeddings_baseline.pt` vs. `patient_embeddings.pt`) so you can run the same downstream evaluation on each and report the delta — that comparison is what actually substantiates the paper's "trial-awareness helps" claim.

## Known simplifications, called out explicitly (not hidden)

- **Criteria extraction is a stub.** `mock_data.generate_mock_trials()` hands the pipeline already-structured `(entity, operator, value)` triplets. Turning ClinicalTrials.gov free text into that structure is a separate clinical NER-RE task (e.g. MedCAT/scispaCy) — out of scope here.
- **Stage B uses full-graph forward passes**, not mini-batched, for simplicity at mock-data scale. On real MIMIC-III/IV scale, this needs the same (or a k-hop) subgraph sampler as Stage A.
- **Positive-pair derivation is weak-supervision only**, using a random *subset* of each trial's inclusion criteria (`derive_weak_positive_pairs`, `train_criteria_fraction=0.7`). This is intentional — it must NOT be reused to build your evaluation set (P@k, ETE@k), or the eval becomes circular. A genuinely held-out, independently-curated eligibility/enrollment sample is still needed for that (flagged in `trial_graph.py`'s docstring).
- **Comorbidity edges use a simple shared-diagnosis-count heuristic** capped by group size (`COMORBIDITY_MAX_GROUP_SIZE`) — a real deployment may want smarter thresholds or ontology-aware comorbidity definitions.

## How to run

```bash
pip install torch torch_geometric pandas numpy pyarrow tqdm

python run.py      # Phase 1 (preprocessing) + Phase 2 (graph construction)
                    # falls back to mock_data.py if config.DATA_DIR doesn't exist

python train.py    # Stage A (contrastive pretraining) + Stage B (trial alignment)
                    # writes patient_embeddings*.pt and trial_embeddings.pt to processed_data/
```

`main.py` from the original submission was dropped — it was fully redundant with `run.py`.
