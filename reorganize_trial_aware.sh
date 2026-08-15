#!/bin/bash
# Reorganize trial_aware repo by objective, preserving git history via `git mv`.
# RUN FROM THE REPO ROOT. Review each section before running -- especially
# the "REVIEW ME" block, since root config.py/trial_graph.py/alignment.py/
# evaluate.py are exact duplicates of C/'s versions and you should decide
# whether to just delete them or keep a thin re-export for backward compat.
set -e

mkdir -p data_pipeline \
         models/claude_active/evaluate/compose_based \
         models/gemini_variant/evaluate \
         models/gpt_variant/evaluate models/gpt_variant/_archive \
         scripts scratch archive

# ---------------------------------------------------------------------------
# 1. Shared data extraction / preprocessing (used by all model tracks)
# ---------------------------------------------------------------------------
git mv clinical_trials_api.py data_pipeline/
git mv extracting_trials/ontology_loader.py data_pipeline/
git mv extracting_trials/load_1000_trials.py data_pipeline/
git mv extracting_trials/preprocessor.py data_pipeline/
git mv criteria_parser.py data_pipeline/
git mv hierarchy.py data_pipeline/
git mv generate_trial_json.py data_pipeline/
git mv graph_constructor.py data_pipeline/
git mv build_pyg_graph.py data_pipeline/
git mv build_real_trials.py data_pipeline/
git mv filter_trials_by_codes.py data_pipeline/
git mv check_coverage.py data_pipeline/
git mv validate_pipeline.py data_pipeline/
git mv trial_extraction.py data_pipeline/
git mv run.py data_pipeline/

# ---------------------------------------------------------------------------
# 2. Claude track (C/) -> models/claude_active/
#    NOTE: two evaluators exist for this track -- the plain one (root
#    evaluate.py, uses trial_graph.py's simple matching) and the
#    COMPOSE-based one (bootstrap CI + p-value, uses matching_engine.py's
#    hierarchy-aware strict match). Kept separate on purpose -- see chat.
# ---------------------------------------------------------------------------
git mv C/config.py C/train.py C/trial_graph.py C/matching_engine.py \
       C/trial_embedding.py C/gcl_framework.py C/alignment.py \
       C/run_anchor_sweep.py C/run_multi_seed.py \
       models/claude_active/
git mv evaluate.py models/claude_active/evaluate/
git mv C/COMPOSE_based_evaluation/evaluate.py models/claude_active/evaluate/compose_based/
git mv C/COMPOSE_based_evaluation/matching_engine.py models/claude_active/evaluate/compose_based/
git mv C/COMPOSE_based_evaluation/config.py models/claude_active/evaluate/compose_based/
rmdir C/COMPOSE_based_evaluation
rmdir C

# ---------------------------------------------------------------------------
# 3. Gemini track (G/) -> models/gemini_variant/
# ---------------------------------------------------------------------------
git mv G/config.py G/train.py G/trial_embedding.py G/__init__.py \
       models/gemini_variant/
git mv G/evaluate.py models/gemini_variant/evaluate/
rmdir G

# ---------------------------------------------------------------------------
# 4. GPT track (B/) -> models/gpt_variant/, drafts to _archive
# ---------------------------------------------------------------------------
git mv B/matching_engine.py models/gpt_variant/
git mv B/train.py models/gpt_variant/
git mv B/trial_embedding.py models/gpt_variant/
git mv B/matching_engine_1.py models/gpt_variant/_archive/
git mv B/train1.py models/gpt_variant/_archive/
git mv B/train2.py models/gpt_variant/_archive/
rmdir B
# NOTE: models/gpt_variant/evaluate/ is left empty on purpose -- the GPT
# track has no evaluator at all in the current repo. Either write one
# (mirroring models/claude_active/evaluate/evaluate.py) or drop the
# folder if you don't plan to score this track the same way.

# ---------------------------------------------------------------------------
# 5. Job scripts
# ---------------------------------------------------------------------------
git mv job_pipeline2.sh scripts/
git mv job_pipeline_C.sh scripts/
git mv job_run.sh scripts/
git mv job_train.sh scripts/
git mv job_evaluate.sh scripts/
git mv job_run_anchor_sweep.sh scripts/
git mv job_pipeline.sh archive/   # calls a root train.py that no longer exists -- dead script

# ---------------------------------------------------------------------------
# 6. Scratch / exploratory work
# ---------------------------------------------------------------------------
git mv extra scratch/extra
git mv extracting_trials/OLD scratch/extracting_trials_OLD
git mv extracting_trials/test.py scratch/
git mv extracting_trials/test2.py scratch/
git mv extracting_trials/test3.py scratch/
git mv extracting_trials/test4.py scratch/
git mv extracting_trials/testing.ipynb scratch/
git mv extracting_trials/test.ipynb scratch/

# ---------------------------------------------------------------------------
# 7. Superseded drafts -> archive/
# ---------------------------------------------------------------------------
git mv extracting_trials/ontology_loader_1.py archive/
git mv extracting_trials/ontology_loader_2.py archive/
git mv extracting_trials/load_1000_trials_1.py archive/

# ---------------------------------------------------------------------------
# 8. Root config.py / trial_graph.py / alignment.py were exact duplicates of
#    C's copies (already moved above as part of step 2) -- remove the
#    leftover root originals so there's a single source of truth.
# ---------------------------------------------------------------------------
git rm config.py
git rm trial_graph.py
git rm alignment.py

echo "Done. Now: fix imports (e.g. 'from config import Config' -> update paths"
echo "or add __init__.py + PYTHONPATH), then git commit."
