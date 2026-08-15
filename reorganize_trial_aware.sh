#!/bin/bash
# Reorganize trial_aware repo by objective, preserving git history via `git mv`.
# RUN FROM THE REPO ROOT (the directory with .git in it).
#
# Hardened version: does NOT abort on the first missing/untracked file.
# Each move is checked individually; problems are collected and printed
# in a summary at the end so you can fix them by hand afterward.

MISSING=()
UNTRACKED=()
MOVED=0

safe_mv() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$src" ]; then
    echo "  SKIP (not found): $src"
    MISSING+=("$src")
    return
  fi
  if git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
    git mv "$src" "$dst"
    MOVED=$((MOVED+1))
  else
    echo "  WARN (exists but untracked by git -- plain mv, no history kept): $src"
    UNTRACKED+=("$src")
    mkdir -p "$(dirname "$dst")" 2>/dev/null
    mv "$src" "$dst"
    MOVED=$((MOVED+1))
  fi
}

safe_rmdir() {
  local d="$1"
  if [ -d "$d" ]; then
    if [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
      rmdir "$d"
    else
      echo "  NOTE: $d not empty, left in place -- check what's still inside:"
      ls -la "$d"
    fi
  fi
}

mkdir -p data_pipeline \
         models/claude_active/evaluate/compose_based \
         models/gemini_variant/evaluate \
         models/gpt_variant/evaluate models/gpt_variant/_archive \
         scripts scratch archive

echo "=== 1. Shared data extraction / preprocessing ==="
safe_mv clinical_trials_api.py            data_pipeline/
safe_mv extracting_trials/ontology_loader.py   data_pipeline/
safe_mv extracting_trials/load_1000_trials.py  data_pipeline/
safe_mv extracting_trials/preprocessor.py      data_pipeline/
safe_mv criteria_parser.py                data_pipeline/
safe_mv hierarchy.py                      data_pipeline/
safe_mv generate_trial_json.py            data_pipeline/
safe_mv graph_constructor.py              data_pipeline/
safe_mv build_pyg_graph.py                data_pipeline/
safe_mv build_real_trials.py              data_pipeline/
safe_mv filter_trials_by_codes.py         data_pipeline/
safe_mv check_coverage.py                 data_pipeline/
safe_mv validate_pipeline.py              data_pipeline/
safe_mv extracting_trials/trial_extraction.py  data_pipeline/
safe_mv run.py                            data_pipeline/

# NOTE: extracting_trials/trial_graph.py is an orphaned duplicate of
# C/trial_graph.py (identical content). hierarchy.py and check_coverage.py
# both import it ("from trial_graph import ..." / "from C.trial_graph import
# TrialStore"). Moving those two files above without fixing their imports
# will break them. Archiving the duplicate rather than deleting silently --
# fix the two import lines by hand after this script finishes (see chat).
safe_mv extracting_trials/trial_graph.py  archive/

echo "=== 2. Claude track (C/) -> models/claude_active/ ==="
safe_mv C/config.py             models/claude_active/
safe_mv C/train.py              models/claude_active/
safe_mv C/trial_graph.py        models/claude_active/
safe_mv C/matching_engine.py    models/claude_active/
safe_mv C/trial_embedding.py    models/claude_active/
safe_mv C/gcl_framework.py      models/claude_active/
safe_mv C/alignment.py          models/claude_active/
safe_mv C/run_anchor_sweep.py   models/claude_active/
safe_mv C/run_multi_seed.py     models/claude_active/
safe_mv evaluate.py             models/claude_active/evaluate/
safe_mv C/COMPOSE_based_evaluation/evaluate.py         models/claude_active/evaluate/compose_based/
safe_mv C/COMPOSE_based_evaluation/matching_engine.py  models/claude_active/evaluate/compose_based/
safe_mv C/COMPOSE_based_evaluation/config.py           models/claude_active/evaluate/compose_based/
safe_rmdir C/COMPOSE_based_evaluation
safe_rmdir C

echo "=== 3. Gemini track (G/) -> models/gemini_variant/ ==="
safe_mv G/config.py          models/gemini_variant/
safe_mv G/train.py           models/gemini_variant/
safe_mv G/trial_embedding.py models/gemini_variant/
safe_mv G/__init__.py        models/gemini_variant/
safe_mv G/evaluate.py        models/gemini_variant/evaluate/
safe_rmdir G

echo "=== 4. GPT track (B/) -> models/gpt_variant/ ==="
safe_mv B/matching_engine.py  models/gpt_variant/
safe_mv B/train.py            models/gpt_variant/
safe_mv B/trial_embedding.py  models/gpt_variant/
safe_mv B/matching_engine_1.py models/gpt_variant/_archive/
safe_mv B/train1.py           models/gpt_variant/_archive/
safe_mv B/train2.py           models/gpt_variant/_archive/
safe_rmdir B
# models/gpt_variant/evaluate/ intentionally left empty -- GPT track has
# no evaluator in the current repo.

echo "=== 5. Job scripts ==="
safe_mv job_pipeline2.sh          scripts/
safe_mv job_pipeline_C.sh         scripts/
safe_mv job_run.sh                scripts/
safe_mv job_train.sh              scripts/
safe_mv job_evaluate.sh           scripts/
safe_mv job_run_anchor_sweep.sh   scripts/
safe_mv job_pipeline.sh           archive/   # dead script, calls a root train.py that no longer exists

echo "=== 6. Scratch / exploratory work ==="
safe_mv extra                              scratch/extra
safe_mv extracting_trials/OLD              scratch/extracting_trials_OLD
safe_mv extracting_trials/test.py          scratch/
safe_mv extracting_trials/test2.py         scratch/
safe_mv extracting_trials/test3.py         scratch/
safe_mv extracting_trials/test4.py         scratch/
safe_mv extracting_trials/testing.ipynb    scratch/
safe_mv extracting_trials/test.ipynb       scratch/

echo "=== 7. Superseded drafts -> archive/ ==="
safe_mv extracting_trials/ontology_loader_1.py   archive/
safe_mv extracting_trials/ontology_loader_2.py   archive/
safe_mv extracting_trials/load_1000_trials_1.py  archive/

echo "=== 8. Remove root duplicates of C's config/trial_graph/alignment ==="
for f in config.py trial_graph.py alignment.py; do
  if [ -f "$f" ]; then
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git rm "$f"
    else
      rm "$f"
    fi
  fi
done

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Moved/handled: $MOVED"
if [ ${#MISSING[@]} -gt 0 ]; then
  echo ""
  echo "NOT FOUND (check these manually -- already moved? renamed? never existed?):"
  printf '  %s\n' "${MISSING[@]}"
fi
if [ ${#UNTRACKED[@]} -gt 0 ]; then
  echo ""
  echo "MOVED BUT WERE UNTRACKED BY GIT (no history preserved for these):"
  printf '  %s\n' "${UNTRACKED[@]}"
fi
echo ""
echo "Next: fix imports (e.g. 'from config import Config' now needs to resolve"
echo "inside models/claude_active/, models/gemini_variant/, etc. -- use PYTHONPATH"
echo "or relative imports), then review with 'git status' and commit."