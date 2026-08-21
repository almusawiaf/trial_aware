#!/usr/bin/env python
"""
run_all.py
==========
Entry point for the patient-trial matching baseline benchmark.

Examples
--------
    # everything, synthetic data, quick smoke test
    python run_all.py --data synthetic --models all --seeds 0 --no-tune

    # the real comparison on MIMIC-III pipeline outputs
    python run_all.py --data real --models all --seeds 0 1 2 3 4 \
        --regimes honest oracle --tune-trials 20

    # just the graph models
    python run_all.py --models gnn --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from src.config import BenchmarkConfig
from src.features import FeatureRegime
from src.models import MODEL_GROUPS, available_models
from src.report import write_report
from src.train_eval import prepare_data, resolve_model_keys, run_benchmark


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Patient-trial matching baselines")
    p.add_argument("--data", default="auto", choices=["auto", "real", "real_ctg", "synthetic"])
    p.add_argument("--models", nargs="+", default=["all"],
                   help=f"model keys or groups {sorted(MODEL_GROUPS)}")
    p.add_argument("--regimes", nargs="+", default=["honest"],
                   choices=[r.value for r in FeatureRegime])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--tune-trials", type=int, default=None,
                   help="randomised-search budget per model (identical for all)")
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--mimic-dir", default=None, help="dir with *_clean.parquet")
    p.add_argument("--trials-dir", default=None, help="dir with structured trial JSON")
    p.add_argument("--reference", default="GraphSAGE",
                   help="model display name used as the comparison reference")
    p.add_argument("--n-patients", type=int, default=None, help="synthetic cohort size")
    p.add_argument("--n-trials", type=int, default=None, help="synthetic trial count")
    p.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples (0 to skip)")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--list-models", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.list_models:
        print("Models:", ", ".join(available_models()))
        print("Groups:")
        for g, ms in MODEL_GROUPS.items():
            print(f"  {g:10s} -> {', '.join(ms)}")
        return 0

    cfg = BenchmarkConfig()
    if args.out:
        cfg.paths.output_dir = args.out
    if args.mimic_dir:
        cfg.paths.mimic_processed_dir = args.mimic_dir
    if args.trials_dir:
        cfg.paths.trials_dir = args.trials_dir
    if args.tune_trials is not None:
        cfg.n_tuning_trials = args.tune_trials
    if args.no_tune:
        cfg.n_tuning_trials = 0
    if args.n_patients:
        cfg.synthetic_n_patients = args.n_patients
    if args.n_trials:
        cfg.synthetic_n_trials = args.n_trials
    if args.bootstrap is not None:
        cfg.eval.n_bootstrap = args.bootstrap
    if args.device:
        cfg.device = args.device
    cfg.paths.ensure()

    model_keys = resolve_model_keys(args.models)
    regimes = [FeatureRegime(r) for r in args.regimes]

    logging.info("Models : %s", ", ".join(model_keys))
    logging.info("Regimes: %s", ", ".join(r.value for r in regimes))
    logging.info("Seeds  : %s", args.seeds or list(cfg.seeds))
    logging.info("Device : %s", cfg.resolve_device())

    prepared = prepare_data(cfg, args.data)

    df = run_benchmark(
        cfg,
        model_keys=model_keys,
        regimes=regimes,
        seeds=args.seeds,
        data_mode=args.data,
        tune=not args.no_tune,
        prepared=prepared,
    )

    note = (
        f"Data source: `{prepared.dataset.source}`. "
        + (
            "**Synthetic data — smoke test only, do not report these numbers.**"
            if "synthetic" in prepared.dataset.source
            else ""
        )
    )
    path = write_report(
        df,
        cfg.paths.output_dir,
        reference_model=args.reference,
        label_audit=prepared.audit,
        dataset_note=note,
    )

    with open(os.path.join(cfg.paths.output_dir, "config_used.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"Report:      {path}")
    print(f"Raw results: {os.path.join(cfg.paths.output_dir, 'raw_results.csv')}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
