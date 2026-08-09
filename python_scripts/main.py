#!/usr/bin/env python3
"""
Single entry point for the privacy_in_ml pipeline.
 See README.md for the full how-to.

Quick reference
---------------
Default behaviour (just `--dataset <name>`):
fits the Original model, runs Input/Output/Internal perturbation across the
dataset's standard epsilon grid, and then runs the membership-inference
attack against the resulting confidence scores.

    python main.py --dataset diabetes

Everything else is opt-in through flags:

    python main.py --dataset bean --variant balanced
    python main.py --dataset creditcard --section input output
    python main.py --dataset diabetes --experiment row_removal --n-iter 5
    python main.py --dataset all --section preprocessing
"""

import argparse
import sys

from cli_utils import print_major_banner
from dataset_config import (
    DATASET_NAMES,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_N_ITER,
    DEFAULT_ATTACK_THRESHOLD,
)

PERTURBATION_SECTIONS = ["original", "input", "output", "internal"]
ROW_REMOVAL_SECTIONS = ["input", "output", "internal"]  # baseline always included automatically
SECTION_CHOICES = ["all", "original", "input", "output", "internal", "membership_attack", "preprocessing"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the differential-privacy logistic-regression pipeline "
                    "(preprocessing / perturbations / row-removal / membership attack).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=DATASET_NAMES + ["all"], required=True,
        help="Which dataset to run. 'all' runs diabetes, bean and creditcard in sequence.",
    )
    parser.add_argument(
        "--variant", choices=["default", "balanced"], default="default",
        help="'default' = the normal (imbalanced) training set, "
             "'balanced' = the RandomOverSampler-balanced training set.",
    )
    parser.add_argument(
        "--experiment", choices=["normal", "row_removal"], default="normal",
        help="'normal' = the standard Original/Input/Output/Internal pipeline, "
             "'row_removal' = repeatedly remove one random training row and rerun "
             "each perturbation type across epsilon values.",
    )
    parser.add_argument(
        "--section", nargs="+", choices=SECTION_CHOICES, default=["all"],
        help="Which part(s) to run. 'all' expands to original+input+output+internal"
             "+membership_attack for --experiment normal (or input+output+internal for "
             "row_removal, which always includes a baseline run). 'preprocessing' is never "
             "included in 'all' - it must be requested explicitly since datasets/*.csv are "
             "already committed to the repo.",
    )
    parser.add_argument(
        "--epsilon", nargs="+", type=float, default=None,
        help="Override the epsilon grid used for Input/Output/Internal/row_removal. "
             "Defaults to each dataset's own standard grid if not given.",
    )
    parser.add_argument(
        "--n-iter", type=int, default=DEFAULT_N_ITER,
        help="Number of row-removal iterations per perturbation type (--experiment row_removal only).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold for the membership-attack's threshold classifier. "
             f"Defaults to {DEFAULT_ATTACK_THRESHOLD} (the value used throughout the notebooks).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help=f"Directory containing the dataset CSVs. Defaults to {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help=f"Directory to write results xlsx / confidence-score pickles to. Defaults to {DEFAULT_RESULTS_DIR}",
    )
    return parser.parse_args(argv)


def _expand_sections(section_arg, experiment):
    """Turns the raw --section list into the concrete set of things to run.
    'all' expands to the standard pipeline for the chosen --experiment;
    'preprocessing' is excluded from 'all'(datasets/*.csv are already committed). Any other explicit section
    names are just passed through, so e.g. `--section all preprocessing`
    still runs preprocessing too."""
    explicit = [s for s in section_arg if s != "all"]
    if "all" in section_arg:
        base = (PERTURBATION_SECTIONS + ["membership_attack"]) if experiment == "normal" else list(ROW_REMOVAL_SECTIONS)
        return base + [s for s in explicit if s not in base]
    return explicit


def run_for_dataset(dataset_name, args):
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    results_dir = args.results_dir or DEFAULT_RESULTS_DIR
    sections = _expand_sections(args.section, args.experiment)

    if "preprocessing" in sections:
        import preprocessing
        print_major_banner(f"{dataset_name}: PREPROCESSING (raw CSV -> train/test/balanced CSVs)")
        preprocessing.run_preprocessing(dataset_name, data_dir=data_dir)

    perturbation_sections = [s for s in sections if s in PERTURBATION_SECTIONS]
    run_attack = "membership_attack" in sections

    if args.experiment == "row_removal":
        row_removal_sections = [s for s in sections if s in ROW_REMOVAL_SECTIONS]
        if row_removal_sections:
            import row_removal
            print_major_banner(f"{dataset_name} ({args.variant}): ROW REMOVAL EXPERIMENT")
            row_removal.run_row_removal(
                dataset_name,
                sections=row_removal_sections,
                variant=args.variant,
                epsilon_values=args.epsilon,
                n_iter=args.n_iter,
                data_dir=data_dir,
                results_dir=results_dir,
            )
    else:
        if perturbation_sections:
            import perturbations
            print_major_banner(f"{dataset_name} ({args.variant}): PERTURBATIONS "
                                f"(Original / Input / Output / Internal)")
            perturbations.run_sections(
                dataset_name,
                sections=perturbation_sections,
                variant=args.variant,
                epsilon_values=args.epsilon,
                data_dir=data_dir,
                results_dir=results_dir,
            )

        if run_attack:
            import membership_attack
            print_major_banner(f"{dataset_name}: MEMBERSHIP ATTACK "
                                f"(who was in the training set?)")
            attack_sections = [s for s in perturbation_sections if s != "original"] or ["input", "output", "internal"]
            membership_attack.run_membership_attack(
                dataset_name,
                sections=attack_sections,
                variant=args.variant,
                threshold=args.threshold,
                data_dir=data_dir,
                results_dir=results_dir,
            )


def main(argv=None):
    args = parse_args(argv)
    targets = DATASET_NAMES if args.dataset == "all" else [args.dataset]

    for dataset_name in targets:
        run_for_dataset(dataset_name, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
