# privacy_in_ml — python_scripts

Self-contained Python:

Differentially private (Input / Output / Internal / objective perturbation) logistic
regression, a row-removal experiment evaluation, and a membership-inference
attack, for three datasets (Pima Diabetes, Dry Bean, Credit Card default).

## Requirements

- **Python 3.11** (a Python version mismatch causing a broken `scikit-learn`/
  `diffprivlib` install is a real issue we hit — see the comment at the top
  of `requirements.txt`).
- Dependencies are in `requirements.txt`

## Setup

```bash
cd python_scripts
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## How to run

The **main file is `main.py`**. There is no need to run any other script directly

### Quickest start (default behaviour)

```bash
python main.py --dataset diabetes
```

With no other flags it fits the plain (non-private) baseline model, runs Input, Output and Internal perturbation across that dataset's standard epsilon grid, and then
runs the membership-inference attack against the resulting confidence scores. 

Since we provide the processed datasets the preprocessing is only run when it is expllicitly called.

Results are written to `../results/model_results_<dataset>.xlsx`; confidence
score pickles (consumed by the membership attack) are written alongside them.

Swap `diabetes` for `bean` or `creditcard`, or use `--dataset all` to run all
three back to back.

### Command-line options

| Flag | Choices / type | Default | Meaning |
|---|---|---|---|
| `--dataset` | `diabetes`, `bean`, `creditcard`, `all` | *(required)* | Which dataset to run. |
| `--variant` | `default`, `balanced` | `default` | `balanced` uses the RandomOverSampler-balanced training set instead of the normal one. |
| `--experiment` | `normal`, `row_removal` | `normal` | `row_removal` repeatedly removes one random training row and reruns each perturbation type across epsilon values, to empirically probe DP's neighboring-datasets guarantee. |
| `--section` | one or more of `all`, `original`, `input`, `output`, `internal`, `membership_attack`, `preprocessing` | `all` | Which part(s) to run. `all` expands to Original+Input+Output+Internal+membership_attack for `--experiment normal` (or Input+Output+Internal, with an automatic baseline, for `row_removal`). `preprocessing` is **never** included in `all` — request it explicitly. |
| `--epsilon` | one or more floats | *(dataset's own grid)* | Overrides the epsilon values used for Input/Output/Internal/row_removal. |
| `--n-iter` | int | `10` | Row-removal iterations per perturbation type (`--experiment row_removal` only). |
| `--threshold` | float | `0.970489258446794` | Confidence threshold for the membership attack's threshold classifier. |
| `--data-dir` | path | `../datasets` | Where the dataset CSVs live. |
| `--results-dir` | path | `../results` | Where to write result spreadsheets / confidence-score pickles. |

Run `python main.py --help` at any time for this same reference.

### Examples

Run just the Original + Input sections for bean, with a custom epsilon grid:

```bash
python main.py --dataset bean --section original input --epsilon 1 10 50
```

Run the full pipeline on the balanced credit-card training set:

```bash
python main.py --dataset creditcard --variant balanced
```

Row-removal sensitivity experiment, 5 iterations, diabetes only:

```bash
python main.py --dataset diabetes --experiment row_removal --n-iter 5
```

Just the membership attack, reusing confidence scores already produced by an
earlier run:

```bash
python main.py --dataset diabetes --section membership_attack
```

Regenerate all three datasets' train/test splits and balanced variants from
the raw CSV:

```bash
python main.py --dataset all --section preprocessing
```

## Package versions

See `requirements.txt` in this folder — every dependency is pinned to an
exact version that's been verified to work together. In particular,
`scikit-learn` is intentionally pinned to `1.7.2` rather than the latest
release, because `diffprivlib==0.6.6` (used for Internal/objective
perturbation) imports private scikit-learn internals that later scikit-learn
versions removed. Don't bump `scikit-learn` without re-verifying
`perturbations.py`'s Internal section still runs.

## What each file does

- `main.py` — CLI entry point; parses arguments and dispatches to the modules below.
- `dataset_config.py` — all per-dataset configuration (feature ranges, sensitivity formulas, model hyperparameters, target column, epsilon grids, file paths, class ordering for the attack). 
- `functions.py` — shared model-fitting/evaluation/result-saving helpers
- `preprocessing.py` — raw CSV → cleaned train/test split → balanced (oversampled) variant, for all three datasets.
- `perturbations.py` — Original / Input / Output / Internal perturbation, including the confidence-score pickles the membership attack consumes.
- `row_removal.py` — the row-removal sensitivity experiment.
- `membership_attack.py` — the confidence/loss/entropy-gap membership-inference evaluation and the threshold attack.
