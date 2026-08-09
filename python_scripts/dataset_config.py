"""
Central per-dataset configuration.

Everything that used to be hardcoded separately inside each
`<dataset>_dataset.ipynb` / `<dataset>_dataset_all.ipynb` / `<dataset>_row_removal.ipynb`
notebook (feature ranges, sensitivity inputs, model hyperparameters, target
column name, class ordering for the membership attack, epsilon grids, output
filenames, dataset-specific quirks) is centralized here so the rest of
python_scripts/ can stay generic across diabetes / bean / creditcard instead
of re-implementing the same loops three times.

Only the *configuration* differs per dataset; the actual perturbation /
row-removal / attack logic (which used to be copy-pasted per-notebook) lives
once in perturbations.py / row_removal.py / membership_attack.py and reads
its behaviour from this file.
"""

from pathlib import Path

# Repository root = the parent of this file's directory (python_scripts/../)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# Default privacy budgets used across the notebooks. Diabetes evolved to
# include an extra 0.01 point; bean/creditcard did not - preserved as-is
# rather than "fixed" to match the original notebooks exactly.
DIABETES_EPSILON_VALUES = [0.01, 0.1, 1, 5, 10, 30, 50, 100]
STANDARD_EPSILON_VALUES = [0.1, 1, 5, 10, 30, 50, 100]

# Same hardcoded attack threshold used in every membership_attack_*.ipynb.
DEFAULT_ATTACK_THRESHOLD = 0.970489258446794

# Default number of row-removal iterations per perturbation type
# (matches what was agreed on for *_row_removal.ipynb).
DEFAULT_N_ITER = 10


DATASETS = {
    "diabetes": {
        "target_column": "Outcome",
        "bivariate": True,
        "train_csv": "diabetes_train.csv",
        "train_csv_balanced": "diabetes_train_balanced.csv",
        "test_csv": "diabetes_test.csv",
        "raw_csv": "diabetes.csv",
        "model_params": {"penalty": "l2", "C": 1, "max_iter": 1000, "random_state": 42},
        # [lower, upper] plausible bound per feature, used both to bound the
        # Laplace noise added under Input perturbation and as the sensitivity
        # (upper - lower) for that same noise.
        "ranges": [[0, 20], [40, 300], [30, 200], [5, 100], [2, 900], [10, 70], [0, 3], [21, 100]],
        "epsilon_values": DIABETES_EPSILON_VALUES,
        # diabetes_dataset_all.ipynb divides the per-column epsilon by the
        # number of dimensions (eps_j = e/d); bean/creditcard do not (their
        # notebooks note this "leads to huge errors" and keep eps_j = e).
        "input_eps_divide_by_dimensions": True,
        # Output perturbation sensitivity: diabetes uses (2*C_x)/(lambda*n)
        # where C_x is the max L2 norm of a training row; bean/creditcard use
        # a fixed numerator of 2 instead of 2*C_x.
        "output_sensitivity_uses_data_norm": True,
        # diabetes scales the intercept noise by the epsilon like the coef
        # noise; bean/creditcard have a fixed /3 divisor instead (kept as-is,
        # not "fixed", to match the original notebooks).
        "output_intercept_noise_uses_epsilon": True,
        # Original/Input/Output confidence-score pickles (for the membership
        # attack) are built from the *raw, unscaled* X_train/X_test in
        # diabetes_dataset_all.ipynb / bean_dataset_all.ipynb, even though the
        # models they came from were trained on scaled features - this looks
        # like a pre-existing inconsistency (creditcard_dataset_all.ipynb
        # uses the scaled features instead, consistently). Preserved as-is
        # rather than silently "fixed", since it changes what the membership
        # attack actually measures.
        "confidence_scores_use_scaled_features": False,
        "results_xlsx": "model_results_diabetes.xlsx",
        "row_removal_xlsx": "row_removal_diabetes.xlsx",
        "pkl_name": "diabetes",
        "class_order": [0, 1],
        "needs_dummies": False,
    },
    "bean": {
        "target_column": "Class",
        "bivariate": False,
        "train_csv": "bean_train.csv",
        "train_csv_balanced": "bean_train_balanced.csv",
        "test_csv": "bean_test.csv",
        "raw_csv": "bean.csv",
        "model_params": {"C": 1, "l1_ratio": 0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42},
        "ranges": [
            [10000, 300000], [500, 2000], [100, 750], [100, 750],
            [1, 3], [0, 1], [10000, 300000], [100, 700], [0, 1],
            [0, 1], [0.2, 1], [0.5, 1], [0, 0.2], [0, 0.1],
            [0.2, 1], [0.9, 1],
        ],
        "epsilon_values": STANDARD_EPSILON_VALUES,
        "input_eps_divide_by_dimensions": False,
        "output_sensitivity_uses_data_norm": False,
        "output_intercept_noise_uses_epsilon": False,
        "confidence_scores_use_scaled_features": False,
        "results_xlsx": "model_results_bean.xlsx",
        "row_removal_xlsx": "row_removal_bean.xlsx",
        "pkl_name": "bean",
        "class_order": ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"],
        "needs_dummies": False,
    },
    "creditcard": {
        "target_column": "default.payment.next.month",
        "bivariate": True,
        "train_csv": "creditcard_train.csv",
        "train_csv_balanced": "creditcard_train_balanced.csv",
        "test_csv": "creditcard_test.csv",
        "raw_csv": "credit_card.csv",
        "model_params": {"penalty": "l2", "C": 1, "max_iter": 1000, "random_state": 42},
        "ranges": [
            [1000, 1000000], [18, 100], [-2, 8], [-2, 8], [-2, 8], [-2, 8], [-2, 8],
            [-2, 8], [-10000, 500000], [-10000, 500000], [-10000, 500000], [-10000, 500000],
            [-10000, 500000], [-10000, 500000], [0, 250000], [0, 250000], [0, 250000],
            [0, 250000], [0, 250000], [0, 250000], [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
        ],
        "epsilon_values": STANDARD_EPSILON_VALUES,
        "input_eps_divide_by_dimensions": False,
        "output_sensitivity_uses_data_norm": False,
        "output_intercept_noise_uses_epsilon": False,
        "confidence_scores_use_scaled_features": True,
        "results_xlsx": "model_results_cc.xlsx",
        "row_removal_xlsx": "row_removal_cc.xlsx",
        "pkl_name": "creditcard",
        "class_order": [0, 1],
        "needs_dummies": True,
        "drop_id": True,
        "id_column": "ID",
        "categorical_columns": ["MARRIAGE", "SEX", "EDUCATION"],
        "numeric_features": [
            "LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4",
            "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
            "PAY_AMT5", "PAY_AMT6",
        ],
        "ordinal_features": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    },
}

DATASET_NAMES = list(DATASETS.keys())


def get_config(dataset_name):
    """Look up a dataset's config dict, raising a clear error for typos."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose one of: {DATASET_NAMES}"
        )
    return DATASETS[dataset_name]


def train_csv_path(dataset_name, data_dir=DEFAULT_DATA_DIR, variant="default"):
    cfg = get_config(dataset_name)
    filename = cfg["train_csv_balanced"] if variant == "balanced" else cfg["train_csv"]
    return Path(data_dir) / filename


def test_csv_path(dataset_name, data_dir=DEFAULT_DATA_DIR):
    cfg = get_config(dataset_name)
    return Path(data_dir) / cfg["test_csv"]


def results_xlsx_path(dataset_name, results_dir=DEFAULT_RESULTS_DIR):
    cfg = get_config(dataset_name)
    return Path(results_dir) / cfg["results_xlsx"]


def row_removal_xlsx_path(dataset_name, results_dir=DEFAULT_RESULTS_DIR):
    cfg = get_config(dataset_name)
    return Path(results_dir) / cfg["row_removal_xlsx"]
