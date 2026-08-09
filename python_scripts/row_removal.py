import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import functions as fc
from cli_utils import print_banner
from dataset_config import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_N_ITER,
    get_config,
    row_removal_xlsx_path,
)
from perturbations import load_dataset, scale_features


def _predict_fns(bivariate):
    if bivariate:
        return fc.predict_binary, fc.predict_binary_save_results
    return fc.predict_multiclass, fc.predict_multiclass_save_results


def run_baseline(dataset_name, X_train, X_test, y_train, y_test, output_file):
    print_banner(f"ROW REMOVAL - BASELINE (full data, no removal) - {dataset_name}")
    cfg = get_config(dataset_name)
    predict_fn, save_fn = _predict_fns(cfg["bivariate"])

    X_train_sc, X_test_sc = scale_features(dataset_name, X_train, X_test)

    baseline_model = LogisticRegression(**cfg["model_params"])
    baseline_model.fit(X_train_sc, y_train)

    print("Baseline (full data):")
    predict_fn(baseline_model, X_train_sc, y_train, conf_matrix=False)

    save_fn(
        baseline_model, X_test_sc, y_test, conf_matrix=False,
        perturbation_type="Original", epsilon=0, row_id=None, output_file=output_file,
    )
    return baseline_model


def run_input(dataset_name, X_train, X_test, y_train, y_test, baseline_model,
              epsilon_values, n_iter, output_file):
    print_banner(f"ROW REMOVAL - INPUT PERTURBATION - {dataset_name}")
    cfg = get_config(dataset_name)
    predict_fn, save_fn = _predict_fns(cfg["bivariate"])
    ranges = cfg["ranges"]
    sensitivity = [r[1] - r[0] for r in ranges]

    row_rng = np.random.default_rng(42)  # dedicated RNG for row selection only
    np.random.seed(42)                   # global state, used for the Laplace noise

    for iteration in range(1, n_iter + 1):
        idx = row_rng.integers(0, X_train.shape[0])
        removed_label = X_train.index[idx]

        X_train_reduced = X_train.drop(index=removed_label)
        y_train_reduced = y_train.drop(index=removed_label)

        # creditcard clips first, then noises the unclipped values and clips again
        if dataset_name == "creditcard":
            X_train_base = X_train_reduced.copy()
            for i, col in enumerate(X_train_reduced.columns):
                lower, upper = ranges[i]
                X_train_base[col] = X_train_base[col].clip(lower=lower, upper=upper)
        else:
            X_train_base = X_train_reduced

        for e in epsilon_values:
            X_train_perturbed = X_train_base.copy()

            for i, var in enumerate(X_train_base.columns):
                eps_j = e / X_train_base.shape[1] if cfg["input_eps_divide_by_dimensions"] else e
                scale = sensitivity[i] / eps_j
                noise = np.random.laplace(loc=0, scale=scale, size=len(X_train_base))

                lower, upper = ranges[i]
                X_train_perturbed[var] = (X_train_reduced[var] + noise).clip(lower=lower, upper=upper)

            scaler = StandardScaler()
            X_train_perturbed_sc = scaler.fit_transform(X_train_perturbed)
            X_test_perturbed_sc = scaler.transform(X_test)

            input_model = copy.deepcopy(baseline_model)
            input_model.fit(X_train_perturbed_sc, y_train_reduced)

            print(f"Input - iteration {iteration}/{n_iter}, epsilon={e} - removed row {removed_label}")
            predict_fn(input_model, X_train_perturbed_sc, y_train_reduced, conf_matrix=False)

            save_fn(
                input_model, X_test_perturbed_sc, y_test, conf_matrix=False,
                perturbation_type="Input", epsilon=e, row_id=removed_label, output_file=output_file,
            )


def run_output(dataset_name, X_train, X_test, y_train, y_test, baseline_model,
               epsilon_values, n_iter, output_file):
    print_banner(f"ROW REMOVAL - OUTPUT PERTURBATION - {dataset_name}")
    cfg = get_config(dataset_name)
    predict_fn, save_fn = _predict_fns(cfg["bivariate"])

    row_rng = np.random.default_rng(42)  # reset: same n_iter rows as Input above
    np.random.seed(42)

    for iteration in range(1, n_iter + 1):
        idx = row_rng.integers(0, X_train.shape[0])
        removed_label = X_train.index[idx]

        X_train_reduced = X_train.drop(index=removed_label)
        y_train_reduced = y_train.drop(index=removed_label)

        scaler = StandardScaler()
        X_train_reduced_sc = scaler.fit_transform(X_train_reduced)
        X_test_reduced_sc = scaler.transform(X_test)

        # Retrain the base model on D' first - output perturbation noises an
        # already-fitted model's coefficients, so removing a row has to
        # affect the model that gets trained before noise is added.
        reduced_model = copy.deepcopy(baseline_model)
        reduced_model.fit(X_train_reduced_sc, y_train_reduced)

        n = X_train_reduced.shape[0]
        lambda_reg = 1 / reduced_model.C
        if cfg["output_sensitivity_uses_data_norm"]:
            C_x = np.max(np.linalg.norm(X_train_reduced_sc, axis=1))
            sensitivity = (C_x * 2) / (lambda_reg * n)
        else:
            sensitivity = 2 / (lambda_reg * n)

        for e in epsilon_values:
            intercept_scale = sensitivity / e if cfg["output_intercept_noise_uses_epsilon"] else sensitivity / 3
            coef_noise = np.random.laplace(loc=0, scale=sensitivity / e, size=reduced_model.coef_.shape)
            intercept_noise = np.random.laplace(loc=0, scale=intercept_scale, size=reduced_model.intercept_.shape)

            output_model = copy.deepcopy(reduced_model)
            output_model.coef_ = reduced_model.coef_ + coef_noise
            output_model.intercept_ = reduced_model.intercept_ + intercept_noise

            print(f"Output - iteration {iteration}/{n_iter}, epsilon={e} - removed row {removed_label}")
            predict_fn(output_model, X_train_reduced_sc, y_train_reduced, conf_matrix=False)

            save_fn(
                output_model, X_test_reduced_sc, y_test, conf_matrix=False,
                perturbation_type="Output", epsilon=e, row_id=removed_label, output_file=output_file,
            )


def run_internal(dataset_name, X_train, X_test, y_train, y_test, baseline_model,
                  epsilon_values, n_iter, output_file, pkl_dir=None):
    print_banner(f"ROW REMOVAL - INTERNAL (OBJECTIVE) PERTURBATION - {dataset_name}")
    cfg = get_config(dataset_name)

    row_rng = np.random.default_rng(42)  # reset: same n_iter rows as Input/Output above

    for iteration in range(1, n_iter + 1):
        idx = row_rng.integers(0, X_train.shape[0])
        removed_label = X_train.index[idx]

        X_train_reduced = X_train.drop(index=removed_label)
        y_train_reduced = y_train.drop(index=removed_label)

        scaler = StandardScaler()
        X_train_reduced_sc = scaler.fit_transform(X_train_reduced)
        X_test_reduced_sc = scaler.transform(X_test)

        data_norm = np.max(np.linalg.norm(np.asarray(X_train_reduced_sc), axis=1))

        print(f"Internal - iteration {iteration}/{n_iter} - removed row {removed_label}, data_norm={data_norm:.4f}")

        fc.internal_perturbation_save_results(
            f"{cfg['pkl_name']}_row_removal",
            X_train_reduced_sc,
            y_train_reduced,
            X_test_reduced_sc,
            y_test,
            epsilon_values=epsilon_values,
            data_norm=data_norm,
            C=baseline_model.C,
            bivariate=cfg["bivariate"],
            row_id=removed_label,
            output_file=output_file,
            perturbation_type="Internal",
            pkl_dir=pkl_dir,
        )


SECTION_RUNNERS = {
    "input": run_input,
    "output": run_output,
    "internal": run_internal,
}


def run_row_removal(
    dataset_name,
    sections=("input", "output", "internal"),
    variant="default",
    epsilon_values=None,
    n_iter=DEFAULT_N_ITER,
    data_dir=DEFAULT_DATA_DIR,
    results_dir=DEFAULT_RESULTS_DIR,
    pkl_dir=None,
):
    cfg = get_config(dataset_name)
    epsilon_values = epsilon_values or cfg["epsilon_values"]
    output_file = str(row_removal_xlsx_path(dataset_name, results_dir))
    pkl_dir = pkl_dir or str(results_dir)

    X_train, X_test, y_train, y_test = load_dataset(dataset_name, variant=variant, data_dir=data_dir)

    baseline_model = run_baseline(dataset_name, X_train, X_test, y_train, y_test, output_file)

    for section in ["input", "output", "internal"]:
        if section not in sections:
            continue
        if section == "internal":
            run_internal(
                dataset_name, X_train, X_test, y_train, y_test, baseline_model,
                epsilon_values, n_iter, output_file, pkl_dir=pkl_dir,
            )
        else:
            SECTION_RUNNERS[section](
                dataset_name, X_train, X_test, y_train, y_test, baseline_model,
                epsilon_values, n_iter, output_file,
            )
