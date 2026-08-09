import copy
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import functions as fc
from cli_utils import print_banner
from dataset_config import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    get_config,
    train_csv_path,
    test_csv_path,
    results_xlsx_path,
)


def get_dummies_all(X_train, X_test, categorical_cols):
    """Converts categorical columns into one-hot dummy columns (drop_first=True)."""
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    return X_train_encoded, X_test_encoded


def load_dataset(dataset_name, variant="default", data_dir=DEFAULT_DATA_DIR):
    """Returns raw (unscaled) X_train, X_test, y_train, y_test."""
    cfg = get_config(dataset_name)

    train_df = pd.read_csv(train_csv_path(dataset_name, data_dir, variant))
    test_df = pd.read_csv(test_csv_path(dataset_name, data_dir))

    if cfg.get("drop_id"):
        train_df = train_df.drop(columns=[cfg["id_column"]])
        test_df = test_df.drop(columns=[cfg["id_column"]])

    target = cfg["target_column"]
    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = train_df[target]
    y_test = test_df[target]

    if cfg.get("needs_dummies"):
        X_train, X_test = get_dummies_all(X_train, X_test, cfg["categorical_columns"])

    return X_train, X_test, y_train, y_test


def scale_features(dataset_name, X_train, X_test):
    """Plain StandardScaler for diabetes/bean; ColumnTransformer (scale
    numeric+ordinal, pass one-hot dummies through) for creditcard - matching
    each dataset's Original section exactly."""
    cfg = get_config(dataset_name)

    if cfg.get("needs_dummies"):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), cfg["numeric_features"]),
                ("ord", StandardScaler(), cfg["ordinal_features"]),
            ],
            remainder="passthrough",
        )
        X_train_sc = pd.DataFrame(
            preprocessor.fit_transform(X_train),
            columns=preprocessor.get_feature_names_out(),
            index=X_train.index,
        )
        X_test_sc = pd.DataFrame(
            preprocessor.transform(X_test),
            columns=preprocessor.get_feature_names_out(),
            index=X_test.index,
        )
        return X_train_sc, X_test_sc

    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def _predict_fns(bivariate):
    if bivariate:
        return fc.predict_binary, fc.predict_binary_save_results
    return fc.predict_multiclass, fc.predict_multiclass_save_results


def _confidence_X_all(cfg, X_train, X_test, X_train_sc, X_test_sc):
    """Which features get fed to predict_proba() for the attack pickle -
    see the `confidence_scores_use_scaled_features` comment in
    dataset_config.py for why this differs per dataset."""
    if cfg["confidence_scores_use_scaled_features"]:
        return np.concatenate([np.asarray(X_train_sc), np.asarray(X_test_sc)], axis=0)
    return pd.concat([X_train, X_test], axis=0)


def _save_pickle(obj, pkl_dir, filename):
    os.makedirs(pkl_dir, exist_ok=True)
    path = os.path.join(pkl_dir, filename)
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {path}")


def run_original(dataset_name, data, results_dir=DEFAULT_RESULTS_DIR, pkl_dir=None):
    """Fits the baseline (non-private) model, saves its metrics, and pickles
    its confidence scores for the membership attack."""
    print_banner(f"ORIGINAL (BASELINE, NO PRIVACY) - {dataset_name}")
    cfg = get_config(dataset_name)
    pkl_dir = pkl_dir or str(results_dir)
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = data
    predict_fn, save_fn = _predict_fns(cfg["bivariate"])
    results_xlsx = str(results_xlsx_path(dataset_name, results_dir))

    model = LogisticRegression(**cfg["model_params"])
    model.fit(X_train_sc, y_train)

    print("Training set: ")
    predict_fn(model, X_train_sc, y_train, conf_matrix=False)

    save_fn(
        model, X_test_sc, y_test, conf_matrix=False,
        perturbation_type="Original", epsilon=0, output_file=results_xlsx,
    )

    fc.cross_validate_model(model, X_train_sc, y_train, cv=10, bivariate=cfg["bivariate"])
    fc.hyperparameterTuning_save_Results(
        X_train_sc, y_train, cv=10, bivariate=cfg["bivariate"],
        perturbation_type="Original_hpt", epsilon=0, output_file=results_xlsx,
    )

    X_all = _confidence_X_all(cfg, X_train, X_test, X_train_sc, X_test_sc)
    orig_confidence_scores = model.predict_proba(X_all)
    _save_pickle(orig_confidence_scores, pkl_dir, f"orig_confidence_scores_{cfg['pkl_name']}.pkl")

    return model


def run_input(dataset_name, data, epsilon_values=None, results_dir=DEFAULT_RESULTS_DIR, pkl_dir=None):
    """Input perturbation: Laplace noise added to the raw training features,
    clipped back into each feature's plausible range, refit from scratch."""
    print_banner(f"INPUT PERTURBATION - {dataset_name}")
    cfg = get_config(dataset_name)
    pkl_dir = pkl_dir or str(results_dir)
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = data
    epsilon_values = epsilon_values or cfg["epsilon_values"]
    predict_fn, save_fn = _predict_fns(cfg["bivariate"])
    results_xlsx = str(results_xlsx_path(dataset_name, results_dir))
    ranges = cfg["ranges"]

    sensitivity = [r[1] - r[0] for r in ranges]

    # creditcard clips the (unperturbed) training data first, then adds
    # noise on top of the *unclipped* values and clips again
    if dataset_name == "creditcard":
        X_train_base = X_train.copy()
        for i, col in enumerate(X_train.columns):
            lower, upper = ranges[i]
            X_train_base[col] = X_train_base[col].clip(lower=lower, upper=upper)
    else:
        X_train_base = X_train

    baseline_model = LogisticRegression(**cfg["model_params"])
    baseline_model.fit(X_train_sc, y_train)

    np.random.seed(42)
    perturbed_datasets = {}
    for e in epsilon_values:
        X_train_perturbed = X_train_base.copy()
        d = X_train_perturbed.shape[1]

        for i, var in enumerate(X_train_base.columns):
            eps_j = e / d if cfg["input_eps_divide_by_dimensions"] else e
            scale = sensitivity[i] / eps_j
            noise = np.random.laplace(loc=0, scale=scale, size=len(X_train_base))

            lower, upper = ranges[i]
            X_train_perturbed[var] = (X_train[var] + noise).clip(lower=lower, upper=upper)

        perturbed_datasets[e] = X_train_perturbed

    input_confidence_scores = {}
    for e in epsilon_values:
        input_model = copy.deepcopy(baseline_model)

        scaler = StandardScaler()
        X_train_input_sc = scaler.fit_transform(perturbed_datasets[e])
        X_test_input_sc = scaler.transform(X_test)

        input_model.fit(X_train_input_sc, y_train)

        print("Training set: ")
        predict_fn(input_model, X_train_input_sc, y_train, conf_matrix=False)

        save_fn(
            input_model, X_test_input_sc, y_test, conf_matrix=False,
            perturbation_type="Input", epsilon=e, output_file=results_xlsx,
        )

        fc.hyperparameterTuning_save_Results(
            X_train_input_sc, y_train, cv=10, bivariate=cfg["bivariate"],
            perturbation_type="Input_hpt", epsilon=e, output_file=results_xlsx,
        )

        X_all = _confidence_X_all(cfg, X_train, X_test, X_train_input_sc, X_test_input_sc)
        input_confidence_scores[e] = input_model.predict_proba(X_all)

    _save_pickle(input_confidence_scores, pkl_dir, f"input_confidence_scores_{cfg['pkl_name']}.pkl")


def run_output(dataset_name, data, epsilon_values=None, results_dir=DEFAULT_RESULTS_DIR, pkl_dir=None):
    """Output perturbation: Laplace noise added to the fitted baseline
    model's coefficients/intercept."""
    print_banner(f"OUTPUT PERTURBATION - {dataset_name}")
    cfg = get_config(dataset_name)
    pkl_dir = pkl_dir or str(results_dir)
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = data
    epsilon_values = epsilon_values or cfg["epsilon_values"]
    predict_fn, save_fn = _predict_fns(cfg["bivariate"])
    results_xlsx = str(results_xlsx_path(dataset_name, results_dir))

    baseline_model = LogisticRegression(**cfg["model_params"])
    baseline_model.fit(X_train_sc, y_train)

    n = X_train.shape[0]
    lambda_reg = 1 / baseline_model.C
    if cfg["output_sensitivity_uses_data_norm"]:
        C_x = np.max(np.linalg.norm(X_train_sc, axis=1))
        sensitivity = (C_x * 2) / (lambda_reg * n)
    else:
        sensitivity = 2 / (lambda_reg * n)

    np.random.seed(42)
    perturbed_coefficients = {}
    for e in epsilon_values:
        intercept_scale = sensitivity / e if cfg["output_intercept_noise_uses_epsilon"] else sensitivity / 3
        coef_noise = np.random.laplace(loc=0, scale=sensitivity / e, size=baseline_model.coef_.shape)
        intercept_noise = np.random.laplace(loc=0, scale=intercept_scale, size=baseline_model.intercept_.shape)
        perturbed_coefficients[e] = {
            "coef": baseline_model.coef_ + coef_noise,
            "intercept": baseline_model.intercept_ + intercept_noise,
        }

    output_confidence_scores = {}
    for e in epsilon_values:
        output_model = copy.deepcopy(baseline_model)
        output_model.coef_ = perturbed_coefficients[e]["coef"]
        output_model.intercept_ = perturbed_coefficients[e]["intercept"]

        print("Training set: ")
        predict_fn(output_model, X_train_sc, y_train, conf_matrix=False)

        save_fn(
            output_model, X_test_sc, y_test, conf_matrix=False,
            perturbation_type="Output", epsilon=e, output_file=results_xlsx,
        )

        X_all = _confidence_X_all(cfg, X_train, X_test, X_train_sc, X_test_sc)
        output_confidence_scores[e] = output_model.predict_proba(X_all)

    _save_pickle(output_confidence_scores, pkl_dir, f"output_confidence_scores_{cfg['pkl_name']}.pkl")


def run_internal(dataset_name, data, epsilon_values=None, results_dir=DEFAULT_RESULTS_DIR, pkl_dir=None):
    """Internal (objective) perturbation via diffprivlib - see
    functions.internal_perturbation_save_results for the DP mechanics."""
    print_banner(f"INTERNAL (OBJECTIVE) PERTURBATION - {dataset_name}")
    cfg = get_config(dataset_name)
    pkl_dir = pkl_dir or str(results_dir)
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = data
    epsilon_values = epsilon_values or cfg["epsilon_values"]
    results_xlsx = str(results_xlsx_path(dataset_name, results_dir))

    baseline_model = LogisticRegression(**cfg["model_params"])
    baseline_model.fit(X_train_sc, y_train)

    data_norm = np.max(np.linalg.norm(np.asarray(X_train_sc), axis=1))
    print("data_norm (max L2 norm of a scaled training row):", data_norm)

    return fc.internal_perturbation_save_results(
        cfg["pkl_name"],
        X_train_sc,
        y_train,
        X_test_sc,
        y_test,
        epsilon_values=epsilon_values,
        data_norm=data_norm,
        C=baseline_model.C,
        bivariate=cfg["bivariate"],
        output_file=results_xlsx,
        perturbation_type="Internal",
        pkl_dir=pkl_dir,
    )


SECTION_RUNNERS = {
    "original": run_original,
    "input": run_input,
    "output": run_output,
    "internal": run_internal,
}


def prepare_data(dataset_name, variant="default", data_dir=DEFAULT_DATA_DIR):
    """Loads + scales a dataset once; reused across whichever sections run."""
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, variant=variant, data_dir=data_dir)
    X_train_sc, X_test_sc = scale_features(dataset_name, X_train, X_test)
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc


def run_sections(
    dataset_name,
    sections=("original", "input", "output", "internal"),
    variant="default",
    epsilon_values=None,
    data_dir=DEFAULT_DATA_DIR,
    results_dir=DEFAULT_RESULTS_DIR,
    pkl_dir=None,
):
    """Runs the requested subset of Original/Input/Output/Internal, in that
    order regardless of how `sections` was given, since Input/Output/Internal
    each need a freshly-fit baseline model (cheap to just refit per section)."""
    os.makedirs(results_dir, exist_ok=True)
    data = prepare_data(dataset_name, variant=variant, data_dir=data_dir)

    order = ["original", "input", "output", "internal"]
    for section in order:
        if section not in sections:
            continue
        runner = SECTION_RUNNERS[section]
        if section == "original":
            runner(dataset_name, data, results_dir=results_dir, pkl_dir=pkl_dir)
        else:
            runner(dataset_name, data, epsilon_values=epsilon_values, results_dir=results_dir, pkl_dir=pkl_dir)
