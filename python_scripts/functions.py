"""
Direct port of jupyter_notebooks/functions.ipynb into a plain importable module.

Logic is unchanged from the notebook version; the only alteration is that the
hardcoded relative default `output_file="results/model_results.xlsx"` (which
only worked if you happened to run Jupyter with a particular working
directory) now defaults to an absolute path resolved from dataset_config, so
these functions behave the same no matter where the script is invoked from.
"""

import time
import copy
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_validate, GridSearchCV

from diffprivlib.models import LogisticRegression as DPLogisticRegression
from diffprivlib.utils import PrivacyLeakWarning

from dataset_config import DEFAULT_RESULTS_DIR

_DEFAULT_OUTPUT_FILE = str(DEFAULT_RESULTS_DIR / "model_results.xlsx")


def cross_validate_model(model, X, y, cv, bivariate=True, print_results=True):
    """Perform cross-validation and print metrics if asked"""
    if bivariate == False:
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=1),
            'recall': make_scorer(recall_score, average='weighted', zero_division=1),
            'f1': make_scorer(f1_score, average='weighted', zero_division=1)
        }
        start = time.time()
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        end = time.time()
        if print_results == True:
            print(f"Accuracy: {results['test_accuracy'].mean():.4f}")
            print(f"Precision: {results['test_precision'].mean():.4f}")
            print(f"Recall: {results['test_recall'].mean():.4f}")
            print(f"F1 Score: {results['test_f1'].mean():.4f}")
            print("Prediction time: ", end - start)
        return results
    else:
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        start = time.time()
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        end = time.time()
        if print_results == True:
            print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
            print(f"Precision: {cv_results['test_precision'].mean():.4f}")
            print(f"Recall: {cv_results['test_recall'].mean():.4f}")
            print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")
            print("Time:", end - start)
        return cv_results


def hyperparameterTuning(X, y, cv, bivariate):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }

    if bivariate == True:
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': make_scorer(precision_score, average='binary', zero_division=1),
            'recall_weighted': make_scorer(recall_score, average='binary', zero_division=1),
            'f1_weighted': make_scorer(f1_score, average='binary', zero_division=1)
        }

        start = time.time()
        lr = LogisticRegression()
        grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring=scoring, refit='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
        print(f"Best cross-validation precision: {grid_search.cv_results_['mean_test_precision_weighted'][grid_search.best_index_]:.4f}")
        print(f"Best cross-validation recall: {grid_search.cv_results_['mean_test_recall_weighted'][grid_search.best_index_]:.4f}")
        print(f"Best cross-validation F1 Score: {grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]:.4f}")
        print("Time for Hypertuning: ", time.time() - start)
    else:
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=1),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=1),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=1)
        }
        start = time.time()
        lr = LogisticRegression()
        grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring=scoring, refit='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
        print(f"Best cross-validation precision: {grid_search.cv_results_['mean_test_precision_weighted'][grid_search.best_index_]:.4f}")
        print(f"Best cross-validation recall: {grid_search.cv_results_['mean_test_recall_weighted'][grid_search.best_index_]:.4f}")
        print(f"Best cross-validation F1 Score: {grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]:.4f}")
        print("Time for Hypertuning: ", time.time() - start)

    best_model = grid_search.best_estimator_  # Best trained model
    best_params = grid_search.best_params_    # Best hyperparameters
    return best_model, best_params


def hyperparameterTuning_save_Results(
    X,
    y,
    cv,
    bivariate=True,
    perturbation_type="Original_hpt",
    epsilon=0,
    output_file=None,
    sheet_name="Results"
):
    if output_file is None:
        output_file = _DEFAULT_OUTPUT_FILE

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100]
    }

    if bivariate:
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score),
            "f1": make_scorer(f1_score)
        }
    else:
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="weighted", zero_division=1),
            "recall": make_scorer(recall_score, average="weighted", zero_division=1),
            "f1": make_scorer(f1_score, average="weighted", zero_division=1)
        }

    start = time.time()

    lr = LogisticRegression(max_iter=1000)

    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X, y)

    elapsed = time.time() - start

    idx = grid_search.best_index_

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy : {grid_search.cv_results_['mean_test_accuracy'][idx]:.4f}")
    print(f"Precision: {grid_search.cv_results_['mean_test_precision'][idx]:.4f}")
    print(f"Recall   : {grid_search.cv_results_['mean_test_recall'][idx]:.4f}")
    print(f"F1 Score : {grid_search.cv_results_['mean_test_f1'][idx]:.4f}")
    print(f"Time     : {elapsed:.2f} s")

    # ---------- Save results ----------
    result = pd.DataFrame([{
        "Type": perturbation_type,
        "Epsilon": epsilon,
        "Best C": grid_search.best_params_["C"],
        "Accuracy": grid_search.cv_results_["mean_test_accuracy"][idx],
        "Precision": grid_search.cv_results_["mean_test_precision"][idx],
        "Recall": grid_search.cv_results_["mean_test_recall"][idx],
        "F1 Score": grid_search.cv_results_["mean_test_f1"][idx],
        "Time (s)": elapsed
    }])

    if os.path.exists(output_file):
        try:
            existing = pd.read_excel(output_file, sheet_name=sheet_name)
            result = pd.concat([existing, result], ignore_index=True)
        except ValueError:
            pass

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="a" if os.path.exists(output_file) else "w",
        if_sheet_exists="replace"
    ) as writer:
        result.to_excel(writer, sheet_name=sheet_name, index=False)

    return grid_search.best_estimator_, grid_search.best_params_


def predict_binary(model, X, y, conf_matrix=True):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print("---------------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("---------------------------------------")

    if conf_matrix:
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
        print(cm_df)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.show()


def predict_binary_save_results(
    model,
    X,
    y,
    conf_matrix=True,
    perturbation_type="Original",
    epsilon=0,
    row_id=None,
    output_file=None,
    sheet_name="Results"
):
    if output_file is None:
        output_file = _DEFAULT_OUTPUT_FILE

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print("---------------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("---------------------------------------")

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    if conf_matrix:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
        print(cm_df)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.show()

    ### Save results ###
    results = pd.DataFrame([{
        "Type": perturbation_type,
        "Epsilon": epsilon,
        "Row_ID": row_id,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }])

    if os.path.exists(output_file):
        existing = pd.read_excel(output_file, sheet_name=sheet_name)
        results = pd.concat([existing, results], ignore_index=True)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="w"
    ) as writer:
        results.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved results to {output_file}")


def internal_perturbation_save_results(
    datasetname,
    X_train,
    y_train,
    X_test,
    y_test,
    epsilon_values,
    data_norm,
    C=1.0,
    bivariate=True,
    max_iter=1000,
    perturbation_type="Internal",
    row_id=None,
    output_file=None,
    sheet_name="Results",
    pkl_dir=None,
):
    """
    `pkl_dir` controls where internal_confidence_scores_<name>.pkl
    gets written, so callers can direct it
    """
    if output_file is None:
        output_file = _DEFAULT_OUTPUT_FILE
    pkl_path = os.path.join(pkl_dir or ".", f"internal_confidence_scores_{datasetname}.pkl")

    all_results = []
    fitted_models = {}
    internal_confidence_scores = {}

    for e in epsilon_values:
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PrivacyLeakWarning)
            dp_model = DPLogisticRegression(
                epsilon=e,
                data_norm=data_norm,
                C=C,
                max_iter=max_iter,
                random_state=42
            )
            dp_model.fit(X_train, y_train)
        elapsed = time.time() - start

        # NEEDED FOR THE MEMBERSHIP ATTACK
        X_all = np.concatenate([X_train, X_test], axis=0)
        internal_confidence_scores[e] = dp_model.predict_proba(X_all)

        y_pred = dp_model.predict(X_test)

        if bivariate:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)
            f1 = f1_score(y_test, y_pred, zero_division=1)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)

        print("---------------------------------------")
        print(f"Epsilon: {e}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Time: {elapsed:.2f} s")
        print("---------------------------------------")

        row = {
            "Type": perturbation_type,
            "Epsilon": e,
            "Row_ID": row_id,
            "Data_norm": data_norm,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Time (s)": elapsed
        }

        if bivariate:
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            row.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})

        all_results.append(row)
        fitted_models[e] = dp_model

    with open(pkl_path, "wb") as f:
        pickle.dump(internal_confidence_scores, f)

    results = pd.DataFrame(all_results)

    if os.path.exists(output_file):
        try:
            existing = pd.read_excel(output_file, sheet_name=sheet_name)
            results = pd.concat([existing, results], ignore_index=True)
        except ValueError:
            pass

    file_exists = os.path.exists(output_file)
    writer_kwargs = {"engine": "openpyxl", "mode": "a" if file_exists else "w"}
    if file_exists:
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        results.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved results to {output_file}")

    return fitted_models


def predict_multiclass(model, X, y, conf_matrix=True):
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=1)

    print("---------------------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("---------------------------------------")

    cm = confusion_matrix(y, y_pred)
    if conf_matrix:
        labels = model.classes_

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels]
        )

        print(cm_df)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


def predict_multiclass_save_results(
    model,
    X,
    y,
    conf_matrix=True,
    perturbation_type="Original",
    epsilon=0,
    row_id=None,
    output_file=None,
    sheet_name="Results"
):
    if output_file is None:
        output_file = _DEFAULT_OUTPUT_FILE

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=1)

    print("---------------------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("---------------------------------------")

    cm = confusion_matrix(y, y_pred)
    if conf_matrix:
        labels = model.classes_

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels]
        )

        print(cm_df)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    ### Save results ###
    results = pd.DataFrame([{
        "Type": perturbation_type,
        "Epsilon": epsilon,
        "Row_ID": row_id,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }])

    if os.path.exists(output_file):
        existing = pd.read_excel(output_file, sheet_name=sheet_name)
        results = pd.concat([existing, results], ignore_index=True)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="w"
    ) as writer:
        results.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved results to {output_file}")
