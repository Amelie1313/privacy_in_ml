import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler

from dataset_config import DEFAULT_DATA_DIR

NA_VALUES = ["NA", "", "NULL", "unknown", "Unknown", "na"]


def summary(dataset, target):
    print("Dimensions: ", dataset.shape)
    print("Datatypes: ", dataset.dtypes.unique())
    print("Missing values:", dataset.isna().sum().sum())

    if dataset[target].isna().sum() > 0:
        dataset = dataset.dropna(subset=[target]).reset_index(drop=True)
        print("Target had missing values. These rows were removed.")
        print("Dimensions: ", dataset.shape)

    print("Target_table:")
    print(dataset[target].value_counts(dropna=False))
    print(dataset[target].value_counts(normalize=True, dropna=False))


def _oversample_and_save(X, y, out_path, random_state=42):
    print("----------------------------------------------------------------------------------------------")
    print("Oversampling using RandomOverSampler")
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print(f"Original class distribution: {y.value_counts()}")
    print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")

    resampled = pd.concat([X_resampled, y_resampled], axis=1)
    resampled.to_csv(out_path, index=False)
    print(f"Saved balanced training set to {out_path}")


def preprocess_diabetes(data_dir=DEFAULT_DATA_DIR, random_state=42):
    print("# DIABETES #")
    diabetes = pd.read_csv(os.path.join(data_dir, "diabetes.csv"), na_values=NA_VALUES)

    print(diabetes.head())
    print(diabetes.dtypes)
    summary(diabetes, "Outcome")

    diabetes_train, diabetes_test = train_test_split(diabetes, test_size=0.2, random_state=random_state)
    print(diabetes_train.shape, diabetes_test.shape)

    # While no values are marked as NA, 0 is treated as missing in every
    # column except Pregnancies, DiabetesPedigreeFunction and Outcome, where
    # 0 is a physiologically impossible value.
    rep_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
    diabetes_train[rep_cols] = diabetes_train[rep_cols].replace(0, np.nan)
    diabetes_test[rep_cols] = diabetes_test[rep_cols].replace(0, np.nan)

    imputer = KNNImputer(n_neighbors=5)
    diabetes_train[rep_cols] = imputer.fit_transform(diabetes_train[rep_cols])
    diabetes_test[rep_cols] = imputer.transform(diabetes_test[rep_cols])

    train_path = os.path.join(data_dir, "diabetes_train.csv")
    test_path = os.path.join(data_dir, "diabetes_test.csv")
    diabetes_train.to_csv(train_path, index=False)
    diabetes_test.to_csv(test_path, index=False)
    print(f"Saved {train_path}, {test_path}")

    _oversample_and_save(
        diabetes_train.drop(["Outcome"], axis=1),
        diabetes_train["Outcome"],
        os.path.join(data_dir, "diabetes_train_balanced.csv"),
        random_state=random_state,
    )


def preprocess_bean(data_dir=DEFAULT_DATA_DIR, random_state=42):
    print("# DRY BEANS #")
    bean = pd.read_csv(os.path.join(data_dir, "bean.csv"), na_values=NA_VALUES)

    print(bean.head())
    print(bean.dtypes)
    summary(bean, "Class")

    bean_train, bean_test = train_test_split(bean, test_size=0.2, random_state=random_state)
    print(bean_train.shape, bean_test.shape)

    train_path = os.path.join(data_dir, "bean_train.csv")
    test_path = os.path.join(data_dir, "bean_test.csv")
    bean_train.to_csv(train_path, index=False)
    bean_test.to_csv(test_path, index=False)
    print(f"Saved {train_path}, {test_path}")

    _oversample_and_save(
        bean_train.drop(["Class"], axis=1),
        bean_train["Class"],
        os.path.join(data_dir, "bean_train_balanced.csv"),
        random_state=random_state,
    )


def preprocess_creditcard(data_dir=DEFAULT_DATA_DIR, random_state=42):
    print("# CREDIT CARD #")
    creditcard = pd.read_csv(os.path.join(data_dir, "credit_card.csv"), na_values=NA_VALUES)

    print(creditcard.head())
    print(creditcard.dtypes)
    summary(creditcard, "default.payment.next.month")

    # 0 isn't a defined category for MARRIAGE/EDUCATION in the codebook -
    # remapped to the "other" bucket for each.
    creditcard["MARRIAGE"] = creditcard["MARRIAGE"].replace(0, 3)
    creditcard["EDUCATION"] = creditcard["EDUCATION"].replace(0, 6)

    creditcard_train, creditcard_test = train_test_split(creditcard, test_size=0.2, random_state=random_state)
    print(creditcard_train.shape, creditcard_test.shape)

    train_path = os.path.join(data_dir, "creditcard_train.csv")
    test_path = os.path.join(data_dir, "creditcard_test.csv")
    creditcard_train.to_csv(train_path, index=False)
    creditcard_test.to_csv(test_path, index=False)
    print(f"Saved {train_path}, {test_path}")

    _oversample_and_save(
        creditcard_train.drop(["default.payment.next.month"], axis=1),
        creditcard_train["default.payment.next.month"],
        os.path.join(data_dir, "creditcard_train_balanced.csv"),
        random_state=random_state,
    )


PREPROCESSORS = {
    "diabetes": preprocess_diabetes,
    "bean": preprocess_bean,
    "creditcard": preprocess_creditcard,
}


def run_preprocessing(dataset_name=None, data_dir=DEFAULT_DATA_DIR, random_state=42):
    """dataset_name=None (or 'all') runs preprocessing for every dataset."""
    targets = PREPROCESSORS.keys() if dataset_name in (None, "all") else [dataset_name]
    for name in targets:
        if name not in PREPROCESSORS:
            raise ValueError(f"Unknown dataset '{name}'. Choose one of: {list(PREPROCESSORS)}")
        PREPROCESSORS[name](data_dir=data_dir, random_state=random_state)
        print()
