
import hashlib
from pathlib import Path
from time import perf_counter

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlab.features import (
    numeric_features,
    categorical_features,
    features,
)

EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def load_dataset(path="data/raw/telco.csv"):
    source = Path(path)

    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    if actual_hash != EXPECTED_SHA256:
        raise ValueError("Dataset SHA-256 mismatch")

    df = pd.read_csv(source)

    required = set(features) | {"customerID", "Churn"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Invalid customer identifiers")

    if df["Churn"].isna().any() or not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Invalid target values")

    return df


def split_dataset(df, seed=42):
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df["Churn"],
        random_state=seed,
    )

    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=0.5,
        stratify=remaining_df["Churn"],
        random_state=seed,
    )

    return train_df, valid_df, test_df


def build_pipeline(classifier):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_model(classifier, X_train, y_train):
    pipeline = build_pipeline(classifier)

    start = perf_counter()
    pipeline.fit(X_train, y_train)
    duration = perf_counter() - start

    return pipeline, duration


def evaluate_model(pipeline, X_valid, y_valid, threshold=0.5):
    positive_index = list(pipeline.classes_).index(1)

    scores = pipeline.predict_proba(X_valid)[:, positive_index]
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_valid, predictions),
        "average_precision": average_precision_score(y_valid, scores),
        "precision": precision_score(y_valid, predictions, zero_division=0),
        "recall": recall_score(y_valid, predictions, zero_division=0),
        "f1": f1_score(y_valid, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
