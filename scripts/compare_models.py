
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from churnlab.features import (
    numeric_features,
    categorical_features,
    prepare_features,
)

from churnlab.training import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)


def main():
    df = load_dataset()
    train_df, valid_df, test_df = split_dataset(df, seed=42)

    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)

    y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
    y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "DummyClassifier": DummyClassifier(
            strategy="most_frequent",
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, classifier in models.items():
        pipeline, duration = train_model(
            classifier, X_train, y_train
        )

        metrics = evaluate_model(
            pipeline, X_valid, y_valid
        )

        fitted_pipelines[name] = pipeline

        results.append({
            "model": name,
            "configuration": str(classifier),
            "fit_seconds": round(duration, 4),
            **metrics,
        })

    comparison = pd.DataFrame(results).sort_values(
        "average_precision",
        ascending=False,
    )

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        report_dir / "tp02_model_comparison.csv",
        index=False,
    )

    print("\nModel Comparison:")
    print(
        comparison[
            [
                "model",
                "accuracy",
                "average_precision",
                "precision",
                "recall",
                "f1",
                "fit_seconds",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nConfusion Matrices:")
    print(
        comparison[
            ["model", "tn", "fp", "fn", "tp"]
        ].to_string(index=False)
    )

    learned_models = comparison[
        comparison["model"] != "DummyClassifier"
    ]

    selected_row = learned_models.iloc[0]
    selected_name = selected_row["model"]
    selected_pipeline = fitted_pipelines[selected_name]

    artifact_dir = Path("artifacts/tp02")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifact_dir / "churn_pipeline.joblib"
    joblib.dump(selected_pipeline, artifact_path)

    allocations = pd.concat([
        pd.DataFrame({
            "customerID": train_df["customerID"],
            "split": "train",
        }),
        pd.DataFrame({
            "customerID": valid_df["customerID"],
            "split": "validation",
        }),
        pd.DataFrame({
            "customerID": test_df["customerID"],
            "split": "test",
        }),
    ])

    split_path = report_dir / "tp02_split_allocations.csv"
    allocations.to_csv(split_path, index=False)

    samples = X_valid.head(5)

    samples.to_json(
        report_dir / "tp02_inference_samples.json",
        orient="records",
        indent=2,
    )

    positive_index = list(
        selected_pipeline.classes_
    ).index(1)

    expected_scores = selected_pipeline.predict_proba(
        samples
    )[:, positive_index]

    (
        report_dir / "tp02_expected_scores.json"
    ).write_text(
        json.dumps(expected_scores.tolist(), indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model": selected_name,
        "artifact": artifact_path.as_posix(),
        "selection_metric": "average_precision",
        "positive_class": "Yes",
        "positive_label": 1,
        "threshold": 0.5,
        "split_seed": 42,
        "split_counts": {
            "train": len(train_df),
            "validation": len(valid_df),
            "test": len(test_df),
        },
        "split_allocations": split_path.as_posix(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_imputation": "median",
        "numeric_scaling": "StandardScaler",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "model_parameters": selected_pipeline.named_steps[
            "classifier"
        ].get_params(),
        "validation_metrics": {
            metric: float(selected_row[metric])
            for metric in [
                "accuracy",
                "average_precision",
                "precision",
                "recall",
                "f1",
            ]
        },
        "source_sha256": hashlib.sha256(
            Path("data/raw/telco.csv").read_bytes()
        ).hexdigest(),
        "code_sha256": {
            str(path): hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest()
            for path in [
                "scripts/compare_models.py",
                "src/churnlab/features.py",
                "src/churnlab/training.py",
            ]
        },
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }

    metadata_path = artifact_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSelected model:", selected_name)
    print("Pipeline saved:", artifact_path)
    print("Metadata saved:", metadata_path)


if __name__ == "__main__":
    main()
