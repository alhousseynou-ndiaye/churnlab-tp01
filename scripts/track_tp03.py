
import hashlib
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from churnlab.features import prepare_features
from churnlab.training import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "telco-churn-tp03"
THRESHOLD = 0.5
SEED = 42


def git_info():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip())
    return commit, dirty


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_dataset()
    train_df, valid_df, test_df = split_dataset(df, seed=SEED)

    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)

    y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
    y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

    # Verify the original TP02 customer allocation.
    allocation_path = Path("reports/tp02_split_allocations.csv")
    original = pd.read_csv(allocation_path)

    allocation = pd.concat([
        pd.DataFrame({
            "customerID": part["customerID"],
            "split": name,
        })
        for name, part in [
            ("train", train_df),
            ("validation", valid_df),
            ("test", test_df),
        ]
    ], ignore_index=True)

    if (
        original["customerID"].astype(str).tolist()
        != allocation["customerID"].astype(str).tolist()
        or original["split"].tolist() != allocation["split"].tolist()
    ):
        raise ValueError("TP02 split allocation mismatch")

    source_hash = hashlib.sha256(
        Path("data/raw/telco.csv").read_bytes()
    ).hexdigest()

    allocation_hash = hashlib.sha256(
        allocation_path.read_bytes()
    ).hexdigest()

    commit, dirty = git_info()

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=SEED
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    train_dataset = mlflow.data.from_pandas(
        train_df,
        name="telco_train",
        source="data/raw/telco.csv",
        targets="Churn",
    )
    validation_dataset = mlflow.data.from_pandas(
        valid_df,
        name="telco_validation",
        source="data/raw/telco.csv",
        targets="Churn",
    )

    report_dir = Path("reports/tp03")
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name, classifier in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({
                "model_type": name,
                "split_seed": SEED,
                "threshold": THRESHOLD,
                **{
                    f"classifier__{key}": value
                    for key, value in classifier.get_params(
                        deep=False
                    ).items()
                },
            })

            mlflow.set_tags({
                "git_commit": commit,
                "git_dirty": str(dirty).lower(),
                "data_sha256": source_hash,
                "split_allocation_sha256": allocation_hash,
            })

            mlflow.log_input(
                train_dataset, context="training"
            )
            mlflow.log_input(
                validation_dataset, context="validation"
            )

            pipeline, duration = train_model(
                classifier, X_train, y_train
            )
            metrics = evaluate_model(
                pipeline, X_valid, y_valid,
                threshold=THRESHOLD,
            )

            mlflow.log_metrics({
                f"val_{key}": float(metrics[key])
                for key in [
                    "accuracy",
                    "average_precision",
                    "precision",
                    "recall",
                    "f1",
                ]
            })
            mlflow.log_metric("fit_seconds", duration)

            # Confusion matrix: rows = actual,
            # columns = predicted; order [0, 1].
            matrix = pd.DataFrame(
                [
                    [metrics["tn"], metrics["fp"]],
                    [metrics["fn"], metrics["tp"]],
                ],
                index=["actual_0", "actual_1"],
                columns=["predicted_0", "predicted_1"],
            )
            matrix.index.name = "actual_class"

            matrix_path = report_dir / f"confusion_{name}.csv"
            matrix.to_csv(matrix_path)
            mlflow.log_artifact(
                str(matrix_path),
                artifact_path="validation",
            )

            provenance = {
                "model": name,
                "features": X_train.columns.tolist(),
                "class_mapping": {"No": 0, "Yes": 1},
                "positive_class": 1,
                "split_seed": SEED,
                "split_counts": {
                    "train": len(train_df),
                    "validation": len(valid_df),
                    "test": len(test_df),
                },
                "split_allocation_file": str(allocation_path),
                "split_allocation_sha256": allocation_hash,
                "original_csv_sha256": source_hash,
                "git_commit": commit,
                "git_dirty": dirty,
                "selection_metric": "val_average_precision",
                "decision_threshold": THRESHOLD,
                "confusion_matrix_order": [0, 1],
                "model_parameters": classifier.get_params(
                    deep=False
                ),
            }
            mlflow.log_dict(provenance, "provenance.json")

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="churn_pipeline",
                input_example=X_train.head(3),
                serialization_format="cloudpickle",
            )

            results.append({
                "model": name,
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "val_average_precision": metrics["average_precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "fit_seconds": duration,
            })

            print(
                f"{name}: AP={metrics['average_precision']:.4f}"
                f" | Run={run.info.run_id}"
            )

    comparison = pd.DataFrame(results).sort_values(
        "val_average_precision", ascending=False
    )
    comparison.to_csv(
        report_dir / "comparison.csv", index=False
    )

    selected = comparison.iloc[0]

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))
    print("\nSELECTED MODEL:", selected["model"])
    print("SOURCE RUN:", selected["run_id"])
    print("MODEL URI:", selected["model_uri"])


if __name__ == "__main__":
    main()
