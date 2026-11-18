
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from churnlab.features import prepare_features


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/telco-churn@candidate"

SAMPLES_PATH = Path("reports/tp02_inference_samples.json")
SCORES_PATH = Path("reports/tp02_expected_scores.json")
EVIDENCE_PATH = Path("reports/tp03/inference_evidence.json")


def main():
    # 1. Connect to the MLflow tracking server.
    mlflow.set_tracking_uri(TRACKING_URI)

    # 2. Retrieve the registered model version.
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(
        "telco-churn", "candidate"
    )

    print("Registered model: telco-churn")
    print("Alias: candidate")
    print("Registered version:", version.version)

    # 3. Load the fitted pipeline without training.
    pipeline = mlflow.sklearn.load_model(MODEL_URI)

    print("\nPipeline loaded successfully from MLflow")

    # 4. Recover TP02 inference samples and expected scores.
    samples = pd.DataFrame(
        json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    )

    expected_scores = np.array(
        json.loads(SCORES_PATH.read_text(encoding="utf-8"))
    )

    # Same input parsing as TP02.
    X = prepare_features(samples)

    # 5. Predict using the recovered pipeline.
    positive_index = list(pipeline.classes_).index(1)

    actual_scores = pipeline.predict_proba(X)[
        :, positive_index
    ]
    predictions = pipeline.predict(X)

    # 6. Check agreement with TP02.
    np.testing.assert_allclose(
        actual_scores,
        expected_scores,
        rtol=0,
        atol=1e-8,
    )

    expected_predictions = (
        expected_scores >= 0.5
    ).astype(int)

    assert np.array_equal(
        predictions,
        expected_predictions,
    ), "Class predictions do not match TP02"

    max_difference = float(
        np.max(np.abs(actual_scores - expected_scores))
    )

    print("\nVALIDATION RESULTS")
    print("Expected probabilities match: YES")
    print("Expected class predictions match: YES")
    print("Maximum probability difference:", max_difference)
    print("Model retrained: NO")

    results = pd.DataFrame({
        "predicted_churn": predictions,
        "churn_probability": actual_scores,
    })

    print("\nINFERENCE RESULTS")
    print(results.to_string(index=False))

    # 7. Predict a new customer using TP02 parsing rules.
    new_customer = pd.DataFrame([{
        "tenure": 12,
        "MonthlyCharges": 79.85,
        "TotalCharges": " ",
        "Contract": "Month-to-month",
        "InternetService": "Satellite",
        "PaymentMethod": "Electronic check",
    }])

    new_X = prepare_features(new_customer)

    new_score = float(
        pipeline.predict_proba(new_X)[0, positive_index]
    )
    new_prediction = int(pipeline.predict(new_X)[0])

    print("\nNEW CUSTOMER")
    print("Predicted churn:", new_prediction)
    print("Churn probability:", round(new_score, 4))

    # 8. Save evidence for the TP03 report.
    EVIDENCE_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    evidence = {
        "model_uri": MODEL_URI,
        "registered_version": version.version,
        "alias": "candidate",
        "samples_checked": len(samples),
        "probabilities_match": True,
        "class_predictions_match": True,
        "maximum_absolute_difference": max_difference,
        "model_retrained": False,
        "new_customer_prediction": new_prediction,
        "new_customer_probability": new_score,
    }

    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    print("\nEvidence saved:", EVIDENCE_PATH)
    print("\nTP03 INFERENCE CHECK PASSED")


if __name__ == "__main__":
    main()
