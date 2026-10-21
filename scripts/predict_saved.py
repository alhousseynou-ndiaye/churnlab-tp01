
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlab.features import prepare_features

artifact_path = Path("artifacts/tp02/churn_pipeline.joblib")
samples_path = Path("reports/tp02_inference_samples.json")
scores_path = Path("reports/tp02_expected_scores.json")

pipeline = joblib.load(artifact_path)

samples = pd.DataFrame(
    json.loads(samples_path.read_text(encoding="utf-8"))
)

expected_scores = np.array(
    json.loads(scores_path.read_text(encoding="utf-8"))
)

X = prepare_features(samples)

positive_index = list(pipeline.classes_).index(1)

actual_scores = pipeline.predict_proba(X)[:, positive_index]
predictions = pipeline.predict(X)

assert np.allclose(
    actual_scores,
    expected_scores,
    rtol=0,
    atol=1e-8,
)

print("Pipeline loaded successfully")
print("Predictions match the original pipeline")

results = pd.DataFrame({
    "predicted_churn": predictions,
    "churn_probability": actual_scores,
})

print("\nInference results:")
print(results.to_string(index=False))

new_customer = pd.DataFrame([{
    "tenure": 12,
    "MonthlyCharges": 79.85,
    "TotalCharges": " ",
    "Contract": "Month-to-month",
    "InternetService": "Satellite",
    "PaymentMethod": "Electronic check",
}])

new_X = prepare_features(new_customer)

new_score = pipeline.predict_proba(new_X)[0, positive_index]
new_prediction = pipeline.predict(new_X)[0]

print("\nNew customer:")
print("Predicted churn:", int(new_prediction))
print("Estimated churn probability:", round(float(new_score), 4))
