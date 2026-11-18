import numpy as np
import pytest

from churnlab.features import prepare_features


def test_missing_required_feature_is_rejected(fitted_pipeline, sample_features):
    """Reject inference when a required feature is missing."""

    broken = sample_features.drop(columns=["Contract"])

    with pytest.raises(
        ValueError,
        match="Missing features:.*Contract",
    ):
        X = prepare_features(broken)
        fitted_pipeline.predict(X)


def test_missing_numeric_and_unknown_category(fitted_pipeline, sample_features):
    """Allow missing numeric values and unseen categories."""

    modified = sample_features.copy()

    # Simulate raw CSV values stored as strings.
    modified["TotalCharges"] = modified["TotalCharges"].astype("string")

    # Introduce a permitted missing numeric value.
    modified.loc[0, "TotalCharges"] = " "

    # Introduce a category absent from the training data.
    unknown_category = "Satellite"

    assert unknown_category not in set(sample_features["InternetService"])

    modified.loc[0, "InternetService"] = unknown_category

    X = prepare_features(modified)

    predictions = fitted_pipeline.predict(X)
    probabilities = fitted_pipeline.predict_proba(X)

    # Every customer must receive one prediction.
    assert len(predictions) == len(modified)

    # Two classes: 0 = No churn, 1 = Churn.
    assert probabilities.shape == (len(modified), 2)

    # Probabilities must remain finite.
    assert np.isfinite(probabilities).all()

    # Predictions must belong to the known classes.
    assert set(predictions).issubset({0, 1})
