import joblib
import numpy as np

from churnlab.features import prepare_features


def test_reload_preserves_predictions(tmp_path, fitted_pipeline, sample_features):
    """Saving and reloading must preserve model behavior."""

    # Prepare the synthetic input data.
    X = prepare_features(sample_features)

    # Predictions before saving.
    predictions_before = fitted_pipeline.predict(X)
    probabilities_before = fitted_pipeline.predict_proba(X)

    # Save the complete fitted pipeline in a temporary folder.
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_pipeline, artifact)

    assert artifact.is_file()

    # Reload without fitting again.
    loaded_pipeline = joblib.load(artifact)

    # Predictions after reloading.
    predictions_after = loaded_pipeline.predict(X)
    probabilities_after = loaded_pipeline.predict_proba(X)

    # Class predictions must be exactly identical.
    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )

    # Class ordering must remain identical.
    np.testing.assert_array_equal(
        fitted_pipeline.classes_,
        loaded_pipeline.classes_,
    )

    # Probabilities must agree within numerical tolerance.
    np.testing.assert_allclose(
        probabilities_before,
        probabilities_after,
        rtol=0,
        atol=1e-8,
    )

    # Verify the complete preprocessing pipeline was saved.
    assert "preprocessor" in loaded_pipeline.named_steps
    assert "classifier" in loaded_pipeline.named_steps
