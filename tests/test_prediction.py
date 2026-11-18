import numpy as np

from churnlab.features import prepare_features


def test_probability_contract(fitted_pipeline, sample_features):
    """Validate the prediction probability contract."""

    X = prepare_features(sample_features)

    probabilities = fitted_pipeline.predict_proba(X)

    # Verify the expected binary classification classes.
    classes = fitted_pipeline.classes_

    assert set(classes) == {0, 1}

    # One probability row per customer, one column per class.
    assert probabilities.shape == (
        len(sample_features),
        len(classes),
    )

    # Probabilities must be finite.
    assert np.isfinite(probabilities).all()

    # Every probability must be between 0 and 1.
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    # Probabilities for each customer must sum to 1.
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(sample_features)),
        rtol=0,
        atol=1e-8,
    )

    # Find the positive class dynamically.
    positive_index = list(classes).index(1)

    positive_scores = probabilities[:, positive_index]

    assert positive_scores.shape == (len(sample_features),)
    assert np.isfinite(positive_scores).all()
    assert np.all((positive_scores >= 0) & (positive_scores <= 1))
