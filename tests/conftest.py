import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churnlab.features import prepare_features
from churnlab.training import train_model


@pytest.fixture
def sample_features():
    """Synthetic customers with the six TP02 features."""
    return pd.DataFrame(
        {
            "tenure": [
                1,
                3,
                6,
                12,
                18,
                24,
                36,
                48,
                2,
                5,
                9,
                15,
                27,
                40,
                55,
                72,
            ],
            "MonthlyCharges": [
                79,
                85,
                90,
                65,
                55,
                45,
                35,
                25,
                95,
                88,
                75,
                60,
                50,
                40,
                30,
                20,
            ],
            "TotalCharges": [
                79,
                255,
                540,
                780,
                990,
                1080,
                1260,
                1200,
                190,
                440,
                675,
                900,
                1350,
                1600,
                1650,
                1440,
            ],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
                "Two year",
                "Two year",
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
                "Two year",
                "Two year",
                "Two year",
            ],
            "InternetService": [
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "DSL",
                "DSL",
                "No",
                "DSL",
                "No",
                "Fiber optic",
                "Fiber optic",
                "DSL",
                "DSL",
                "No",
                "DSL",
                "No",
                "No",
            ],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }
    )


@pytest.fixture
def sample_targets():
    """Binary target: 1 = churn, 0 = no churn."""
    return pd.Series(
        [
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )


@pytest.fixture
def fitted_pipeline(sample_features, sample_targets):
    """Train the real TP02 pipeline on synthetic data."""
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    X_train = prepare_features(sample_features)

    pipeline, _ = train_model(
        classifier,
        X_train,
        sample_targets,
    )

    return pipeline
