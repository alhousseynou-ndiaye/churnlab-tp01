
import numpy as np
import pandas as pd

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]
features = numeric_features + categorical_features


def prepare_features(frame):
    missing = sorted(set(features) - set(frame.columns))

    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = frame[features].copy()

    for column in numeric_features:
        values = X[column].astype("string").str.strip()
        values = values.mask(values.eq(""))

        X[column] = pd.to_numeric(
            values, errors="raise"
        ).to_numpy(dtype="float64", na_value=np.nan)

    for column in categorical_features:
        values = X[column].astype("string").str.strip()
        values = values.mask(values.eq(""))

        X[column] = values.to_numpy(
            dtype=object, na_value=np.nan
        )

    return X
