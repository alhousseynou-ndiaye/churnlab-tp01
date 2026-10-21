
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]
features = numeric_features + categorical_features

df = pd.read_csv("data/raw/telco.csv")

train_df, remaining_df = train_test_split(
    df,
    test_size=0.4,
    stratify=df["Churn"],
    random_state=42,
)

valid_df, test_df = train_test_split(
    remaining_df,
    test_size=0.5,
    stratify=remaining_df["Churn"],
    random_state=42,
)

def prepare_features(frame):
    missing = sorted(set(features) - set(frame.columns))

    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = frame[features].copy()

    X["tenure"] = pd.to_numeric(X["tenure"], errors="raise")
    X["MonthlyCharges"] = pd.to_numeric(
        X["MonthlyCharges"], errors="raise"
    )

    total = X["TotalCharges"].astype("string").str.strip()
    total = total.mask(total.eq(""))

    X["TotalCharges"] = pd.to_numeric(
        total, errors="raise"
    ).to_numpy(dtype="float64", na_value=np.nan)

    return X

X_train = prepare_features(train_df)
X_valid = prepare_features(valid_df)

y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_valid)

print("Training customers:", len(X_train))
print("Validation customers:", len(X_valid))
print("Missing TotalCharges in training:", X_train["TotalCharges"].isna().sum())

print("\nPipeline Accuracy:", round(accuracy_score(y_valid, predictions), 4))

matrix = confusion_matrix(y_valid, predictions, labels=[0, 1])

print("\nConfusion Matrix:")
print(pd.DataFrame(
    matrix,
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"],
))
