
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("data/raw/telco.csv")

train_df, remaining_df = train_test_split(
    df,
    test_size=0.4,
    stratify=df["Churn"],
    random_state=42
)

valid_df, test_df = train_test_split(
    remaining_df,
    test_size=0.5,
    stratify=remaining_df["Churn"],
    random_state=42
)

train_ids = set(train_df["customerID"])
valid_ids = set(valid_df["customerID"])
test_ids = set(test_df["customerID"])

assert train_ids.isdisjoint(valid_ids)
assert train_ids.isdisjoint(test_ids)
assert valid_ids.isdisjoint(test_ids)

print(f"Training: {len(train_df)} customers")
print(f"Validation: {len(valid_df)} customers")
print(f"Test: {len(test_df)} customers")

print("\nTraining churn distribution:")
print(train_df["Churn"].value_counts(normalize=True))

print("\nValidation churn distribution:")
print(valid_df["Churn"].value_counts(normalize=True))

print("\nTest churn distribution:")
print(test_df["Churn"].value_counts(normalize=True))



features = ["tenure", "MonthlyCharges"]

X_train = train_df[features]
X_valid = valid_df[features]

y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

models = {
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)
    matrix = confusion_matrix(y_valid, predictions, labels=[0, 1])

    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"],
        )
    )
