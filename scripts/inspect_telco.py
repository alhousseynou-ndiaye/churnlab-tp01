
import hashlib
import json
from pathlib import Path

import pandas as pd

source = Path("data/raw/telco.csv")
expected_sha256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

if actual_sha256 != expected_sha256:
    raise ValueError("Source hash mismatch: investigate before continuing.")

df = pd.read_csv(source)

required = {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}
missing_columns = sorted(required - set(df.columns))

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

target_counts = df["Churn"].value_counts(dropna=False)

blank_total_charges = (
    df["TotalCharges"].astype("string").str.strip().eq("").sum()
)

report = {
    "source_file": source.as_posix(),
    "sha256": actual_sha256,
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "target_counts": {str(k): int(v) for k, v in target_counts.items()},
    "target_rates": {
        str(k): float(v / len(df)) for k, v in target_counts.items()
    },
    "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
    "blank_total_charges": int(blank_total_charges),
    "dtypes": {name: str(dtype) for name, dtype in df.dtypes.items()},
}

out = Path("reports/tp01_data_overview.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
