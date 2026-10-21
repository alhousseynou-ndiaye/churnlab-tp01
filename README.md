# ChurnLab - Reproducible Customer Churn Project

## Project Overview

This project is part of the ESILV Machine Learning Operations course.

The objective is to build a reproducible machine learning project for customer churn prediction.

The business question is: Can we predict which telecom customers are likely to leave the company?

The target variable is `Churn`, with two possible values: `Yes` and `No`.

TP01 focuses on environment reproducibility, data inspection and Git collaboration. Model training will be covered in later practical sessions.

## Requirements

- Python 3.12
- uv
- Git

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/alhousseynou-ndiaye/churnlab-tp01.git
cd churnlab-tp01
```

Install the dependencies:

```bash
uv sync --locked
```

Verify the package:

```bash
uv run python -c "import churnlab; print(churnlab.__version__)"
```

## Dataset

The project uses the IBM Telco Customer Churn dataset.

Source repository:
https://github.com/IBM/telco-customer-churn-on-icp4d

The dataset is retrieved from a specific Git commit to ensure source consistency.

Expected SHA-256:

`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

Download and verify the dataset:

```bash
uv run python scripts/download_data.py
```

Inspect the dataset:

```bash
uv run python scripts/inspect_telco.py
```

The dataset contains 7,043 customer records and 21 columns.

## Project Structure

| Directory       | Purpose                      |
| --------------- | ---------------------------- |
| src/churnlab/   | Python package source code   |
| scripts/        | Executable scripts           |
| data/raw/       | Original input data          |
| data/processed/ | Future processed datasets    |
| evidence/tp01/  | Reproducibility evidence     |
| reports/        | Generated inspection reports |
| artifacts/      | Future trained models        |

## Environment Reproducibility

The project uses `pyproject.toml` to declare dependencies and `uv.lock` to record resolved package versions.

Record the environment:

```bash
uv run python scripts/record_environment.py
```

The environment information is saved in `evidence/tp01/environment.json`.

## Data Limitations

The dataset represents a customer snapshot, not a chronological event log.

It does not provide a complete customer event history or randomized retention interventions.

The original dataset contains whitespace-only values in `TotalCharges`.

Data cleaning, feature engineering and model training will be addressed in later practical sessions.


## TP02 - Reusable Training Pipeline

### Objective

Build a reproducible customer churn prediction workflow using scikit-learn.

The target is Churn, where Yes = 1 and No = 0.

### Dataset Splitting

The dataset is split into three stratified subsets using random_state=42:

- Training: 4,225 customers (60%)
- Validation: 1,409 customers (20%)
- Test: 1,409 customers (20%)

The test set is reserved and is not used for model selection.

### First Baseline

Run the initial two-feature classification experiment:

```bash
uv run python scripts/train_baseline.py
```

This compares LogisticRegression with a majority-class DummyClassifier using tenure and MonthlyCharges.

### Full Training Pipeline

The complete pipeline uses six features:

Numeric:
- tenure
- MonthlyCharges
- TotalCharges

Categorical:
- Contract
- InternetService
- PaymentMethod

Numeric preprocessing uses median imputation and StandardScaler.

Categorical preprocessing uses most-frequent imputation and OneHotEncoder with handle_unknown="ignore".

The original CSV is not modified.

### Model Comparison

Run the complete training workflow:

```bash
uv run python scripts/compare_models.py
```

The workflow compares:

- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- DummyClassifier

All models use identical training and validation splits and the same six input features.

The primary selection metric is Average Precision, computed on validation scores.

Accuracy, precision, recall, F1-score and confusion matrices are also recorded.

RandomForest was selected with a validation Average Precision of approximately 0.6663.

### Saved Pipeline

The selected fitted pipeline is saved to:

artifacts/tp02/churn_pipeline.joblib

Metadata is saved to:

artifacts/tp02/metadata.json

The artifact contains the fitted preprocessing and classifier.

### Inference

First run the training workflow to generate the artifact and inference reference files.

Then run:

```bash
uv run python scripts/predict_saved.py
```

The script loads the saved pipeline and verifies that its prediction scores match the original scores within an absolute tolerance of 1e-8.

No model training is performed during inference.

Unknown categorical values are ignored by the encoder. Missing numeric inputs are handled using training-derived imputation statistics.

Malformed nonblank numeric values and missing required features produce errors.

### Reliability Tests

Run:

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

The six reliability tests verify:

- Disjoint customer allocations
- Training-only imputation statistics
- No changes to fitted statistics during prediction
- Missing numeric and unknown categorical inputs
- Missing required feature detection
- Prediction consistency after model reloading

### Reproducibility

The project uses pyproject.toml and uv.lock for environment management.

TP02 evidence is available in evidence/tp02/.

The original dataset can be obtained using:

```bash
uv run python scripts/download_data.py
```

Model artifacts and generated reports are excluded from Git and must be regenerated or distributed separately.

### Limitations

Model selection uses validation data only. Final test evaluation has not been performed.

Prediction probabilities are not guaranteed to be calibrated.

Handling unseen categories does not guarantee reliable predictions for those categories.

Training duration measurements are descriptive and depend on the execution environment.



