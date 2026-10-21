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
