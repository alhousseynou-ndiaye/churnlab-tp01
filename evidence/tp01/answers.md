# TP01 - Answers

## Part C3 - Data Inspection

### 1. Why is churn prediction a classification task?

Churn prediction is a binary classification task because the target variable has two categories: Yes and No. We want to predict whether a customer will leave the company.

### 2. What does the class balance suggest about using accuracy alone?

The dataset is imbalanced. About 73.46% of customers did not churn, while 26.54% did. A model predicting No for every customer would achieve 73.46% accuracy without identifying any churned customers. We should also consider precision, recall and F1-score.

### 3. Why can TotalCharges be read as a string?

TotalCharges contains 11 whitespace-only values. Because these values cannot be interpreted directly as numbers, pandas reads the column as a string.

### 4. Why may isna() fail to detect whitespace-only values?

The isna() function detects missing values such as NaN, but spaces are still considered valid strings. We can use strip() to remove whitespace and check whether the result is empty.

### 5. Which evidence proves that a teammate used the same dataset?

We can compare the dataset source URL, the Git commit and the SHA-256 hash. If the hashes match, we have strong evidence that the files contain identical bytes.

## Part D2 - Reproducibility

| Item               | Evidence                                     | Remaining limitation                                  |
| ------------------ | -------------------------------------------- | ----------------------------------------------------- |
| Code               | Git commit                                   | Uncommitted changes are not tracked by the commit     |
| Package selections | pyproject.toml and uv.lock                   | Different platforms may affect installation           |
| Environment        | environment-before.json and environment.json | Not all system and hardware details are captured      |
| Source data        | Source URL and SHA-256                       | The hash does not guarantee data quality              |
| Trained model      | Not available in TP01                        | Model training and tracking will be implemented later |

## Part E - Git Collaboration

Initial commit: `427b5d2`

Documentation commit: `fc9e261`

Pull Request:
https://github.com/alhousseynou-ndiaye/churnlab-tp01/pull/1

The README was updated in the docs/telco-onboarding branch.

The pull request contains the environment setup instructions, dataset provenance, SHA-256 verification and project structure.

The project was completed individually. The pull request was created successfully, but peer review is still pending.

## Part F - Git Recovery

### F1 - Discard an unstaged change

I added a temporary sentence to README.md and checked the changes using git diff.

I then used git restore README.md to discard the modification.

The working tree was clean after restoration.

### F2 - Unstage without deleting the change

I added another temporary sentence and staged the file using git add.

I used git restore --staged README.md to remove the change from the staging area.

The modification remained in the working directory.

Finally, I used git restore README.md to discard the temporary change and switched back to main.

### Difference

git restore README.md discards unstaged changes.

git restore --staged README.md removes changes from staging without deleting them from the working directory.
