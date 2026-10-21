
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlab.features import numeric_features, prepare_features


class TestPipelineReliability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv("data/raw/telco.csv")
        cls.allocations = pd.read_csv(
            "reports/tp02_split_allocations.csv"
        )
        cls.pipeline = joblib.load(
            "artifacts/tp02/churn_pipeline.joblib"
        )

    def test_disjoint_splits(self):
        splits = {
            name: set(
                self.allocations.loc[
                    self.allocations["split"] == name,
                    "customerID"
                ]
            )
            for name in ["train", "validation", "test"]
        }

        self.assertTrue(splits["train"].isdisjoint(splits["validation"]))
        self.assertTrue(splits["train"].isdisjoint(splits["test"]))
        self.assertTrue(splits["validation"].isdisjoint(splits["test"]))
        self.assertEqual(
            self.allocations["customerID"].nunique(),
            len(self.df)
        )
        self.assertEqual(
            set(self.allocations["customerID"]),
            set(self.df["customerID"])
        )

    def test_training_only_imputation(self):
        train_ids = set(
            self.allocations.loc[
                self.allocations["split"] == "train",
                "customerID"
            ]
        )

        train_df = self.df[
            self.df["customerID"].isin(train_ids)
        ]

        X_train = prepare_features(train_df)

        numeric_pipeline = (
            self.pipeline.named_steps["preprocessor"]
            .named_transformers_["num"]
        )

        actual = numeric_pipeline.named_steps["imputer"].statistics_
        expected = X_train[numeric_features].median().to_numpy()

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_prediction_does_not_change_statistics(self):
        numeric_pipeline = (
            self.pipeline.named_steps["preprocessor"]
            .named_transformers_["num"]
        )

        imputer = numeric_pipeline.named_steps["imputer"]
        scaler = numeric_pipeline.named_steps["scaler"]

        before_medians = imputer.statistics_.copy()
        before_means = scaler.mean_.copy()

        X = prepare_features(self.df.head(5))
        self.pipeline.predict(X)

        np.testing.assert_array_equal(
            before_medians, imputer.statistics_
        )
        np.testing.assert_array_equal(
            before_means, scaler.mean_
        )

    def test_missing_numeric_and_unknown_category(self):
        sample = self.df.head(1).copy()

        sample["TotalCharges"] = " "
        sample["InternetService"] = "__unseen_service_tp02__"

        X = prepare_features(sample)
        scores = self.pipeline.predict_proba(X)

        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores.shape, (1, 2))

    def test_missing_required_feature(self):
        sample = self.df.head(1).drop(columns=["Contract"])

        with self.assertRaisesRegex(ValueError, "Missing features"):
            prepare_features(sample)

    def test_reloaded_scores(self):
        samples = pd.DataFrame(
            json.loads(
                Path("reports/tp02_inference_samples.json")
                .read_text(encoding="utf-8")
            )
        )

        expected = np.array(
            json.loads(
                Path("reports/tp02_expected_scores.json")
                .read_text(encoding="utf-8")
            )
        )

        X = prepare_features(samples)
        positive_index = list(self.pipeline.classes_).index(1)

        actual = self.pipeline.predict_proba(X)[:, positive_index]

        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
