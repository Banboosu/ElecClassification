from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tcn_moment.controlled_svm import StatisticalBranchScaler, fit_controlled_rbf
from tcn_moment.feature_transforms import BranchScaler


class ControlledSVMTests(unittest.TestCase):
    def setUp(self):
        x, y = make_classification(
            n_samples=60, n_features=8, n_informative=6, n_redundant=0, n_classes=3, random_state=42
        )
        self.x, self.y = x.astype(np.float32), y

    def fit_control(self, x, reference, kind="fusion"):
        return fit_controlled_rbf(
            x,
            self.y,
            reference,
            kind=kind,
            reference_dim=5,
            c_values=[0.1, 1],
            gamma_tokens=["reference_scale", 0.05, 5.0],
            cv_folds=3,
            n_jobs=1,
            device="cpu",
        )

    def test_statistics_geometry_is_exactly_the_original_branch(self):
        original = BranchScaler(5).fit_transform(self.x)
        statistics = StatisticalBranchScaler().fit_transform(self.x[:, 5:])
        np.testing.assert_array_equal(statistics, original[:, 5:])

    def test_reference_search_matches_original_pipeline(self):
        pipeline = Pipeline([("scale", BranchScaler(5)), ("svc", SVC())])
        expected = GridSearchCV(
            pipeline,
            {"svc__C": [0.1, 1], "svc__gamma": ["scale", 0.05, 5.0]},
            cv=StratifiedKFold(3),
            scoring=make_scorer(f1_score, average="macro", zero_division=0),
        ).fit(self.x, self.y)
        actual = self.fit_control(self.x, self.x)
        for fold in range(3):
            np.testing.assert_allclose(
                actual.cv_results_[f"split{fold}_test_score"],
                expected.cv_results_[f"split{fold}_test_score"],
                atol=1e-12,
                rtol=0,
            )
        np.testing.assert_array_equal(actual.predict(self.x), expected.predict(self.x))
        self.assertEqual(actual.best_params_["C"], expected.best_params_["svc__C"])

    def test_bandwidth_reference_never_fits_on_held_out_fold(self):
        reference = self.x.copy()
        original = self.fit_control(self.x[:, 5:], reference, "statistics")
        _, held_out = next(StratifiedKFold(3).split(self.x, self.y))
        reference[held_out] += 1000
        modified = self.fit_control(self.x[:, 5:], reference, "statistics")
        self.assertEqual(
            original.geometry_["folds"][0]["reference_scaler_sha256"],
            modified.geometry_["folds"][0]["reference_scaler_sha256"],
        )
        self.assertNotEqual(
            original.geometry_["folds"][1]["reference_scaler_sha256"],
            modified.geometry_["folds"][1]["reference_scaler_sha256"],
        )

    def test_final_statistics_model_needs_no_reference_at_prediction(self):
        result = self.fit_control(self.x[:, 5:], self.x, "statistics")
        self.assertEqual(result.best_estimator_.n_features_in_, 3)
        self.assertIsInstance(result.best_estimator_.named_steps["svc"].gamma, float)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "classifier.joblib"
            joblib.dump(result.best_estimator_, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(
                loaded.predict(self.x[:, 5:]), result.predict(self.x[:, 5:])
            )


if __name__ == "__main__":
    unittest.main()
