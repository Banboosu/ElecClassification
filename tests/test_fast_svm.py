from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tcn_moment.fast_svm import FastRBFGridSearchCV, gamma_value, squared_distances
from tcn_moment.feature_transforms import BranchScaler


class FastSVMTests(unittest.TestCase):
    def test_distances_and_gamma_match_double_precision_definitions(self):
        x = np.array([[1, 3], [2, 6], [4, 5]], dtype=np.float64)
        distances = squared_distances(x, None, "cpu")
        np.testing.assert_allclose(distances, ((x[:, None] - x[None]) ** 2).sum(axis=2))
        np.testing.assert_array_equal(distances.diagonal(), np.zeros(3))
        self.assertEqual(gamma_value("scale", x), 1 / (x.shape[1] * x.var()))
        self.assertEqual(gamma_value("auto", x), 0.5)
        self.assertEqual(gamma_value("scale", np.ones((3, 2))), 1)

    def test_cpu_search_matches_sklearn_with_fold_local_transforms(self):
        self.check_search_equivalence("cpu")

    def test_cuda_search_matches_sklearn_when_cuda_available(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.check_search_equivalence("cuda")

    def check_search_equivalence(self, device):
        x, y = make_classification(
            n_samples=90,
            n_features=8,
            n_informative=6,
            n_redundant=0,
            n_classes=3,
            random_state=42,
        )
        x = x.astype(np.float32)
        for transformer in (None, StandardScaler(), BranchScaler(6)):
            with self.subTest(device=device, transformer=type(transformer).__name__):
                estimator = SVC()
                prefix = ""
                if transformer is not None:
                    estimator = Pipeline([("scale", transformer), ("svc", estimator)])
                    prefix = "svc__"
                grid = {prefix + "C": [0.1, 1, 10], prefix + "gamma": ["scale", "auto", 0.1]}
                cv = StratifiedKFold(3, shuffle=False)
                scoring = (
                    "accuracy"
                    if transformer is None
                    else make_scorer(
                        f1_score,
                        average="macro",
                        zero_division=0,
                    )
                )
                expected = GridSearchCV(estimator, grid, cv=cv, scoring=scoring).fit(x, y)
                fast = FastRBFGridSearchCV(estimator, grid, cv=cv, scoring=scoring, n_jobs=1)
                fast.kernel_device = device
                fast.fit(x, y)
                self.assertEqual(fast.best_params_, expected.best_params_)
                for fold in range(3):
                    np.testing.assert_allclose(
                        fast.cv_results_[f"split{fold}_test_score"],
                        expected.cv_results_[f"split{fold}_test_score"],
                        atol=1e-12,
                        rtol=0,
                    )
                np.testing.assert_array_equal(fast.predict(x), expected.predict(x))
                svc = (
                    fast.best_estimator_.steps[-1][1]
                    if transformer is not None
                    else fast.best_estimator_
                )
                self.assertEqual(svc.kernel, "rbf")


if __name__ == "__main__":
    unittest.main()
