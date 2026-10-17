"""Importable transformers, kept separate from CLI modules for joblib portability."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class BranchScaler(TransformerMixin, BaseEstimator):
    """Fit each branch on a CV training fold, balancing total branch variance."""

    def __init__(self, split_at: int, stats_weight: float = 1.0):
        self.split_at = split_at
        self.stats_weight = stats_weight

    def fit(self, x: np.ndarray, y: Any = None) -> BranchScaler:
        del y
        if x.ndim != 2 or not 0 < self.split_at < x.shape[1]:
            raise ValueError("Fusion requires two nonempty feature branches.")
        if self.stats_weight <= 0:
            raise ValueError("stats_weight must be positive.")
        self.n_features_in_ = x.shape[1]
        self.moment_scaler_ = StandardScaler().fit(x[:, : self.split_at])
        self.stats_scaler_ = StandardScaler().fit(x[:, self.split_at :])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Fusion feature dimensions changed after fitting.")
        moment = self.moment_scaler_.transform(x[:, : self.split_at])
        stats = self.stats_scaler_.transform(x[:, self.split_at :])
        return np.concatenate(
            [
                moment / np.sqrt(self.split_at),
                stats * self.stats_weight / np.sqrt(x.shape[1] - self.split_at),
            ],
            axis=1,
        )
