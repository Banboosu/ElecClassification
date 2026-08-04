from __future__ import annotations

import unittest

import numpy as np

from tcn_moment.config import load_config
from tcn_moment.evaluate_retrieval import (
    cosine_topk,
    l2_normalize,
    neighbor_overlap,
    raw_resampled_features,
    retrieval_metrics,
    statistical_features,
)


class RetrievalFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 0.0]], dtype=np.float32
        )
        self.input_mask = np.asarray([[1, 1, 1, 1, 1, 0]], dtype=np.uint8)
        self.observation_mask = np.asarray([[1, 1, 0, 1, 1, 0]], dtype=np.uint8)

    def test_hidden_values_do_not_affect_label_free_baselines(self) -> None:
        changed = self.values.copy()
        changed[0, 2] = 1_000_000.0

        raw = raw_resampled_features(
            self.values, self.input_mask, self.observation_mask, 8
        )
        changed_raw = raw_resampled_features(
            changed, self.input_mask, self.observation_mask, 8
        )
        stats = statistical_features(
            self.values, self.input_mask, self.observation_mask
        )
        changed_stats = statistical_features(
            changed, self.input_mask, self.observation_mask
        )

        np.testing.assert_allclose(raw, changed_raw)
        np.testing.assert_allclose(stats, changed_stats)

    def test_l2_normalize_handles_zero_rows(self) -> None:
        normalized = l2_normalize(np.asarray([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(normalized[0], [0.6, 0.8])
        np.testing.assert_allclose(normalized[1], [0.0, 0.0])


class RetrievalMetricTests(unittest.TestCase):
    def test_exact_cosine_topk_and_semantic_metrics(self) -> None:
        gallery = l2_normalize(
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        )
        queries = l2_normalize(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        indices, scores, _ = cosine_topk(gallery, queries, 2, 1)

        np.testing.assert_array_equal(indices, [[0, 1], [2, 1]])
        metrics = retrieval_metrics(
            indices,
            scores,
            gallery_labels=np.asarray([0, 0, 1]),
            query_labels=np.asarray([0, 1]),
            gallery_lengths=np.asarray([10, 11, 20]),
            query_lengths=np.asarray([10, 20]),
            k_values=(1, 2),
        )
        self.assertEqual(metrics["macro_precision_at_1"], 1.0)
        self.assertEqual(metrics["precision_at_2"], 0.75)

    def test_neighbor_overlap(self) -> None:
        clean = np.asarray([[1, 2, 3], [4, 5, 6]])
        changed = np.asarray([[1, 8, 3], [7, 5, 6]])
        result = neighbor_overlap(clean, changed, (1, 3))
        self.assertEqual(result["clean_neighbor_overlap_at_1"], 0.5)
        self.assertAlmostEqual(result["clean_neighbor_overlap_at_3"], 2 / 3)

    def test_retrieval_config_matches_thesis_protocol(self) -> None:
        config = load_config("configs/experiments/moment_retrieval_zero_shot.yaml")
        self.assertEqual(config.data.normalize, "none")
        self.assertEqual(config.retrieval.k_values, (1, 5, 10))
        self.assertEqual(config.retrieval.query_mask_rate, 0.4)
        self.assertEqual(config.retrieval.mask_seeds, (42, 43, 44, 45, 46))


if __name__ == "__main__":
    unittest.main()
