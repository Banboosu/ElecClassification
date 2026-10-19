import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tcn_moment import run_pooling_pilot as pilot
from tcn_moment.pooling import (
    forward_poolings,
    length_groups,
    pool_embeddings,
    representation,
    training_length_edges,
)


class PoolingPilotTests(unittest.TestCase):
    def test_masked_mean_max_and_valid_time_thirds(self):
        tokens = torch.tensor([[[1.0], [3.0], [5.0], [7.0], [1000000.0]]])
        mask = torch.zeros((1, 40))
        mask[:, :32] = 1
        values = pool_embeddings(tokens, mask, 8, 8)
        torch.testing.assert_close(values["mean"], torch.tensor([[4.0]]))
        torch.testing.assert_close(values["max"], torch.tensor([[7.0]]))
        torch.testing.assert_close(values["thirds"], torch.tensor([[2.0, 5.0, 7.0]]))

    def test_incomplete_tail_excluded_and_empty_third_zero(self):
        tokens = torch.tensor([[[1.0], [3.0], [10000.0]]])
        mask = torch.zeros((1, 24))
        mask[:, :19] = 1
        out = pool_embeddings(tokens, mask, 8, 8)
        torch.testing.assert_close(out["mean"], torch.tensor([[2.0]]))
        torch.testing.assert_close(out["max"], torch.tensor([[3.0]]))
        torch.testing.assert_close(out["thirds"], torch.tensor([[1.0, 3.0, 0.0]]))

    def test_pooling_operator_padding_invariance(self):
        tokens = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        mask = torch.ones((1, 32))
        a = pool_embeddings(tokens, mask, 8, 8)
        padded = torch.cat([tokens, torch.full((1, 8, 3), 1e8)], dim=1)
        b = pool_embeddings(padded, torch.cat([mask, torch.zeros((1, 64))], dim=1), 8, 8)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_constant_and_single_complete_patch_finite(self):
        x = torch.full((2, 3, 4), 5.0)
        mask = torch.zeros((2, 24))
        mask[:, :8] = 1
        out = pool_embeddings(x, mask, 8, 8)
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
        torch.testing.assert_close(out["thirds"][:, 4:], torch.zeros((2, 8)))

    def test_invalid_masks_shapes_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            pool_embeddings(torch.ones((1, 3, 2)), torch.zeros((1, 24)), 8, 8)
        with self.assertRaises(ValueError):
            pool_embeddings(torch.ones((1, 2, 2)), torch.ones((1, 24)), 8, 8)
        with self.assertRaises(ValueError):
            pool_embeddings(torch.full((1, 3, 2), torch.nan), torch.ones((1, 24)), 8, 8)

    def test_forward_keeps_encoder_reduction_none(self):
        class Model:
            patch_len = 8
            config = SimpleNamespace(patch_stride_len=8)
            head = SimpleNamespace(linear=SimpleNamespace(in_features=2))

            def embed(self, *, x_enc, input_mask, reduction):
                assert reduction == "none"
                return SimpleNamespace(embeddings=torch.ones((len(x_enc), 1, 3, 2)))

        out = forward_poolings(Model(), torch.zeros((1, 1, 24)), torch.ones((1, 24)))
        self.assertEqual(out["thirds"].shape, (1, 6))

    def test_dimensions_and_training_only_length_bins(self):
        data = {
            "mean": np.zeros((3, 1024)),
            "max": np.ones((3, 1024)),
            "thirds": np.zeros((3, 3072)),
        }
        self.assertEqual(representation(data, "mean_max").shape, (3, 2048))
        self.assertEqual(representation(data, "thirds").shape, (3, 3072))
        edges = training_length_edges([18, 24, 30, 36])
        np.testing.assert_allclose(edges, [24, 30])
        np.testing.assert_array_equal(
            length_groups([18, 24, 25, 30, 31, 1000], edges), [0, 0, 1, 1, 2, 2]
        )
        with self.assertRaises(ValueError):
            training_length_edges([])

    def test_scope_six_cases_and_test_features_not_accessed(self):
        p = pilot.protocol(pilot.DEFAULT_PROTOCOL)
        self.assertEqual(len(pilot.cases(p)), 6)

        class Bundle:
            x_train = np.zeros((4, 8))
            mask_train = np.ones((4, 8))
            y_train = np.array([0, 1, 2, 0])
            ids_train = np.arange(4)
            lengths_train = np.full(4, 8)
            x_val = x_train[:2]
            mask_val = mask_train[:2]
            y_val = y_train[:2]
            ids_val = ids_train[:2]
            lengths_val = lengths_train[:2]

            @property
            def x_test(self):
                raise AssertionError("test encoding is forbidden")

        self.assertEqual(
            set(pilot.extraction_inputs(Bundle(), np.array([0, 1, 2]))), {"train", "validation"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            p["test_evaluated"] = True
            file = Path(tmp) / "p.json"
            file.write_text(json.dumps(p))
            with self.assertRaises(ValueError):
                pilot.protocol(file)

    def test_pilot_rule_must_beat_existing_optimized_head(self):
        p = pilot.protocol(pilot.DEFAULT_PROTOCOL)
        scores = {
            ("mean", "fusion"): 0.91,
            ("mean_max", "fusion"): 0.916,
            ("thirds", "fusion"): 0.915,
        }
        result = pilot.candidate_rule(p, scores, 0.9172)
        self.assertFalse(result["eligible_for_separately_frozen_replication"])
        scores[("mean_max", "fusion")] = 0.921
        self.assertTrue(
            pilot.candidate_rule(p, scores, 0.9172)["eligible_for_separately_frozen_replication"]
        )
        scores[("thirds", "fusion")] = 0.921
        self.assertEqual(pilot.candidate_rule(p, scores, 0.9172)["selected_pooling"], "mean_max")

    def test_empty_length_stratum_is_retained(self):
        result = pilot.stratified_metrics(
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([18, 18, 18]),
            [18, 18],
            ["0", "1", "2"],
        )
        self.assertEqual([r["n"] for r in result], [3, 0, 0])
        self.assertIsNone(result[1]["metrics"])

    def test_fit_persistence_uses_value_then_path(self):
        p = pilot.protocol(pilot.DEFAULT_PROTOCOL)

        class Search:
            best_score_ = 1.0
            best_estimator_ = object()

            def __init__(self, **kwargs):
                self.best_params_ = {"svc__C": 1.0, "svc__gamma": 0.05}
                self.backend_timings_ = {"unit": True}
                self.cv_results_ = {"mean_test_score": [1.0]}

            def fit(self, x, y):
                return self

            def predict(self, x):
                return np.arange(len(x)) % 3

        with tempfile.TemporaryDirectory() as tmp:
            p = copy.deepcopy(p)
            p["output_dir"] = tmp
            pilot.root_directory(p).mkdir()
            values = {
                "mean": np.zeros((9, 1024), dtype=np.float32),
                "max": np.ones((9, 1024), dtype=np.float32),
                "statistics": np.zeros((9, 15), dtype=np.float32),
                "labels": np.arange(9) % 3,
                "sample_ids": np.array([f"id{i}" for i in range(9)]),
                "lengths": np.arange(18, 27),
            }

            def dump(value, path):
                self.assertIs(value, Search.best_estimator_)
                Path(path).write_bytes(b"synthetic estimator")

            with (
                patch.object(pilot, "FastRBFGridSearchCV", Search),
                patch.object(pilot, "_atomic_joblib_dump", side_effect=dump),
            ):
                pilot.fit_case(
                    p,
                    {"protocol_sha256": "unit"},
                    {"train": values, "validation": values},
                    {
                        "classes": ["0", "1", "2"],
                        "train_ids_sha256": "unit",
                        "length_edges": [20, 23],
                    },
                    "mean_max",
                    "fusion",
                )
            self.assertEqual(
                pilot.read(pilot.case_directory(p, "mean_max", "fusion") / "status.json")["status"],
                "completed",
            )


if __name__ == "__main__":
    unittest.main()
