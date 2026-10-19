"""E06 controls: synthetic inputs only, no project curves or outcomes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment import tcn_moment_complement as experiment
from tcn_moment.training_utils import seed_everything


class ComplementTests(unittest.TestCase):
    def test_scope_rejects_test_seed_budget_and_expansion(self):
        base = json.loads(experiment.DEFAULT.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "protocol.json"
            file.write_text(json.dumps(base))
            self.assertEqual(experiment.protocol(file)["seeds"], [42, 44, 45, 46])
            for key, value in (("test_evaluated", True), ("automatic_next_stage", True),
                               ("pi_wake_enabled", True), ("claim_total_label_efficiency", True),
                               ("seeds", [42, 43, 44, 45]), ("train_fraction", 0.4),
                               ("maximum_head_fits", 24)):
                file.write_text(json.dumps({**base, key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    experiment.protocol(file)
            other = copy.deepcopy(base)
            other["head"]["epochs"] = 200
            file.write_text(json.dumps(other))
            with self.assertRaises(ValueError):
                experiment.protocol(file)
            other = copy.deepcopy(base)
            other["success_rules"]["minimum_positive_seeds_each"] = 1
            file.write_text(json.dumps(other))
            with self.assertRaises(ValueError):
                experiment.protocol(file)

    def test_tcn_extraction_and_zero_residual_restore(self):
        result = experiment.synthetic_check()
        self.assertTrue(result["pooled_logits_equal"])
        self.assertTrue(result["zero_residual_restores_parent"])
        self.assertFalse(result["project_data_used"])
        seed_everything(torch, 42)
        model = pilot.LegacyTCN(masked=True).eval()
        d = {"x": np.ones((5, 64), np.float32), "mask": np.ones((5, 64), np.float32),
             "y": np.array([0, 1, 2, 0, 1])}
        before = pilot.state_hash(model)
        features, logits = experiment.extract_tcn(model, d, 3, "cpu")
        self.assertEqual(features.shape, (5, 128))
        with torch.no_grad():
            expected = model(torch.from_numpy(d["x"][:, None, :]), torch.from_numpy(d["mask"]))
        torch.testing.assert_close(torch.from_numpy(logits), expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(before, pilot.state_hash(model))

    def test_scalers_fit_train_only_and_zero_control_dimensions_match(self):
        rng = np.random.default_rng(4)
        t, v = rng.normal(size=(8, 128)), rng.normal(size=(4, 128)) + 1000
        a, b = rng.normal(size=(8, 1024)), rng.normal(size=(4, 1024)) + 500
        x, y, scalers = experiment.branch_features(t, v, a, b)
        x2, y2, other = experiment.branch_features(t, v * 100, a, b * 100)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(scalers["side"].mean_, other["side"].mean_)
        np.testing.assert_array_equal(scalers["tcn"].mean_, other["tcn"].mean_)
        self.assertGreater(np.max(np.abs(y2 - y)), 1)
        z, q, control = experiment.branch_features(t, v, None, None)
        self.assertEqual(z.shape, x.shape)
        self.assertEqual(q.shape, y.shape)
        self.assertTrue(np.all(z[:, 128:] == 0))
        self.assertIsNone(control["side"])
        np.testing.assert_array_equal(z[:, :128], x[:, :128])
        with self.assertRaises(ValueError):
            experiment.branch_features(t, v, a[:, :-1], b)
        with self.assertRaises(FloatingPointError):
            experiment.branch_features(t * float("nan"), v, None, None)

    def test_all_arms_share_initialization_and_can_learn(self):
        hashes = []
        torch.set_num_threads(4)
        for _ in experiment.CONDITIONS:
            seed_everything(torch, 44)
            model = experiment.ResidualHead()
            hashes.append(pilot.state_hash(model))
            parent = torch.randn(6, 3)
            x = torch.randn(6, 1152)
            self.assertTrue(torch.equal(model(x, parent), parent))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            old = model.output.weight.detach().clone()
            loss = torch.nn.functional.cross_entropy(model(x, parent), torch.tensor([0, 1, 2, 0, 1, 2]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            self.assertFalse(torch.equal(old, model.output.weight))
        self.assertEqual(len(set(hashes)), 1)

    def test_archive_requires_order_dimension_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "features.npz"
            ids = np.array(["a", "b"])
            experiment.npz_save(file, sample_ids=ids, features=np.ones((2, 1024), np.float32))
            self.assertEqual(experiment.checked_archive(file, ids).shape, (2, 1024))
            with self.assertRaises(ValueError):
                experiment.checked_archive(file, ids[::-1])
            with self.assertRaises(ValueError):
                experiment.checked_archive(file, ids, [2, 12])
            experiment.npz_save(file, sample_ids=ids, features=np.full((2, 1024), np.nan))
            with self.assertRaises(FloatingPointError):
                experiment.checked_archive(file, ids)

    def test_selected_ids_must_match_original_tcn_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "parent.json"
            base_file = Path(tmp) / "base.json"
            base_file.write_text("{}")
            file.write_text(json.dumps({"parent_protocol": str(base_file), "split_manifests": {"42": "unused"}}))
            d = {"train": {"ids": np.array([f"i{i}" for i in range(30)]),
                            "y": np.tile(np.arange(3), 10), "x": np.ones((30, 32)),
                            "mask": np.ones((30, 32))},
                 "validation": {"ids": np.array(["v0", "v1", "v2"]), "y": np.arange(3)}}
            p = {"parent_protocol": str(file), "parent_root": tmp, "parent_condition": "case",
                 "train_fraction": 0.1, "selected_train_count": 3, "validation_count": 3}
            parent = experiment.parent_case(p, 42)
            parent.mkdir(parents=True)
            ix = experiment.stratified_train_subset_indices(d["train"]["y"], d["train"]["ids"], 0.1, 42)
            np.save(parent / "train_ids.npy", d["train"]["ids"][ix])
            with patch.object(pilot, "load_train_validation", side_effect=lambda _: copy.deepcopy(d)):
                selected, full_ids, _ = experiment.load_selected_data(p, 42)
                self.assertEqual(len(selected["train"]["ids"]), 3)
                self.assertEqual(len(full_ids), 30)
                np.save(parent / "train_ids.npy", np.array(["wrong"] * 3))
                with self.assertRaises(ValueError):
                    experiment.load_selected_data(p, 42)

    def test_success_requires_all_fixed_references_and_directions(self):
        p = {"seeds": [42, 44, 45, 46]}
        rows = []
        for seed in p["seeds"]:
            for name, value in (("original_tcn", .93), ("tcn_only", .931),
                                ("tcn_moment", .94), ("tcn_random", .932)):
                rows.append({"seed": seed, "condition": name, "metrics": {"macro_f1": value}})
        self.assertTrue(experiment.summarize_rows(rows, p)["all_success_rules_passed"])
        for row in rows:
            if row["condition"] == "tcn_random":
                row["metrics"]["macro_f1"] = .939
        self.assertFalse(experiment.summarize_rows(rows, p)["all_success_rules_passed"])
        self.assertFalse(experiment.paired([3, -1, -1, 3], .5, 3)["passed"])

    def test_budget_exhaustion_fails_instead_of_expanding(self):
        with self.assertRaises(TimeoutError):
            experiment.check_deadline(0)


if __name__ == "__main__":
    unittest.main()
