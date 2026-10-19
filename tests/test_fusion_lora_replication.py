"""E07 scientific scope/selection controls; synthetic fixtures only."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tcn_moment import replicate_fusion_lora as experiment
from tcn_moment.io_utils import atomic_write_json, sha256_file


class FusionLoRAReplicationTests(unittest.TestCase):
    def test_scope_rejects_pilot_seed_test_extra_conditions_and_retuning(self):
        base = experiment.read_protocol(experiment.DEFAULT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            for key, value in (("seeds", [42, 43, 44, 45]), ("test_evaluated", True),
                               ("automatic_test", True), ("automatic_budget_expansion", True),
                               ("pi_wake_enabled", True), ("maximum_training_conditions", 16),
                               ("maximum_additional_epochs_total", 480), ("claim_total_label_efficiency", True)):
                path.write_text(json.dumps({**base, key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    experiment.read_protocol(path)
            for section, key, value in (("training", "head_learning_rate", .001),
                                        ("training", "maximum_epochs", 50),
                                        ("architecture", "rank", 16),
                                        ("architecture", "adapter_seed_offset", 0),
                                        ("success_rules", "minimum_positive_seeds_each", 1),
                                        ("technical_gate", "maximum_initial_prediction_disagreements", 100)):
                p = copy.deepcopy(base)
                p[section][key] = value
                path.write_text(json.dumps(p))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    experiment.read_protocol(path)

    def test_seed_specific_online_batch_and_optimizer_schedule(self):
        p = experiment.read_protocol(experiment.DEFAULT)
        for seed in p["seeds"]:
            pp = experiment.training_protocol(p, seed)
            self.assertEqual(pp["seeds"], [seed])
            self.assertEqual(pp["training"]["head_learning_rate"], .0001)
            self.assertEqual(pp["training"]["adapter_learning_rate"], .0001)
            self.assertEqual(pp["training"]["maximum_epochs"], 30)
            self.assertEqual(pp["training"]["minimum_epochs"], 30)
            self.assertEqual(pp["training"]["patience"], 31)
            self.assertEqual(pp["training"]["physical_batch_size"], 64)
            self.assertEqual(pp["training"]["frozen_head_batch_size"], 64)
            self.assertEqual(pp["training"]["effective_batch_size"], 128)
            self.assertEqual(pp["architecture"]["adapter_seed_offset"] + seed, 30000 + seed)
            self.assertEqual(pp["architecture"]["training_seed_offset"] + seed, 20000 + seed)
        with self.assertRaises(ValueError):
            experiment.training_protocol(p, 43)

    def test_selection_statistics_and_normalization_use_selected_train_only(self):
        p = {"train_fraction": .4}
        rng = np.random.default_rng(42)
        data = {"train": {"x": rng.normal(size=(30, 32)).astype(np.float32), "mask": np.ones((30, 32), np.float32),
                          "y": np.tile(np.arange(3), 10), "ids": np.array([f"a{i}" for i in range(30)])},
                "validation": {"x": rng.normal(size=(6, 32)).astype(np.float32), "mask": np.ones((6, 32), np.float32),
                               "y": np.tile(np.arange(3), 2), "ids": np.array([f"v{i}" for i in range(6)])}}
        before = data["train"]["x"].copy()
        a = experiment.select_arrays(p, 42, data)
        data["validation"]["x"] *= 1000
        b = experiment.select_arrays(p, 42, data)
        for key in a["train"]:
            np.testing.assert_array_equal(a["train"][key], b["train"][key])
        np.testing.assert_array_equal(before, data["train"]["x"])
        self.assertEqual(len(a["train"]["labels"]), 12)
        np.testing.assert_allclose(a["train"]["x"].mean(1), 0, atol=1e-6)
        np.testing.assert_allclose(a["train"]["x"].std(1), 1, atol=1e-6)
        self.assertFalse(np.array_equal(a["validation"]["statistics"], b["validation"]["statistics"]))
        self.assertEqual(set(a), {"train", "validation"})

    def test_migration_gates_are_not_relaxed_after_result(self):
        p = experiment.read_protocol(experiment.DEFAULT)
        a = np.eye(3, dtype=float)[np.arange(9) % 3]
        proof = experiment.migration_check(p, a, a.copy(), (.9, .9))
        self.assertEqual(proof["prediction_disagreements"], 0)
        b = a.copy()
        b[:2] = np.roll(b[:2], 1, axis=1)
        with self.assertRaises(ValueError):
            experiment.migration_check(p, a, b, (.9, .9))
        with self.assertRaises(ValueError):
            experiment.migration_check(p, a, a, (.9, .901))

    def test_all_four_gates_required_before_training(self):
        p = experiment.read_protocol(experiment.DEFAULT)
        with tempfile.TemporaryDirectory() as tmp:
            p.update(output_dir=tmp, suite="synthetic")
            f = {"protocol_sha256": "scope"}
            for seed in p["seeds"]:
                directory = experiment.root(p) / "technical_gates" / f"seed{seed}"
                directory.mkdir(parents=True)
                (directory / "pred.csv").write_text("synthetic")
                atomic_write_json(directory / "result.json", {"seed": seed, "status": "passed", "protocol_sha256": "scope",
                                                              "output_sha256": {"pred.csv": sha256_file(directory / "pred.csv")}})
            self.assertEqual(len(experiment.all_gates(p, f)), 4)
            target = experiment.root(p) / "technical_gates/seed46/result.json"
            target.unlink()
            with self.assertRaises(FileNotFoundError):
                experiment.all_gates(p, f)

    def test_predeclared_max_reference_and_both_gates(self):
        p = experiment.read_protocol(experiment.DEFAULT)
        scores, references = {}, {}
        for seed in p["seeds"]:
            scores[seed, "frozen_fusion"] = .92
            scores[seed, "lora8_fusion"] = .93
            references[str(seed)] = {name: {"macro_f1": .915}
                                     for name in p["success_rules"]["strongest_nonadapted_set"]
                                     if name != "continued_frozen_fusion"}
            references[str(seed)]["strong_tcn"] = {"macro_f1": .98}
        result = experiment.decisions(p, scores, references)
        self.assertTrue(result["replication_rule_passed"])
        self.assertLess(result["lora_minus_strong_tcn_context"]["mean_pp"], 0)
        for seed in p["seeds"]:
            references[str(seed)]["parent_fusion_nn"]["macro_f1"] = .929
        result = experiment.decisions(p, scores, references)
        self.assertTrue(result["continued_frozen_rule"]["rule_passed"])
        self.assertFalse(result["replication_rule_passed"])
        self.assertFalse(experiment.paired([3, -1, 3, -1], .5)["rule_passed"])

    def test_prediction_audit_supports_nn_and_tcn_formats(self):
        ids = np.array(["a", "b", "c", "d", "e", "f"])
        y = np.array([0, 1, 2, 0, 1, 2])
        prob = np.eye(3)[y] * .8 + .2 / 3
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "pred.csv"
            experiment.pilot.save_predictions(file, {"sample_ids": ids, "labels": y}, prob)
            metrics, loaded = experiment.read_predictions(file, ids, y)
            self.assertEqual(metrics["macro_f1"], 1)
            np.testing.assert_allclose(prob, loaded)
            with self.assertRaises(ValueError):
                experiment.read_predictions(file, ids[::-1], y)
            frame = experiment.pd.read_csv(file)
            frame = frame.rename(columns={"true_index": "y_true", "predicted_index": "y_pred",
                                           **{f"probability_{i}": f"p{i}" for i in range(3)}})
            frame.to_csv(file, index=False)
            self.assertEqual(experiment.read_predictions(file, ids, y)[0]["battery_AP"], 1)
            frame.loc[0, "p0"] = -1
            frame.to_csv(file, index=False)
            with self.assertRaises(ValueError):
                experiment.read_predictions(file, ids, y)

    def test_exhausted_original_budget_does_not_start_child(self):
        with patch.object(experiment.subprocess, "Popen") as start:
            with self.assertRaises(TimeoutError):
                experiment.launch_child(["prepare"], 0)
            start.assert_not_called()

    def test_child_failure_is_not_automatically_retried(self):
        process = unittest.mock.Mock()
        process.wait.return_value = 2
        process.poll.return_value = 2
        with patch.object(experiment.subprocess, "Popen", return_value=process) as start:
            with self.assertRaises(RuntimeError):
                experiment.launch_child(["prepare"], experiment.time.time() + 100)
            self.assertEqual(start.call_count, 1)


if __name__ == "__main__":
    unittest.main()
