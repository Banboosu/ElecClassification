from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from tcn_moment.config import load_config
from tcn_moment.io_utils import sha256_file
from tcn_moment.train_baselines import FEATURE_NAMES
from tcn_moment.train_feature_fusion import (
    BranchScaler,
    assemble_features,
    build_search,
    raw_statistics,
    validate_cache,
    validation_gate,
    verified_checkpoint,
)


class FeatureFusionTests(unittest.TestCase):
    def test_branch_scaling_balances_variance_and_never_refits_on_validation(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(100, 12)) * np.arange(1, 13)
        scaler = BranchScaler(10).fit(train)
        transformed = scaler.transform(train)
        np.testing.assert_allclose(transformed[:, :10].var(axis=0).sum(), 1)
        np.testing.assert_allclose(transformed[:, 10:].var(axis=0).sum(), 1)
        before = scaler.stats_scaler_.mean_.copy()
        validation = train[:5] + 1000
        shifted = scaler.transform(validation)
        np.testing.assert_array_equal(before, scaler.stats_scaler_.mean_)
        self.assertGreater(float(shifted.mean()), 10)
        with self.assertRaises(ValueError):
            scaler.transform(np.zeros((1, 13)))

    def test_assemble_selects_raw_mean_std_without_losing_moment(self):
        moment = np.arange(12).reshape(3, 4)
        stats = np.arange(45).reshape(3, 15)
        fused = assemble_features("fusion_meanstd_svm", moment, stats)
        np.testing.assert_array_equal(fused[:, :4], moment)
        np.testing.assert_array_equal(
            fused[:, 4:],
            stats[:, [FEATURE_NAMES.index("mean"), FEATURE_NAMES.index("std")]],
        )
        self.assertEqual(assemble_features("fusion_stats_svm", moment, stats).shape, (3, 19))
        with self.assertRaises(ValueError):
            assemble_features("moment_tuned_svm", None, stats)

    def test_search_uses_fold_local_scaling_and_common_folds(self):
        config = load_config("configs/experiments/improvements/feature_fusion.yaml")
        tuned = build_search("moment_tuned_svm", 4, 4, config, n_jobs=1, seed=42)
        fusion = build_search("fusion_stats_svm", 19, 4, config, n_jobs=1, seed=42)
        paper = build_search("moment_paper_svm", 4, 4, config, n_jobs=1, seed=42)
        self.assertIsInstance(tuned.estimator, Pipeline)
        self.assertIsInstance(fusion.estimator.named_steps["scale"], BranchScaler)
        self.assertEqual(paper.scoring, "accuracy")
        self.assertEqual(paper.param_grid["C"], list(config.svm.c_values))
        x = np.arange(120).reshape(30, 4)
        y = np.repeat([0, 1, 2], 10)
        for a, b in zip(tuned.cv.split(x, y), paper.cv.split(x, y), strict=True):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
        # Exercise sklearn clone/fit/joblib-compatible estimator API with a tiny grid.
        tiny = GridSearchCV(fusion.estimator, {"svc__C": [1]}, cv=2)
        tiny.fit(np.column_stack([x, np.tile(x, (1, 4))[:, :15]]), y)
        self.assertEqual(tiny.predict(np.zeros((2, 19))).shape, (2,))

    def test_cache_rejects_protocol_sample_order_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            features = np.ones((2, 4), dtype=np.float32)
            ids = np.array(["a", "b"])
            path = directory / "train.npz"
            np.savez(path, features=features, sample_ids=ids)
            manifest = {
                "signature": {"version": 1},
                "splits": {
                    "train": {"sha256": sha256_file(path), "shape": [2, 4]},
                },
            }
            (directory / "manifest.json").write_text(json.dumps(manifest))
            actual = validate_cache(directory, {"version": 1}, {"train": ids})
            np.testing.assert_array_equal(actual["train"], features)
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                validate_cache(directory, {"version": 2}, {"train": ids})
            with self.assertRaisesRegex(ValueError, "sample order"):
                validate_cache(directory, {"version": 1}, {"train": ids[::-1]})
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                validate_cache(directory, {"version": 1}, {"train": ids})

    def test_saved_fusion_transformer_loads_in_a_fresh_process(self):
        import subprocess
        import sys

        import joblib

        self.assertEqual(BranchScaler.__module__, "tcn_moment.feature_transforms")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scaler.joblib"
            fitted = BranchScaler(2).fit(np.arange(24).reshape(6, 4))
            joblib.dump(fitted, path)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import joblib,numpy as np,sys; "
                        "m=joblib.load(sys.argv[1]); assert m.transform(np.ones((2,4))).shape==(2,4)"
                    ),
                    str(path),
                ],
                check=True,
                capture_output=True,
            )

    def test_checkpoint_is_pinned_and_checksummed_without_network(self):
        from unittest.mock import patch

        config = load_config("configs/experiments/improvements/feature_fusion.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            weight = directory / "fixedrevision" / "model.safetensors"
            weight.parent.mkdir()
            weight.write_bytes(b"synthetic weights")
            record = {
                "model_id": config.model.model_id,
                "filename": weight.name,
                "revision": "fixedrevision",
                "bytes": weight.stat().st_size,
                "sha256": sha256_file(weight),
            }
            manifest = directory / "fingerprint.json"
            manifest.write_text(json.dumps(record))
            with patch("huggingface_hub.try_to_load_from_cache", return_value=str(weight)):
                self.assertEqual(verified_checkpoint(config, manifest), record)
                weight.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    verified_checkpoint(config, manifest)

    def test_validation_gate_ignores_poisoned_test_metrics(self):
        class TestMustNotBeRead(dict):
            def __getitem__(self, key):
                raise AssertionError("Test metrics were accessed")

        rows = []
        for fraction in (0.01, 0.05, 0.1):
            for name, score in [
                ("stats_rf", 0.80),
                ("stats_svm", 0.79),
                ("moment_tuned_svm", 0.78),
                ("fusion_meanstd_svm", 0.82),
                ("fusion_stats_svm", 0.805),
            ]:
                rows.append(
                    {
                        "model": name,
                        "train_fraction": fraction,
                        "validation_metrics": {"macro_f1": score},
                        "test_metrics": TestMustNotBeRead(),
                    }
                )
        gate = validation_gate(rows)
        self.assertTrue(gate["candidates"]["fusion_meanstd_svm"]["expand"])
        self.assertFalse(gate["candidates"]["fusion_stats_svm"]["expand"])
        self.assertIn("stats_rf", gate["models_to_expand"])

    def test_raw_statistics_reload_unnormalized_data_and_exclude_test_features(self):
        from dataclasses import replace
        from unittest.mock import patch

        from tcn_moment.data import load_dataset

        # Use synthetic input and a private split path, never the real dataset.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            csv_path = directory / "tiny.csv"
            lines = ["charging_station_id InsertedColumn charging_powers_str"]
            for i in range(90):
                values = ",".join(str(50 + i + j * 2) for j in range(24))
                lines.append(f"id{i} {i % 3} {values}")
            csv_path.write_text("\n".join(lines))
            config = load_config("configs/experiments/improvements/feature_fusion.yaml")
            config = replace(
                config,
                data=replace(
                    config.data,
                    csv_path=csv_path,
                    split_path=directory / "split.json",
                ),
            )
            bundle = load_dataset(config.data)
            observed = []

            def recording_load(data):
                observed.append(data.normalize)
                return load_dataset(data)

            with patch("tcn_moment.train_feature_fusion.load_dataset", recording_load):
                stats = raw_statistics(config, bundle)
            self.assertEqual(observed, ["none"])
            self.assertEqual(set(stats), {"train", "validation"})
            self.assertTrue((stats["train"][:, FEATURE_NAMES.index("mean")] > 50).all())


if __name__ == "__main__":
    unittest.main()
