"""Freeze and audit the bounded E03 LoRA pilot; never launch replication/test."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from tcn_moment.analyze_e01c import DEFAULT_PROTOCOL as PARENT_PROTOCOL
from tcn_moment.analyze_e01c import verify as verify_parent
from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.config import load_config
from tcn_moment.experiment import utc_now
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.train_feature_fusion import source_hashes

DEFAULT_PROTOCOL = "configs/experiments/improvements/e03_peft_pilot_protocol.json"
FILES = ("moment_peft.py", "train_e03.py", "analyze_e03.py")
PACKAGES = ("torch", "momentfm", "transformers", "numpy", "scikit-learn")
CONDITIONS = [
    ("statistics_mlp", "statistics", 0),
    ("frozen_moment_mlp", "moment", 0),
    ("frozen_fusion_mlp", "fusion", 0),
    ("lora8_moment_mlp", "moment", 8),
    ("lora8_fusion_mlp", "fusion", 8),
    ("lora16_moment_mlp", "moment", 16),
    ("lora16_fusion_mlp", "fusion", 16),
]


def read_protocol(path: Path):
    p = json.loads(path.read_text())
    if (
        p["experiment_id"] != "E03P"
        or p["test_evaluated"]
        or p["seeds"] != [43]
        or p["fractions"] != [0.4]
        or [(c["name"], c["branch"], c["rank"]) for c in p["conditions"]] != CONDITIONS
    ):
        raise ValueError("Unexpected pilot scope; no test or automatic expansion permitted.")
    t, a, rules = p["training"], p["architecture"], p["decision_rules"]
    required = {
        "maximum_epochs": 30,
        "minimum_epochs": 10,
        "patience": 7,
        "head_learning_rate": 0.001,
        "adapter_learning_rate": 0.0001,
        "effective_batch_size": 128,
        "physical_batch_size": 64,
        "frozen_head_batch_size": 128,
        "precision": "amp_fp16",
        "gradient_checkpointing": False,
    }
    if any(t[k] != v for k, v in required.items()):
        raise ValueError("Training budget differs from the declared pilot.")
    if a["lora_targets"] != ["q", "v"] or a["encoder_blocks"] != 24 or a["hidden_dim"] != 128:
        raise ValueError("Unexpected adapter/head architecture.")
    if (
        rules["automatically_evaluate_test"]
        or rules["automatically_expand_replication"]
        or rules["minimum_gain_over_same_head_frozen_fusion_pp"] != 0.5
        or rules["minimum_gain_over_best_nonadapted_reference_pp"] != 0.5
        or rules["minimum_gain_over_rank_matched_lora_moment_pp"] != 0.25
    ):
        raise ValueError("Pilot gates changed.")
    return p


def plan_directory(p):
    return Path(p["output_dir"]) / f"{p['suite']}_analysis"


def run_directory(p, name):
    return Path(p["output_dir"]) / f"{p['suite']}_{name}_seed43_fraction40"


def verify(path: Path):
    p = read_protocol(path)
    f = json.loads((plan_directory(p) / "frozen_protocol.json").read_text())
    if f["protocol"] != p or f["protocol_sha256"] != sha256_file(path):
        raise ValueError("E03 pilot protocol changed after freezing.")
    for filename, digest in f["file_sha256"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"Frozen E03 input changed: {filename}")
    if f["packages"] != {k: importlib.metadata.version(k) for k in PACKAGES}:
        raise ValueError("E03 package versions changed.")
    return f


def freeze(path: Path):
    p = read_protocol(path)
    directory = plan_directory(p)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "frozen_protocol.json").exists():
        verify(path)
        print("Existing E03 pilot freeze verified.")
        return
    if any(run_directory(p, c["name"]).exists() for c in p["conditions"]):
        raise ValueError("Do not freeze after pilot outcomes/runs already exist.")
    parent = verify_parent(Path(PARENT_PROTOCOL))
    smoke_path = Path(p["smoke_report"])
    smoke = json.loads(smoke_path.read_text())
    if (
        smoke["status"] != "passed"
        or not smoke["synthetic_input_only"]
        or smoke["test_evaluated"]
        or not smoke["all_frozen_parameters_unchanged"]
        or not smoke["adapter_gradients_finite_and_nonzero"]
        or smoke["zero_adapter_max_logit_error"] != 0
        or smoke["reload_max_logit_error"] != 0
        or smoke["precision"] != p["training"]["precision"]
        or smoke["batch_size"] != p["training"]["physical_batch_size"]
        or smoke["source_sha256"] != sha256_file(Path(__file__).with_name("moment_peft.py"))
    ):
        raise ValueError("Real model smoke and current core source/precision/batch must match.")
    config = load_config(p["training_config"])
    files = [
        path,
        smoke_path,
        Path(p["training_config"]),
        config.model.config_path,
        Path("scripts/run_e03_remote.sh"),
        Path("scripts/check_e03_peft.py"),
        Path("tests/test_moment_peft.py"),
        Path("tests/test_e03_pilot.py"),
        Path("docs/experiment_records/e03_peft_pilot_protocol_20261017.md"),
    ]
    files.extend(Path(name) for name in parent["file_sha256"])
    files.extend(Path(__file__).with_name(name) for name in [*FILES, *source_hashes()])
    fusion = Path(p["parent_fusion_run"]) / "fusion_stats_svm_fraction_40"
    control = Path(p["parent_control_run"]) / "statistics_matched_kernel_fraction_40"
    files.extend(
        root / name
        for root in (fusion, control)
        for name in ("metrics.json", "validation_predictions.csv")
    )
    references = {
        "parent_pretrained_fusion_svm": json.loads((fusion / "metrics.json").read_text())[
            "validation_metrics"
        ]["macro_f1"],
        "parent_matched_statistics_svm": json.loads((control / "metrics.json").read_text())[
            "validation_metrics"
        ]["macro_f1"],
    }
    atomic_write_json(
        directory / "frozen_protocol.json",
        {
            "frozen_at": utc_now(),
            "protocol": p,
            "protocol_sha256": sha256_file(path),
            "file_sha256": {str(name): sha256_file(name) for name in files},
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            "smoke": smoke,
            "reference_macro_f1": references,
            "boundary": "Single-seed exploratory pilot; prior reference validation seen; no automatic replication/test.",
        },
    )
    print(f"Frozen E03 pilot: seven conditions, seed43, 40%, max30 epochs; {directory}")


def judge_candidates(p, scores, references):
    rules = p["decision_rules"]
    strongest = max(scores[k] for k in ("statistics_mlp", "frozen_moment_mlp", "frozen_fusion_mlp"))
    strongest = max(strongest, *references.values())
    candidates = []
    for rank in (8, 16):
        score = scores[f"lora{rank}_fusion_mlp"]
        differences = {
            "over_same_head_frozen_fusion_pp": (score - scores["frozen_fusion_mlp"]) * 100,
            "over_best_nonadapted_reference_pp": (score - strongest) * 100,
            "over_rank_matched_lora_moment_pp": (score - scores[f"lora{rank}_moment_mlp"]) * 100,
        }
        passed = all(
            differences[key.removeprefix("minimum_gain_")] + 1e-12 >= rules[key]
            for key in (
                "minimum_gain_over_same_head_frozen_fusion_pp",
                "minimum_gain_over_best_nonadapted_reference_pp",
                "minimum_gain_over_rank_matched_lora_moment_pp",
            )
        )
        candidates.append(
            {"rank": rank, "macro_f1": score, **differences, "pilot_rule_passed": passed}
        )
    qualifying = [c for c in candidates if c["pilot_rule_passed"]]
    selected = None
    if qualifying:
        best = max(c["macro_f1"] for c in qualifying)
        selected = min(
            c["rank"]
            for c in qualifying
            if (best - c["macro_f1"]) * 100 <= rules["rank_tie_tolerance_pp"] + 1e-12
        )
    return {
        "best_nonadapted_reference_macro_f1": strongest,
        "candidates": candidates,
        "selected_rank_for_proposed_replication": selected,
        "automatic_replication_started": False,
    }


def summarize(path: Path):
    frozen = verify(path)
    p = frozen["protocol"]
    rows, scores, initial_heads = [], {}, set()
    parent_predictions = (
        Path(p["parent_fusion_run"]) / "fusion_stats_svm_fraction_40/validation_predictions.csv"
    )
    with parent_predictions.open(newline="") as handle:
        target = list(csv.DictReader(handle))
    expected_ids = [(r["sample_id"], r["true_index"]) for r in target]
    for condition in p["conditions"]:
        run = run_directory(p, condition["name"])
        row = json.loads((run / "metrics.json").read_text())
        if (
            json.loads((run / "status.json").read_text())["status"] != "completed"
            or row["test_evaluated"]
            or row["protocol_sha256"] != frozen["protocol_sha256"]
            or row["condition"] != condition
            or row["seed"] != 43
            or row["train_fraction"] != 0.4
            or sha256_file(run / "model_best.pt") != row["best_checkpoint_sha256"]
        ):
            raise ValueError("Incomplete/mismatched pilot condition.")
        np.testing.assert_array_equal(
            np.load(run / "train_ids.npy"),
            np.load(Path(p["parent_fusion_run"]) / "train_ids_40.npy"),
        )
        with (run / "validation_predictions.csv").open(newline="") as handle:
            predictions = list(csv.DictReader(handle))
        if [(r["sample_id"], r["true_index"]) for r in predictions] != expected_ids:
            raise ValueError("Pilot validation IDs or targets changed.")
        matrix = [[0] * 3 for _ in range(3)]
        for r in predictions:
            actual, predicted = int(r["true_index"]), int(r["predicted_index"])
            probabilities = np.asarray([float(r[f"probability_{i}"]) for i in range(3)])
            if (
                not np.isfinite(probabilities).all()
                or abs(probabilities.sum() - 1) > 1e-5
                or (probabilities < 0).any()
                or predicted != probabilities.argmax()
            ):
                raise ValueError("Invalid pilot probabilities/predictions.")
            matrix[actual][predicted] += 1
        if row["validation_metrics"]["confusion_matrix"] != matrix:
            raise ValueError("Pilot confusion matrix differs from predictions.")
        for key, actual in metrics_from_confusion(matrix).items():
            if abs(actual - row["validation_metrics"][key]) > 1e-12:
                raise ValueError("Pilot metric mismatch.")
        history = json.loads((run / "history.json").read_text())
        best = max(r["validation_macro_f1"] for r in history)
        if abs(best - row["validation_metrics"]["macro_f1"]) > 1e-12:
            raise ValueError("Best checkpoint does not match the epoch selection rule.")
        if condition["rank"] and (
            not row["all_frozen_parameters_unchanged"]
            or row["parameters"]["adapter_parameters"] != 98304 * condition["rank"]
        ):
            raise ValueError("Frozen backbone/adapters scope changed.")
        initial_heads.add(row["initial_head_sha256"])
        scores[condition["name"]] = row["validation_metrics"]["macro_f1"]
        rows.append(
            {
                "condition": condition["name"],
                "macro_f1_percent": scores[condition["name"]] * 100,
                "accuracy_percent": row["validation_metrics"]["accuracy"] * 100,
                "best_epoch": row["best_epoch"],
                "epochs": len(history),
                "trainable_parameters": row["parameters"]["trainable_parameters"],
                "training_seconds": row["training_seconds"],
            }
        )
    if len(initial_heads) != 1:
        raise ValueError("Heads did not share initialization/buffers.")
    result = {
        "audit_status": "passed",
        "recorded_at": utc_now(),
        "test_evaluated": False,
        "conditions": rows,
        **judge_candidates(p, scores, frozen["reference_macro_f1"]),
        "limitations": p["limitations"],
    }
    root = plan_directory(p)
    atomic_write_json(root / "summary.json", result)
    lines = [
        "# E03 exploratory pilot (validation only)",
        "",
        "| Condition | Macro-F1 % | Accuracy % | Best epoch |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['condition']} | {r['macro_f1_percent']:.2f} | {r['accuracy_percent']:.2f} | {r['best_epoch']} |"
        for r in rows
    )
    lines.extend(
        [
            "",
            f"Proposed replication rank: {result['selected_rank_for_proposed_replication']}",
            "No automatic replication or test evaluation.",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "summarize"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "freeze":
        freeze(path)
    elif args.action == "verify":
        verify(path)
        print("E03 pilot inputs verified.")
    else:
        summarize(path)


if __name__ == "__main__":
    main()
