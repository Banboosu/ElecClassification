"""E03F: bounded, post-test exploratory strong-head LoRA; no test execution path."""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import shutil
import signal
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from tcn_moment import train_e03
from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import (
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.experiment import collect_environment, prepare_run, utc_now
from tcn_moment.head_diagnostics import GainFusionHead, fan_in_gain
from tcn_moment.head_diagnostics_stable import FixedHeadGradScaler
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json, sha256_file
from tcn_moment.moment_peft import (
    PeftClassifier,
    adapter_parameters,
    capture_learned_state,
    frozen_parameter_digest,
    optimizer_groups,
    restore_learned_state,
    trainable_summary,
)
from tcn_moment.replicate_gain_heads import normalize_masked_rows
from tcn_moment.run_e03d_stable import cases as parent_cases
from tcn_moment.run_e03d_stable import run_directory as parent_directory
from tcn_moment.run_e03d_stable import verify as verify_parent
from tcn_moment.train_baselines import extract_statistical_features
from tcn_moment.train_feature_fusion import verified_checkpoint
from tcn_moment.training_utils import capture_random_state, seed_everything

DEFAULT_PROTOCOL = "configs/experiments/improvements/e03f_strong_head_lora_protocol.json"
CONDITIONS = [
    ("frozen_fusion", "fusion", 0),
    ("lora8_fusion", "fusion", 8),
    ("frozen_moment", "moment", 0),
    ("lora8_moment", "moment", 8),
]
PACKAGES = ("torch", "momentfm", "transformers", "numpy", "scikit-learn")


def read_protocol(path):
    p = json.loads(Path(path).read_text())
    if (
        p["experiment_id"] != "E03F"
        or p["seed"] != 43
        or p["train_fraction"] != 0.4
        or p["test_evaluated"]
        or not p["prior_test_results_known"]
        or [(c["name"], c["branch"], c["rank"]) for c in p["conditions"]] != CONDITIONS
    ):
        raise ValueError("Unexpected strong-head exploratory scope.")
    expected = {
        "maximum_epochs": 30,
        "early_stopping": False,
        "head_learning_rate": 1e-4,
        "adapter_learning_rate": 1e-4,
        "physical_batch_size": 64,
        "effective_batch_size": 128,
        "evaluation_batch_size": 64,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "precision": "amp_fp16",
        "loss_scale": 1024,
        "loss_scale_growth_interval": 1000000000,
        "warmup_fraction": 0.1,
        "minimum_lr_ratio": 0.1,
    }
    if any(p["training"][k] != v for k, v in expected.items()):
        raise ValueError("Training budget/precision must not be retuned.")
    a = p["architecture"]
    if any(
        a[k] != v
        for k, v in {
            "rank": 8,
            "alpha": 16,
            "lora_dropout": 0.05,
            "adapter_seed": 30043,
            "training_seed": 20043,
            "all_conditions_online": True,
            "gradient_checkpointing": False,
            "lora_targets": ["q", "v"],
        }.items()
    ):
        raise ValueError("Architecture must match fixed rank8 design.")
    rules = p["success_rules"]
    if rules != {
        "minimum_over_continued_frozen_fusion_pp": 0.5,
        "minimum_over_strongest_nonadapted_pp": 0.5,
        "minimum_over_rank_matched_moment_pp": 0.25,
        "require_all": True,
        "automatic_replication": False,
        "automatic_test": False,
        "automatic_budget_expansion": False,
    }:
        raise ValueError("No changed gates or automatic next stage.")
    if (
        p["technical_gate"]["maximum_initial_prediction_disagreements"] != 1
        or p["technical_gate"]["maximum_initial_macro_f1_difference_pp"] != 0.05
    ):
        raise ValueError("Migration tolerances fixed before outcomes.")
    return p


def plan_directory(p):
    return Path(p["output_dir"]) / (p["suite"] + "_analysis")


def run_directory(p, name):
    return Path(p["output_dir"]) / (p["suite"] + "_" + name)


def parent_run(p, branch):
    pp = json.loads(Path(p["parent_protocol"]).read_text())
    return parent_directory(pp, {"name": branch + "_fanin_gain"})


def runtime_identity():
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch_build": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def verify(path):
    p = read_protocol(path)
    f = json.loads((plan_directory(p) / "frozen_protocol.json").read_text())
    if p != f["protocol"] or sha256_file(Path(path)) != f["protocol_sha256"]:
        raise ValueError("Protocol changed after freeze.")
    for filename, h in f["file_sha256"].items():
        if sha256_file(Path(filename)) != h:
            raise ValueError("Frozen input changed: " + filename)
    if (
        f["packages"] != {k: importlib.metadata.version(k) for k in PACKAGES}
        or f["runtime"] != runtime_identity()
    ):
        raise ValueError("Packages/GPU/precision changed.")
    if verified_checkpoint(load_config(p["training_config"])) != f["public_checkpoint"]:
        raise ValueError("Public checkpoint identity changed.")
    return f


def freeze(path):
    p = read_protocol(path)
    root = plan_directory(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(path)
    if any(run_directory(p, c[0]).exists() for c in CONDITIONS):
        raise ValueError("Freeze before new condition outcomes.")
    parent = verify_parent(Path(p["parent_protocol"]))
    smoke = json.loads(Path(p["smoke_report"]).read_text())
    if (
        smoke["status"] != "passed"
        or not smoke["synthetic_input_only"]
        or smoke["test_evaluated"]
        or smoke["batch_size"] != 64
        or smoke["runtime"] != runtime_identity()
        or smoke["source_sha256"] != sha256_file(Path(__file__))
    ):
        raise ValueError("Exact strong-head batch64 synthetic smoke required.")
    files = [Path(k) for k in parent["file_sha256"]]
    files += [
        Path(path),
        Path(__file__),
        Path(p["smoke_report"]),
        Path("src/tcn_moment/replicate_gain_heads.py"),
        Path("scripts/run_e03f_remote.sh"),
        Path("scripts/autodl_env.sh"),
        Path("tests/test_strong_head_lora.py"),
        Path("docs/experiment_records/e03f_strong_head_lora_protocol_20261019.md"),
    ]
    pp = parent["protocol"]
    references = dict(parent["reference_macro_f1"])
    for case in parent_cases(pp):
        r = parent_directory(pp, case)
        for name in (
            "model_best.pt",
            "metrics.json",
            "history.json",
            "status.json",
            "validation_predictions_200.csv",
        ):
            files.append(r / name)
        if case["mode"] == "fanin_gain":
            if json.loads((r / "status.json").read_text())["status"] != "completed":
                raise ValueError("Strong-head parent incomplete.")
            ck = torch.load(r / "model_best.pt", map_location="cpu", weights_only=True)
            references[case["name"]] = ck["validation_macro_f1"]
    prepared = Path(pp["output_dir"]) / (pp["suite"] + "_analysis/prepared")
    files.extend(
        prepared / n for n in ("manifest.json", "normalizer.json", "train.npz", "validation.npz")
    )
    old_pilot = json.loads(Path(pp["parent_protocol"]).read_text())
    files.append(Path(old_pilot["parent_fusion_run"]) / "split_manifest.json")
    f = {
        "protocol": p,
        "protocol_sha256": sha256_file(Path(path)),
        "frozen_at": utc_now(),
        "file_sha256": {str(k): sha256_file(k) for k in files},
        "packages": {k: importlib.metadata.version(k) for k in PACKAGES},
        "runtime": runtime_identity(),
        "environment": collect_environment(torch),
        "reference_macro_f1": references,
        "public_checkpoint": verified_checkpoint(load_config(p["training_config"])),
        "prior_test_results_known": True,
        "boundary": "Post-test exploratory warm-start on reused validation, seed43 only; no test/replication/expansion.",
    }
    atomic_write_json(root / "frozen_protocol.json", f)
    print("Frozen E03F: four conditions, seed43, +30 epochs, epoch0 eligible; no test.", flush=True)
    return f


def load_prepared(p, f):
    root = plan_directory(p) / "prepared"
    m = json.loads((root / "manifest.json").read_text())
    if m["protocol_sha256"] != f["protocol_sha256"]:
        raise ValueError("Prepared scope changed.")
    arrays = {}
    for split in ("train", "validation"):
        path = root / (split + ".npz")
        if sha256_file(path) != m["splits"][split]:
            raise ValueError("Prepared inputs changed.")
        with np.load(path, allow_pickle=False) as z:
            arrays[split] = {k: z[k].copy() for k in z.files}
        a = arrays[split]
        n = len(a["labels"])
        if (
            a["x"].shape != (n, 1024)
            or a["mask"].shape != (n, 1024)
            or a["statistics"].shape != (n, 15)
        ):
            raise ValueError("Invalid prepared shape.")
        if any(not np.isfinite(a[k]).all() for k in ("x", "mask", "statistics")):
            raise ValueError("Nonfinite prepared inputs.")
    if len(arrays["train"]["labels"]) != 9827 or len(arrays["validation"]["labels"]) != 3510:
        raise ValueError("Unexpected budget/split.")
    return arrays, m


def prepare(path):
    f = verify(path)
    p = f["protocol"]
    root = plan_directory(p) / "prepared"
    root.mkdir(exist_ok=True)
    if (root / "manifest.json").exists():
        load_prepared(p, f)
        return
    config = with_random_seed(load_config(p["training_config"]), 43)
    bundle = load_dataset(replace(config.data, normalize="none"))
    index = stratified_train_subset_indices(bundle.y_train, bundle.ids_train, 0.4, 43)
    parent = parent_run(p, "fusion")
    np.testing.assert_array_equal(bundle.ids_train[index], np.load(parent / "train_ids.npy"))
    pp = json.loads(Path(p["parent_protocol"]).read_text())
    old_pilot = json.loads(Path(pp["parent_protocol"]).read_text())
    if sha256_file(bundle.split_path) != sha256_file(
        Path(old_pilot["parent_fusion_run"]) / "split_manifest.json"
    ):
        raise ValueError("Parent split mismatch.")
    oldprep = Path(pp["output_dir"]) / (pp["suite"] + "_analysis/prepared")
    norm = json.loads((oldprep / "normalizer.json").read_text())
    splits = {}
    for split, x, mask, y, ids in (
        (
            "train",
            bundle.x_train[index],
            bundle.mask_train[index],
            bundle.y_train[index],
            bundle.ids_train[index],
        ),
        ("validation", bundle.x_val, bundle.mask_val, bundle.y_val, bundle.ids_val),
    ):
        stats = extract_statistical_features(x, mask)
        with np.load(oldprep / (split + ".npz"), allow_pickle=False) as old:
            np.testing.assert_array_equal(old["sample_ids"], ids.astype(str))
            np.testing.assert_array_equal(old["labels"], y)
            np.testing.assert_array_equal(old["statistics"], stats)
        if split == "train":
            sc = StandardScaler().fit(stats)
            np.testing.assert_allclose(sc.mean_, norm["mean"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(sc.scale_, norm["scale"], rtol=0, atol=1e-12)
        file = root / (split + ".npz")
        temp = file.with_suffix(".npz.tmp")
        with temp.open("wb") as h:
            np.savez(
                h,
                x=normalize_masked_rows(x, mask),
                mask=mask,
                statistics=stats,
                labels=y,
                sample_ids=ids.astype(str),
            )
        temp.replace(file)
        splits[split] = sha256_file(file)
    save_label_encoder(bundle.label_encoder, root)
    shutil.copy2(bundle.split_path, root / "split_manifest.json")
    atomic_write_json(
        root / "manifest.json",
        {
            "protocol_sha256": f["protocol_sha256"],
            "splits": splits,
            "classes": bundle.label_encoder.classes_.tolist(),
            "train_ids_sha256": train_subset_sha256(bundle.ids_train[index]),
            "dataset_sha256": bundle.dataset_sha256,
            "test_encoded": False,
        },
    )
    load_prepared(p, f)
    print("Prepared selected train + validation raw inputs only; no test encoded.", flush=True)


def training_protocol(p):
    pp = json.loads(
        Path("configs/experiments/improvements/e03_peft_pilot_protocol.json").read_text()
    )
    pp = copy.deepcopy(pp)
    t = p["training"]
    pp["training"].update(
        {
            k: t[k]
            for k in (
                "maximum_epochs",
                "head_learning_rate",
                "adapter_learning_rate",
                "physical_batch_size",
                "effective_batch_size",
                "evaluation_batch_size",
                "weight_decay",
                "gradient_clip_norm",
                "warmup_fraction",
                "minimum_lr_ratio",
            )
        }
    )
    pp["training"].update(minimum_epochs=30, patience=31, grad_scaler_initial_scale=1024)
    return pp


def make_model(p, branch, rank):
    config = with_random_seed(load_config(p["training_config"]), 43)
    pp = training_protocol(p)
    backbone = train_e03.build_backbone(config, rank, pp)
    parent = torch.load(
        parent_run(p, branch) / "model_best.pt", map_location="cpu", weights_only=True
    )
    state = parent["learned_state"]
    if state["adapters"]:
        raise ValueError("Strong-head parent must not have adapted encoder.")
    head = GainFusionHead(
        state["classifier"]["stats_mean"],
        state["classifier"]["stats_scale"],
        branch=branch,
        input_gain=fan_in_gain(branch),
    )
    head.load_state_dict(state["classifier"], strict=True)
    return PeftClassifier(backbone, head).cuda(), parent


def seed_initial_checkpoint(model, p, root, metadata, initial_score):
    """Epoch0 is an eligible candidate; fresh optimizer and deterministic new RNG."""
    t = p["training"]
    seed_everything(torch, p["architecture"]["training_seed"])
    generator = torch.Generator().manual_seed(p["architecture"]["training_seed"])
    opt = torch.optim.AdamW(
        optimizer_groups(model, t["head_learning_rate"], t["adapter_learning_rate"]),
        weight_decay=t["weight_decay"],
        fused=False,
    )
    scaler = FixedHeadGradScaler("cuda")
    state = capture_learned_state(model)
    atomic_torch_save(
        torch,
        {
            "metadata": metadata,
            "learned_state": state,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": capture_random_state(torch, generator),
            "epoch": 0,
            "history": [],
            "best": initial_score,
            "best_epoch": 0,
            "best_learned_state": state,
            "patience_best": initial_score,
            "bad": 0,
        },
        root / "checkpoint_last.pt",
    )


def check_migration(p, root, branch, initial, probs):
    old = pd.read_csv(
        parent_run(p, branch) / "validation_predictions_200.csv", dtype={"sample_id": str}
    )
    disagreement = int(np.sum(old.predicted_index.to_numpy() != probs.argmax(axis=1)))
    parent = torch.load(
        parent_run(p, branch) / "model_best.pt", map_location="cpu", weights_only=True
    )
    delta = abs(initial["macro_f1"] - parent["validation_macro_f1"]) * 100
    g = p["technical_gate"]
    if (
        disagreement > g["maximum_initial_prediction_disagreements"]
        or delta > g["maximum_initial_macro_f1_difference_pp"] + 1e-12
    ):
        raise ValueError(
            "Cross-GPU epoch0 validation migration exceeded frozen tolerance; no training permitted."
        )
    atomic_write_json(
        root / "migration_gate.json",
        {
            "status": "passed",
            "initial_prediction_disagreements": disagreement,
            "macro_f1_abs_difference_pp": delta,
        },
    )


def save_predictions(path, a, probs):
    pd.DataFrame(
        {
            "sample_id": a["sample_ids"],
            "true_index": a["labels"],
            "predicted_index": probs.argmax(axis=1),
            **{f"probability_{i}": probs[:, i] for i in range(3)},
        }
    ).to_csv(path, index=False)


def run(path, name, resume=False):
    f = verify(path)
    p = f["protocol"]
    matches = [c for c in p["conditions"] if c["name"] == name]
    if len(matches) != 1:
        raise ValueError("Condition outside frozen scope.")
    c = matches[0]
    root = run_directory(p, name)
    with (plan_directory(p) / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            resume
            and (root / "status.json").exists()
            and json.loads((root / "status.json").read_text())["status"] == "completed"
        ):
            print("Preserved completed " + name)
            return
        if root.exists() != resume:
            raise ValueError("Existing run requires explicit resume; missing run cannot resume.")
        config_path = Path(p["training_config"])
        config = with_random_seed(load_config(config_path), 43)
        context = prepare_run(
            model_name="E03F_STRONG_HEAD_LORA",
            base_output_dir=Path(p["output_dir"]),
            config=config,
            config_path=config_path,
            torch=torch,
            run_name=root.name,
            resume_dir=root if resume else None,
        )
        try:
            a, dm = load_prepared(p, f)
            model, parent = make_model(p, c["branch"], c["rank"])
            metadata = {
                "protocol_sha256": f["protocol_sha256"],
                "condition": c,
                "seed": 43,
                "train_fraction": 0.4,
                "parent_checkpoint_sha256": sha256_file(
                    parent_run(p, c["branch"]) / "model_best.pt"
                ),
                "parent_best_epoch": parent["epoch"],
                "initial_head_sha256": train_e03.head_digest(model.classifier),
                "prepared_manifest_sha256": sha256_file(
                    plan_directory(p) / "prepared/manifest.json"
                ),
                "prior_test_results_known": True,
                "test_evaluated": False,
            }
            classes = [str(v) for v in dm["classes"]]
            data = {
                k: train_e03.gpu_dataset(v["x"][:, None], v["statistics"], v["labels"], v["mask"])
                for k, v in a.items()
            }
            if not resume:
                atomic_write_json(root / "experiment_plan.json", metadata)
                shutil.copy2(
                    plan_directory(p) / "frozen_protocol.json", root / "frozen_protocol.json"
                )
                shutil.copy2(
                    plan_directory(p) / "prepared/label_encoder.pkl", root / "label_encoder.pkl"
                )
                np.save(root / "train_ids.npy", a["train"]["sample_ids"])
                (root / "source").mkdir()
                shutil.copy2(Path(__file__), root / "source" / Path(__file__).name)
            elif json.loads((root / "experiment_plan.json").read_text()) != metadata:
                raise ValueError("Resume metadata changed.")
            if not (root / "checkpoint_last.pt").exists():
                initial, probs = train_e03.evaluate(
                    model, data["validation"], 64, classes, details=True
                )
                check_migration(p, root, c["branch"], initial, probs)
                atomic_write_json(root / "initial_metrics.json", initial)
                save_predictions(root / "initial_predictions.csv", a["validation"], probs)
                seed_initial_checkpoint(model, p, root, metadata, initial["macro_f1"])
            params = trainable_summary(model)
            with patch.object(torch.amp, "GradScaler", FixedHeadGradScaler):
                result, probs = train_e03.train(
                    model,
                    data["train"],
                    data["validation"],
                    training_protocol(p),
                    root,
                    metadata,
                    classes,
                    resume=True,
                )
            save_predictions(root / "validation_predictions.csv", a["validation"], probs)
            last = torch.load(root / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            restore_learned_state(model, last["learned_state"])
            endpoint, endpoint_probs = train_e03.evaluate(
                model, data["validation"], 64, classes, details=True
            )
            save_predictions(root / "endpoint_predictions.csv", a["validation"], endpoint_probs)
            atomic_write_json(
                root / "metrics.json",
                {
                    **metadata,
                    **result,
                    "parameters": params,
                    "endpoint_validation_metrics": endpoint,
                    "model_best_sha256": sha256_file(root / "model_best.pt"),
                    "checkpoint_last_sha256": sha256_file(root / "checkpoint_last.pt"),
                    "train_ids_sha256": dm["train_ids_sha256"],
                },
            )
            context.set_status("completed", test_evaluated=False, prior_test_results_known=True)
        except BaseException as e:
            context.set_status(
                "interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
                error_type=type(e).__name__,
                message=str(e),
            )
            raise


def decision(p, scores, references):
    strongest = max(scores["frozen_fusion"], scores["frozen_moment"], *references.values())
    deltas = {
        "over_continued_frozen_fusion_pp": (scores["lora8_fusion"] - scores["frozen_fusion"]) * 100,
        "over_strongest_nonadapted_pp": (scores["lora8_fusion"] - strongest) * 100,
        "over_rank_matched_moment_pp": (scores["lora8_fusion"] - scores["lora8_moment"]) * 100,
    }
    passed = all(
        deltas[k.removeprefix("minimum_")] + 1e-12 >= p["success_rules"][k]
        for k in p["success_rules"]
        if k.startswith("minimum_")
    )
    return {
        "strongest_nonadapted_macro_f1": strongest,
        **deltas,
        "pilot_gate_passed": passed,
        "automatic_next_stage": False,
        "prior_test_results_known": True,
    }


def audit_predictions(file, a, metrics):
    df = pd.read_csv(file, dtype={"sample_id": str})
    np.testing.assert_array_equal(df.sample_id, a["sample_ids"])
    np.testing.assert_array_equal(df.true_index, a["labels"])
    prob = df[[f"probability_{i}" for i in range(3)]].to_numpy()
    if (
        not np.isfinite(prob).all()
        or (prob < 0).any()
        or not np.allclose(prob.sum(axis=1), 1, rtol=0, atol=1e-5)
    ):
        raise ValueError("Invalid probability output.")
    np.testing.assert_array_equal(df.predicted_index, prob.argmax(axis=1))
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (df.true_index.to_numpy(), df.predicted_index.to_numpy()), 1)
    if cm.tolist() != metrics["confusion_matrix"]:
        raise ValueError("Confusion differs from predictions.")
    for k, v in metrics_from_confusion(cm.tolist()).items():
        if abs(v - metrics[k]) > 1e-12:
            raise ValueError("Metric mismatch: " + k)
    return prob


def summarize(path):
    f = verify(path)
    p = f["protocol"]
    a, dm = load_prepared(p, f)
    rows, scores, heads, initials = [], {}, {}, {}
    for c in p["conditions"]:
        root = run_directory(p, c["name"])
        m = json.loads((root / "metrics.json").read_text())
        if (
            json.loads((root / "status.json").read_text())["status"] != "completed"
            or m["test_evaluated"]
            or m["condition"] != c
            or m["protocol_sha256"] != f["protocol_sha256"]
            or not m["all_frozen_parameters_unchanged"]
        ):
            raise ValueError("Incomplete/mismatched run.")
        for n, k in (
            ("model_best.pt", "model_best_sha256"),
            ("checkpoint_last.pt", "checkpoint_last_sha256"),
        ):
            if sha256_file(root / n) != m[k]:
                raise ValueError("Model hash mismatch.")
        np.testing.assert_array_equal(np.load(root / "train_ids.npy"), a["train"]["sample_ids"])
        if m["train_ids_sha256"] != dm["train_ids_sha256"]:
            raise ValueError("Training IDs differ.")
        if m["prepared_manifest_sha256"] != sha256_file(plan_directory(p) / "prepared/manifest.json"):
            raise ValueError("Prepared manifest differs across conditions.")
        h = json.loads((root / "history.json").read_text())
        if [r["epoch"] for r in h] != list(range(1, 31)) or any(
            r["grad_scaler_scale"] != 1024 or r["skipped_optimizer_steps"] or r["early_stop"]
            for r in h
        ):
            raise ValueError("Incomplete/numerically revised training budget.")
        if c["rank"] and (
            m["parameters"]["adapter_parameters"] != 786432
            or any(not r["adapter_B_gradient_norm_first_window"] > 0 for r in h)
        ):
            raise ValueError("Adapter parameters/gradients invalid.")
        init = json.loads((root / "initial_metrics.json").read_text())
        probabilities = audit_predictions(root / "initial_predictions.csv", a["validation"], init)
        audit_predictions(
            root / "validation_predictions.csv", a["validation"], m["validation_metrics"]
        )
        audit_predictions(
            root / "endpoint_predictions.csv", a["validation"], m["endpoint_validation_metrics"]
        )
        values = [init["macro_f1"]] + [r["validation_macro_f1"] for r in h]
        if (
            abs(max(values) - m["validation_metrics"]["macro_f1"]) > 1e-12
            or int(np.argmax(values)) != m["best_epoch"]
        ):
            raise ValueError("Epoch0-inclusive selection mismatch.")
        if abs(h[-1]["validation_macro_f1"] - m["endpoint_validation_metrics"]["macro_f1"]) > 1e-12:
            raise ValueError("Endpoint mismatch.")
        heads.setdefault(c["branch"], set()).add(m["initial_head_sha256"])
        if c["branch"] in initials:
            np.testing.assert_array_equal(initials[c["branch"]], probabilities)
        initials[c["branch"]] = probabilities
        scores[c["name"]] = m["validation_metrics"]["macro_f1"]
        rows.append(
            {
                "condition": c["name"],
                "macro_f1_percent": scores[c["name"]] * 100,
                "accuracy_percent": m["validation_metrics"]["accuracy"] * 100,
                "initial_macro_f1_percent": values[0] * 100,
                "endpoint_macro_f1_percent": m["endpoint_validation_metrics"]["macro_f1"] * 100,
                "best_epoch": m["best_epoch"],
                "epochs": len(h),
                "training_seconds": m["training_seconds"],
                "trainable_parameters": m["parameters"]["trainable_parameters"],
            }
        )
    if any(len(v) != 1 for v in heads.values()):
        raise ValueError("Within-branch initial head mismatch.")
    summary = {
        "audit_status": "passed",
        "recorded_at": utc_now(),
        "test_evaluated": False,
        "prior_test_results_known": True,
        "prediction_rows_audited": 4 * 3 * 3510,
        "epochs_audited": 120,
        "conditions": rows,
        **decision(p, scores, f["reference_macro_f1"]),
        "limitations": p["limitations"],
    }
    atomic_write_json(plan_directory(p) / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def smoke(path):
    """Synthetic ONLY; pre-freeze numerical check, trained heads but no project labels."""
    p = read_protocol(path)
    seed_everything(torch, 43)
    verified_checkpoint(with_random_seed(load_config(p["training_config"]), 43))
    batch = 64
    positions = torch.arange(1024, device="cuda")[None, :]
    lengths = 18 + (torch.arange(batch, device="cuda") * 37) % 799
    mask = (positions < lengths[:, None]).float()
    x = (torch.sin(positions.float() / 13).expand(batch, -1) * mask)[:, None]
    x[0].zero_()
    reports = []
    for branch in ("fusion", "moment"):
        model, _ = make_model(p, branch, 0)
        model.eval()
        stats = (
            model.classifier.stats_mean[None]
            + torch.randn(batch, 15, device="cuda") * model.classifier.stats_scale[None]
        )
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            baseline = model(x, stats, mask).float().cpu()
        del model
        torch.cuda.empty_cache()
        model, _ = make_model(p, branch, 8)
        model.eval()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            zero = model(x, stats, mask).float().cpu()
        err = (zero - baseline).abs().max().item()
        if (
            err > p["technical_gate"]["zero_adapter_logit_tolerance"]
            or not torch.isfinite(zero).all()
        ):
            raise AssertionError("Strong head zero adapter failed.")
        before = frozen_parameter_digest(model)
        opt = torch.optim.AdamW(optimizer_groups(model, 1e-4, 1e-4), weight_decay=0.01, fused=False)
        scaler = FixedHeadGradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            model.train_selected()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(
                    model(x, stats, mask), torch.arange(batch, device="cuda") % 3
                )
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            adapters = adapter_parameters(model.backbone)
            if any(v.grad is None or not torch.isfinite(v.grad).all() for v in adapters.values()):
                raise AssertionError("Nonfinite adapter gradient.")
            for suffix in ["lora_B"] if not step else ["lora_A", "lora_B"]:
                if not any(v.grad.abs().sum() > 0 for k, v in adapters.items() if suffix in k):
                    raise AssertionError("Zero adapter gradient.")
            norm = torch.nn.utils.clip_grad_norm_(
                [v for v in model.parameters() if v.requires_grad], 1
            )
            if not torch.isfinite(norm):
                raise AssertionError("Nonfinite gradient norm.")
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() != 1024:
                raise AssertionError("Scale changed.")
        if frozen_parameter_digest(model) != before:
            raise AssertionError("Frozen parameters changed.")
        model.eval()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            expected = model(x, stats, mask).float()
        state = capture_learned_state(model)
        with torch.no_grad():
            for v in model.parameters():
                if v.requires_grad:
                    v.zero_()
        restore_learned_state(model, state)
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            actual = model(x, stats, mask).float()
        reload_err = (expected - actual).abs().max().item()
        if reload_err != 0:
            raise AssertionError("Reload error.")
        reports.append(
            {
                "branch": branch,
                "zero_adapter_max_error": err,
                "reload_max_error": reload_err,
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "gradients_finite": True,
                "frozen_unchanged": True,
            }
        )
        del opt, scaler, model, state, expected, actual
        torch.cuda.empty_cache()
    result = {
        "status": "passed",
        "synthetic_input_only": True,
        "test_evaluated": False,
        "batch_size": 64,
        "steps_per_branch": 2,
        "runtime": runtime_identity(),
        "source_sha256": sha256_file(Path(__file__)),
        "branches": reports,
    }
    atomic_write_json(Path(p["smoke_report"]), result)
    print(json.dumps(result, indent=2))


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Interrupted; resume committed epoch.")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("smoke", "freeze", "verify", "prepare", "run", "summarize")
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--condition")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "run":
        run(path, args.condition, args.resume)
    else:
        globals()[args.action](path)


if __name__ == "__main__":
    main()
