"""E05C/N1: fixed TCN replication and length diagnostics; no test evaluation."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import math
import platform
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.data import stratified_train_subset_indices, train_subset_sha256
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.train_baselines import extract_statistical_features
from tcn_moment.train_tcn import make_loader

DEFAULT = Path("configs/experiments/improvements/e05c_tcn_replication_protocol.json")
CONDITIONS = ["current_fixed50", "legacy_masked50", "legacy_unmasked50"]
PACKAGES = (*pilot.PACKAGES, "scipy", "joblib")


def protocol(path):
    p = pilot.read(path)
    if (p["experiment_id"] != "E05C_N1" or p["seeds"] != [42, 44, 45, 46]
        or p["train_fractions"] != [0.1, 0.4, 1.0] or p["conditions"] != CONDITIONS
        or p["epochs_per_condition"] != 50 or p["maximum_training_conditions"] != 36
        or p["wall_time_limit_hours"] != 8
        or any(p[k] for k in ("seed43_in_replication_mean", "test_evaluated", "automatic_next_stage",
                              "automatic_shutdown", "pi_wake_enabled"))):
        raise ValueError("Unexpected bounded validation-only replication scope")
    if p["diagnostics"] != {
        "seed": 43, "train_fractions": [0.1, 1.0], "padding_widths": [1024, 1280, 1536],
        "baseline_probability_atol": 0.00001, "baseline_argmax_disagreements": 0,
        "length_quantiles": [1 / 3, 2 / 3], "tree_max_depth": [1, 2, 3, 5],
        "tree_min_samples_leaf": 20, "cv_folds": 5,
        "features": ["length", "statistics15"], "length_bypass_training": False,
    }:
        raise ValueError("Length diagnostic budget changed")
    return p


def root(p):
    return Path(p["output_dir"]) / p["suite"]


def seed_config(p, seed):
    parent = pilot.protocol(Path(p["parent_protocol"]))
    return {**parent, "seed": seed, "output_dir": str(root(p)), "suite": f"seed{seed}",
            "split_manifest": p["split_manifests"][str(seed)]}


def validate_split(m, parent, seed):
    expected = {**pilot.read(parent["split_manifest"])["protocol"], "random_state": seed}
    if m["protocol"] != expected:
        raise ValueError("Split protocol differs from the original data protocol")
    for name, count in (("train", 24569), ("validation", 3510), ("test", 7020)):
        ids = m["splits"][name]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("Unexpected/duplicated split IDs")
    sets = [set(m["splits"][key]) for key in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Overlapping split IDs")


def verify(path):
    p = protocol(path)
    f = pilot.read(root(p) / "frozen_protocol.json")
    if f["protocol"] != p or f["protocol_sha256"] != sha256_file(path):
        raise ValueError("Replication protocol changed after freeze")
    for file, expected in f["file_sha256"].items():
        if sha256_file(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    if f["packages"] != {key: importlib.metadata.version(key) for key in PACKAGES}:
        raise ValueError("Replication dependencies changed")
    runtime = f["runtime"]
    if runtime["host"] != platform.node() or runtime["gpu"] != torch.cuda.get_device_name():
        raise ValueError("Runtime changed; a separate migration protocol is required")
    return f


def freeze(path):
    p = protocol(path)
    r = root(p)
    r.mkdir(parents=True, exist_ok=True)
    if (r / "frozen_protocol.json").exists():
        return verify(path)
    if list(r.glob("seed*")) or (r / "diagnostics").exists():
        raise ValueError("Freeze must precede new diagnostic/training outcomes")
    parent = pilot.verify(Path(p["parent_protocol"]))
    pp = parent["protocol"]
    if Path(p["pilot_root"]) != pilot.root(pp):
        raise ValueError("Pilot root mismatch")
    if pilot.read(Path(p["pilot_root"]) / "summary.json")["audit_status"] != "passed":
        raise ValueError("Pilot audit incomplete")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen replication requires CUDA")
    files = set(Path(name) for name in parent["file_sha256"])
    files.update(Path("src/tcn_moment").glob("*.py"))
    files.update([path, Path(p["parent_protocol"]), Path(p["pilot_root"]) / "frozen_protocol.json",
                  Path(p["pilot_root"]) / "summary.json", Path("tests/test_replicate_tcn.py"),
                  Path("scripts/run_e05c_remote.sh"),
                  Path("docs/experiment_records/e05c_tcn_replication_protocol_20261019.md")])
    for seed in p["seeds"]:
        split = Path(p["split_manifests"][str(seed)])
        validate_split(pilot.read(split), pp, seed)
        files.add(split)
    for fraction in p["diagnostics"]["train_fractions"]:
        for condition in CONDITIONS:
            d = Path(p["pilot_root"]) / pilot.case_name(fraction, condition)
            result = pilot.read(d / "result.json")
            for name, expected in result["output_sha256"].items():
                if sha256_file(d / name) != expected:
                    raise ValueError(f"Pilot output changed: {d/name}")
            files.update(d / name for name in ("result.json", "model_selected.pt", "validation_selected.csv", "train_ids.npy"))
    checks = pilot.synthetic_check()
    pilot.configure()
    f = {"protocol": p, "protocol_sha256": sha256_file(path), "frozen_at": pilot.now(),
         "file_sha256": {str(file): sha256_file(file) for file in sorted(files)},
         "packages": {key: importlib.metadata.version(key) for key in PACKAGES},
         "runtime": {"host": platform.node(), "python": platform.python_version(),
                     "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda},
         "synthetic_checks": checks, "new_real_data_outcomes_observed": False,
         "prior_pilot_and_historical_test_outcomes_known": True}
    for file in files:
        if file.parts[0] in {"src", "configs", "tests", "scripts", "docs", "legacy"}:
            destination = r / "source_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    atomic_write_json(r / "frozen_protocol.json", f)
    print(f"Frozen N1: {len(files)} inputs; 36 TCN conditions, no test, no wake plugin", flush=True)
    return f


def subset(data, fraction, seed):
    d = data["train"]
    return stratified_train_subset_indices(d["y"], d["ids"], fraction, seed)


def prediction_metrics(y, probabilities):
    pred = probabilities.argmax(1)
    cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3)
    metrics = {**metrics_from_confusion(cm.tolist()), "confusion_matrix": cm.tolist()}
    metrics["class_f1"] = [float(2 * cm[i, i] / (cm[i].sum() + cm[:, i].sum()))
                           if cm[i].sum() + cm[:, i].sum() else 0.0 for i in range(3)]
    # Original label2 is battery abnormality (encoded class index2).
    battery = y == 2
    metrics["battery_AP"] = float(average_precision_score(battery, probabilities[:, 2]))
    metrics["battery_recall"] = float(np.mean(pred[battery] == 2))
    metrics["battery_FPR"] = float(np.mean(pred[~battery] == 2))
    return metrics


def read_predictions(file, v, expected=None):
    frame = pd.read_csv(file, dtype={"sample_id": str})
    probabilities = frame[["p0", "p1", "p2"]].to_numpy()
    if (not np.array_equal(frame.sample_id, v["ids"]) or not np.array_equal(frame.y_true, v["y"])
        or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(1), 1, atol=2e-7, rtol=0)
        or not np.array_equal(frame.y_pred, probabilities.argmax(1))):
        raise ValueError(f"Ordered prediction/probability mismatch: {file}")
    metrics = prediction_metrics(v["y"], probabilities)
    if expected is not None and abs(metrics["macro_f1"] - expected) > 1e-12:
        raise ValueError(f"Independent metric mismatch: {file}")
    return probabilities, metrics


def early_stop_replay(history, delta=0.0001, patience=7):
    selected, stale, selected_epoch = -1.0, 0, 0
    for h in history:
        if h["macro_f1"] > selected + delta:
            selected, selected_epoch, stale = h["macro_f1"], h["epoch"], 0
        else:
            stale += 1
        if stale >= patience:
            return {"stopped_epoch": h["epoch"], "selected_epoch": selected_epoch, "macro_f1": selected}
    return {"stopped_epoch": history[-1]["epoch"], "selected_epoch": selected_epoch, "macro_f1": selected}


def length_edges(lengths):
    return np.quantile(lengths, [1 / 3, 2 / 3], method="linear")


def stratified_metrics(y, probabilities, lengths, edges):
    groups = np.searchsorted(edges, lengths, side="left")
    rows = []
    for i in range(3):
        mask = groups == i
        if mask.any():
            cm = np.bincount(3 * y[mask] + probabilities[mask].argmax(1), minlength=9).reshape(3, 3).tolist()
            metrics = metrics_from_confusion(cm)
        else:
            metrics = None
        rows.append({"group": i, "n": int(mask.sum()),
                     "class_counts": np.bincount(y[mask], minlength=3).tolist(), "metrics": metrics})
    return rows


def infer_logits(model, values, mask, batch_size=64):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = torch.as_tensor(values[start:start + batch_size, None, :], device="cuda")
            m = torch.as_tensor(mask[start:start + batch_size], device="cuda")
            logits = model(x, m)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite diagnostic logits")
            result.append(logits.cpu())
    return torch.cat(result)


def diagnostic_hashes(directory):
    return {str(file.relative_to(directory)): sha256_file(file) for file in sorted(directory.rglob("*"))
            if file.is_file() and file.name != "summary.json"}


def diagnose(path):
    f = verify(path)
    p = f["protocol"]
    directory = root(p) / "diagnostics"
    if (directory / "summary.json").exists():
        summary = pilot.read(directory / "summary.json")
        for name, want in summary["output_sha256"].items():
            if sha256_file(directory / name) != want:
                raise ValueError("Diagnostic output changed")
        return summary
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Incomplete diagnostic outputs retained; refuse silent recomputation")
    directory.mkdir(exist_ok=True)
    pilot.configure()
    pp = pilot.protocol(Path(p["parent_protocol"]))
    data = pilot.load_train_validation(pp)
    d, v = data["train"], data["validation"]
    train_lengths, val_lengths = d["mask"].sum(1), v["mask"].sum(1)
    stats_train = extract_statistical_features(d["x"], d["mask"])
    stats_val = extract_statistical_features(v["x"], v["mask"])
    summary = {"test_evaluated": False, "seed": 43, "replication_mean_includes_this_seed": False,
               "length_distributions": [], "padding": [], "tree_controls": []}
    for fraction in p["diagnostics"]["train_fractions"]:
        ix = subset(data, fraction, 43)
        edges = length_edges(train_lengths[ix])
        counts = {}
        for name, labels, lengths in (("train", d["y"][ix], train_lengths[ix]), ("validation", v["y"], val_lengths)):
            counts[name] = [{"class": i, "n": int((labels == i).sum()),
                             "length_quantiles": np.quantile(lengths[labels == i], [0, .25, .5, .75, 1]).tolist()}
                            for i in range(3)]
        summary["length_distributions"].append({"fraction": fraction, "train_edges": edges.tolist(), "by_class": counts,
                                                 "train_ids_sha256": train_subset_sha256(d["ids"][ix])})
        for condition in CONDITIONS:
            old = Path(p["pilot_root"]) / pilot.case_name(fraction, condition)
            if not np.array_equal(np.load(old / "train_ids.npy"), d["ids"][ix]):
                raise ValueError("Pilot diagnostic subset mismatch")
            saved, _ = read_predictions(old / "validation_selected.csv", v)
            model = pilot.build_model(condition).to("cuda")
            model.load_state_dict(torch.load(old / "model_selected.pt", map_location="cuda", weights_only=True))
            baseline = infer_logits(model, v["x"], v["mask"])
            baseline_prob = baseline.softmax(1).numpy()
            drift = float(np.max(np.abs(baseline_prob - saved)))
            disagreements = int(np.count_nonzero(baseline_prob.argmax(1) != saved.argmax(1)))
            if drift > p["diagnostics"]["baseline_probability_atol"] or disagreements:
                raise ValueError("Pilot checkpoint migration failed; do not relax tolerance")
            for width in p["diagnostics"]["padding_widths"]:
                logits = baseline if width == 1024 else infer_logits(model, np.pad(v["x"], ((0, 0), (0, width - 1024))),
                                                                     np.pad(v["mask"], ((0, 0), (0, width - 1024))))
                probabilities = logits.softmax(1).numpy()
                stem = f"{pilot.case_name(fraction, condition)}_padding{width}"
                np.savez_compressed(directory / f"{stem}.npz", sample_ids=v["ids"], labels=v["y"], logits=logits.numpy(), probabilities=probabilities)
                pilot.save_predictions(directory / f"{stem}.csv", v["ids"], v["y"], probabilities)
                summary["padding"].append({"fraction": fraction, "condition": condition, "width": width,
                    "pilot_probability_drift": drift, "pilot_argmax_disagreements": disagreements,
                    "max_logit_difference": float((logits - baseline).abs().max()),
                    "max_probability_difference": float(np.abs(probabilities - baseline_prob).max()),
                    "changed_predictions": int(np.count_nonzero(probabilities.argmax(1) != baseline_prob.argmax(1))),
                    "metrics": prediction_metrics(v["y"], probabilities),
                    "length_strata": stratified_metrics(v["y"], probabilities, val_lengths, edges), "stem": stem})
            del model
            torch.cuda.empty_cache()
        for feature in p["diagnostics"]["features"]:
            x, xv = (train_lengths[:, None], val_lengths[:, None]) if feature == "length" else (stats_train, stats_val)
            search = GridSearchCV(DecisionTreeClassifier(min_samples_leaf=20, random_state=43),
                                  {"max_depth": [1, 2, 3, 5]}, scoring="f1_macro",
                                  cv=StratifiedKFold(5, shuffle=False), n_jobs=1, error_score="raise")
            started = time.perf_counter()
            search.fit(x[ix], d["y"][ix])
            prob = search.predict_proba(xv)
            stem = f"fraction{round(100*fraction):03d}_{feature}_tree"
            pilot.save_predictions(directory / f"{stem}.csv", v["ids"], v["y"], prob)
            pd.DataFrame(search.cv_results_).to_csv(directory / f"{stem}_cv.csv", index=False)
            joblib.dump(search.best_estimator_, directory / f"{stem}.joblib")
            summary["tree_controls"].append({"fraction": fraction, "features": feature, "stem": stem,
                "best_params": search.best_params_, "cv_macro_f1": float(search.best_score_),
                "metrics": prediction_metrics(v["y"], prob), "training_seconds": time.perf_counter() - started,
                "length_strata": stratified_metrics(v["y"], prob, val_lengths, edges)})
        print(f"Completed seed43 length/padding diagnosis fraction={fraction}", flush=True)
    summary["output_sha256"] = diagnostic_hashes(directory)
    atomic_write_json(directory / "summary.json", summary)
    return summary


def audit_case(directory, cp, fraction, condition, data, protocol_sha):
    result, history = pilot.read(directory / "result.json"), pilot.read(directory / "history.json")
    if (result["protocol_sha256"] != protocol_sha or result["condition"] != condition
        or result["fraction"] != fraction or result["test_evaluated"]
        or pilot.read(directory / "status.json")["status"] != "completed"):
        raise ValueError("Case result scope mismatch")
    ix = subset(data, fraction, cp["seed"])
    if (not np.array_equal(np.load(directory / "train_ids.npy"), data["train"]["ids"][ix])
        or result["train_ids_sha256"] != train_subset_sha256(data["train"]["ids"][ix])):
        raise ValueError("Training subset mismatch")
    for name, expected in result["output_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Output changed: {directory/name}")
    if result["epochs"] != 50 or [h["epoch"] for h in history] != list(range(1, 51)):
        raise ValueError("Incomplete fixed50 trajectory")
    steps = math.ceil(len(ix) / 32)
    if result["optimizer_steps"] != 50 * steps or any(h["optimizer_steps"] != steps for h in history):
        raise ValueError("Optimizer budget mismatch")
    selected, epoch = -1.0, 0
    for h in history:
        if not all(math.isfinite(h[key]) for key in ("train_loss", "val_loss", "macro_f1")):
            raise ValueError("Nonfinite history")
        if abs(metrics_from_confusion(h["confusion_matrix"])["macro_f1"] - h["macro_f1"]) > 1e-12:
            raise ValueError("Epoch confusion metrics mismatch")
        if h["macro_f1"] > selected + .0001:
            selected, epoch = h["macro_f1"], h["epoch"]
    peak = max(history, key=lambda h: h["macro_f1"])
    if (selected != result["selected_macro_f1"] or epoch != result["selected_epoch"]
        or peak["macro_f1"] != result["peak_macro_f1"] or peak["epoch"] != result["peak_epoch"]):
        raise ValueError("Selected/peak rule mismatch")
    evaluated = {}
    for suffix, expected in (("selected", selected), ("peak", peak["macro_f1"]), ("endpoint", history[-1]["macro_f1"])):
        _, evaluated[suffix] = read_predictions(directory / f"validation_{suffix}.csv", data["validation"], expected)
    return {"seed": cp["seed"], "fraction": fraction, "condition": condition,
            "train_count": len(ix), "epochs": 50, "selected_epoch": epoch,
            "selected": evaluated["selected"], "peak": evaluated["peak"], "endpoint": evaluated["endpoint"],
            "early7_replay": early_stop_replay(history) if condition == "current_fixed50" else None,
            "initial_model_sha256": result["initial_model_sha256"],
            "parameters": result["total_parameters"], "training_seconds": result["training_seconds"]}


def aggregate(rows):
    tables, differences = [], []
    for fraction in (.1, .4, 1.0):
        for condition in CONDITIONS:
            cells = [r for r in rows if r["fraction"] == fraction and r["condition"] == condition]
            if sorted(r["seed"] for r in cells) != [42, 44, 45, 46]:
                raise ValueError("Missing replication seeds or pilot contamination")
            scores = [100 * r["selected"]["macro_f1"] for r in cells]
            tables.append({"fraction": fraction, "condition": condition,
                           "macro_f1_mean_percent": statistics.mean(scores),
                           "macro_f1_sample_sd_percent": statistics.stdev(scores)})
        for a, b in (("legacy_masked50", "current_fixed50"), ("legacy_unmasked50", "legacy_masked50"),
                     ("legacy_unmasked50", "current_fixed50")):
            deltas = []
            for seed in (42, 44, 45, 46):
                cells = {r["condition"]: r for r in rows if r["fraction"] == fraction and r["seed"] == seed}
                deltas.append(100 * (cells[a]["selected"]["macro_f1"] - cells[b]["selected"]["macro_f1"]))
            mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
            half = 3.182446305284263 * sd / 2
            differences.append({"fraction": fraction, "comparison": f"{a}-{b}", "per_seed_pp": deltas,
                                "mean_pp": mean, "positive_seeds": sum(d > 0 for d in deltas),
                                "descriptive_ci95_pp": [mean - half, mean + half]})
    return tables, differences


def audit(path):
    f = verify(path)
    p, rows, output_manifest = f["protocol"], [], {}
    diagnostic = diagnose(path)
    # Check all diagnostic rows and full persisted logits, including the controls.
    pp = pilot.protocol(Path(p["parent_protocol"]))
    pilot_data = pilot.load_train_validation(pp)
    for item in diagnostic["padding"] + diagnostic["tree_controls"]:
        probs, metrics = read_predictions(root(p) / "diagnostics" / (item["stem"] + ".csv"), pilot_data["validation"], item["metrics"]["macro_f1"])
        if "width" in item:
            with np.load(root(p) / "diagnostics" / (item["stem"] + ".npz")) as saved:
                if (not np.array_equal(saved["sample_ids"], pilot_data["validation"]["ids"])
                    or not np.array_equal(saved["labels"], pilot_data["validation"]["y"])
                    or not np.allclose(saved["probabilities"], probs, rtol=0, atol=1e-12)
                    or not np.allclose(torch.from_numpy(saved["logits"]).softmax(1).numpy(), probs, atol=1e-7, rtol=0)):
                    raise ValueError("Saved padding logits/rows mismatch")
        else:
            cv = pd.read_csv(root(p) / "diagnostics" / (item["stem"] + "_cv.csv"))
            scores = cv[[f"split{i}_test_score" for i in range(5)]].to_numpy()
            if len(cv) != 4 or not np.isfinite(scores).all() or not np.allclose(scores.mean(1), cv.mean_test_score, atol=1e-12, rtol=0):
                raise ValueError("Tree CV budget/score mismatch")
            best = int(np.argmax(scores.mean(1)))
            if int(cv.iloc[best].param_max_depth) != item["best_params"]["max_depth"]:
                raise ValueError("Tree CV selection mismatch")
    del pilot_data
    for seed in p["seeds"]:
        cp = seed_config(p, seed)
        data = pilot.load_train_validation(cp)
        hashes = {}
        for fraction in p["train_fractions"]:
            for condition in CONDITIONS:
                directory = pilot.root(cp) / pilot.case_name(fraction, condition)
                row = audit_case(directory, cp, fraction, condition, data, f["protocol_sha256"])
                rows.append(row)
                hashes[condition] = row["initial_model_sha256"]
                for file in directory.iterdir():
                    if file.is_file():
                        output_manifest[str(file.relative_to(root(p)))] = sha256_file(file)
            if hashes["legacy_masked50"] != hashes["legacy_unmasked50"]:
                raise ValueError("Legacy initialization mismatch")
        del data
    tables, differences = aggregate(rows)
    summary = {"audit_status": "passed", "recorded_at": pilot.now(), "test_evaluated": False,
               "prior_test_results_known": True, "seed43_in_replication_mean": False,
               "conditions": len(rows), "epochs_audited": len(rows) * 50,
               "prediction_rows_audited": len(rows) * 3 * 3510, "per_condition": rows,
               "aggregate": tables, "paired_differences": differences,
               "diagnostic_tree_conditions": len(diagnostic["tree_controls"]),
               "diagnostic_padding_conditions": len(diagnostic["padding"]),
               "automatic_next_stage": False,
               "limitations": "Exploratory reused random splits/validation; overlapping seeds, pilot43 excluded; Torch structural port, no external/test confirmation or global architecture optimum."}
    atomic_write_json(root(p) / "output_manifest.json", output_manifest)
    atomic_write_json(root(p) / "summary.json", summary)
    report(p, summary, diagnostic)
    return summary


def report(p, s, d):
    lines = ["# E05C / N1：TCN四seed复核与长度诊断结果", "",
             "seed42/44/45/46；pilot43独立诊断。全部validation-only，既有test已知后的探索，没有新test评价。", "",
             "| 训练预算 | 条件 | Macro-F1均值 % | 样本SD % |", "|---:|---|---:|---:|"]
    for row in s["aggregate"]:
        lines.append(f"| {100*row['fraction']:g}% | {row['condition']} | {row['macro_f1_mean_percent']:.4f} | {row['macro_f1_sample_sd_percent']:.4f} |")
    lines += ["", "| 预算 | 配对差 | 均值pp | 正向seed | 描述性95%区间pp |", "|---:|---|---:|---:|---|"]
    for row in s["paired_differences"]:
        lo, hi = row["descriptive_ci95_pp"]
        lines.append(f"| {100*row['fraction']:g}% | {row['comparison']} | {row['mean_pp']:+.4f} | {row['positive_seeds']}/4 | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "区间按四seed描述性配对t计算；split重叠，不作独立总体置信保证。selected按min_delta0.0001规则，另存真峰值/端点及各类/电池AP、Recall、FPR。", "",
              "## seed43长度与padding诊断", "", "| 标签预算 | 控制 | 最佳树深 | Validation F1 % |", "|---:|---|---:|---:|"]
    for row in d["tree_controls"]:
        lines.append(f"| {100*row['fraction']:g}% | {row['features']} | {row['best_params']['max_depth']} | {100*row['metrics']['macro_f1']:.4f} |")
    lines += ["", "两树同4候选×5折，15统计包含length；不代表最强统计SVM，也不能据此证明长度是唯一机制。", "",
              "| 预算 | 条件 | Padding | 最大logit差 | 预测改变数 | F1 % |", "|---:|---|---:|---:|---:|---:|"]
    for row in d["padding"]:
        lines.append(f"| {100*row['fraction']:g}% | {row['condition']} | {row['width']} | {row['max_logit_difference']:.3g} | {row['changed_predictions']} | {100*row['metrics']['macro_f1']:.4f} |")
    lines += ["", f"审计{s['conditions']}条件、{s['epochs_audited']}epoch及{s['prediction_rows_audited']}行复核预测，另有18组padding及4树控制。固定权重padding敏感性不等于新观测，不能据此挑padding长度。", "",
              "legacy结构是统一优化器/初始化规则的PyTorch移植，未复现原Keras全部训练；无超参搜索、无长度旁路训练。current轨迹仅回放early7，不另训。", "",
              "本轮只更新现有数据上的结构/池化参照，不改E03T/E05A，不自动开始N2、LoRA或test；不使用pi-wake。", "",
              f"证据：`{root(p)}/summary.json`、`output_manifest.json`及`diagnostics/summary.json`。"]
    Path("docs/experiment_records/e05c_tcn_replication_results_20261019.md").write_text("\n".join(lines) + "\n")


def run(path):
    p = protocol(path)
    r = root(p)
    r.mkdir(parents=True, exist_ok=True)
    with (r / "suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = pilot.read(r / "suite_status.json") if (r / "suite_status.json").exists() else {}
        started_at = previous.get("started_at", pilot.now())
        try:
            f = verify(path)
            if (r / "summary.json").exists():
                return
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
            deadline = time.monotonic() + p["wall_time_limit_hours"] * 3600 - elapsed
            atomic_write_json(r / "suite_status.json", {"status": "running", "started_at": started_at,
                "updated_at": pilot.now(), "stage": "diagnostics_seed43", "test_evaluated": False, "pi_wake_enabled": False})
            diagnose(path)
            for seed in p["seeds"]:
                pilot.configure()
                cp = seed_config(p, seed)
                pilot.root(cp).mkdir(exist_ok=True)
                data = pilot.load_train_validation(cp)
                for fraction in p["train_fractions"]:
                    for condition in CONDITIONS:
                        atomic_write_json(r / "suite_status.json", {"status": "running", "started_at": started_at,
                            "updated_at": pilot.now(), "stage": f"seed{seed}/{pilot.case_name(fraction, condition)}",
                            "test_evaluated": False, "pi_wake_enabled": False})
                        directory = pilot.root(cp) / pilot.case_name(fraction, condition)
                        if (directory / "result.json").exists():
                            audit_case(directory, cp, fraction, condition, data, f["protocol_sha256"])
                        else:
                            pilot.run_case(cp, f, fraction, condition, data, deadline)
                del data
            audit(path)
            atomic_write_json(r / "suite_status.json", {"status": "completed", "started_at": started_at,
                "updated_at": pilot.now(), "test_evaluated": False, "automatic_next_stage": False, "pi_wake_enabled": False})
        except BaseException as error:
            atomic_write_json(r / "suite_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "started_at": started_at, "updated_at": pilot.now(), "test_evaluated": False,
                "error_type": type(error).__name__, "error": str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "diagnose", "run", "audit"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.action](args.protocol)


if __name__ == "__main__":
    main()
