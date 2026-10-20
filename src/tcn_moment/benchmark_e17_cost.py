"""E17: same-hardware end-to-end cost of the clean-trained final pipeline.

Times the full online path (preprocessing, DYN, MOMENT original/diff, TCN, heads),
not cached classifier heads.  Validation only, no test.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from tcn_moment import run_e08
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.dynamics_features import extract_dynamics_features
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.run_e13_robustness import _diff_pad, _load_tcn, _normalize_pad, _tcn_prob
from tcn_moment.train_moment import (
    build_model,
    forward_features,
    require_torch_and_moment,
    set_num_classes,
)


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    scores = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return (
        np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64),
        scores,
        np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64),
    )


def _time_cuda(fn: Any) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array) * 1000),
        "p95_ms": float(np.quantile(array, 0.95) * 1000),
        "mean_ms": float(array.mean() * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/improvements/e17_cost_protocol.json")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if protocol.get("test_evaluated") or protocol.get("automatic_expansion"):
        raise ValueError("E17 is validation-only")
    root = Path(protocol["output_dir"]) / protocol["suite"]
    root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("E17 requires CUDA")
    device = torch.device("cuda")

    seed = int(protocol["seed"])
    tag = protocol["tag"]
    config = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
    raw = load_dataset(replace(config.data, normalize="none"))
    indices = stratified_train_subset_indices(raw.y_train, raw.ids_train, 0.4, seed)
    with np.load(Path(protocol["e08_suite"]) / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
        dyn_train = archive["train_features"]
    seed_results = json.loads(
        (Path(protocol["e08_suite"]) / f"seed{seed}" / "seed_results.json").read_text()
    )
    hgb_params = next(
        record["selected_params"]
        for record in seed_results["conditions"]
        if record["method"] == "dyn_hgb" and record["tag"] == tag
    )
    hgb = run_e08._hgb_factory(hgb_params, seed).fit(dyn_train[indices], raw.y_train[indices])
    moment_train, _ = run_e08._load_moment_split(
        Path(protocol["e01_cache"]), seed=seed, split="train", ids=raw.ids_train
    )
    diff_train, _ = run_e08._load_moment_split(
        Path(protocol["e09_cache"]), seed=seed, split="train", ids=raw.ids_train
    )
    md_model = run_e08._logit_factory({"C": 1.0}, seed).fit(
        np.hstack([moment_train[indices], diff_train[indices]]), raw.y_train[indices]
    )
    with np.load(
        Path(protocol["e12_suite"]) / f"seed{seed}" / f"oof_{tag}.npz", allow_pickle=False
    ) as archive:
        oof_dyn = archive["oof_dyn"]
        oof_md = archive["oof_md"]
        oof_tcn = archive["oof_tcn"]
        y_oof = archive["y_train"]
    meta_grid = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    tcn_dyn_selected, _ = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        meta_grid,
        np.hstack([oof_tcn, oof_dyn]),
        y_oof,
        folds=3,
        seed=seed,
    )
    tcn_dyn_meta = LogisticRegression(
        C=tcn_dyn_selected["C"], max_iter=5000, random_state=seed
    ).fit(np.hstack([oof_tcn, oof_dyn]), y_oof)
    full_selected, _ = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        meta_grid,
        np.hstack([oof_dyn, oof_md, oof_tcn]),
        y_oof,
        folds=3,
        seed=seed,
    )
    full_meta = LogisticRegression(
        C=full_selected["C"], max_iter=5000, random_state=seed
    ).fit(np.hstack([oof_dyn, oof_md, oof_tcn]), y_oof)

    torch_mod, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    moments = build_model(config, MOMENTPipeline, 3)
    set_num_classes(moments, 3)
    moments.init()
    moments.to(device)
    moments.eval()
    for parameter in moments.parameters():
        parameter.requires_grad = False
    tcn = _load_tcn(
        Path(protocol["e12_suite"]) / f"seed{seed}" / "models" / tag / "full.pt", device
    )

    x_val = raw.x_val
    mask_val = raw.mask_val.astype(bool)
    ids_val = raw.ids_val.astype(str)
    y_val = raw.y_val
    checks = {}

    def preprocess(batch_x: np.ndarray, batch_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_norm = np.zeros_like(batch_x)
        diff_rows = []
        diff_masks = []
        for index in range(len(batch_x)):
            values = batch_x[index, batch_mask[index]]
            x_norm[index] = _normalize_pad(values, batch_x.shape[1])
            row, row_mask = _diff_pad(values, batch_x.shape[1])
            diff_rows.append(row)
            diff_masks.append(row_mask)
        return x_norm, np.stack(diff_rows), np.stack(diff_masks), batch_x

    def embed(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            features = forward_features(
                moments,
                torch.as_tensor(x[:, None, :], dtype=torch.float32, device=device),
                torch.as_tensor(mask, dtype=torch.float32, device=device),
            ).float().cpu().numpy()
        return features

    def run_batch(batch_x: np.ndarray, batch_mask: np.ndarray) -> dict[str, np.ndarray]:
        x_norm, diff_matrix, diff_mask, raw_batch = preprocess(batch_x, batch_mask)
        dyn = extract_dynamics_features(raw_batch.astype(np.float32), batch_mask)
        moment = embed(x_norm, batch_mask)
        diff = embed(diff_matrix, diff_mask)
        tcn_prob = _tcn_prob(tcn, raw_batch.astype(np.float32), batch_mask.astype(np.float32), device)
        dyn_prob = hgb.predict_proba(dyn)
        md_prob = md_model.predict_proba(np.hstack([moment, diff]))
        tcn_dyn_prob = tcn_dyn_meta.predict_proba(np.hstack([tcn_prob, dyn_prob]))
        full_prob = full_meta.predict_proba(np.hstack([dyn_prob, md_prob, tcn_prob]))
        return {
            "dyn_prob": dyn_prob,
            "md_prob": md_prob,
            "tcn_prob": tcn_prob,
            "tcn_dyn_prob": tcn_dyn_prob,
            "full_prob": full_prob,
        }

    # warmup
    for _ in range(int(protocol["warmup"])):
        run_batch(x_val[:32], mask_val[:32])
    torch.cuda.reset_peak_memory_stats(device)

    timings: dict[str, Any] = {}
    for batch_size in protocol["batch_sizes"]:
        repeats = int(protocol["repeats"][str(batch_size)])
        rows = []
        for start in range(0, len(x_val) - batch_size + 1, batch_size):
            if len(rows) >= repeats * 4:
                break
            batch_x = x_val[start : start + batch_size]
            batch_mask = mask_val[start : start + batch_size]
            total = _time_cuda(lambda: run_batch(batch_x, batch_mask))
            rows.append(total / batch_size)
        percentile = _percentiles(rows)
        timings[str(batch_size)] = {
            "batch_size": batch_size,
            "batches": len(rows),
            "median_ms_per_sample": percentile["median_ms"],
            "p95_ms_per_sample": percentile["p95_ms"],
            "median_ms_per_batch": percentile["median_ms"] * batch_size,
            "throughput_samples_per_s": float(1000.0 / percentile["median_ms"]),
        }

    single = []
    single_components = []
    for index in range(int(protocol["single_sample_count"])):
        batch_x = x_val[index : index + 1]
        batch_mask = mask_val[index : index + 1]
        total = _time_cuda(lambda: run_batch(batch_x, batch_mask))
        single.append(total)
        # component timings on the same sample
        started = time.perf_counter()
        x_norm, diff_matrix, diff_mask, _ = preprocess(batch_x, batch_mask)
        preprocess_s = time.perf_counter() - started
        started = time.perf_counter()
        extract_dynamics_features(batch_x.astype(np.float32), batch_mask)
        dyn_s = time.perf_counter() - started
        moment_s = _time_cuda(lambda: embed(x_norm, batch_mask))
        diff_s = _time_cuda(lambda: embed(diff_matrix, diff_mask))
        tcn_s = _time_cuda(lambda: _tcn_prob(tcn, batch_x.astype(np.float32), batch_mask.astype(np.float32), device))
        dyn = extract_dynamics_features(batch_x.astype(np.float32), batch_mask)
        moment = embed(x_norm, batch_mask)
        diff = embed(diff_matrix, diff_mask)
        tcn_prob = _tcn_prob(tcn, batch_x.astype(np.float32), batch_mask.astype(np.float32), device)
        started = time.perf_counter()
        dyn_prob = hgb.predict_proba(dyn)
        md_prob = md_model.predict_proba(np.hstack([moment, diff]))
        full_meta.predict_proba(np.hstack([dyn_prob, md_prob, tcn_prob]))
        heads_s = time.perf_counter() - started
        single_components.append(
            {
                "preprocess_ms": preprocess_s * 1000,
                "dyn_ms": dyn_s * 1000,
                "moment_orig_ms": moment_s * 1000,
                "moment_diff_ms": diff_s * 1000,
                "tcn_ms": tcn_s * 1000,
                "heads_ms": heads_s * 1000,
            }
        )

    def run_dyn(batch_x: np.ndarray, batch_mask: np.ndarray) -> np.ndarray:
        return hgb.predict_proba(extract_dynamics_features(batch_x.astype(np.float32), batch_mask))

    def run_tcn(batch_x: np.ndarray, batch_mask: np.ndarray) -> np.ndarray:
        return _tcn_prob(tcn, batch_x.astype(np.float32), batch_mask.astype(np.float32), device)

    def run_tcn_dyn(batch_x: np.ndarray, batch_mask: np.ndarray) -> np.ndarray:
        dyn_prob = run_dyn(batch_x, batch_mask)
        tcn_prob = run_tcn(batch_x, batch_mask)
        return tcn_dyn_meta.predict_proba(np.hstack([tcn_prob, dyn_prob]))

    method_timings: dict[str, Any] = {}
    for method, function in (
        ("dyn", run_dyn),
        ("tcn", run_tcn),
        ("tcn_dyn", run_tcn_dyn),
        ("full", run_batch),
    ):
        method_timings[method] = {}
        for batch_size in protocol["batch_sizes"]:
            repeats = int(protocol["repeats"][str(batch_size)])
            rows = []
            for start in range(0, len(x_val) - batch_size + 1, batch_size):
                if len(rows) >= repeats:
                    break
                batch_x = x_val[start : start + batch_size]
                batch_mask = mask_val[start : start + batch_size]
                total = _time_cuda(lambda: function(batch_x, batch_mask))
                rows.append(total / batch_size)
            method_timings[method][str(batch_size)] = {
                "median_ms_per_sample": float(np.median(rows) * 1000),
                "p95_ms_per_sample": float(np.quantile(rows, 0.95) * 1000),
                "throughput_samples_per_s": float(1.0 / np.median(rows)),
            }

    # accuracy of the full online path
    predictions = []
    for start in range(0, len(x_val), 128):
        result = run_batch(x_val[start : start + 128], mask_val[start : start + 128])
        predictions.append(result["full_prob"])
    full_prob = np.concatenate(predictions)
    full_pred = full_prob.argmax(1)
    from sklearn.metrics import f1_score

    checks["full_macro_f1"] = float(f1_score(y_val, full_pred, average="macro"))
    checks["full_accuracy"] = float((full_pred == y_val).mean())
    reference_y, reference_prob, reference_pred = _read_predictions(
        Path(protocol["e12_suite"]) / f"seed{seed}" / "conditions" / tag / "stack_tcn" / "predictions.csv"
    )
    checks["reference_macro_f1"] = float(f1_score(reference_y, reference_pred, average="macro"))
    checks["reference_prob_max_abs_diff"] = float(np.abs(reference_prob - full_prob).max())

    component_summary = {
        key: _percentiles([row[key] / 1000 for row in single_components]) for key in single_components[0]
    }
    moment_params = sum(parameter.numel() for parameter in moments.parameters())
    tcn_params = sum(parameter.numel() for parameter in tcn.parameters())
    model_sizes = {
        "moment_params": int(moment_params),
        "moment_fp32_mb": float(moment_params * 4 / 1024**2),
        "tcn_params": int(tcn_params),
        "tcn_fp32_mb": float(tcn_params * 4 / 1024**2),
        "hgb_pickle_mb": len(pickle.dumps(hgb)) / 1024**2,
        "md_pickle_mb": len(pickle.dumps(md_model)) / 1024**2,
        "full_meta_pickle_mb": len(pickle.dumps(full_meta)) / 1024**2,
    }
    summary = {
        "test_evaluated": False,
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "host": platform.node(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "settings": {
            "seed": seed,
            "tag": tag,
            "batch_sizes": protocol["batch_sizes"],
            "single_sample_count": protocol["single_sample_count"],
            "warmup": protocol["warmup"],
        },
        "timings": timings,
        "method_timings": method_timings,
        "single_sample": _percentiles(single),
        "single_components": component_summary,
        "gpu_peak_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "host_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "model_sizes": model_sizes,
        "checks": checks,
    }
    atomic_write_json(root / "summary.json", summary)
    lines = [
        "# E17：同硬件端到端成本（validation-only）",
        "",
        "**seed42 / 40% checkpoint；完整在线路径，不含训练；0 test。**",
        "",
        f"- GPU：{summary['hardware']['gpu']}；PyTorch {summary['hardware']['torch']}；CUDA {summary['hardware']['cuda']}",
        f"- batch=1 延迟中位数：**{summary['single_sample']['median_ms']:.2f} ms/sample**（p95 {summary['single_sample']['p95_ms']:.2f}）",
        f"- batch=32：**{timings['32']['median_ms_per_sample']:.2f} ms/sample**，batch≈{timings['32']['median_ms_per_batch']:.0f} ms，{timings['32']['throughput_samples_per_s']:.1f} samples/s",
        f"- batch=128：**{timings['128']['median_ms_per_sample']:.2f} ms/sample**，batch≈{timings['128']['median_ms_per_batch']:.0f} ms，{timings['128']['throughput_samples_per_s']:.1f} samples/s",
        f"- GPU 峰值显存：**{summary['gpu_peak_mb']:.1f} MB**；host max RSS {summary['host_max_rss_mb']:.0f} MB",
        f"- 参数量：MOMENT {moment_params/1e6:.2f}M（{model_sizes['moment_fp32_mb']:.1f} MB FP32），TCN {tcn_params/1e6:.2f}M（{model_sizes['tcn_fp32_mb']:.1f} MB FP32）",
        f"- pickle 大小：HGB {model_sizes['hgb_pickle_mb']:.2f} MB、MD logistic {model_sizes['md_pickle_mb']:.2f} MB、full meta {model_sizes['full_meta_pickle_mb']:.3f} MB",
        "",
        "## 单样本组件（batch=1，中位数）",
        "",
        "| 组件 | ms |",
        "|---|---:|",
        f"| 预处理（normalize+diff） | {component_summary['preprocess_ms']['median_ms']:.3f} |",
        f"| DYN 60 维提取 | {component_summary['dyn_ms']['median_ms']:.3f} |",
        f"| MOMENT 原序列 | {component_summary['moment_orig_ms']['median_ms']:.3f} |",
        f"| MOMENT ΔP | {component_summary['moment_diff_ms']['median_ms']:.3f} |",
        f"| TCN 前向 | {component_summary['tcn_ms']['median_ms']:.3f} |",
        f"| HGB+MD+meta 头 | {component_summary['heads_ms']['median_ms']:.3f} |",
        "",
        "## 方法级端到端成本（ms/sample 中位数）",
        "",
        "| 方法 | batch=1 | batch=32 | batch=128 |",
        "|---|---:|---:|---:|",
        f"| DYN+HGB | {method_timings['dyn']['1']['median_ms_per_sample']:.3f} | {method_timings['dyn']['32']['median_ms_per_sample']:.3f} | {method_timings['dyn']['128']['median_ms_per_sample']:.3f} |",
        f"| TCN | {method_timings['tcn']['1']['median_ms_per_sample']:.3f} | {method_timings['tcn']['32']['median_ms_per_sample']:.3f} | {method_timings['tcn']['128']['median_ms_per_sample']:.3f} |",
        f"| TCN+DYN | {method_timings['tcn_dyn']['1']['median_ms_per_sample']:.3f} | {method_timings['tcn_dyn']['32']['median_ms_per_sample']:.3f} | {method_timings['tcn_dyn']['128']['median_ms_per_sample']:.3f} |",
        f"| Full | {method_timings['full']['1']['median_ms_per_sample']:.3f} | {method_timings['full']['32']['median_ms_per_sample']:.3f} | {method_timings['full']['128']['median_ms_per_sample']:.3f} |",
        "",
        "## 在线路径精度核对",
        "",
        f"- full macro-F1：{checks['full_macro_f1']*100:.2f}%；E12 保存参考：{checks['reference_macro_f1']*100:.2f}%",
        f"- full 概率与参考最大差异：{checks['reference_prob_max_abs_diff']:.2e}",
        "",
        "## 边界",
        "",
        "- 只测 seed42/40% 的推理，不是多 seed 成本均值；batch=1 受 Python/CPU 开销影响。",
        "- 包含预处理、DYN、双流 MOMENT、TCN 和 heads，不是缓存分类头速度。",
        "- 没有并发、服务化、磁盘 I/O、数据解析、网络或端侧硬件；不是部署 SLA。",
        "- 没有做能耗、模型量化/编译或 TensorRT；GPU 只测了这一张卡。",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
