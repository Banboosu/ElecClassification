"""TODO1102-3D: int8 quantization of the low-cost TCN+DYN tier.

`todo1102.md` §3d asks for one concrete deployment number: quantize the TCN to
int8/ONNX, measure the latency gain and the accuracy cost.  The frozen MOMENT
encoder is *not* touched — the whole point of the low-cost tier is that it runs
without the 341 M-parameter backbone.

What this script measures, per seed:

* TCN accuracy: fp32 (torch), ONNX fp32, ONNX int8 (static QDQ with calibration);
* fused accuracy: the meta-learner is refitted on the frozen E12 OOF
  probabilities and applied to `[dyn60, md, tcn]`, swapping only the TCN block.
  This isolates the quantization cost from any retraining;
* latency: batch 1 / 32 per-sample timings for torch-GPU, torch-CPU, ORT-CPU
  fp32 and ORT-CPU int8.

Validation only; the test split is never read.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcn_moment import run_e08
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.data import parse_power_sequence, prepare_sequence, sequence_mask
from tcn_moment.label_semantics import BATTERY_LABEL, class_index

CLASSES = ["0", "1", "2"]
BATTERY_INDEX = class_index(CLASSES, BATTERY_LABEL)
SEEDS = (42, 44, 45, 46)
TAG = "40p"
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
E09 = Path("artifacts/improvements/e09_diff_stream")
E12 = Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_20261020_v1")
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/quantization")


class TCNExport(torch.nn.Module):
    """ONNX-friendly wrapper: (B, 1, T) signal + (B, T) mask -> logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, mask)


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    probs = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return ids, y, probs


def _align(reference: np.ndarray, ids: np.ndarray, probs: np.ndarray) -> np.ndarray:
    if np.array_equal(ids, reference):
        return probs
    mapping = {value: index for index, value in enumerate(ids)}
    if set(ids.tolist()) != set(reference.tolist()):
        raise ValueError("Prediction ID sets differ")
    return probs[[mapping[value] for value in reference]]


def _sequences(
    dataset: Path, ids: np.ndarray, max_length: int
) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    frame = pd.read_csv(dataset, sep=" ", dtype=str).set_index("charging_station_id")
    rows = frame.loc[list(ids)]
    values = []
    masks = []
    for text in rows["charging_powers_str"]:
        sequence = parse_power_sequence(text)
        values.append(prepare_sequence(sequence, max_length, "none"))
        masks.append(sequence_mask(len(sequence), max_length))
    return np.stack(values), np.stack(masks)


def _macro_f1(y: np.ndarray, prediction: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, prediction, average="macro"))


def _meta(oof: np.ndarray, y: np.ndarray, evaluation: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    grid = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    selected, _ = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        grid,
        oof,
        y,
        folds=3,
        seed=seed,
    )
    model = LogisticRegression(C=selected["C"], max_iter=5000, random_state=seed).fit(oof, y)
    return model.predict(evaluation)


def _export(model: torch.nn.Module, path: Path, max_length: int) -> None:
    wrapper = TCNExport(model).eval()
    dummy_x = torch.zeros(2, 1, max_length, dtype=torch.float32)
    dummy_mask = torch.ones(2, max_length, dtype=torch.float32)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "tcn.onnx"
        torch.onnx.export(
            wrapper,
            (dummy_x, dummy_mask),
            temporary.as_posix(),
            input_names=["signal", "mask"],
            output_names=["logits"],
            dynamic_axes={"signal": {0: "batch"}, "mask": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
        Path(path).write_bytes(temporary.read_bytes())


def _quantize(source: Path, destination: Path, calibration: list[dict[str, np.ndarray]]) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self, items: list[dict[str, np.ndarray]]) -> None:
            self._iterator = iter(items)

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self._iterator, None)

    quantize_static(
        source.as_posix(),
        destination.as_posix(),
        Reader(calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def _ort_logits(path: Path, x: np.ndarray, mask: np.ndarray, threads: int) -> np.ndarray:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(path.as_posix(), options, providers=["CPUExecutionProvider"])
    chunks = []
    for start in range(0, len(x), 64):
        batch = {
            "signal": x[start : start + 64, None, :].astype(np.float32),
            "mask": mask[start : start + 64].astype(np.float32),
        }
        chunks.append(session.run(["logits"], batch)[0])
    return np.concatenate(chunks)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _latency(fn: Any, batch: int, repeats: int) -> float:
    fn(batch)  # warm-up
    started = time.perf_counter()
    for _ in range(repeats):
        fn(batch)
    return (time.perf_counter() - started) / repeats / batch * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/最新多.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--calibration-rows", type=int, default=256)
    parser.add_argument("--e08", type=Path, default=E08)
    parser.add_argument("--e09", type=Path, default=E09)
    parser.add_argument("--e12", type=Path, default=E12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot.configure()
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    max_length = 1024

    for seed in args.seeds:
        archive = np.load(args.e08 / f"seed{seed}" / "features.npz", allow_pickle=True)
        val_ids = archive["validation_ids"].astype(str)
        x_val, mask_val = _sequences(args.dataset, val_ids, max_length)
        y_val = archive["validation_labels"]

        model = pilot.build_model("legacy_masked50")
        state = torch.load(
            args.e12 / f"seed{seed}" / "models" / args.tag / "full.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        model_gpu = copy.deepcopy(model).to(device)

        x_tensor = torch.as_tensor(x_val[:, None, :], dtype=torch.float32)
        mask_tensor = torch.as_tensor(mask_val, dtype=torch.float32)
        with torch.no_grad():
            torch_logits = model_gpu(x_tensor.to(device), mask_tensor.to(device)).cpu().numpy()

        onnx_path = args.output / f"tcn_seed{seed}_{args.tag}_fp32.onnx"
        int8_path = args.output / f"tcn_seed{seed}_{args.tag}_int8.onnx"
        _export(model, onnx_path, max_length)

        calibration_ids = np.load(
            args.e08 / f"seed{seed}" / f"train_ids_{args.tag}.npy", allow_pickle=True
        ).astype(str)[: args.calibration_rows]
        x_cal, mask_cal = _sequences(args.dataset, calibration_ids, max_length)
        calibration = [
            {
                "signal": x_cal[start : start + 32, None, :].astype(np.float32),
                "mask": mask_cal[start : start + 32].astype(np.float32),
            }
            for start in range(0, len(x_cal), 32)
        ]
        _quantize(onnx_path, int8_path, calibration)

        onnx_logits = _ort_logits(onnx_path, x_val, mask_val, threads=8)
        int8_logits = _ort_logits(int8_path, x_val, mask_val, threads=8)
        drift_onnx = float(np.max(np.abs(onnx_logits - torch_logits)))
        drift_int8 = float(np.max(np.abs(int8_logits - torch_logits)))
        fp32_scale = float(np.max(np.abs(torch_logits)))
        int8_scale = float(np.max(np.abs(int8_logits)))

        torch_f1 = _macro_f1(y_val, torch_logits.argmax(1))
        onnx_f1 = _macro_f1(y_val, onnx_logits.argmax(1))
        int8_f1 = _macro_f1(y_val, int8_logits.argmax(1))

        # fused system: refit the meta-learner on E12 OOF, swap only the TCN block
        oof = np.load(args.e12 / f"seed{seed}" / f"oof_{args.tag}.npz", allow_pickle=False)
        y_train = oof["y_train"]
        dyn_ids, _, dyn_val = _read_predictions(
            args.e08 / f"seed{seed}" / "conditions" / args.tag / "dyn_hgb" / "predictions.csv"
        )
        md_ids, md_y, md_val = _read_predictions(
            args.e09 / f"e09_diff_stream_20261020_seed{seed}" / "conditions" / args.tag
            / "moment_diff_logit" / "predictions.csv"
        )
        dyn_val = _align(val_ids, dyn_ids, dyn_val)
        md_val = _align(val_ids, md_ids, md_val)
        if not np.array_equal(md_y, y_val):
            raise ValueError("Validation labels disagree")

        fused = {}
        for label, probabilities in (
            ("fp32", torch_logits),
            ("onnx_fp32", onnx_logits),
            ("onnx_int8", int8_logits),
        ):
            prediction = _meta(
                np.hstack([oof["oof_dyn"], oof["oof_md"], oof["oof_tcn"]]),
                y_train,
                np.hstack([dyn_val, md_val, _softmax(probabilities)]),
                seed,
            )
            fused[label] = _macro_f1(y_val, prediction)
            rows.append(
                {
                    "seed": seed,
                    "tag": args.tag,
                    "backend": label,
                    "tcn_macro_f1": torch_f1 if label == "fp32" else onnx_f1 if label == "onnx_fp32" else int8_f1,
                    "fused_macro_f1": fused[label],
                    "max_abs_logit_drift": 0.0 if label == "fp32" else drift_onnx if label == "onnx_fp32" else drift_int8,
                    "onnx_size_mb": onnx_path.stat().st_size / 1e6 if label != "fp32" else float("nan"),
                    "fp32_max_abs_logit": fp32_scale,
                    "int8_max_abs_logit": int8_scale,
                }
            )
            if label == "onnx_int8":
                rows[-1]["onnx_size_mb"] = int8_path.stat().st_size / 1e6
        print(
            f"seed {seed}: TCN fp32={100 * torch_f1:.2f} onnx={100 * onnx_f1:.2f} "
            f"int8={100 * int8_f1:.2f} | fused fp32={100 * fused['fp32']:.2f} "
            f"int8={100 * fused['onnx_int8']:.2f} | drift onnx={drift_onnx:.2e} int8={drift_int8:.2f}",
            flush=True,
        )

        if seed == args.seeds[0]:
            import onnxruntime as ort

            sample_x = torch.as_tensor(x_val[:64, None, :], dtype=torch.float32)
            sample_mask = torch.as_tensor(mask_val[:64], dtype=torch.float32)
            sessions = {
                "ort_fp32": ort.InferenceSession(
                    onnx_path.as_posix(), providers=["CPUExecutionProvider"]
                ),
                "ort_int8": ort.InferenceSession(
                    int8_path.as_posix(), providers=["CPUExecutionProvider"]
                ),
            }

            def torch_call(batch: int, target: torch.device) -> None:
                network = model_gpu if target.type == "cuda" else model
                with torch.no_grad():
                    network(
                        sample_x[:batch].to(target),
                        sample_mask[:batch].to(target),
                    )

            def ort_call(batch: int, session: Any) -> None:
                session.run(
                    ["logits"],
                    {
                        "signal": sample_x[:batch].numpy(),
                        "mask": sample_mask[:batch].numpy(),
                    },
                )

            for batch in (1, 32):
                latency_rows.append({"backend": "torch_gpu_fp32", "batch": batch, "ms_per_sample": _latency(lambda b: torch_call(b, torch.device("cuda")), batch, 50)})
                latency_rows.append({"backend": "torch_cpu_fp32", "batch": batch, "ms_per_sample": _latency(lambda b: torch_call(b, torch.device("cpu")), batch, 50)})
                latency_rows.append({"backend": "ort_cpu_fp32", "batch": batch, "ms_per_sample": _latency(lambda b: ort_call(b, sessions["ort_fp32"]), batch, 50)})
                latency_rows.append({"backend": "ort_cpu_int8", "batch": batch, "ms_per_sample": _latency(lambda b: ort_call(b, sessions["ort_int8"]), batch, 50)})

    with (args.output / "accuracy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (args.output / "latency.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["backend", "batch", "ms_per_sample"])
        writer.writeheader()
        for row in latency_rows:
            writer.writerow(row)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "torch": torch.__version__,
        "seeds": args.seeds,
        "tag": args.tag,
        "calibration_rows": args.calibration_rows,
        "accuracy": {},
        "latency": latency_rows,
        "test_evaluated": False,
    }
    for backend in ("fp32", "onnx_fp32", "onnx_int8"):
        selected = [row for row in rows if row["backend"] == backend]
        summary["accuracy"][backend] = {
            "n": len(selected),
            "tcn_macro_f1_mean": float(np.mean([row["tcn_macro_f1"] for row in selected]) * 100),
            "fused_macro_f1_mean": float(np.mean([row["fused_macro_f1"] for row in selected]) * 100),
            "max_abs_logit_drift": float(np.nanmax([row["max_abs_logit_drift"] for row in selected])),
            "size_mb": float(np.nanmean([row["onnx_size_mb"] for row in selected])),
            "fp32_max_abs_logit": float(np.max([row["fp32_max_abs_logit"] for row in selected])),
            "int8_max_abs_logit": float(np.max([row["int8_max_abs_logit"] for row in selected])),
        }
    for backend in ("onnx_fp32", "onnx_int8"):
        diffs = [
            (
                next(
                    row["fused_macro_f1"]
                    for row in rows
                    if row["seed"] == seed and row["backend"] == backend
                )
                - next(
                    row["fused_macro_f1"]
                    for row in rows
                    if row["seed"] == seed and row["backend"] == "fp32"
                )
            )
            * 100
            for seed in args.seeds
        ]
        summary["accuracy"][f"{backend}_minus_fp32_pp"] = {
            "mean": float(np.mean(diffs)),
            "std": float(np.std(diffs)),
            "value": diffs,
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["# TODO1102-3D int8 quantization of the TCN tier", ""]
    lines += [
        f"- 主机：{summary['host']}，torch {summary['torch']}",
        f"- 预算：{args.tag}，种子 {args.seeds}，校准样本 {args.calibration_rows}",
        "",
        "| 后端 | TCN Macro-F1 | 融合 Macro-F1 | 最大 logit 偏差 | 模型大小 |",
        "|---|---:|---:|---:|---:|",
    ]
    for backend, name in (("fp32", "torch fp32"), ("onnx_fp32", "ONNX fp32"), ("onnx_int8", "ONNX int8")):
        entry = summary["accuracy"][backend]
        size = "—" if np.isnan(entry["size_mb"]) else f"{entry['size_mb']:.1f} MB"
        lines.append(
            f"| {name} | {entry['tcn_macro_f1_mean']:.2f} | {entry['fused_macro_f1_mean']:.2f} | "
            f"{entry['max_abs_logit_drift']:.2f} | {size} |"
        )
    lines += [
        "",
        f"- fp32 最大 |logit|：{summary['accuracy']['fp32'].get('fp32_max_abs_logit', float('nan')):.1f}；"
        f"int8 输出被压到 {summary['accuracy']['onnx_int8'].get('int8_max_abs_logit', float('nan')):.1f}，"
        "说明激活动态范围超出 per-tensor int8 的可表示区间。",
        "",
        "| 后端 | batch | ms/样本 |",
        "|---|---:|---:|",
    ]
    for row in latency_rows:
        lines.append(f"| {row['backend']} | {row['batch']} | {row['ms_per_sample']:.2f} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
