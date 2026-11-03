"""TODO1102-3D latency benchmark for the TCN tier (torch vs ONNX runtime).

Accuracy is reported by ``todo1102_quantization.py``; this script only measures
throughput, with thread counts made explicit so the numbers are reproducible
across runs (the first version inherited whatever OMP_NUM_THREADS happened to be
set to, which moved the CPU figures by 4x).

Reports median per-sample latency over repeated timings for:
  torch-CUDA fp32, torch-CPU fp32 (1 / 8 threads), ORT-CPU fp32 and int8 (1 / 8).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.data import parse_power_sequence, prepare_sequence, sequence_mask

DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/quantization")
E12 = Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_20261020_v1")
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")


def _batch(dataset: Path, ids: np.ndarray, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    frame = pd.read_csv(dataset, sep=" ", dtype=str).set_index("charging_station_id")
    rows = frame.loc[list(ids)]
    values, masks = [], []
    for text in rows["charging_powers_str"]:
        sequence = parse_power_sequence(text)
        values.append(prepare_sequence(sequence, max_length, "none"))
        masks.append(sequence_mask(len(sequence), max_length))
    return np.stack(values), np.stack(masks)


def _time(fn: Callable[[], None], repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/最新多.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="40p")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    max_length = 1024
    pilot.configure()
    archive = np.load(E08 / f"seed{args.seed}" / "features.npz", allow_pickle=True)
    val_ids = archive["validation_ids"].astype(str)
    x_val, mask_val = _batch(args.dataset, val_ids, max_length)

    model = pilot.build_model("legacy_masked50")
    model.load_state_dict(
        torch.load(
            E12 / f"seed{args.seed}" / "models" / args.tag / "full.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    model_gpu = copy.deepcopy(model).to("cuda") if torch.cuda.is_available() else None

    fp32_path = args.output / f"tcn_seed{args.seed}_{args.tag}_fp32.onnx"
    int8_path = args.output / f"tcn_seed{args.seed}_{args.tag}_int8.onnx"
    if not fp32_path.exists():
        raise FileNotFoundError(f"Run todo1102_quantization.py first: {fp32_path}")

    import onnxruntime as ort

    sessions: dict[tuple[str, int], Any] = {}
    for name, path in (("ort_cpu_fp32", fp32_path), ("ort_cpu_int8", int8_path)):
        for threads in (1, 8):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            sessions[(name, threads)] = ort.InferenceSession(
                path.as_posix(), options, providers=["CPUExecutionProvider"]
            )

    rows: list[dict[str, Any]] = []
    for batch in args.batches:
        x_tensor = torch.as_tensor(x_val[:batch, None, :], dtype=torch.float32)
        mask_tensor = torch.as_tensor(mask_val[:batch], dtype=torch.float32)
        feed = {
            "signal": x_val[:batch, None, :].astype(np.float32),
            "mask": mask_val[:batch].astype(np.float32),
        }

        def run_torch_cpu(target_threads: int) -> None:
            torch.set_num_threads(target_threads)
            with torch.no_grad():
                model(x_tensor, mask_tensor)

        for threads in (1, 8):
            seconds = _time(lambda: run_torch_cpu(threads), args.repeats)
            rows.append(
                {
                    "backend": "torch_cpu_fp32",
                    "threads": threads,
                    "batch": batch,
                    "ms_per_sample": seconds / batch * 1000.0,
                }
            )

        if model_gpu is not None:
            gpu_x = x_tensor.to("cuda")
            gpu_mask = mask_tensor.to("cuda")

            def run_torch_gpu() -> None:
                with torch.no_grad():
                    model_gpu(gpu_x, gpu_mask)
                torch.cuda.synchronize()

            seconds = _time(run_torch_gpu, args.repeats)
            rows.append(
                {
                    "backend": "torch_cuda_fp32",
                    "threads": 0,
                    "batch": batch,
                    "ms_per_sample": seconds / batch * 1000.0,
                }
            )

        for (name, threads), session in sessions.items():
            def run_ort(session: Any = session) -> None:
                session.run(["logits"], feed)

            seconds = _time(run_ort, args.repeats)
            rows.append(
                {
                    "backend": name,
                    "threads": threads,
                    "batch": batch,
                    "ms_per_sample": seconds / batch * 1000.0,
                }
            )
        print(f"batch {batch} done", flush=True)

    with (args.output / "latency.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["backend", "threads", "batch", "ms_per_sample"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    payload = {
        "seed": args.seed,
        "tag": args.tag,
        "repeats": args.repeats,
        "torch": torch.__version__,
        "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "rows": rows,
    }
    (args.output / "latency.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["# TODO1102-3D latency (median of repeated timings)", "",
             "| 后端 | 线程 | batch | ms/样本 | 样本/秒 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['threads']} | {row['batch']} | "
            f"{row['ms_per_sample']:.2f} | {1000 / row['ms_per_sample']:.0f} |"
        )
    (args.output / "latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
