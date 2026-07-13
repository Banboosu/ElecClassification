from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tcn_moment.io_utils import atomic_write_json


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key != "loss"
    }


def _group_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def _read_records(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    run_name = str(payload.get("run_name", run_dir.name))
    if payload.get("model") == "STATISTICAL_BASELINES":
        return [
            {
                "group": _group_name(run_name),
                "run_name": run_name,
                "model": name,
                **_scalar_metrics(result["test_metrics"]),
            }
            for name, result in payload["results"].items()
        ]
    return [
        {
            "group": _group_name(run_name),
            "run_name": run_name,
            "model": str(payload["model"]),
            **_scalar_metrics(payload["test_metrics"]),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated experiment results.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/summaries"))
    args = parser.parse_args()

    records = []
    for run_dir in args.runs:
        if not (run_dir / "metrics.json").is_file():
            raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
        records.extend(_read_records(run_dir))
    frame = pd.DataFrame(records)
    metric_columns = [
        column for column in frame.columns if column not in {"group", "run_name", "model"}
    ]
    summary = (
        frame.groupby(["group", "model"])[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"summary_{timestamp}.csv"
    json_path = args.output_dir / f"summary_{timestamp}.json"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    atomic_write_json(
        json_path,
        {
            "runs": [str(path) for path in args.runs],
            "records": records,
            "summary": summary.to_dict(orient="records"),
        },
    )
    print(f"Saved summaries to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
