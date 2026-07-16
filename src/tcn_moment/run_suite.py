from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tcn_moment.io_utils import atomic_write_json


MODULES = {
    "moment": "tcn_moment.train_moment",
    "tcn": "tcn_moment.train_tcn",
    "cnn": "tcn_moment.train_cnn",
    "baseline": "tcn_moment.train_baselines",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment preset for multiple seeds.")
    parser.add_argument("--model", choices=sorted(MODULES), required=True)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--suite-name", help="Tag included in every run name.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    suite_started = datetime.now().isoformat()
    suite_name = args.suite_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []
    for config_path in args.configs:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config does not exist: {config_path}")
        for seed in args.seeds:
            run_name = f"{config_path.stem}_{suite_name}_seed{seed}"
            command = [
                sys.executable,
                "-m",
                MODULES[args.model],
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--run-name",
                run_name,
            ]
            print("Running:", " ".join(command))
            completed = subprocess.run(command, check=False)
            record = {
                "model": args.model,
                "config": str(config_path),
                "seed": seed,
                "run_name": run_name,
                "return_code": completed.returncode,
            }
            records.append(record)
            if completed.returncode != 0 and not args.continue_on_error:
                break
        if records[-1]["return_code"] != 0 and not args.continue_on_error:
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("artifacts/suites") / f"suite_{timestamp}.json"
    atomic_write_json(
        output_path,
        {
            "started_at": suite_started,
            "finished_at": datetime.now().isoformat(),
            "suite_name": suite_name,
            "records": records,
        },
    )
    failures = [record for record in records if record["return_code"] != 0]
    print(f"Saved suite summary to {output_path}")
    if failures:
        raise SystemExit(f"{len(failures)} experiment(s) failed.")


if __name__ == "__main__":
    main()
