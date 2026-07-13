from __future__ import annotations

import argparse
from pathlib import Path

from tcn_moment.config import load_config, with_random_seed
from tcn_moment.train_tcn import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 1D-CNN baseline.")
    parser.add_argument(
        "--config",
        default="configs/experiments/cnn_baseline.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--seed", type=int, help="Override random seed and use its split manifest.")
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run-name", help="Unique output directory name for a new run.")
    run_group.add_argument("--resume", type=Path, help="Existing run directory to resume.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.seed is not None:
        config = with_random_seed(config, args.seed)
    train(
        config,
        config_path,
        run_name=args.run_name,
        resume_dir=args.resume,
        model_name="CNN",
    )


if __name__ == "__main__":
    main()
