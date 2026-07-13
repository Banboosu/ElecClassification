from __future__ import annotations

import argparse
import json

import momentfm
import torch

from tcn_moment.experiment import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the locked training environment.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit with an error when CUDA is unavailable.",
    )
    args = parser.parse_args()
    environment = collect_environment(torch)
    print(json.dumps(environment, indent=2, ensure_ascii=False))
    if environment["packages"]["momentfm"] != "0.1.4":
        raise SystemExit("Expected momentfm 0.1.4.")
    if environment["packages"]["torch"] != "2.12.1":
        raise SystemExit("Expected torch 2.12.1.")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false.")
    if not hasattr(momentfm, "MOMENTPipeline"):
        raise SystemExit("momentfm.MOMENTPipeline is unavailable.")


if __name__ == "__main__":
    main()
