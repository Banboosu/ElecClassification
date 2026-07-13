# Charging Power MOMENT

This project experiments with charging-power time-series classification.

The original TCN scripts and outputs are preserved, while the new project layout uses `uv`,
`pyproject.toml`, and a package under `src/`.

## Layout

```text
.
├── configs/
│   └── moment.yaml                 # shared data protocol and model configs
├── data/
│   └── raw/
│       └── 最新多.csv              # source charging-power data
├── src/
│   └── tcn_moment/
│       ├── config.py               # YAML config loader
│       ├── data.py                 # CSV parsing and dataset preparation
│       ├── metrics.py              # metrics shared by all classifiers
│       ├── train_moment.py         # MOMENT classifier training entrypoint
│       └── train_tcn.py            # PyTorch TCN training entrypoint
├── artifacts/
│   └── tcn/                        # previous TCN models, metrics, and plots
└── legacy/                         # previous standalone scripts
```

## Environment

The stable branch is locked for a reproducible MOMENT run:

- Python `3.11`
- `momentfm 0.1.4`
- `torch 2.12.1`
- `numpy 1.25.2`

Install the exact lock-file environment on the Linux CUDA machine:

```bash
uv sync --frozen
uv run moment-check-environment --require-cuda
```

The environment check prints the Python, package, CUDA, cuDNN, NVIDIA driver, and GPU details and
fails early if the locked versions or CUDA are unavailable.

## Inspect Data

This command does not require MOMENT or PyTorch:

```bash
uv run moment-inspect-data --config configs/moment.yaml
```

It parses `charging_powers_str`, filters short sequences, pads/truncates each series to
`max_length`, encodes `InsertedColumn`, and prints the train/validation/test shapes and label
counts.

The first inspection creates `artifacts/splits/unified_split.json`. It stores the exact sample IDs
for every subset, their class counts, the filtering protocol, and the source CSV SHA-256. Later
runs reuse this file and fail if the data or split protocol changed. Rebuild it only intentionally:

```bash
uv run moment-inspect-data --config configs/moment.yaml --rebuild-split
```

Both TCN and MOMENT read the same `data` section in `configs/moment.yaml`. The default protocol
uses a stratified 70%/10%/20% train/validation/test split with random state 42. Validation data is
used during training; test data is reserved for the final report.

Rows whose label is listed in `data.invalid_labels` are separated before training. By default,
label `5` is treated as invalid/incomplete data and is not included as a classification class.

## Train MOMENT Classifier

```bash
uv run moment-train --config configs/moment.yaml --run-name moment_zscore_seed42
```

## Train TCN Baseline

Run this command on the CUDA machine to train the TCN under exactly the same data protocol:

```bash
uv run tcn-train --config configs/moment.yaml --run-name tcn_zscore_seed42
```

The two trainers report the same metric set: accuracy, balanced accuracy, macro precision/recall/F1,
weighted precision/recall/F1, confusion matrix, and per-class classification results. Outputs are
written to unique subdirectories under `artifacts/moment/` and `artifacts/tcn/`.

The initial results recorded before protocol unification are documented in
`docs/experiment_records/initial_baseline_results.md`.

Each run directory is self-contained:

```text
artifacts/<model>/<run_name>/
├── checkpoint_latest.pt
├── config.yaml
├── environment.json
├── label_encoder.pkl
├── metrics.json
├── metrics_partial.json
├── resolved_config.json
├── split_manifest.json
├── status.json
└── <model>_classifier_best.pt
```

`status.json` records whether the run is `running`, `completed`, `interrupted`, or `failed`.
Configuration and metric JSON files are written atomically. To resume an interrupted run, first
increase the epoch limit in the config if necessary, then pass its directory:

```bash
uv run moment-train --config configs/moment.yaml \
  --resume artifacts/moment/moment_zscore_seed42

uv run tcn-train --config configs/moment.yaml \
  --resume artifacts/tcn/tcn_zscore_seed42
```

## Notes

- The default model is `AutonLab/MOMENT-1-large`.
- The input is treated as a single-channel time series with shape `[batch, 1, length]`.
- Both models receive an explicit valid-timestep mask. MOMENT uses `input_mask`; TCN uses masked
  global pooling, so padded values are excluded from the final feature average.
- `configs/moment.yaml` controls sequence length, normalization, train/validation/test split, epochs,
  batch size, and learning rate.
- `data.invalid_labels` defaults to `["5"]`, so incomplete samples are reported separately
  instead of being used as a model class.
- The legacy TCN scripts still use old relative paths internally. If you want to run them again,
  either update their paths or run them from a copied layout matching the original top-level files.
