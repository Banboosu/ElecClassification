# Charging Power MOMENT

This project experiments with charging-power time-series classification.

The original TCN scripts and outputs are preserved, while the new project layout uses `uv`,
`pyproject.toml`, and a package under `src/`.

## Layout

```text
.
├── configs/
│   └── moment.yaml                 # MOMENT experiment config
├── data/
│   └── raw/
│       └── 最新多.csv              # source charging-power data
├── src/
│   └── tcn_moment/
│       ├── config.py               # YAML config loader
│       ├── data.py                 # CSV parsing and dataset preparation
│       └── train_moment.py         # MOMENT classifier training entrypoint
├── artifacts/
│   └── tcn/                        # previous TCN models, metrics, and plots
└── legacy/                         # previous standalone scripts
```

## Environment

The stable branch is locked for a reproducible MOMENT run:

- Python `3.11`
- `momentfm 0.1.4`
- `torch 2.3.1`
- `numpy 1.25.2`

The `experiment/latest-software` branch is for dependency experiments that still resolve with
the current `momentfm 0.1.4` package:

- Python `3.11`
- `momentfm >=0.1.4`
- `torch >=2.7`
- `numpy 1.25.2`

Install all project dependencies with one command:

```powershell
uv sync
```

The project does not use optional dependency groups. `uv sync` installs the data, MOMENT,
PyTorch, and plotting dependencies together.

Newer Python, PyTorch, and CUDA versions may work, but they are not the stable reproducible
environment yet. Treat them as an upgrade experiment: regenerate `uv.lock`, then verify that
`momentfm`, `transformers`, and the training entrypoint still import and run correctly.
NumPy is still pinned to `1.25.2` because `momentfm 0.1.4` requires that exact version.
Python 3.12 was tested as an experiment, but dependency resolution failed because current
pandas builds for Python 3.12 require a newer NumPy than MOMENT allows.
This machine has no CUDA GPU, so the experiment uses the CPU PyTorch wheel.

## Inspect Data

This command does not require MOMENT or PyTorch:

```powershell
uv run moment-inspect-data --config configs/moment.yaml
```

It parses `charging_powers_str`, filters short sequences, pads/truncates each series to
`max_length`, encodes `InsertedColumn`, and prints the train/test shape and label counts.

Rows whose label is listed in `data.invalid_labels` are separated before training. By default,
label `5` is treated as invalid/incomplete data and is not included as a classification class.

## Train MOMENT Classifier

```powershell
uv run moment-train --config configs/moment.yaml
```

Outputs are written to:

```text
artifacts/moment/
├── checkpoint_latest.pt
├── label_encoder.pkl
├── metrics.json
├── metrics_partial.json
└── moment_classifier.pt
```

## Notes

- The default model is `AutonLab/MOMENT-1-large`.
- The input is treated as a single-channel time series with shape `[batch, 1, length]`.
- `configs/moment.yaml` controls sequence length, normalization, train/test split, epochs,
  batch size, and learning rate.
- `data.invalid_labels` defaults to `["5"]`, so incomplete samples are reported separately
  instead of being used as a model class.
- The legacy TCN scripts still use old relative paths internally. If you want to run them again,
  either update their paths or run them from a copied layout matching the original top-level files.
