# Troubleshooting

## `MOMENTPipeline.__init__() missing ... config`

Observed on the remote CUDA host before training started:

```text
TypeError: MOMENTPipeline.__init__() missing 1 required positional argument: 'config'
```

`momentfm 0.1.4` uses `PyTorchModelHubMixin`. Although the official model repository contains a
`config.json`, the Hub loader did not inject it into the constructor in this run. The project now
stores an exact copy of the official `AutonLab/MOMENT-1-large` configuration at
`configs/models/moment-1-large.json` and explicitly passes it to `from_pretrained`.

After updating the project, verify model loading before starting a suite:

```bash
uv sync --frozen
uv run moment-check-model --config configs/moment.yaml
```

The old failed run directory can be kept for audit purposes. New suite runs include a timestamp in
their names, so retrying does not collide with `moment_linear_probe_seed42`.
