# Experiment presets

Each file inherits `configs/moment.yaml` and changes only the variable named in the filename.
The resolved configuration is saved in every run directory.

- `normalization_*`: compare raw, min-max, and z-score inputs.
- `length_*`: compare sequence lengths 256, 512, 816, and 1024.
- `moment_linear_probe.yaml`: train only the classification head.
- `moment_svm_rbf.yaml`: freeze MOMENT embeddings and use the paper-aligned RBF-SVM protocol.
- `moment_partial_finetune.yaml`: unfreeze the final two encoder blocks.
- `moment_full_finetune.yaml`: train the full MOMENT model.
- `moment_head_lr_*`: linear-probe head learning-rate comparison.

Run a single preset with `--config`, or use `experiment-suite` for multiple seeds.
