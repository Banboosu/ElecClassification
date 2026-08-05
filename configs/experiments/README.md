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
- `few_shot/`: nested, stratified 1%, 5%, 10%, 20%, and 40% label-budget
  comparisons for TCN and frozen MOMENT + RBF-SVM.
- `pretraining_ablation/`: M01 same-architecture random MOMENT encoder control at
  1%, 5%, and 10% label budgets; it directly constructs the model and never loads
  pretrained checkpoint tensors.
- `moment_imputation_zero_shot.yaml`: evaluate the pretrained reconstruction head
  without parameter updates on random-patch and contiguous-block missingness.
- `moment_retrieval_zero_shot.yaml`: evaluate frozen MOMENT representations for
  label-free similar-sequence retrieval and masked-query robustness.

Run a single preset with `--config`, or use `experiment-suite` for multiple seeds.
