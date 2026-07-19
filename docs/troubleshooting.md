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

## TCN 在训练后期出现 `Non-finite loss`

V100 的 CUDA AMP 使用 FP16。TCN 的多层扩张卷积可能在训练后期产生超出 FP16 表示范围的
中间激活；梯度裁剪发生在反向传播阶段，无法阻止前向传播先溢出。因此即使输入已经做 z-score
归一化并启用了梯度裁剪，仍可能出现：

```text
FloatingPointError: Non-finite loss detected at epoch 22.
```

项目现在将 TCN/CNN 默认设为 FP32（`tcn_training.amp: false`）。如果实验配置显式开启 AMP，
训练循环在检测到非有限损失后会用 FP32 重算当前 batch，并在本次运行的后续训练中关闭 AMP。
若 FP32 重算仍失败，错误信息会同时报告输入和 logits 是否有限，以便区分数据异常与模型参数
异常。

失败运行在第 21 轮结束时已保存 `checkpoint_latest.pt`。如需快速验证修复，可以先从 suite 汇总中
找到失败记录的 `config`、`seed` 和 `run_name`：

```bash
cat artifacts/suites/suite_20260719_082722.json
```

更新代码后，使用该记录中的原配置和随机种子恢复；下面的尖括号内容需要替换：

```bash
uv sync --frozen
uv run tcn-train \
  --config <config> \
  --seed <seed> \
  --resume artifacts/tcn/<run_name>
```

不要在 `--resume` 时省略 `--seed`，否则 seed 不为 42 的运行会加载错误的数据划分清单。

上述恢复方式只用于验证和调试，因为该运行会混合“前 21 轮 FP16、后续 FP32”两种数值协议。
正式论文的多随机种子结果必须在更新代码后，用新的 `suite-name` 从第 1 轮统一重跑，确保所有
配置和随机种子都使用 `tcn_training.amp: false`。最终 `metrics.json` 会记录
`amp_requested`、`amp_enabled`、`amp_fallback_triggered` 和 `amp_fallback_epoch`；正式汇总时不应
纳入触发过 AMP 回退的运行。
