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

## MOMENT v1 线性探测耗时过长且 pooling 未排除 padding

`momentfm 0.1.4` 的分类编码器会接收 `input_mask`，但默认 `ClassificationHead` 最后直接对全部
patch embeddings 求均值，没有在池化阶段排除 padding patch。当前数据的序列长度与类别存在
关联，因此旧 `thesis_moment_strategy_v1` 只能作为诊断结果，不能进入正式论文主表。

项目的 MOMENT v2 训练路径会读取模型返回的 patch embeddings，使用由 `input_mask` 转换得到的
完整有效 patch mask 做加权平均，再调用原分类线性层。线性探测还会把三个 split 的冻结特征各
提取一次，后续 epoch 不再重复运行 backbone。

冻结和部分微调策略只在 checkpoint 中保存可训练参数；完全微调仍保存完整模型。旧 v1
checkpoint 缺少 v2 协议标记，不能通过 `--resume` 混入新运行。更新代码后先执行：

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run moment-check-model --config configs/experiments/moment_linear_probe.yaml
```

然后只做一个 seed 的冒烟运行：

```bash
uv run moment-train \
  --config configs/experiments/moment_linear_probe.yaml \
  --seed 42 \
  --run-name moment_linear_probe_v2_smoke_seed42
```

日志中应先出现 `Caching frozen MOMENT features with mask-aware pooling...`。最终
`metrics.json` 中应满足 `mask_aware_pooling=true`、`feature_cache_enabled=true` 和
`checkpoint_model_state_scope="trainable"`，并记录 `moment_protocol_version=2`。

## V100 32GB 上的 MOMENT 吞吐与显存

当前默认值按 6 vCPU、25GB 内存和单卡 V100 32GB 设置：训练加载使用 4 个 worker，FP16、
pin-memory 和 fused AdamW 默认开启；验证同样使用 FP16。线性探测的冻结特征以 batch 64 提取
一次、分类头以 `32 x 1` 训练，partial 和 full 都使用 `32 x 1`，三种策略的有效 batch 都是
32。full 默认关闭梯度检查点，利用 V100 32GB 的显存换取更少的反向重算。
考虑到数据盘只有 50GB，成功运行默认删除包含优化器状态的 `checkpoint_latest.pt`，仅保留最佳
模型；中断/失败运行不会删除。若需要保存成功运行的完整恢复点，设置
`training.keep_completed_checkpoint: true`。

每轮在 `metrics_partial.json` 和最终 `metrics.json` 中记录 `train_seconds`、
`validation_seconds`、`train_samples_per_second`、`peak_gpu_memory_mb`、
`physical_batch_size` 和 `effective_batch_size`。比较优化效果时应至少完成一个完整 epoch，不要只用
模型下载和首次 CUDA kernel 初始化阶段计时。

若 full fine-tune 出现 CUDA OOM，只修改物理 batch 和累积步数：

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 2
  gradient_checkpointing: true
```

这样仍保持有效 batch 32。若 OOM 出现在 validation，则另把 `evaluation_batch_size` 从 64 降到
32。不要通过缩短序列长度解决同一主实验的显存问题，否则会同时改变实验自变量。
