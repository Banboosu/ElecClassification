# MOMENT 零样本缺失值插补实验（2026-08-03）

## 1. 实验目的

验证 MOMENT-1-large 在不使用标签、不更新参数的情况下，能否利用预训练重构头恢复充电功率
序列中的缺失区间。该实验用于补充分类结果，体现同一预训练模型的跨任务复用能力。

MOMENT 在预训练阶段通过随机遮挡 patch 并重构输入学习时序表示，因此缺失值插补与其预训练
目标直接一致。参考：

- [MOMENT 论文](https://arxiv.org/abs/2402.03885)
- [官方插补教程](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/imputation.ipynb)

## 2. 数据与防泄漏协议

- 数据集继续使用统一过滤协议和 seed 42 划分清单。
- 只评价 7,020 条测试序列；类别标签不进入模型、遮挡生成或参数选择。
- 仅遮挡真实有效区间内的完整 MOMENT patch，不评价 padding 和不完整尾部 patch。
- 最少要求 3 个完整 patch，每条序列至少保留 1 个可见 patch。
- 输入保持原始功率尺度（`normalize: none`）。
- MOMENT 内部 RevIN 的均值和标准差仅由 `observation_mask × input_mask` 指定的可见点计算，
  被遮挡真值不参与归一化。
- 所有指标只在人工遮挡位置计算。

数据集 SHA-256 预计继续为：
`5615a96a7894caed5d14463c77167af8098bdc1e1ebf32a33a89a12c3c5cf6e6`。

## 3. 遮挡条件

- 遮挡模式：
  - `random_patches`：随机选择不连续 patch；
  - `contiguous_block`：选择连续 patch 区间，模拟设备断连或连续缺测。
- 目标遮挡率：10%、25%、40%、60%。
- 遮挡随机种子：42、43、44、45、46。
- 同一样本和种子下，随机 patch 遮挡随遮挡率增加保持嵌套；连续遮挡围绕同一中心扩展。
- 每个条件保存遮挡 SHA-256，确保所有方法使用完全相同的缺失位置。

总计 `2 × 4 × 5 = 40` 个遮挡条件。

## 4. 对比方法与指标

| 方法 | 训练或参数更新 | 说明 |
|---|---:|---|
| MOMENT zero-shot | 无 | 使用预训练 reconstruction head |
| Mean | 无 | 使用可见点均值 |
| Forward Fill | 无 | 使用最近的历史可见值，前端缺失回填首个可见值 |
| Linear | 无 | 线性插值，边缘使用最近可见值 |
| PCHIP | 无 | 保形分段三次插值 |

主要指标为逐序列标准化后取平均的 Macro-NRMSE；同时报告原始功率尺度的 MAE、RMSE、
Macro-MAE、负值预测比例。五个遮挡种子按均值 ± 样本标准差统计。

## 5. 计算优化与产物

- V100 32GB，AMP，batch size 64。
- MOMENT 骨干和重构头均不更新，`trainable_parameters=0`。
- 每个模式、比例在首个遮挡种子保存 8 条可视化样本。
- 正式输出目录：`artifacts/moment_imputation/moment_imputation_zero_shot_thesis_v2/`
- 主要产物：
  - `metrics.json`
  - `condition_metrics.csv`
  - `summary.csv`
  - `examples_*.npz`
  - `status.json`

## 6. 远程执行

```bash
tmux new-session -d -s moment_imputation_20260803 \
  "cd /root/autodl-tmp/ElecClassification && bash scripts/run_imputation_remote.sh"
```

日志：`artifacts/logs/moment_imputation_zero_shot_v2_20260803.log`

## 7. 运行状态

首次运行 `moment_imputation_zero_shot_thesis_v1` 在第一个条件失败，未产生正式结果。
根因是测试样本 `C4MNI1TP5964GDJA` 为长度 147、所有有效值均为 98 的常量序列；其
可见点标准差为 0，MOMENT RevIN 在 CUDA FP16 下产生非有限值。同一批次改用 FP32 后
输出全部有限，因此不是数据文件损坏或模型权重错误。

修复后，程序会检测 AMP 非有限输出，在 FP32 中重算当前批次，并让后续条件继续使用 FP32；
回退状态和触发条件将写入 `metrics.json`。失败的 v1 目录保留用于审计，正式结果使用 v2。

- tmux 会话：`moment_imputation_20260803_v2`
- 正式运行名称：`moment_imputation_zero_shot_thesis_v2`
- 相关远程单元测试、语法检查和 FP32 诊断均已通过。
- v2 已于 2026-08-03 15:52（Asia/Shanghai）重新启动。
- 首个条件 `random_patches / 10% / seed 42` 已成功完成并写入
  `metrics_partial.json`，确认程序已越过 v1 的失败位置；正式实验继续运行中。

## 8. 论文判定标准

- 若 MOMENT 在连续缺失或 40%–60% 高缺失率下获得最低 Macro-NRMSE，且误差随缺失率增长
  更慢，可以作为预训练模型学习全局时序结构的主要证据。
- 若线性/PCHIP 在低缺失率下更好，而 MOMENT 在长连续缺失下更好，也属于符合预期且有价值的
  结果：局部平滑插值适合短缺口，预训练重构适合上下文跨度较大的缺失。
- 若 MOMENT 未超过简单插值，应如实报告领域差异，并把结果解释为通用预训练在本数据上的
  迁移边界，不能只展示对 MOMENT 有利的条件。
