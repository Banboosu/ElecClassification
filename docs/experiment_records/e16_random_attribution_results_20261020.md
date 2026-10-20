# E16：预训练 vs 随机初始化归因（validation-only）

**状态：4 seed × 2 预算 × 2 初始化 × 3 流 = 48 条件 / 168,480 行；仅 validation，0 test。**

同一 MOMENT 架构、mask-aware mean pooling、1024 维特征；随机初始化每个数据 seed 一次，原序列/差分/双流共用同一个随机 backbone；预训练加载被 runtime guard 禁止。Logistic 固定 C=1.0，三流同协议；原序列逐序列 z-score，差分逐序列 z-score。

**核心结论：**

| 预算 | 流 | 预训练 | 随机初始化 | 预训练−随机 |
|---|---|---:|---:|---:|
| 10% | orig | 81.27 ± 0.35 | 58.79 ± 1.13 | **+22.48 ± 0.91（4/4）** |
| 10% | diff | 88.31 ± 0.84 | 78.94 ± 1.07 | **+9.36 ± 1.18（4/4）** |
| 10% | dual | 91.24 ± 0.57 | 81.27 ± 0.91 | **+9.97 ± 0.73（4/4）** |
| 40% | orig | 84.02 ± 0.53 | 63.45 ± 1.14 | **+20.57 ± 1.10（4/4）** |
| 40% | diff | 89.20 ± 0.51 | 83.03 ± 1.02 | **+6.17 ± 1.06（4/4）** |
| 40% | dual | 92.77 ± 0.58 | 86.77 ± 0.69 | **+6.00 ± 0.56（4/4）** |

- **预训练在每条流上都显著优于随机初始化，全部 4/4。** 原序列差距最大（+20 至 +22pp），双流差距 +6 至 +10pp；差分流的预训练收益小于原序列，但仍稳定。
- 电池 40%：预训练 dual recall/FPR **93.33/3.81%**；随机 dual 88.45/5.75%。预训练同时提高召回、降低 FPR。
- 这说明 MOMENT 分支的收益不能只用“随机特征/架构容量”解释；**预训练权重本身有实质贡献**。
- 但注意：E16 固定 C=1.0，因此预训练 diff 的绝对值（89.20/40%）低于 E09 的 91.04%（E09 在 C=1.0/0.1 中训练内选择）。E16 的目的是同协议下 pretrained vs random，不是复现 E09 的最优绝对值。
- 与 E14 的关系：E14 的 MultiRocket 在 10% 超过完整融合，说明“预训练+融合”不是低标签精度的必要条件；E16 说明“在 MOMENT 分支内部，预训练确实有用”。两件事不矛盾：预训练有价值，但并非唯一可行路线。

## Macro-F1 %

| 预算 | 流 | 预训练 | 随机初始化 | 预训练−随机 |
|---|---|---:|---:|---:|
| 10p | orig | 81.27 ± 0.35 | 58.79 ± 1.13 | +22.48 ± 0.91 (4/4) |
| 10p | diff | 88.31 ± 0.84 | 78.94 ± 1.07 | +9.36 ± 1.18 (4/4) |
| 10p | dual | 91.24 ± 0.57 | 81.27 ± 0.91 | +9.97 ± 0.73 (4/4) |
| 40p | orig | 84.02 ± 0.53 | 63.45 ± 1.14 | +20.57 ± 1.10 (4/4) |
| 40p | diff | 89.20 ± 0.51 | 83.03 ± 1.02 | +6.17 ± 1.06 (4/4) |
| 40p | dual | 92.77 ± 0.58 | 86.77 ± 0.69 | +6.00 ± 0.56 (4/4) |

## 电池异常（40%，默认 argmax）

| 流 | 预训练 Recall/FPR | 随机 Recall/FPR |
|---|---:|---:|
| orig | 78.80/9.81 | 59.47/21.36 |
| diff | 90.49/5.50 | 84.92/7.58 |
| dual | 93.33/3.81 | 88.45/5.75 |

## 审计

- 48 条件 / 168480 行；Macro-F1 复算最大误差 0；0 test。

## 边界

- 这是同一架构内的预训练归因；随机初始化只用了一个 draw/seed，不能代表所有随机权重。
- 只改变初始化，不改变池化、特征维度、logistic C、数据与选择预算；backbone 全程冻结。
- 四 seed split 重叠、validation 反复使用；无 test、无外部数据；不是部署保证。
- 没有做随机初始化多次重复；因此“预训练无差异”只能作为本数据上的条件性结论。
## 保全与验证

- 本地：`artifacts/imports/e16_random_attribution_20261020/`（suite 13MB + random feature cache 892MB；共 168,480 行预测，全部 validation，0 test）。
- 远端：`artifacts/improvements/e16_random_attribution/e16_random_attribution_20261020_v1/`、`artifacts/improvements/e16_random_features/`。
- 协议：`configs/experiments/improvements/e16_random_attribution_protocol.json`；代码：`src/tcn_moment/run_e16_random_attribution.py`；测试：`tests/test_e16_random_attribution.py`。
- 随机初始化 provenance：每个 seed 的 `manifest.json` + `pretrained_loading_guard.json`；random 原序列特征与既有 E01C random cache 在 seed44/45/46 上最大差异约 1.8e-6–2.9e-6，说明随机 backbone 与 guard 一致。
- 审计：48 条件 / 168,480 行，Macro-F1 复算最大误差 **0.0**，0 test。
