# E18 100%：全标签统一基准（validation-only）

**状态：4 seed × 12 方法 = 48 条件 / 168,480 行；仅 validation，0 test。**

在 RTX 5090 新环境、与 E18 相同的固定配置下补跑 100% 训练标签（24,569 条）；validation 仍 3,510 条。HGB 固定参数、MOMENT logistic C=1.0、TCN 50 epoch/5折OOF+全量、Multi/Mini interp_z 256 + RidgeCV。

| 方法 | 100% |
|---|---:|
| F15+HGB | 91.87 ± 0.31 |
| DYN+HGB | 96.06 ± 0.20 |
| MOMENT 原序列 | 85.12 ± 0.55 |
| MOMENT ΔP | 91.38 ± 0.59 |
| MOMENT 双流 | 93.03 ± 0.31 |
| TCN | 98.33 ± 0.11 |
| TCN+DYN | **98.54 ± 0.18** |
| DYN+MOMENT | 96.13 ± 0.27 |
| DYN+MOMENT/ΔP | 96.58 ± 0.25 |
| Full | **98.53 ± 0.16** |
| MultiRocket | 97.67 ± 0.25 |
| MiniRocket | 94.94 ± 0.33 |

## 与 40% 的对比

| 方法 | 40% | 100% | 差值 |
|---|---:|---:|---:|
| Full | 98.14 | 98.53 | +0.39 |
| TCN+DYN | 98.09 | 98.54 | +0.45 |
| TCN | 97.62 | 98.33 | +0.71 |
| MultiRocket | 97.25 | 97.67 | +0.42 |
| MiniRocket | 94.39 | 94.94 | +0.55 |
| DYN+HGB | 95.00 | 96.06 | +1.06 |
| MOMENT 双流 | 92.92 | 93.03 | +0.11 |

- 100% 时 **TCN+DYN 98.54% 与 Full 98.53% 基本相同**；完整融合的 MOMENT 分支没有带来净增益。
- Full 相对 40% 只升 +0.39pp；TCN 升 +0.71pp；DYN 升 +1.06pp。低预算差距在 100% 收敛。
- MultiRocket 97.67%，低于 TCN/TCN+DYN/Full 约 0.7–0.9pp；MiniRocket 仍弱。
- MOMENT 单支在 100% 仍远低于 DYN/TCN（双流 93.03%），说明“预训练单支替代监督模型”不成立。

## 边界

- 新主机 RTX 5090、torch 2.8.0+cu128；与旧 4080 结果有 0.1–0.5pp 差异，不混列。
- 100% 是完整训练池（24,569），不是“更多外部数据”；test 仍未打开。
- 固定超参、未逐预算调参；这是同一配置下的全标签对照，不是各模型的全标签上限。
- 四 seed split 重叠；validation 反复使用；不是独立外部确认。

## 保全

- 本地：`artifacts/imports/e18_low_label_100p_20261020/`（102 文件 / 13.2MB / 168,480 行，全部 validation）。
- 远端：`ubuntu:~/workspace/banboosu_ws/ElecClassification/artifacts/improvements/e18_low_label_100p/e18_low_label_100p_20261020_v1/`。
- 协议：`configs/experiments/improvements/e18_low_label_100p_protocol.json`。
- 代码：`src/tcn_moment/run_e18_low_label.py`。
