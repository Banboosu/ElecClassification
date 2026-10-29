# E12b（修正版）：完整融合消融（validation-only）

> **2026-10-29 语义纠错**：正确电池为类别1，充电器为类别2；下方电池表已从64条件保存预测重算，Macro-F1不变。40% Full电池Recall/FPR=98.23/0.67%，TCN+DYN=98.18/0.67%。原JSON/CSV和冻结协议不覆盖；详见[后续纠错](label_semantics_correction_followup_20261029.md)与[当前全表](label_semantics_reassessment_expanded_20261029.md)。

**状态：4 seed × 2 预算 × 8 方法 = 64 条件 / 224,640 行；仅 validation，0 test。**

> **修正两处问题：**
> 1. 增加 `tcn_meta` 控制：TCN 概率经同型 meta 头，不接 DYN/MOMENT；用于区分“meta 头本身”与“融合”的增益。
> 2. `tcn+md` 的 MD validation 分支此前直接复用 E09 保存的 `moment_diff_logit`（通常 C=0.1），而 E12 OOF 的 MD 分支用 C=1.0；本版按 C=1.0 重算 MD validation，使 OOF/validation 参数一致。TCN+DYN、dyn+md、full 不受此问题影响（它们用 E12 的 C=1.0 管线）。

**核心结果：**

| 方法 | 10% | 40% |
|---|---:|---:|
| TCN | 89.04 ± 0.38 | 97.53 ± 0.29 |
| TCN + 同型 meta（`tcn_meta`） | 90.71 ± 0.99 | 97.65 ± 0.24 |
| DYN | 92.81 ± 0.30 | 95.00 ± 0.17 |
| MOM/ΔP（C=1.0） | 91.24 ± 0.57 | 92.77 ± 0.58 |
| TCN+DYN | 94.26 ± 0.47 | 97.93 ± 0.22 |
| TCN+MOM/ΔP（修正后） | 93.81 ± 0.48 | 97.82 ± 0.25 |
| DYN+MOM/ΔP | 94.26 ± 0.56 | 96.02 ± 0.28 |
| **Full** | **95.02 ± 0.33** | **98.09 ± 0.23** |

- `tcn_meta − tcn`：10% **+1.67 ± 1.08（4/4）**、40% **+0.12 ± 0.05（4/4）**；同型 meta 头本身有小幅增益，但远小于融合。
- `full − tcn_meta`：10% **+4.31 ± 1.22（4/4）**、40% **+0.44 ± 0.13（4/4）**；完整融合不是“只加了一个 meta 头”。
- `full − tcn+dyn`：10% **+0.76 ± 0.45（4/4）**、40% **+0.16 ± 0.12（3/4）**。
- `full − tcn+md`：10% **+1.21 ± 0.34（4/4）**、40% **+0.27 ± 0.15（4/4）**。
- 40% 下 MOMENT 相对 TCN+DYN 的净增益仍只有 +0.16pp/3/4；10% 下三路互补稳健 +0.76pp/4/4。旧版 56 条件表（无 tcn_meta、tcn+md 参数错配）不再使用，保留为历史记录。

只重拟合廉价 meta logistic；TCN、DYN-HGB、MOMENT/ΔP 基模型均复用。**无 test。**

## 主结果（Macro-F1 %，均值 ± 样本 SD）

| 方法 | 10% | 40% |
|---|---:|---:|
| tcn | 89.04 ± 0.38 | 97.53 ± 0.29 |
| tcn_meta | 90.71 ± 0.99 | 97.65 ± 0.24 |
| dyn | 92.81 ± 0.30 | 95.00 ± 0.17 |
| md | 91.24 ± 0.57 | 92.77 ± 0.58 |
| tcn+dyn | 94.26 ± 0.47 | 97.93 ± 0.22 |
| tcn+md | 93.81 ± 0.48 | 97.82 ± 0.25 |
| dyn+md | 94.26 ± 0.56 | 96.02 ± 0.28 |
| full | 95.02 ± 0.33 | 98.09 ± 0.23 |

## 配对差（pp）

| 比较 | 预算 | 均值 ± SD | 正向 seed |
|---|---:|---:|---:|
| full − tcn+dyn | 10p | +0.76 ± 0.45 | 4/4 |
| full − tcn+dyn | 40p | +0.16 ± 0.12 | 3/4 |
| full − tcn+md | 10p | +1.21 ± 0.34 | 4/4 |
| full − tcn+md | 40p | +0.27 ± 0.15 | 4/4 |
| full − dyn+md | 10p | +0.76 ± 0.46 | 4/4 |
| full − dyn+md | 40p | +2.07 ± 0.32 | 4/4 |
| full − tcn | 10p | +5.98 ± 0.25 | 4/4 |
| full − tcn | 40p | +0.56 ± 0.14 | 4/4 |
| full − tcn_meta | 10p | +4.31 ± 1.22 | 4/4 |
| full − tcn_meta | 40p | +0.44 ± 0.13 | 4/4 |
| tcn_meta − tcn | 10p | +1.67 ± 1.08 | 4/4 |
| tcn_meta − tcn | 40p | +0.12 ± 0.05 | 4/4 |
| tcn+dyn − tcn | 10p | +5.22 ± 0.54 | 4/4 |
| tcn+dyn − tcn | 40p | +0.40 ± 0.07 | 4/4 |
| tcn+md − tcn | 10p | +4.77 ± 0.45 | 4/4 |
| tcn+md − tcn | 40p | +0.29 ± 0.05 | 4/4 |
| dyn+md − tcn | 10p | +5.22 ± 0.61 | 4/4 |
| dyn+md − tcn | 40p | -1.51 ± 0.28 | 0/4 |
| tcn+dyn − dyn | 10p | +1.45 ± 0.52 | 4/4 |
| tcn+dyn − dyn | 40p | +2.93 ± 0.24 | 4/4 |
| tcn+md − md | 10p | +2.57 ± 0.61 | 4/4 |
| tcn+md − md | 40p | +5.05 ± 0.34 | 4/4 |

## 电池异常（40%，默认 argmax）

| 方法 | Recall % | FPR % |
|---|---:|---:|
| tcn | 97.83 | 0.77 |
| tcn_meta | 97.88 | 0.74 |
| dyn | 95.06 | 2.10 |
| md | 92.61 | 2.88 |
| tcn+dyn | 98.18 | 0.67 |
| tcn+md | 97.95 | 0.69 |
| dyn+md | 96.01 | 1.65 |
| full | 98.23 | 0.67 |

上述为类别1、四seed均值、默认argmax；旧97.38/0.85属于充电器。10%时Full电池Recall95.28%略低于TCN+DYN95.33%，并非各分项均最优。

## 审计

- 64 条件 / 224640 行；最大 Macro-F1 复算误差 1.11e-16。
- 所有行 split=validation，0 条 test。

## 边界

- 这是**同 validation 行、同 OOF 折**的廉价 meta 消融，不是重新训练基模型；10% 的 TCN 端点是固定 50 轮无选模。
- `full` 与 `dyn+md` 直接使用 E12 已保存的 `stack_tcn`/`stack_no_tcn`，其余两路组合重新拟合 meta；C 仍只在 OOF 上选。
- `tcn_meta` 是 TCN 概率经同型 meta 头的控制；`tcn+md` 的 MD validation 已重算为 C=1.0，与 E12 OOF 的 MD 分支一致（E09 保存的 validation 通常用 C=0.1）。
- 四 seed split 重叠；不能把差值写成独立确认。没有 test、校准、鲁棒性。
## 保全

- 本地：`artifacts/imports/e12b_fusion_ablation_corrected_20261020/`（130 文件 / 17.6MB / 224,640 行，全部 validation，0 test）。
- 远端：`artifacts/improvements/e12b_fusion_ablation_corrected/`。
- 脚本：`scripts/analyze_e12b_fusion_ablation.py`（当前默认输出 semantics_v2 目录，拒绝覆盖旧目录）。
- 旧版：`artifacts/imports/e12b_fusion_ablation_20261020/`、旧文档，仅作历史记录。
- 测试：`tests/test_e12b_analysis.py`（`tcn_meta` 存在、`_meta` 使用选中 C，OOF/validation 同参数）。
