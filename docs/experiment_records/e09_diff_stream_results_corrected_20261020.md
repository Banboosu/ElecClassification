# E09：冻结 MOMENT 原序列 vs 一阶差分双流（四 seed，validation-only）

四 seed42/44/45/46；1%/10%/40% 训练标签；冻结 backbone，仅改变 MOMENT 输入流；无 test。

**修复说明：** 历史 `stack_three` 的 OOF 列写成了 3:6 与 4:7，第二块覆盖第一块且 7:9 未写入；
本轮只重算 `stack_three`（3:6 与 6:9），其他 E09 条件不变。

## 主结果（Macro-F1 %，均值 ± 样本 SD）

| 方法 | 1% | 10% | 40% |
|---|---:|---:|---:|
| moment_logit | 70.71 ± 1.75 | 81.61 ± 0.36 | 83.91 ± 0.44 |
| diff_logit | 80.64 ± 0.63 | 89.08 ± 1.06 | 91.04 ± 0.69 |
| moment_diff_logit | 84.09 ± 1.88 | 91.64 ± 0.57 | 93.57 ± 0.56 |
| stack_moment | 87.12 ± 1.37 | 93.27 ± 0.44 | 95.46 ± 0.19 |
| stack_moment_diff | 88.69 ± 1.79 | 94.26 ± 0.56 | 96.01 ± 0.31 |
| stack_three | 88.94 ± 0.94 | 94.19 ± 0.34 | 95.78 ± 0.30 |

## 配对差（pp）

| 比较 | 预算 | 均值 ± SD | 正向 seed |
|---|---:|---:|---:|
| moment_diff_logit − moment_logit | 1p | +13.38 ± 1.79 | 4/4 |
| moment_diff_logit − moment_logit | 10p | +10.03 ± 0.39 | 4/4 |
| moment_diff_logit − moment_logit | 40p | +9.66 ± 0.80 | 4/4 |
| stack_moment_diff − stack_moment | 1p | +1.57 ± 1.17 | 4/4 |
| stack_moment_diff − stack_moment | 10p | +0.99 ± 0.35 | 4/4 |
| stack_moment_diff − stack_moment | 40p | +0.55 ± 0.16 | 4/4 |
| stack_moment_diff − stack_three | 1p | -0.25 ± 1.07 | 2/4 |
| stack_moment_diff − stack_three | 10p | +0.07 ± 0.44 | 3/4 |
| stack_moment_diff − stack_three | 40p | +0.23 ± 0.12 | 4/4 |
| stack_moment_diff − diff_logit | 1p | +8.05 ± 1.86 | 4/4 |
| stack_moment_diff − diff_logit | 10p | +5.18 ± 0.69 | 4/4 |
| stack_moment_diff − diff_logit | 40p | +4.97 ± 0.59 | 4/4 |

## 与 E08 强特征 stack / TCN 的上下文

| 比较 | 预算 | 均值 ± SD | 正向 seed |
|---|---:|---:|---:|
| stack_moment_diff - e08_stack | 1p | +1.57 ± 1.17 | 4/4 |
| stack_moment_diff - tcn | 1p | +33.18 ± 14.53 | 4/4 |
| stack_moment_diff - e08_stack | 10p | +0.98 ± 0.38 | 4/4 |
| stack_moment_diff - tcn | 10p | +0.58 ± 1.93 | 2/4 |
| stack_moment_diff - e08_stack | 40p | +0.55 ± 0.16 | 4/4 |
| stack_moment_diff - tcn | 40p | -1.76 ± 0.32 | 0/4 |

## 电池异常（40%，默认 argmax）

| 方法 | Recall % | FPR % |
|---|---:|---:|
| moment_logit | 77.87 | 9.60 |
| diff_logit | 92.50 | 4.55 |
| moment_diff_logit | 94.31 | 3.43 |
| stack_moment | 95.51 | 2.44 |
| stack_moment_diff | 96.35 | 2.18 |
| stack_three | 96.30 | 2.38 |

## 事前门槛

| Gate | 比较 | 均值 pp | 正向 | 要求 | 状态 |
|---|---|---:|---:|---|---|
| G1 | moment_diff - moment @40p | +9.66 | 4/4 | ≥5.0pp, ≥3/4 | **passed** |
| G2 | stack_moment_diff - stack_moment @40p | +0.55 | 4/4 | ≥0.3pp, ≥3/4 | **passed** |
| G3 | stack_moment_diff - stack_moment @10p | +0.99 | 4/4 | ≥0.3pp, ≥3/4 | **passed** |

## 解释边界

- 四个 seed 的 split 重叠、validation 反复使用；这不是独立外部确认，也没有 test。
- 差分流是在有效观测上计算并逐序列 z-score；幅值/尺度信息仍由 DYN 统计分支提供。
- `stack_three` 把 MOM 与 DIFF 分成两个 logistic 分支；修正历史 OOF 列重叠后，它仍略低于拼接的 `stack_moment_diff`（约 0.1–0.3pp），
  说明当前数据量下提前分开两支的代价很小，不能据此说差分信息无用。历史损坏版本（约 72%/91%/95%）已替换。
- 与 TCN 的差值是同训练标签/同 validation 行的管线比较；TCN 的训练/选择预算不同。
- 1% 的 MOMENT 单支仍弱；差分流的主要收益在 10%/40% 和电池类。
- 所有逐样本预测在聚合时重新核对 ID/真值/有限分数/行数，并复算 Macro-F1。

## 保全与审计

- 修正套件：`artifacts/improvements/e09_stack_three_fix/e09_stack_three_fix_20261020_seed{42,44,45,46}/`，12 条件 / 42,120 行。
- 本分析为 72 条件 / 252,720 行；逐样本预测复算 Macro-F1 最大误差 **0.0**；全部 validation，0 test。
- 本地保全：`artifacts/imports/e09_stack_three_fix_20261020/`（88 文件 / 5.5MB / 42,120 行）、`artifacts/imports/e09_analysis_corrected_20261020/`。
