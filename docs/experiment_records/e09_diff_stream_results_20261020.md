# E09：冻结 MOMENT 原序列 vs 一阶差分双流（四 seed，validation-only）

**状态：四 seed × 3 预算 × 6 方法 = 72 条件全部完成；三条事前门槛全部通过；仅 validation，未构造或评价 test。**

> **2026-10-20 更正（P0）**：历史 `stack_three` 的 OOF 列写错为 `3:6` 与 `4:7`，第二块覆盖第一块且
> `7:9` 未写入；本文的 `stack_three` 数字已被修正版替换（只重算 `stack_three`，其余条件不变）。
> 修正后 1%/10%/40% 为 **88.94/94.19/95.78%**，与拼接版 `stack_moment_diff` 的差只有 −0.25/+0.07/+0.23pp。
> 另外，1% 的 TCN 上下文此前误取了 N2 的 5% 记录；修正后 1% 的 stack−TCN 为 +33.18pp（TCN 1% 坍塌背景），
> 不再是 +4.66pp。修正产物：`artifacts/improvements/e09_stack_three_fix/`、
> `artifacts/improvements/e09_diff_stream/analysis_four_seed_corrected/`。

设计要点：原始序列仍走 E01 的冻结 MOMENT + mask-aware mean pooling；差分序列在有效观测上计算
`ΔP`、逐序列 z-score、单独走同一条冻结 backbone；两条 1024 维表征拼接后再与 DYN 特征做概率堆叠。
没有训练 backbone，也没有改 DYN 特征。

- 每个 seed 的协议在各自 suite 内冻结；差分 cache 为 `artifacts/improvements/e09_diff_features/seed<seed>/`。
- 72 个条件 / 252,720 行逐样本预测在聚合时重新核对 ID、真值、有限分数、行数并复算 Macro-F1，
  最大误差 **0.0**；无 test 行。
- 与 TCN 的对照使用同一训练子集、同一 validation 行；TCN 的训练/选择预算不同，差值只作管线比较。


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
| stack_moment_diff − stack_three | 1p | −0.25 ± 1.07 | 2/4 |
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
- `stack_three` 把 MOM 与 DIFF 分成两个 logistic 分支；修正列重叠后仍略低于拼接的 `stack_moment_diff`
  （40% −0.23pp，10% −0.07pp，1% +0.25pp），差距很小，不能据此说差分信息无用。
- 与 TCN 的差值是同训练标签/同 validation 行的管线比较；TCN 的训练/选择预算不同。
- 1% 的 MOMENT 单支仍弱；差分流的主要收益在 10%/40% 和电池类。

## 对论文的影响

- 差分流是让冻结 MOMENT 真正有用的关键：MOM+ΔP 单支在 40% 从 83.91% 升到 **93.57%**（+9.66pp，4/4），
  在 10% 从 81.61% 升到 **91.64%**（+10.03pp，4/4）。
- 在 E08 强特征堆叠上再加 ΔP：40% **95.46 → 96.01%**（+0.55pp，4/4），10% **93.27 → 94.26%**
  （+0.99pp，4/4），1% **87.12 → 88.69%**（+1.57pp，4/4）。三条门槛均通过。
- 与 TCN 的同预算上下文：1% stack 高 +33.18pp（4/4，但 TCN 1% 本身坍塌）；10% 高 +0.58pp
  （仅 2/4，不能宣称稳健超过）；40% 低 −1.76pp（0/4，TCN 仍领先）。
- 40% 电池异常：stack 的 recall/FPR 为 **96.35/2.18%**；MOM+ΔP 单支 94.31/3.43%。这是默认 argmax，
  没有重新选阈值。
- `stack_three`（MOM 与 ΔP 分成两个 logistic 分支）修正后在 1%/10%/40% 为 88.94/94.19/95.78%，
  与拼接后的 88.69/94.26/96.01% 只差 −0.25/+0.07/+0.23pp；分开两支的代价很小，不能据此说差分信息无用。
- 结论应按“强特征 + 双流 MOMENT 堆叠”组织：40% 仍低于 TCN 1.76pp；10% 与 TCN 基本持平但方向不稳；
  1% 的领先主要来自 TCN 坍塌，不能作为稳健优势。

## 必须保留的边界

- 四个 seed 的 split 相互重叠，validation 在此前研究中已反复使用；没有新的独立 test 或外部数据。
- 差分序列的 z-score 会去掉差分幅值，幅值/尺度信息仍由 DYN 分支提供；这是有意的分工，不是“MOMENT 看到了全部信息”。
- 1% 只有约 244–246 个训练标签，但仍使用 3,510 条 validation 选模；不能声称全流程只需 244 条标签。
- 没有做校准/保形、鲁棒性扰动、同机端到端延迟；这些留给 E10。
- TCN + DYN + MOMENT/ΔP 的联合堆叠见 E12/E12b；E12b 的完整融合消融显示 40% 下 TCN+DYN 已达 97.93%，
  完整三路 98.09%（+0.16pp，3/4），10% 下完整三路 95.02% 相对 TCN+DYN 94.26%（+0.76pp，4/4）。
