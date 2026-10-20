# E12 10%：cross-fitted TCN + DYN + MOMENT/ΔP 堆叠（validation-only）

**状态：四 seed、5 折交叉拟合全部完成；16 条件 / 56,160 行审计误差 0.0；无 test。**

协议与 40% E12 相同，只把训练预算换成 10%（2,457 个训练标签）；TCN 每折/全量固定 50 轮 cosine，
不使用 validation 选 checkpoint；meta 只在 OOF 上拟合。

**重要解释边界：本轮 TCN 端点是固定 50 轮 cosine、无 validation 选模的 10% 模型，validation**
**只有 89.04%；E05C 的 93.68% 是 validation 选模后的值。因此 stack 相对本轮 tcn_full 的 +5.98pp**
**不是“stack 比强 TCN 高 6pp”，而是“stack 比本轮弱 TCN 端点高”；更干净的比较是 stack 相对**
**无 TCN 的同折 stack（+0.76pp）以及相对 E05C 选模 TCN 的四 seed 均值上下文（+1.34pp）。**


| 方法 | Macro-F1 均值 ± SD % | 逐 seed | 电池 recall/FPR % |
|---|---:|---|---:|
| tcn_full | 89.04 ± 0.38 | 89.29, 89.16, 89.24, 88.48 | 77.78/0.91 |
| tcn_foldavg | 88.80 ± 1.98 | 87.00, 91.23, 89.58, 87.38 | 78.03/1.12 |
| stack_no_tcn | 94.26 ± 0.56 | 94.62, 94.80, 93.56, 94.06 | 95.22/3.49 |
| stack_tcn | 95.02 ± 0.33 | 95.06, 95.42, 95.00, 94.62 | 94.66/2.59 |

## 配对差

| 比较 | 均值 ± SD pp | 正向 seed |
|---|---:|---:|
| stack_tcn − tcn_full | +5.98 ± 0.25 | 4/4 |
| stack_tcn − tcn_foldavg | +6.23 ± 1.75 | 4/4 |
| stack_tcn − stack_no_tcn | +0.76 ± 0.46 | 4/4 |
| tcn_full − tcn_foldavg | +0.24 ± 1.88 | 2/4 |
| stack_tcn − E08 stack | +1.74 ± 0.48 | 4/4 |

## Gate 与审计

- Gate（stack_tcn − tcn_full ≥ +0.3pp 且 ≥3/4 正向）：**通过**。
- 审计：16 条件 / 56160 行；Macro-F1 最大误差 0.000e+00；全部 validation，0 test。

## 边界

- 四 seed split 重叠、validation 反复使用；这是已有 test 已知后的 validation 探索，不是独立盲测。
- TCN 与 stack 的训练/选择预算不同；stack 的 meta 只在 OOF 上拟合，TCN 未使用 validation 选 checkpoint。
- 本轮只执行 10p 预算；其他预算未在本轮执行，不能外推。
- 没有做校准/保形、鲁棒性或同机端到端延迟。

## 对论文的影响

- 10% 标签下，最终系统 **DYN+MOM/ΔP+TCN stack = 95.02 ± 0.33%**；无 TCN 的同折 stack 94.26 ± 0.56%，
  逐 seed 配对 **+0.76pp（4/4）**；E05C 选模 TCN 93.68 ± 1.93%（上下文，+1.34pp，非同轮配对）。
- 40% 标签下，同一系统为 98.09 ± 0.23%，相对同轮 TCN 端点 97.53% +0.56pp（4/4），相对 E05C 选模
  TCN 97.78% 高 +0.31pp（上下文）。
- 因此跨预算的主表可写成：10% **95.02**、40% **98.09**；对应 E05C TCN 93.68/97.78。10% 的领先
  是稳健的（相对同折无 TCN stack +0.76/4/4）；40% 的领先相对选模 TCN 只有 +0.31pp，属于小幅。
- E12b 进一步显示：10% 完整三路相对最好的两路 `TCN+DYN`（94.26%）为 **+0.76pp（4/4）**；
  40% 相对 `TCN+DYN`（97.93%）只有 **+0.16pp（3/4）**。低标签下三路互补稳健，40% 下 MOMENT 的
  净增益很小。
- 电池 10%：stack recall/FPR **94.66/2.59%**；无 TCN stack 95.22/3.49%；本轮 TCN 端点 77.78/0.91%。
  stack 比无 TCN 降低 FPR，但 recall 也略低；不能只挑一个指标。
- TCN 固定 50 轮端点在 10% 明显弱于 validation 选模的 E05C（89.04 vs 93.68），说明 TCN 在低标签
  仍依赖选模/训练预算；这不是架构上限，也提醒不能把 +5.98pp 当成对 TCN 的真实领先。

## 必须保留的边界

- validation-only；test 已消费，本轮没有打开 test；没有新的独立外部数据。
- 四 seed split 重叠、validation 反复使用；不是独立盲测。
- 只执行 10% 预算；1%/100% 未执行。TCN 端点为固定 50 轮、无 validation 选模，弱于 E05C 选模值。
- 没有做校准/保形、鲁棒性或同机端到端延迟。
