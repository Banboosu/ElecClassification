# E12：cross-fitted TCN + DYN + MOMENT/ΔP 堆叠（40% 标签，validation-only）

**状态：四 seed、5 折交叉拟合全部完成；gate 通过；16 条件 / 56,160 行审计误差 0.0；无 test。**

设计：40% 训练子集内做 5 折分层交叉拟合。每折训练 legacy masked TCN 50 轮（cosine，无 validation
选 checkpoint），得到折内 OOF 概率和 global validation 概率；DYN-HGB 与 MOMENT+ΔP logistic 使用
完全相同的折分配；meta logistic 只在 9 维 OOF 概率（DYN 3 + MOM/ΔP 3 + TCN 3）上拟合，validation
只评一次。`tcn_full` 是在整个 40% 子集上再训 50 轮的 TCN 对照，不参与 meta 拟合。


| 方法 | Macro-F1 均值 ± SD % | 逐 seed | 电池 recall/FPR % |
|---|---:|---|---:|
| tcn_full | 97.53 ± 0.29 | 97.61, 97.90, 97.27, 97.33 | 95.72/0.62 |
| tcn_foldavg | 97.50 ± 0.15 | 97.37, 97.71, 97.50, 97.41 | 95.60/0.59 |
| stack_no_tcn | 96.02 ± 0.28 | 96.10, 96.21, 96.16, 95.61 | 96.35/2.16 |
| stack_tcn | 98.09 ± 0.23 | 97.99, 98.43, 97.89, 98.06 | 97.38/0.85 |

## 配对差

| 比较 | 均值 ± SD pp | 正向 seed |
|---|---:|---:|
| stack_tcn − tcn_full | +0.56 ± 0.14 | 4/4 |
| stack_tcn − tcn_foldavg | +0.59 ± 0.14 | 4/4 |
| stack_tcn − stack_no_tcn | +2.07 ± 0.32 | 4/4 |
| tcn_full − tcn_foldavg | +0.03 ± 0.22 | 2/4 |
| stack_tcn − E08 stack | +2.63 ± 0.27 | 4/4 |

## Gate 与审计

- Gate（stack_tcn − tcn_full ≥ +0.3pp 且 ≥3/4 正向）：**通过**。
- 审计：16 条件 / 56160 行；Macro-F1 最大误差 0.000e+00；全部 validation，0 test。

## 边界

- 四 seed split 重叠、validation 反复使用；这是已有 test 已知后的 validation 探索，不是独立盲测。
- TCN 与 stack 的训练/选择预算不同；stack 的 meta 只在 OOF 上拟合，TCN 未使用 validation 选 checkpoint。
- 40% 是唯一预算；10%/1% 未在本轮执行，不能外推。
- 没有做校准/保形、鲁棒性或同机端到端延迟。

## 对论文的影响

- 当前最佳系统为 **DYN + MOMENT/ΔP + TCN 的 OOF 堆叠**：40% 标签 validation **98.09 ± 0.23%**，
  相对同预算 TCN full **+0.56pp（4/4）**，相对 E08 stack **+2.63pp（4/4）**，相对无 TCN 的同折
  stack **+2.07pp（4/4）**；gate（≥+0.3pp 且 ≥3/4）通过。
- `tcn_full` 97.53%、`tcn_foldavg` 97.50%，两者差 +0.03pp（2/4），说明折模型平均与全量模型
  在这个预算下基本等价；stack 的增益不是“用了更强的 TCN”，而是三路错误的互补。
- 40% 电池异常：stack recall/FPR **97.38/0.85%**；TCN full 95.72/0.62%。stack 提高了 recall，
  代价是 FPR 从 0.62% 升到 0.85%，仍是低误报区间；没有调阈值。
- 40% 标签 98.09% 已接近 E05C 完整标签 TCN 的 98.40%，但两者数据角色/选择协议不同，不能直接
  说“40% 标签达到全标签性能”；只能说在本 validation 协议下非常接近。
- 这使论文主线可以收束为：**强动力学特征 + 冻结 MOMENT 双流 + TCN 的交叉拟合堆叠**；
  MOMENT 的贡献是小而稳定的互补（+0.55pp over E08 stack at 40%），不是单独领先。

## 必须保留的边界

- validation-only；test 在历史研究中已被消费，本轮没有打开 test，也没有新的独立外部数据。
- 四 seed 的 split 相互重叠，validation 反复使用；不是独立盲测或外部泛化确认。
- 只执行 40% 预算；10%/1%/100% 未在本轮执行（10% 另开 suite），不能外推整条曲线。
- TCN 固定 50 轮 cosine、无 validation 选 checkpoint；`tcn_full` 端点略低于 E05C 的 validation
  选模结果（97.53 vs 97.78），因此 stack 的 +0.56pp 是相对本轮 TCN 端点，不是相对 E05C 选模值。
- 没有做校准/保形、鲁棒性扰动、同机端到端延迟；这些留给 E10。
- meta 只在 OOF 上拟合，但 DYN/MOM 的折模型与全量模型存在常规 stacking 的分布差；没有额外做
  nested 选择或重复交叉拟合。
