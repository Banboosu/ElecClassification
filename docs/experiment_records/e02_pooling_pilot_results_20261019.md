# E02P：冻结表征池化与长度诊断结果（validation-only）

日期：2026-10-19。只用现有数据，探索seed43、40%标签；没有test推理或encoder监督训练。

## 协议与技术对照

- 2026-10-19T03:03:33.551578+00:00在新真实池化结果前冻结240输入；协议SHA `4728d78d3b90efde875b6f9c37cfabf6e4696bda0e03ce87828820bb5824541b`。
- 三种池化×仅MOMENT/原始统计融合，共6条件；各9827 train、3510 validation，5C×3gamma×5折，折内标准化，FP32骨干和float64核距离。
- mean与mean+max/thirds来自同一批未适配patch token；完整有效patch掩码不改。多段输入维数增加，不能声称所有模型复杂度完全相同。

| 池化 | 分支 | 维数（不含统计） | Validation Macro-F1 % | Accuracy % |
|---|---|---:|---:|---:|
| mean | moment | 1024 | 84.5781 | 84.1880 |
| mean | fusion | 1024 | 91.1459 | 91.0256 |
| mean_max | moment | 2048 | 84.5989 | 84.1026 |
| mean_max | fusion | 2048 | 89.6769 | 89.5157 |
| thirds | moment | 3072 | 84.4232 | 84.0171 |
| thirds | fusion | 3072 | 90.9733 | 90.8547 |

mean两对照均通过原best params、validation逐行预测、75个fold分数的预设兼容性检查；新mean特征亦通过旧FP32缓存核对。

## 固定pilot门槛

候选按预定顺序选择 **thirds/fusion**，validation Macro-F1 **90.9733%**。
- 相对本轮mean融合SVM **-0.1726pp**，门槛≥0.5pp。
- 相对既有最强非适配管线 **-0.7493pp**，门槛≥0.25pp。
- 两条同时满足的复制建议规则：**未通过**。不自动扩seed、改模型头、LoRA、加预算或test。

| 既有参照（同seed validation） | Macro-F1 % |
|---|---:|
| prior_gain_fusion_nn | 91.7225 |
| mean_fusion_svm | 91.1459 |
| matched_statistics_svm | 89.4826 |
| historical_tcn | 91.3836 |

所选候选相对历史同预算TCN validation差 **-0.4104pp**；不据单seed、不同管线选择预算和历史权重未重核这一上下文宣称普遍优于TCN。

## 长度分层

仅选定train的1/3、2/3线性分位点：**[265.0, 399.0]**。组0≤首点，组1介于两点（含右边界），组2>第二点。

| 池化 | 分支 | 长度组 | N | 真值类别计数0/1/2 | Macro-F1 % |
|---|---|---:|---:|---|---:|
| mean | moment | 0 | 1231 | [670, 182, 379] | 83.9138 |
| mean | moment | 1 | 1145 | [356, 372, 417] | 84.4167 |
| mean | moment | 2 | 1134 | [186, 448, 500] | 81.1721 |
| mean | fusion | 0 | 1231 | [670, 182, 379] | 89.5792 |
| mean | fusion | 1 | 1145 | [356, 372, 417] | 92.2303 |
| mean | fusion | 2 | 1134 | [186, 448, 500] | 89.0129 |
| mean_max | moment | 0 | 1231 | [670, 182, 379] | 83.4993 |
| mean_max | moment | 1 | 1145 | [356, 372, 417] | 83.5643 |
| mean_max | moment | 2 | 1134 | [186, 448, 500] | 82.8753 |
| mean_max | fusion | 0 | 1231 | [670, 182, 379] | 87.1923 |
| mean_max | fusion | 1 | 1145 | [356, 372, 417] | 90.3482 |
| mean_max | fusion | 2 | 1134 | [186, 448, 500] | 88.4487 |
| thirds | moment | 0 | 1231 | [670, 182, 379] | 81.7759 |
| thirds | moment | 1 | 1145 | [356, 372, 417] | 84.8390 |
| thirds | moment | 2 | 1134 | [186, 448, 500] | 82.0470 |
| thirds | fusion | 0 | 1231 | [670, 182, 379] | 88.6683 |
| thirds | fusion | 1 | 1145 | [356, 372, 417] | 92.5020 |
| thirds | fusion | 2 | 1134 | [186, 448, 500] | 89.0074 |

各组使用固定3类别Macro-F1；如缺类别或样本少需结合计数解释，不据分组分数再选方法。

## 边界与padding

固定合成有效长度[18, 19, 24, 64, 128, 384]，包含短、常量及非整patch尾部。比较512/1024 padding及掩码外junk。差异：`{'512': {'mean': 2.384185791015625e-07, 'max': 2.980232238769531e-07, 'thirds': 4.172325134277344e-07}, 'junk_padding': {'mean': 0.0, 'max': 0.0, 'thirds': 0.0}}`。
全部在描述性容差1e-4内：**True**。
- thirds按完整有效patch的时间排名分段，不按padding长度切三段；空段补零，不复制观测。max也排除invalid token。
- 不完整尾部不参与池化，并不代表对编码器无作用：外部z-score/内部RevIN仍可能读取这些有效原始点；未声称整条管线对尾部不变。

## 证据与限制

- 6条件全部保留，21,060行validation预测、混淆指标、CV/模型/cache哈希与长度组通过审计。所有模型仅在train标签上拟合，validation用于候选比较，未输出新test指标。
- 这是已知历史/E03T test之后的validation探索；seed43也被多次用于方案发现，不是独立确认。不得据此修改E03T已固定结果或声称新外部泛化。
- 旧TCN只作validation上下文；原E05A关于整体优越性未证明的限制不回改。
源summary SHA `d96f61f1cce7f2143d39d4e729712c08a08165e59d3d5bcd685171eedbdd711a`；机器可读源`artifacts/improvements/e02_pooling_pilot/e02_pooling_pilot_20261019_v1_analysis/summary.json`。缓存/模型/预测及冻结源码快照均保留。
- 特征准备/统计/合成诊断合计计时2.47分钟，6次CV/拟合/validation预测合计12.02分钟；这些不是线上延迟。
