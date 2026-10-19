# E02P：现有数据上的池化诊断pilot（validation-only）

用户明确要求继续todo2.md中不依赖新数据的未完成项。本阶段不再等待外部元数据，先执行E02：masked mean、mean+max、有效时间轴三段池化、短/常量序列与padding/长度分层检查。

## 范围

- 探索seed43、40%训练标签（9,827），validation3,510。seed43是已反复使用的方案探索seed；后续若建议复制，必须另立协议，不把pilot混入此前四seed确认。
- 三种池化×仅MOMENT/原始15统计融合，共6次RBF-SVM训练搜索。未适配公共MOMENT只提取选定train及validation一次；同一批patch token同时计算全部池化，不持久化全部token、不编码test。
- 全FP32提取，内部RevIN、patch embedding、attention mask均不改。只聚合完整有效patch，保持原`.all()`语义，不把`.all()`换成`.any()`。
- mean保持原1024维；mean+max拼接2048维；thirds按完整有效patch的时间顺序，以`floor(3*rank/n_valid)`分为三段，各masked mean后拼接3072维。短序列若某段为空，保留零向量，不复制观测或伪造token。

## 公平对照与技术关卡

- 复用原RBF搜索实现：5C×3gamma×5折Macro-F1，折内拟合transform；无监督适配骨干，不存在先用所有训练标签适配再只CV分类器的泄漏。
- 仅MOMENT分支使用StandardScaler和gamma=[scale,0.1/D,10/D]；融合使用BranchScaler(D)，gamma=[scale,0.05,5]。各分支归一化只在fold train拟合。SVM避免将多段更大输入维数直接变成更大MLP参数量，但输入维数/核复杂度仍不同，不能声称完全等容量。
- 新mean特征与原FP32缓存最大绝对差需≤1e-4；两个mean条件的best params和validation预测必须逐项相同，75个CV fold分数最大差≤1e-12。先完成两个mean对照，未通过就停止，不继续把技术差异当作池化收益。
- 既有强参照包括seed43增益融合头（E03D-v2）、匹配统计SVM、原mean融合SVM，以及历史同预算TCN的validation表现。只读取各参照validation指标/预测，不重新推理test，不根据历史test来修改本轮门槛。

## 长度/边界诊断

- 长度分层阈值仅由选定训练子集的1/3、2/3线性分位数确定。validation不用于定阈值，所有分组包括空组/缺失类别都披露；小分组Macro-F1不能当独立稳定收益。
- 合成有效长度固定为18/19/24/64/128/384，包含常量和非整patch尾部；比较512和1024 padding，以及掩码外10000的junk padding。
- 池化算子层面测试无效token不参与mean/max、三段不按padded轴切分。完整模型的padding差异另行记录，1e-4是描述性阈值，发现非不变不静默改模型。
- 不完整尾部不进入池化，并不意味着它对整条管线完全无影响：外部z-score和内部RevIN仍可能读取这些有效原始观测。不能把局部池化性质写成整条编码器的尾部不变性。

## 固定pilot判断

候选只在mean_max/thirds的fusion条件中按validation Macro-F1选择，平分按mean_max优先。要建议另立复制阶段，需同时满足：

1. 相对本轮复现的mean融合SVM至少+0.5pp；
2. 相对既有最强非适配管线（包括已优化增益融合头）至少+0.25pp。

无论候选是否通过，都报告全部6条件及与TCN的validation差，不以弱mean-SVM为唯一优越性参照。不自动追加四seed、LoRA、test、预算或关机。

这是已有test结果之后的train/validation探索，不能把后续在同一test上追分当作新的独立确认。旧E03T门槛、模型和结论保持不变。

机器协议：`configs/experiments/improvements/e02_pooling_pilot_protocol.json`。
输出：`artifacts/improvements/e02_pooling_pilot/`。入口：`scripts/run_e02_pooling_remote.sh`。
