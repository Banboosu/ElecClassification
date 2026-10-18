# E03E 启动前协议：增益融合头的四种子复制

> 2026-10-18；本文件只记录启动前方案，不回填结果。
> 机器协议：`configs/experiments/improvements/e03e_gain_replication_protocol.json`。
> 固定读出器复制，非LoRA试验；validation-only，不自动评价test。

## 1. 起点与边界

E03D-v2 在seed43、40%标签上取得增益融合头Macro-F1 91.7225%，比原融合SVM高0.5766pp，刚过建议复制门槛。但200轮端点仅91.5515%，且validation已反复用于选择，不能据此宣称稳健领先。

因此固定同一方法到 **seed42、44、45、46**，将选方案的seed43排除在复制判定之外。四个划分仍来自同一数据集、彼此重叠；旧SVM验证结果已知，seed42还曾用于E01探索。这里的“复制”只指新读出器结果在其余固定数据种子上按事先约定检验，不是独立外部确认。

## 2. 固定范围

- 标签预算40%，每个seed 9,827条训练、3,510条validation。
- 每seed三条件：统计增益头、仅MOMENT增益头、MOMENT+统计增益融合头，共 **12次NN训练**。
- 只使用未经过本任务监督适配的公共MOMENT基座。没有LoRA训练或适配后表征，也不使用test特征、预测或指标。
- 继承原nested stratified训练ID及split；类编码与父实验一致。
- 每seed只提取一次选定训练/validation的AMP表征，共享给头训练。不把旧FP32缓存冒充AMP缓存。
- 额外补seed42缺失的匹配带宽统计SVM，其他三个seed复用E01C已完成的同条件控制。

## 3. 不再调参的NN设置

- 原LayerNorm(moment)/sqrt1024，训练子集拟合统计标准化/sqrt15，缺失支路置零；拼接1039维→Linear128/GELU/dropout0.1/Linear3。
- 固定全局增益 sqrt(1039/有效支路数)，不搜索增益、列标准化、隐藏宽度或分支权重。
- 头初始化seed为10000+data seed；训练/shuffle/dropout seed为20000+data seed。三支路的去增益初值哈希必须相同。
- 固定200epochs，无早停；最多200轮validation选最高Macro-F1，完全同分取最早epoch；另报200轮端点，不替代主要指标。
- 原30轮10%warmup+cosine前缀，之后固定LR1e-4，不重新拉伸scheduler。
- AdamW非fused，基准LR1e-3、weight decay0.01、batch128、验证batch64、clip norm1。
- AMP FP16、固定loss scale1024、growth_interval1000000000；非有限数仍中止，不静默重试或切精度。
- 所有条件完整完成。原E03D-v1数值故障和原E03P失败都保留，不恢复为成功。

## 4. 强基线与缺失控制

每seed比较：

1. 父统计RF、父统计SVM。
2. 匹配核带宽统计SVM。
3. 父冻结预训练融合SVM。
4. 本轮统计NN、仅MOMENT NN。

取六者validation Macro-F1的最大值作为该seed最强参照。不是只挑弱NN头作比较。

父融合：seed42复用E01-v2-pilot，其他seed复用E01B；旧统计模型复用E01-v2。

seed42新增匹配统计SVM严格沿用E01C：15维统计按融合统计支路几何变换；五折不shuffle，C=[0.1,1,10,100,1000]，gamma候选=[reference_scale,0.05,5]；每折的scale仅由该CV训练折的**未适配FP32预训练参照**求得。最终仍是普通15维输入RBF Pipeline。它在训练时借用基础模型参照带宽，不称作完全foundation-free的独立调参基线。FP32参照未用目标标签适配，故不存在先用所有标签训练encoder再做头CV的问题。

新控制须在完整汇总前完成，不根据它的结果改变NN范围。

## 5. 复制成功规则

只用四个复制seed，不加入pilot43：

- **主要实用增量**：融合相对每seed最强参照，平均Macro-F1差至少 **+0.5pp**，且至少 **3/4 seed** 严格正向。
- **统计支路增量**：融合相对同seed仅MOMENT增益头，平均至少 **+0.25pp**，且至少 **3/4 seed** 严格正向。
- 两项同时满足，才记复制规则通过。不得用平均Accuracy、只挑正向seed或加入pilot补足门槛。
- 同时报告相对父融合SVM的差值、四seed完整表、Accuracy、端点和描述性配对t区间。区间不作为额外成功门槛，不解读为独立总体显著性。

即使通过，本套件不自动增加标签比例、训练LoRA或运行test。最终方法/测试列表应另外冻结。

## 6. 工程、恢复与审计

- 独立新目录，不编辑前序受保护源码或配置。
- 冻结协议、代码、依赖、pilot输入、各seed父指标/预测/训练ID/split与FP32参照缓存。
- 原始数据只加载一次/seed；用已有normalize_sequence对有效前缀做zscore、padding仍为0，单元测试要求与原prepare_sequence逐元素一致。仅编码选定train/validation，不把完整无标签训练池加入NN拟合。
- 原始统计只用选定train拟合标准化，保存normalizer和标签编码器。
- 每epoch保留学习状态、optimizer/scaler、CPU/CUDA/shuffle RNG，按epoch恢复；best重载须复现分数。
- 汇总核验四种子12NN状态与逐行预测：每条件3,510条，总 **42,120条NN validation预测**，另核对所有参照预测及新增统计控制。
- 核对全部2,400个训练epoch的固定scale与无跳步、每seed三头的相同初值、200轮历史最优/端点及checkpoint哈希。
- 比较是既定管线比较。NN每轮validation选择机会多于SVM的外部validation次数，预算披露，不声称搜索预算完全一致。

入口：`scripts/run_e03e_remote.sh`。
输出：`artifacts/improvements/e03e_gain_replication/`。
日志：`artifacts/logs/e03e_gain_replication_20261018_v1.log`。
