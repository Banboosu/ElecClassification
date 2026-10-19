# E03T：冻结最终方法后的正式test协议

2026-10-19，用户明确选择“1：冻结方案、固定40%标签预算及对照清单，执行一次正式test”。此前E03E的禁止自动test约束不回改；本阶段为独立的新授权、新文件、新输出目录。

## 固定范围

- 四seed **42、44、45、46**；不补回方案探索pilot43。每seed原始训练集的40%即9,827个标签，validation 3,510、test 7,020；沿用各seed原split及selected train IDs。
- **7方法×4seed=28项推理，0项重训**：增益融合NN、统计NN、仅MOMENT NN、父融合SVM、匹配带宽统计SVM、原统计SVM、原统计RF。
- NN只加载E03E按validation选定的model_best.pt（200轮预算内最佳），冻结具体epoch、模型哈希、标准化器及类别顺序。SVM/RF只加载已完成训练CV后refit的classifier.joblib。不重新搜索、不合并train+validation、不按test重选epoch/seed/阈值、不改端点或集成。
- 历史全标签TCN/完全微调test不纳入同预算正式排名；本阶段结论仅限此40%预算及这套数据协议。

## 冻结与新硬件关卡

任何test特征/预测生成前冻结机器协议、源代码/测试/启动器、父阶段完整冻结依赖、28个模型及其metrics/validation预测、label encoders、training normalizers、selected IDs、split manifests、公共权重指纹与实际权重、包版本及新机环境。独立源代码快照和SHA清单保存在本阶段analysis。

新机为SM8.9 RTX4080SUPER（显示32GB），旧模型/缓存来自V100，不能假定跨GPU逐位相同。因此**四seed全部通过validation-only技术关卡后才能编码任何test**：

1. 每个NN用旧AMP validation缓存检查完整3,510行的预测/概率/指标恢复：预测一致率≥99.9%，概率最大绝对差≤0.005，Macro-F1差≤0.1pp。
2. 统计RF/SVM/匹配统计SVM与父融合SVM用原validation特征重现保存预测，必须逐行一致。
3. 每seed仅取原validation顺序的前256行，在新机从raw序列重提AMP和FP32特征。NN预测一致率≥99%，概率差≤0.02；FP32特征最大绝对差≤0.005；融合SVM预测一致率≥99%。这些容差只检查硬件/加载/预处理一致性，不选择更优方法。
4. 任意失败即在test前停止，保留事故，不自动扩大容差或切换精度。不会因validation分数有涨跌而选择新模型。

推理沿用原方程：原始有效前缀外部z-score；内部RevIN及完整有效patch mask-aware mean不改；NN骨干AMP-FP16，融合SVM骨干FP32；各64batch，TF32关闭；训练时已有15统计量scaler/BranchScaler原样加载。新缓存独立保存，不向V100旧cache补写。统计方法直接用raw统计，不走外部z-score。

## 预先固定判定和报告

- 主指标：四seed配对Macro-F1，融合NN减**预先固定的父融合SVM**。沿用复制所需实用差：平均≥0.5pp且≥3/4严格正向。
- 消融指标：融合NN减仅MOMENT NN，平均≥0.25pp且≥3/4正向。
- 两条均通过才记“本阶段test确认规则通过”；未通过仍报告全部28项，不改阈值、不pool43、不按test换参照。原validation成功判定不回改。
- 另完整报告与统计NN、匹配统计SVM、原统计SVM/RF的固定配对差，以及Accuracy、全部类别P/R/F1、混淆矩阵、电池类别2的默认argmax Recall/FPR。有概率的NN/RF报告AP；未开启probability的SVM不临时校准或伪造AP。
- 区间仅描述性配对t区间，不把重叠随机split当独立重复样本，不做外部泛化/安全部署保证。已有历史实验曾使用该数据的test；这是**本阶段一次固定评价**，不是全项目从未看过的全新测试集。
- 所有28项预测、来源模型哈希、具体best epoch、cache及预测哈希、技术关卡和成本均保留。摘要从保存的CSV重算，检查ID/真值/类别顺序与指标一致。

## 一次评价与技术恢复边界

一个逻辑评价套件、固定28模型，不因结果再训练/调参/改变推理路径。中断时只复用已提交且hash通过的cache/预测；没有完整提交的技术步骤可按同一冻结协议重试并留日志，这不赋予按test选方案的权利。

本阶段完成后停止、审计、备份并报告，无论正负均不自动执行新实验、追加seed/budget、LoRA或远程关机。硬件耗时单列，不与V100训练时间拼接成同硬件benchmark。

机器协议：`configs/experiments/improvements/e03t_final_test_protocol.json`。
入口：`scripts/run_e03t_remote.sh`。
模块：`src/tcn_moment/final_test.py`。
