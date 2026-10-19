# E05C / N1：TCN四seed结构复核与长度诊断

日期：2026-10-19。用户在更新todo2后要求继续，执行N1；不进入N2–N6、不使用pi-wake、不启动test。E05B原协议/代码/产物保留，复用其冻结训练实现到新的独立目录。

## 复核范围

- seed42/44/45/46，排除方案发现seed43；训练标签10%/40%/100%，每seed依次运行current_fixed50、legacy_masked50、legacy_unmasked50。
- 固定36次训练、每次50epoch，最多1800epoch；无early7重训，无LR/dropout搜索，无长度旁路训练。最大8小时套件预算，失败/超时停止并保留，不能自动加预算。
- 精确继承E05B raw序列、padding1024、三类0/1/2、min_length18、AdamW LR0.001/WD0.0001、batch32/eval64、梯度裁剪1、F1 plateau scheduler(factor0.5,patience3)、dropout0.3、FP32并关闭TF32。使用各数据seed同时固定模型初始化/打乱RNG；legacy两个池化条件起点完全相同。
- 沿用既有E01 split和data.py嵌套分层训练子集；训练标签分别2456/9827/24569，validation3510。loader在序列解析前排除test行，不构造test张量、不调用通用train入口。
- 每epoch按min_delta0.0001规则保存selected，另保存真实peak及epoch50端点。复核只报告描述性多seed结果，不为本轮事后加入显著性/涨点门槛；不能将单seed差推广为总体优越性。

## seed43机制诊断（独立于复核均值）

1. 只复用E05B两个预算×三结构的selected checkpoint和validation预测；验证其SHA、模型重载1024padding概率漂移≤1e-5且argmax必须完全一致，失败即停止，不能按新分数放宽。
2. 有效观测不变，额外补零到1280/1536；分别保存全部logits/probability/预测。记录相对1024的最大logit/概率差、预测改变数、F1；对masked亦记录实际浮点误差，不因微小误差静默改计算精度。不依据结果选择新的部署padding长度。
3. 长度边界来自各选定train的1/3和2/3线性分位点；报告train/validation类别计数、长度摘要和validation短/中/长分组，固定三类Macro-F1；不以分组分数挑模型。
4. 对两个预算分别训练“仅length”和“原15维raw统计”浅决策树控制，共4次GridSearchCV：max_depth=[1,2,3,5]、min_samples_leaf20、random_state43、5折固定分层无shuffle、按Macro-F1选择、精确同分按较浅树；共80个CV fit及4个refit。两控制具有相同候选数/折/标签预算，但不声称等有效复杂度。树不需要标准化；只在train拟合。15维中已有length，不能把与length的差解释为去长度消融。
5. 决策树是廉价信息来源诊断，不能冒充E01已优化SVM或最强统计基线；不以一个浅树分数证明TCN只学到了长度。length与类别相关性不自动等于泄漏，也不是因果机制证明。

本轮不启动todo2中可选的length旁路训练。先完成以上诊断与固定复核，再决定是否需要独立4次训练消融。

## 冻结与运行

首次新业务数据诊断/训练前，核对E05B冻结输入与checkpoint、数据SHA、各seed split protocol及ID不重叠；保存新协议、完整本地src Python依赖、测试/启动器/本文、父冻结/结果/selected权重/预测以及软件和GPU哈希快照。旧冻结文件不修改。测试使用合成数据或临时toy CSV，不提前试新真实seed的效果。

先诊断seed43，再逐seed训练；suite.lock防重入，使用E05B每epoch原子checkpoint恢复optimizer/scheduler/RNG。训练输出位于`seed<seed>/fraction<budget>_<condition>/`。无后台唤醒插件和自动下一阶段；长训练由单一远端套件进程运行，状态写JSON，当前会话可按需读取。

## 审计与交付

全部36条件完成才汇总：每个selected/peak/endpoint各3510行，合计379080行；核对ID/真值/概率/argmax、混淆重算、各类与电池AP/Recall/FPR、50轮历史/更新数、selected/peak规则、子集指纹及嵌套关系、同seed两legacy初始化、模型SHA。current轨迹仅回放early7停止点，不运行额外模型。诊断另核验6模型×3padding及4树的保存输出、CV选择和hash。

按预算报告四seed均值/样本SD、逐seed差、正向seed数和描述性95%配对t区间（df3）；交叉seed split有重叠，区间不代表独立外部实验。pilot43不进入任何复核平均。报告与完整产物保全，无论正负都结束，不修改E03T/E05A结果。

数据test已知、validation与seed43反复使用；这是后续探索性benchmark。legacy是控制优化器/初始化规则的PyTorch结构移植，与原Keras仍不同，不能称完整复现或性能上限。40%结果为后续MOMENT同预算比较提供参照，不与历史完整标签test或不同GPU计时混排名。

协议：`configs/experiments/improvements/e05c_tcn_replication_protocol.json`。
入口：`scripts/run_e05c_remote.sh` / `python -m tcn_moment.replicate_tcn {freeze,verify,diagnose,run,audit}`。
