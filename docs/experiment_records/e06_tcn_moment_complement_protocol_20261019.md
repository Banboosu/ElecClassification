# E06：强TCN与冻结MOMENT的互补实验（结果前方案）

## 目标与边界

用户2026-10-19授权开始上一轮建议的“冻结TCN表征 + MOMENT互补”实验。固定10%训练池标签（每seed2456条）、seed42/44/45/46，不加入pilot43，不重新训练编码器。已有E05C和历史test结果已知，validation被反复使用；本阶段是post-test探索，不是独立确认，不重新访问旧test序列、表征或预测。

问题：在已较强的legacy masked TCN之上，MOMENT预训练表征是否仍提供增量？不保证正结果，不以得到优势为由追加预算。

## 固定三个条件与原TCN参照

复用E05C对应10%训练预算的legacy masked selected模型，保持raw输入、mask与1024长度。提取masked平均池化的128维TCN特征及原分类器logits；恢复validation预测必须达到argmax零差异、概率最大绝对差≤1e-5。原TCN所有文件只读、不改写。

每个seed训练三个同形残差头：

1. `tcn_only`：TCN特征 + 1024维全零旁路。
2. `tcn_moment`：TCN特征 + 1024维冻结预训练MOMENT表征。
3. `tcn_random`：TCN特征 + 同架构随机初始化MOMENT表征。

三个条件统一为 `原TCN logits + MLP([TCN, side])`，MLP为1152→128→ReLU→Dropout(0.1)→3，输出层权重/偏置初始化0。每seed相同初始化、打乱/Dropout随机种子，所有条件epoch0与原TCN完全相同。零旁路控制保持参数数目一致，但有效输入信息/可用自由度并不相同；随机表征控制用于补充区别“预训练信息”与“随便加高维特征”。

每个非零分支的StandardScaler只拟合所选2456个训练样本，并除以对应维度平方根。每条件固定100epoch、batch128、AdamW(lr=1e-3, weight_decay=1e-4)、clip1.0、无scheduler/早停，FP32且禁用TF32。最佳头仅按validation Macro-F1严格提高选择（平分保留先到点），epoch0可选，保存最佳与第100轮输出。四seed×三条件，共12个头/1200epoch，不扩标签比例、结构、学习率或seed。

原始TCN输出作为额外固定参照，不计新拟合次数。不会拿弱头重新拟合结果替代原强TCN。

## 数据与缓存

严格复用E05C split、有序训练ID和嵌套2456样本；仅加载train/validation序列。test ID仅用于划分互斥验证，不解析test曲线，不生成test张量。

预训练MOMENT使用现有E01 FP32、z-score、mask-aware池化缓存；冻结前核对数据/split/ID/配置/预训练指纹、源文件与依赖版本及NPZ哈希。随机控制复用E01C seed44/45/46缓存及防止加载预训练的标记；seed42原缓存不存在，直接随机构造相同模型，仅对所选train和validation提取，不下载/加载预训练权重。禁止from_pretrained的加载防护全程启用。不得将本阶段随机提取计时与旧缓存提取混为同机比较。

TCN已经在完整validation上选过checkpoint，新头也用同validation选轮数，因此存在重复选模和开发偏差。监督TCN在全部所选train上训练过，本阶段不把头的训练内部CV当作无泄漏的全管线验证；不执行这种CV。训练标签2456、选择/开发validation标签3510分别披露，不声称总标注仅2456条。

## 预先固定成功规则

四seed全部完成后，`tcn_moment`：

- 相对同形`tcn_only`平均Macro-F1至少+0.5pp，且至少3/4seed正向；
- 相对原始TCN平均至少+0.5pp，且至少3/4正向；
- 相对`tcn_random`平均至少+0.25pp，且至少3/4正向。

三条全部满足才称达到本轮探索性实用增量标准。否则如实报告未通过，不更换指标、seed或参照追求通过。差值/均值/样本SD和配对t区间仅为重叠split上的描述性统计，非独立外部推断。类别F1及电池AP/Recall/FPR全部报告，不调阈值，不宣称安全保证。

## 技术关卡、审计与结束

新模块/测试/文字及机器协议在真实新头结果前冻结；保存源码快照、所有输入哈希、包版本、host/GPU、原模型/缓存/ID指纹。先通过合成测试，再冻结，运行时再次校验。TCN提取必须恢复原输出；新头epoch0必须恢复原TCN；训练或特征出现非有限直接停止，不自动改精度。

保存特征、训练ID、scaler、初始/最佳/端点头、逐轮history、原/最佳/端点validation预测、完整混淆指标及模型哈希。最佳模型重载后验证概率一致性，最终从保存CSV独立重算指标并检查输出哈希。异常和中断现场保留，拒绝覆盖不完整运行。两小时硬墙钟预算，结束不自动下一阶段、test、扩参、关机或启动pi-wake。

入口：`scripts/run_e06_remote.sh`；模块：`src/tcn_moment/tcn_moment_complement.py`；机器协议：`configs/experiments/improvements/e06_tcn_moment_complement_protocol.json`。
