# E05B：legacy TCN 结构与训练长度诊断（validation-only）

日期：2026-10-19。用户连续要求继续后，对原始TCN是否发挥性能进行有边界的真实训练诊断。E05A只补审历史保存预测，本阶段单独建立新模型与输出，不覆盖历史实验。

## 固定问题与范围

探索seed43，10%与100%训练标签，沿用既有seed43 split。两个标签预算各四条件，最多8×50=400epoch；每个条件batch32，所选train按现有嵌套分层函数选取；validation固定3510。原始未归一化输入，统一padding1024、三类0/1/2、长度至少18。loader只构造train/validation序列，不构造test张量、不调用通用train入口或test评价。读取已有split的test ID只用于排除重叠，不读取其序列或标签。

| 条件 | 结构 | 池化 | 停止 |
|---|---|---|---|
| current_early7 | 当前四个双卷积残差块，64/64/128/128，dilation1/2/4/8 | masked mean | 最多50，patience7 |
| current_fixed50 | 与上同结构、同初始化/批次顺序/初始RNG | masked mean | 固定50 |
| legacy_masked50 | 六层普通因果卷积，128通道，dilation1/2/4/8/16/32 | masked mean | 固定50 |
| legacy_unmasked50 | 与上同结构、同初始化/批次顺序/初始RNG | 全padding轴mean | 固定50 |

统一kernel3、每卷积后ReLU/dropout0.3、最后线性三分类；交叉熵直接作用logits等价于softmax后的分类交叉熵。局部感受野当前61、legacy127；全局池化仍聚合整个有效时间轴，不能仅从感受野断言优劣。

## 为什么这样控制

当前两条件只改变早停，完整50轮同时回放历史patience7的停止位置；共享轨迹前缀必须完全一致。两legacy条件只改变池化，初始化参数哈希必须相同。current_fixed50与legacy_masked50比较结构；结构不同导致参数数目、初始化随机数消耗与dropout掩码不同，不声称逐层相同权重或等容量。

统一沿用当前AdamW：LR0.001、weight decay0.0001、梯度裁剪1、按validation Macro-F1的ReduceLROnPlateau(factor0.5,patience3)。全部FP32，关闭TF32；任何非有限loss/logit/gradient立即停止，不自动改精度或换超参。所有条件使用PyTorch默认初始化规则。固定50轮条件保留每轮曲线、端点指标、真正最高F1 checkpoint，以及按min_delta0.0001规则选择的checkpoint；不让min_delta的选择语义掩盖微小峰值。early7复刻当前选择规则。

**这是legacy结构的受控PyTorch移植，不是原Keras训练逐位复现。** 原脚本Glorot初始化、Adam、按val_loss降学习率、以训练集最长序列padding、Weighted-F1与每轮使用名义test均不同。此处统一优化器/初始化/padding/划分以定位结构、池化与早停差异；不能据此证明任一架构的全局最优性能，也没有开展学习率或正则化搜索。

## 冻结、运行和审计

在第一步真实数据训练前保留机器协议、数据/split SHA、模块及其导入的本地依赖、单测和本文的SHA与源码快照、软件包版本、GPU与合成检查结果。协议冻结后修改输入即拒绝运行/恢复。套件文件锁排除并发；每epoch原子保存含模型/optimizer/scheduler/RNG的恢复checkpoint。最多8小时；超时保存已有轨迹并标记不完整，不扩大预算。

所有8条件完整后才汇总。审计每个条件的train ID嵌套/一致性、validation有序ID与真值、保存预测的概率/argmax、独立混淆矩阵指标、历史峰值/端点与更新步数、checkpoint哈希；验证current共享轨迹前缀与legacy初始化。无test推理。汇总只报告描述性validation差、训练成本、早停是否错过后续收益；不设事后显著性门槛、不自动复制seed、追加训练、测试或关机。

seed43与validation均已多次用于发现方案，本轮设计前也已知历史/E03T test，必须写成后续探索。现有数据缺少设备/时间组，不能推断独立外部泛化。完整标签与10%标签之间训练步数不同；同标签预算内部对比才用于诊断。TCN与MOMENT成本并非同硬件重训，不混用计时。

机器协议：`configs/experiments/improvements/e05b_tcn_legacy_pilot_protocol.json`。
入口：`python -m tcn_moment.tcn_legacy_pilot {freeze,run,audit}`。
结果目录：`artifacts/improvements/e05b_tcn_legacy_pilot/e05b_tcn_legacy_pilot_20261019_v1/`。
