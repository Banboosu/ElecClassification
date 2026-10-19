# E03F：强分类头 warm-start 的 LoRA 对照（validation-only探索）

日期：2026-10-19。用户选择继续validation研究、用改进后的强分类头重新做LoRA公平对照。本阶段独立于E03P、E03D/E03E与E03T；不修改其结果或失败门控。

## 已知信息与适用边界

工作区存在E03T最终test结果报告，本轮设计前已被看到。**本阶段是test结果已知后的后续探索，不是独立确认，也不能再把现有test当从未看过的留出集。** 这不授权新的test评价。代码没有test编码/预测/评价入口，不导入final_test，不启动复制、不追加预算、不关机。

此前E03P只有弱的30轮无增益分类头，无法据此判断LoRA在强冻结参照下是否有实用价值。E03D-v2/E03E证明解析增益和充分训练的冻结头明显更强。本次问：在既有强头上，有限的q/v适配是否比同起点继续优化分类头更有用？不是搜索通用最佳LoRA。

## 固定数据与起点

- 单seed43、40%训练标签（9,827），validation3,510；仅pilot探索，seed43本就多次用于方案选择。
- 固定原数据指纹、三类定义、train/validation/test划分、嵌套训练ID。重叠随机划分、validation多次复用及历史test暴露全部披露。
- 使用E03D-v2 `moment_fanin_gain`、`fusion_fanin_gain` 各自**200轮预算内按validation选择的model_best.pt**（而非弱E03P头）。同支路的rank0/rank8使用逐位相同分类头和训练统计归一化缓冲。
- 不同支路的warm-start头本来不同，不能声称跨支路同一参数起点；它们最初训练都花过200轮。父best epoch不一定是200，恢复其最佳状态不等于从第200轮端点继续。
- 不继承旧optimizer、scheduler或RNG。两处理都新建AdamW，固定相同训练/打乱seed20043。恢复中断仅取本阶段已提交epoch状态，不重置已完成轨迹。

## 四个条件与预算

执行顺序：`frozen_fusion`、`lora8_fusion`、`frozen_moment`、`lora8_moment`。全部需要完成后才判定，不按中间分数删减。

| 条件 | 头起点 | Encoder更新 |
|---|---|---|
| frozen_fusion | 父gain fusion最佳头 | 不更新 |
| lora8_fusion | 同一父gain fusion最佳头 | 24层共48个q/v的rank8 LoRA |
| frozen_moment | 父gain moment最佳头 | 不更新 |
| lora8_moment | 同一父gain moment最佳头 | 同上 |

- 每条件固定追加 **30epoch**，不早停、不根据结果延长，总120epoch。
- 所有条件都使用在线相同骨干前向，不把旧V100缓存与新卡梯度路径直接混用；有效batch128，物理batch64（两微批，末尾精确按样本数累积），eval batch64。
- 同为GainFusionHead：LayerNorm(MOMENT)/sqrt1024、仅所选train拟合15维统计/sqrt15、拼接后全局gain sqrt(1039/有效支路数)、1039→128/GELU/dropout0.1→3。
- 冻结骨干参数、关闭base dropout；只开LoRA和头的dropout。rank8、alpha16、adapter dropout0.05、初始化seed30043；零B初始化。checkpointing关闭，避免冻结输入梯度脱离。
- head基础LR **1e-4**（原父训练尾部终值），adapter基础LR **1e-4**（原E03P值）。对两处理都明确重置为本阶段30轮、10%warmup+cosine、末值比例0.1。不把调度重置只用于LoRA。
- 非fused AdamW，weight_decay0.01、梯度裁剪1、未加权CE。AMP-FP16，固定scale1024、growth_interval1e9。任何非有限loss/logits/梯度立即报错；不自动换精度、缩batch、换scale或跳步。
- 候选包含**epoch0**与1..30最高validation Macro-F1，精确同分优先早epoch。若适配退步，允许选回零适配器的强起点；同时报告末轮端点、每轮曲线与成本，不能用best掩盖退化。

## 技术检查与冻结

先对两支路用实际strong head做batch64的合成输入检查：零适配器输出、A/B梯度、固定scale、冻结参数哈希、学习状态重载。此检查不用项目样本/标签，不是选择LR的科学实验。检查通过后，首次新数据训练前冻结协议、源码、父模型/预测/指标、父输入与依赖、runtime、smoke与文档/测试。

新卡epoch0的online validation与原V100父预算200最佳预测核对：每支路允许至多 **1/3510个argmax差异**、Macro-F1绝对漂移 **0.05pp**。超过即停，不能按分数放宽。每对处理初始概率须完全一致，训练ID/统计scaler严格相同。旧validation结果本来已知，检查只用于迁移数值一致性。模型公共权重仍核验原revision/大小/SHA。

## 冻结判定

只有四条件完成、120epoch数值及预测/模型审计通过后，rank8 fusion同时满足才可推荐另立复制阶段：

1. 相对本轮继续训练的frozen fusion至少 **+0.5pp**；
2. 相对最强非适配参照至少 **+0.5pp**。参照包括本轮两frozen控制、父gain统计/MOMENT/融合best200及父融合/匹配统计SVM，不能只选弱参照；
3. 相对rank匹配的LoRA moment至少 **+0.25pp**。

这些沿用原pilot的实用阈值，不因旧test结果另设目标。即使通过也不自动复制或test；失败保留，不能追加rank、seed或预算救门槛。

## 公平性与成本限制

同支路同头、同数据、同追加预算和头LR排除主要弱头混淆，但不表示处理的有效容量、dropout随机数消耗或算力成本相等。LoRA额外消耗adapter dropout RNG并反向传播，不能声称两处理每个头dropout掩码相同。不同支路先前分别选过最佳epoch，本轮也有31个候选，属于warm-start后续优化，不是从头端到端训练的公平结论。

GPU现为RTX4080SUPER；原V100训练成本不直接合并成同硬件测速。真实test没有新的评价许可，已知test结果不能证明本轮泛化。

机器协议：`configs/experiments/improvements/e03f_strong_head_lora_protocol.json`。
入口：`scripts/run_e03f_remote.sh` / `python -m tcn_moment.strong_head_lora`。
独立目录：`artifacts/improvements/e03f_strong_head_lora/`。
