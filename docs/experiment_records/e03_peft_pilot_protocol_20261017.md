# E03P 启动前协议：LoRA 与统计支路的配对消融

> 本文件仅记录启动前设计，不回填实验结果。状态见 `todo2.md` 与套件状态 JSON。
> 机器协议：`configs/experiments/improvements/e03_peft_pilot_protocol.json`。
> 这是单 seed 的探索性 pilot，不是独立确认，也不是正式 test 评价。

## 1. 为什么进入 LoRA

E01C 中，预训练融合相对随机融合提高约 2.50/2.63 pp，支持预训练权重在固定架构中的贡献；但相对更严格统计控制只提高 0.64/1.05 pp，20% 未通过预设实用门槛。不能继续只扩大冻结 SVM 网格来追求有利结论。

本轮检验：保持原始尺度统计支路，在冻结公共骨干的基础上，让少量注意力参数适应当前分类任务，是否优于**同分类头的冻结模型**及既有强管线；同时检查统计支路在适配后是否仍有增量。

## 2. 七个条件，固定样本预算

- seed **43**，训练标签比例 **40%**：9,827 条训练、3,510 条 validation。
- 沿用 E01B/E01C 同 seed 的 split、训练 ID 和标签编码，不重新抽样。
- 验证标签也计入标注预算。不能声称全流程只标注 9,827 条；validation 被反复用于选轮数。
- 不新增无标签领域预训练，不生成 test 表征、预测或指标。

| 条件 | 可训练部分 | 输入支路 |
|---|---|---|
| statistics_mlp | 共享 MLP 头 | 原始统计 |
| frozen_moment_mlp | 共享 MLP 头 | 冻结 MOMENT |
| frozen_fusion_mlp | 共享 MLP 头 | 冻结 MOMENT + 统计 |
| lora8_moment_mlp | rank8 q/v LoRA + 共享头 | MOMENT |
| lora8_fusion_mlp | rank8 q/v LoRA + 共享头 | MOMENT + 统计 |
| lora16_moment_mlp | rank16 q/v LoRA + 共享头 | MOMENT |
| lora16_fusion_mlp | rank16 q/v LoRA + 共享头 | MOMENT + 统计 |

rank8/rank16 都有同 rank 的单 MOMENT 对照，不能用 rank8 单分支与 rank16 融合冒充单因素统计支路消融。七个条件完整运行，不因中途结果改变范围或只保留最好者。

## 3. 适配器和分类头

### 3.1 保留原骨干

- 公共 checkpoint 沿用已校验的 `AutonLab/MOMENT-1-large`，revision `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`，SHA-256 `a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572`。
- 原逐序列 z-score、内部 RevIN 和完整有效 patch 的 mask-aware mean 不改动。
- 原骨干的全部参数冻结，不能悄悄打开 LayerNorm、patch embedding 或旧分类头。
- 仅 24 个 encoder block 的 SelfAttention.q/v 共 **48 个投影**注入 LoRA。
- A 使用 Kaiming 初始化，B 初始化为零；alpha=2×rank，LoRA dropout=0.05。
- 基础骨干保持 eval，训练时只启用适配器与共享分类头的 dropout。eval 并不禁止适配器梯度。
- 原模块的接口与权重保留；不依赖新装 PEFT 库，不修改已冻结的 E01/B/C 源文件和配置。

### 3.2 同分类头对照

- MOMENT 1,024 维向量按样本 LayerNorm 后除以 sqrt(1024)。
- 15 维原始统计仅用**选定的 9,827 条训练样本**拟合标准化，再除以 sqrt(15)。均值和尺度保存在分类头 buffer 中。
- 拼接 1,039 维，Linear(1039,128) → GELU → dropout(0.1) → Linear(128,3)。
- 单分支条件将被排除的支路置零，保持相同头拓扑、参数初值与 buffer；所有条件核验初始头哈希相同。
- 零输入通道没有有效信息，所以“名义参数量相同”不代表有效容量相同，不用这一点声称统计 NN 与大模型在容量上完全公平。
- 头 seed 为 10000+data seed，适配器 seed 为 30000+data seed，训练/shuffle/dropout seed 为 20000+data seed，分别固定。

原分类头仍作为旧特征提取接口的一部分存在，但被冻结且不参与本轮预测。实际输出来自新的共享头。

## 4. 精度与预算

- FP32 主权重 + CUDA AMP FP16；不是 BF16（V100 不具备原生 BF16 训练能力）。
- 可训练参数：rank8 约 **921,987**，其中 adapter 786,432、共享头 135,555；约占含新头模型参数的 **0.27%**。rank16 adapter 为 1,572,864，共约 1,708,419 可训练参数。
- 全部基座参数保持 FP32；参数少不意味着推理骨干变小、一定比 TCN 更快。
- 冻结 NN 使用专门的**未适配 AMP 特征缓存**，不能直接把 E01 FP32 特征当作同精度缓存；只缓存选定训练子集和 validation。
- 统计 NN 不需要 MOMENT 信息，被排除的 MOMENT 支路为零。
- LoRA 需逐批经过整个骨干前后向，不复用静态特征缓存。
- 物理 batch=64，累积两步达到 effective batch=128；短尾组按实际样本数加权，不固定除以 128。
- 冻结头每步 batch=128，保持相同每 epoch 优化步数、shuffle 种子及学习率计划。
- 验证 batch=64；输入和 mask 直接驻留 GPU，减少重复数据复制与 worker 负担。

真实合成输入冒烟（不是目标验证集结果）：

| 物理 batch | 精度 | 训练吞吐 | 峰值 allocated 显存 |
|---:|---|---:|---:|
| 32 | AMP FP16 | 82.5 samples/s | 约 10.3 GiB |
| 64 | AMP FP16 | 111.4 samples/s | 约 18.6 GiB |

包含短/常量/不同有效长度输入；零适配器输出误差为 0，适配器梯度有限且非零，全部冻结参数哈希前后一致，学习状态保存重载后输出误差为 0。实际训练吞吐以运行记录为准，不能将这些数字当线上推理性能。

## 5. 训练和选择

- 每条件最多 **30 epochs**，至少 10 epochs；patience=7，patience min_delta=0.0001 Macro-F1。
- 最优模型按 validation Macro-F1 的**实际最高值**保存；min_delta 只用于 patience，不丢弃幅度小但更好的 checkpoint。完全相同分数保留更早 epoch。
- 未加权交叉熵；不临时增加类别权重、数据增强或新正则化。
- AdamW 非 fused，weight decay=0.01；head LR=1e-3，adapter LR=1e-4。
- 固定最大优化步数的 10% 线性 warmup + cosine，最低 LR 比例=0.1；不能根据中途分数临时延长训练。
- GradScaler 初始 scale=1024，gradient norm clip=1；非有限 logits/loss/梯度范数直接中止，不静默切换精度、扩大网格或跳过坏数据。
- gradient checkpointing 关闭；避免冻结输入配合 reentrant checkpoint 导致适配器梯度被截断。
- 记录每 epoch train loss、validation Macro-F1/Accuracy/loss、LR、适配器梯度见证、GradScaler、耗时和显存；保存 validation 三类概率。

本轮改变了分类头与训练方式，相对旧 SVM 的比较是整条管线比较；只有同 MLP 头的冻结/LoRA 配对才直接控制了头家族差异。不能宣称 NN 与 SVM 的搜索预算完全一致。

## 6. 启动前固定的扩展建议门槛

分别对 rank8/rank16 的融合候选判断，以下三项**同时满足**才列为可建议复制的融合候选：

1. 相对同头 `frozen_fusion_mlp` 至少提高 **0.5 pp** Macro-F1。
2. 相对全部非适配强参照中的最高分至少提高 **0.5 pp**：统计 NN、冻结 MOMENT NN、冻结融合 NN、父预训练融合 SVM、父匹配带宽统计 SVM。
3. 相对**同 rank** 的仅 MOMENT LoRA 至少提高 **0.25 pp**，否则不能宣称“统计支路在 LoRA 后仍有实用增量”。

满足的候选中选 validation Macro-F1 最高者；相差不超过 **0.01 pp** 时优先低 rank。若无候选满足，完整记录各项差值，不回改门槛。仅 MOMENT LoRA 的正结果也应保留，但不冒充融合门控通过。

上述只是单 seed 的**探索扩展建议**，不具有多种子稳健性或统计显著性含义。即使通过，本套件也不自动开始复制；后续需单独冻结其余 seed 与规则。

## 7. 恢复、审计和停止

- 每 epoch 原子保存学习状态（仅 adapter + 新头，含统计标准化 buffer）、optimizer、GradScaler、CPU/CUDA/shuffle RNG 和历史。
- last checkpoint 保存已提交的最优状态；崩溃后即使更晚未提交的 `model_best.pt`/history 已被覆盖，也能从 last 恢复正确最优点。
- 重建基座时必须重新核验公共 checkpoint；metadata 不匹配则拒绝 resume。
- 每个 LoRA 条件训练前后哈希**全部冻结参数**，必须不变；最优模型重载须复现选择分数。
- 最终核对训练 ID、validation ID/真值、预测/概率、指标、历史最佳 epoch、初始共享头哈希和 adapter 参数数量。
- 模型、历史、源码快照和配置放入独立 E03 输出，不覆盖历史结果。
- 本轮结束只审计汇总，不自动开始复制、全量微调或 test。不能预先承诺达到 95% 或优于 TCN。

入口：`scripts/run_e03_remote.sh`。

输出根目录：`artifacts/improvements/e03_peft_pilot/`；AMP 缓存：`artifacts/improvements/e03_amp_features_v1/`；日志：`artifacts/logs/e03_peft_pilot_20261017_v1.log`。
