# E07：强融合头LoRA精简复制（结果前方案）

## 授权、目标和范围

用户在看到E06结果并讨论后授权“开始实验”，执行上一轮明确建议的精简复制：seed42/44/45/46，固定40%训练池标签，每seed仅`frozen_fusion`和`lora8_fusion`两个条件，从对应E03E最佳融合头出发各追加30轮。共8条件、240轮，不加入seed43、不搜索rank/LR、不增加训练轮数。

目标是验证“分类头已优化后，适配MOMENT编码器是否还有可重复增量”，不是要求打败TCN，也不保证正结果。E03F-v2 seed43的+0.7272pp仅是提出假设的探索信号。旧test和四seed历史validation结果均已知；本轮仍是post-test探索，不是全新盲测或外部确认。未授权重新评价旧test。

## 固定模型与训练

- 父模型：每seed E03E `fusion/model_best.pt`（200轮预算内按validation选中），两处理恢复完全相同分类头参数和normalizer buffers。
- Backbone：本地经SHA核验的MOMENT-1-large公共权重，不下载移动版本；原基础参数全部冻结。
- LoRA：24个encoder block的q/v、rank8、alpha16、dropout0.05、adapter seed=30000+data seed。不搜索其他位置或rank。
- 冻结控制：复用已验证的OnlineFrozenClassifier兼容层，仍逐批在线执行同一个encoder，非缓存头控制；基础dropout保持eval。
- 两处理：head LR=adapter LR=1e-4、全新nonfused AdamW/WD0.01，物理batch64/有效128、clip1、FP16 AMP/固定scale1024、不启用gradient checkpointing，固定30轮，无早停。全新30轮10% warmup/cosine、末端LR比例0.1，两处理均重置，不继承父optimizer/scheduler/RNG。
- training seed=20000+data seed。LoRA额外dropout会消耗额外随机数，因此不能声称全处理随机轨迹或计算预算相同；数据和分类头起点/更新机会匹配。
- epoch0到30都可选择，沿用原训练器`score > best + 1e-12`；末轮端点另报。
- 所有非有限loss/logits/梯度直接停止，不自动换精度、缩batch或允许跳步；逐轮保留scale和skipped-step记录。

## 数据与技术关卡

- 与E03E相同固定split、有序9827条训练ID、3510条validation ID、标签0/1/2。仅train/validation序列被解析和张量化；test ID仅用于互斥检查。
- raw统计15维与selected-train拟合normalizer逐项恢复父缓存；外部逐序列z-score及mask沿用现有实现。验证统计量只transform，不能fit。
- 所有四seed在任何真实optimizer.step前全部通过迁移关卡：原V100父头与新机online epoch0最多1条argmax变化、Macro-F1绝对漂移≤0.05pp，不按新结果放宽；每seed两处理的epoch0概率必须完全相同。
- 结果前真实模型、实际强头、batch64合成数值检查：零适配logit误差≤1e-5、两次合成更新中B及随后A/B梯度有限且非零、scale1024、冻结权重不变、learned-state重载输出完全一致。只用合成曲线和合成标签，不做新真实训练。
- 冻结协议前核对E03E原freeze及所有输入哈希、四seed父checkpoint与预测、强非适配参照、同40% E05C legacy masked TCN保存预测/训练ID，保存源码、包版本和host/GPU；本轮硬件和先前V100/RTX4080SUPER区分。

## 固定参照和成功标准

主比较是LoRA融合减继续冻结融合：四seed均值≥+0.5pp，至少3/4严格正向。

强化比较是LoRA融合减每seed最强非适配参照：均值≥+0.5pp、至少3/4正向。参照集合事先固定为本轮继续冻结、父E03E融合/统计/仅MOMENT NN、父融合SVM、统计RF/SVM和匹配带宽统计SVM。只在这份固定集合内取最大，不按结果增删。

两项同时满足才称达到本轮适配实用增量标准。E05C同40%强TCN单独固定展示、报告逐seed差值，但不是此“适配增量”问题的通过门槛；即使通过也不等于胜过TCN。本次省略新rank-matched moment-only支路以节约约一半预算，不能据此新增“统计旁路与适配存在交互增益”的归因主张。E03F原四条件及旧失败规则不回改。

报告均值、样本SD、逐seed差及描述性配对t区间、所有类别F1和电池AP/Recall/FPR，无阈值搜索；不得用多次validation选出的点声称部署安全或独立泛化。训练9827与validation开发/选择3510标签分别披露。

## 执行与收尾

2×4条件顺序执行，套件8小时墙钟上限含准备/技术关卡与中断停机时间，不通过自动重启重置预算。每epoch保存learned state、optimizer/scaler、CPU/CUDA/shuffle RNG和history；中断只允许显式恢复已提交epoch，不覆盖旧结果、不自动恢复技术失败。协调器向子进程转发终止信号；结束审计全部8条件/240epoch以及初始/最佳/端点84240行保存预测，生成汇总和输出manifest。保存所有失败及负结果。

不启动外部观察器/pi-wake，不自动追加实验、test或关机。初始冻结和技术关卡先备份本地；长训练完成后的完整终态备份在后续明确检查时执行，不能把“已启动”写成“已完成”。

入口：`scripts/run_e07_remote.sh`；模块：`src/tcn_moment/replicate_fusion_lora.py`；机器协议：`configs/experiments/improvements/e07_fusion_lora_replication_protocol.json`。
