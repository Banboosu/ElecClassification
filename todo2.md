# MOMENT 改进实验备忘与执行台账

> 建立日期：2026-10-17。训练在 `ssh autodl` 的远程 CUDA 环境执行。
> 目标：在不夸大已有结论的前提下，提高 MOMENT 的分类效果，验证其低标签价值，并形成“方法—消融—成本—演示”的毕业设计闭环。
> 本文件是新阶段计划；旧正式结果及 `todo.md` 保留，不覆盖、不混合汇总。

## 0. 当前证据与问题

- 完整标签：TCN Macro-F1 为 **95.99% ± 0.73%**，MOMENT 完全微调为 **95.37% ± 0.39%**；差距只有 0.62 个百分点，但后者训练成本明显更高。
- 1%/5%/10% 训练标签：冻结 MOMENT-SVM Macro-F1 为 **68.05/77.40/80.77%**，比既有 TCN 高 **6.43/10.86/12.55** 个百分点。此处优势仅相对既有 TCN，尚需低标签三分类统计强基线。
- 同架构预训练与随机初始化对照已支持预训练贡献；不必重复做成新的“创新点”。
- 统计特征对电池异常很有效，`diff_std`、`max_abs_diff` 排名靠前。不能将总体三分类优势外推为电池安全优势。
- `momentfm==0.1.4` 的 `embed()` 内部执行 RevIN；仅修改外部 `normalize: none` 不能保证分类器获得原始幅值信息。优先尝试原始统计信息旁路，而不是盲目移除内部归一化。
- 零样本插补未胜过线性插值，检索也未全面胜过统计特征；保留负结果，避免只挑对 MOMENT 有利的任务。

## 1. 全阶段必须遵守的实验协议

- [x] 远程确认 GPU、驱动、磁盘、内存、Python 与锁定依赖；记录实际配置，不静默替换环境。
- [x] 同步代码与原始数据，校验数据 SHA-256；正式数据哈希应为 `5615a96a7894caed5d14463c77167af8098bdc1e1ebf32a33a89a12c3c5cf6e6`。
- [ ] 新实验使用 `artifacts/improvements/` 等独立目录；旧划分、旧运行和旧模型不可覆盖。
- [ ] 同 seed、同标签预算的模型使用相同 split manifest、训练样本 ID 与类别计数。
- [ ] 特征标准化、PCA、分支权重与超参数选择只使用训练集；交叉验证时放在每折 Pipeline 内拟合。
- [ ] 探索阶段默认不计算或输出 test 指标，只根据训练 CV/validation 决策；冻结候选与协议后再执行正式测试。
- [ ] 既有测试结果已被查看，新研究不能声称这些测试集从未被接触。若有新增数据，保留新的锁定外部测试集。
- [ ] 每个运行保存配置、环境、源码哈希/快照、split、训练 ID、CV 结果、模型、状态和耗时。
- [ ] 扩展搜索前固定候选范围和计算预算；失败及负结果同样留档，不以单 seed 最好结果代替五种子结论。
- [ ] 正式汇总报告 Accuracy、Macro-F1、每类表现、配对差值与不确定性；电池类别另外报告 AP、Recall、FPR，不用 Accuracy 替代安全评价。
- [ ] 标签效率明确区分训练标签、验证标签、无标签数据预算。当前 1% 训练为 244 条，完整 validation 为 3,510 条，并非全流程只标注 244 条。

## 2. E01：低标签强基线与统计信息融合（第一批启动）

### 假设

原始功率的幅值、差分和趋势统计能够补充冻结 MOMENT 的归一化表示；融合的提升应同时超过统计分支与 MOMENT 单分支，不能只是把更强的统计模型加入结果。

### 实验条件

- 标签比例：1%、5%、10%、20%、40%；探索先 seed 42，正式使用 42–46。
- 使用既有嵌套分层子集规则；最大 40% 约 9,827 条，不暗中二次抽样。
- 原始统计特征直接复用 `train_baselines.extract_statistical_features()` 的 15 维定义，输入必须为原始尺度、排除 padding。
- 冻结 MOMENT 特征沿用原 z-score + mask-aware mean 协议，单 seed 只提取一次，持久缓存并核验配置、数据、split、源码与样本 ID。

候选（相同训练子集与 CV 折）：

- [x] 统计特征 + Random Forest：五比例、五种子 validation-only 完成。
- [x] 统计特征 + RBF-SVM：五比例、五种子 validation-only 完成。
- [x] 原 MOMENT + RBF-SVM 官方对齐协议：五比例、五种子完成；无标准化、只搜 C、按训练 CV Accuracy 选模。
- [x] MOMENT + 任务优化 RBF-SVM：五比例、五种子完成；折内标准化、按 Macro-F1 选择 C/gamma。
- [x] MOMENT + 原始均值/标准差：seed 42 五比例 pilot 完成，未通过低标签扩展门控。
- [x] MOMENT + 全部 15 维统计特征：seed 42 五比例 pilot 完成，未通过低标签扩展门控；较高预算信号另列 E01B，不回改原门控。

分支融合要求：先分别标准化，再按分支维数归一并拼接，避免 1,024 维表征仅因维数压过 15 维统计；分支权重固定或纳入训练 CV，不看 test 调权。分支维数归一仅用于融合实验，优化 MOMENT 单分支保留为独立基线。

### 探索验收与扩展门控

- [x] 新增测试覆盖：统计分支用 raw、相同 CV 折、标准化不在 validation 重拟合、缓存错配/损坏拒绝、检查点指纹、validation-only 门控；运行目录复用既有禁止覆盖机制。全部 64 项测试通过。
- [x] seed 42 的 6 方法 × 5 比例 validation-only pilot 全部完成。
- [x] 原低标签扩展门控已执行：均值/标准差融合三个预算均未通过；全统计融合只有 1% 通过，因此两种融合均未扩展。负结果保留，不修改阈值。
- [x] 四种基线无论胜负均完成五种子，补齐统计强基线审计。
- [x] 110 个条件完成，386,100 行 validation 预测核验通过；测试集仍未评价。
- [x] 已输出主表、配对差值、置信区间、拟合耗时和分支 pilot 结果，详见 `docs/experiment_records/e01_feature_fusion_validation_20261017.md`。
- [ ] 统计特征置换重要性与完整标签方法的后续解释性分析。

## 3. E02：池化与时序局部信息

- [ ] 比较 masked mean、mean+max、有效时间轴三段池化；首先缓存冻结特征，避免重复骨干计算。
- [ ] 测试短序列、常量序列、不同 padding 长度及不完整尾部 patch。
- [ ] 不直接将完整 patch 判定 `.all()` 改为 `.any()`；如保留尾部，必须同时处理内部 patch embedding、attention mask 和 pooling 的一致性。
- [ ] 简单池化有效后再尝试注意力池化，并控制 head 容量，增加相近参数量的 MLP 对照。
- [ ] 对最长区间和短序列区间单独报告效果；长度分层阈值只在训练集上确定。

## 4. E03：参数高效领域适配

- [ ] 基线：冻结头、最后 2 层、最后 8/12 层、完全微调；复用既有结果时明确其环境与协议差异。
- [x] encoder q/v 原生 LoRA 工程实现；梯度、全部冻结权重、零初始化等价、学习状态保存重载与恢复测试通过。当前 E03P 仅固定 rank 8/16，rank4 不在本轮范围。
- [ ] 再视需要扩展 q/k/v/o、LayerNorm/patch embedder 适配，不一开始铺开全组合。
- [ ] 比较验证效果、可训练参数、总参数、峰值显存、实际训练耗时与推理延迟；不能从参数少直接推导更快。
- [ ] 评估 warmup + cosine、分层学习率衰减；搜索预算对照，不能只给 MOMENT 额外无限调参。
- [ ] 融合统计分支与 PEFT 采用逐项消融，区分表征适配收益与旁路信息收益。

## 5. E04：自监督领域适配（时间与算力充分后的重点）

- [ ] 只在 training split 做遮挡重构适配，验证重构仅用于选轮数；test 不参与自监督训练。
- [ ] 明确两种数据预算：仅低标签样本本身 vs. 全部训练样本去标签；不可混称同一种 few-shot。
- [ ] 对比原始预训练、目标域适配、同架构随机初始化；适配后先冻结 + 同一分类器，再考虑端到端微调。
- [ ] 给 TCN 配置公平的无标签学习对照（或明确结论仅为管线比较）。
- [ ] 同时检查重构提升是否转化为分类提升，避免重构更好但判别更差。

## 6. E05：公平性、泛化、早期诊断与鲁棒性

- [ ] 补充低标签 TCN 的合理学习率、正则化与优化步数搜索；为各模型声明相近调参预算。
- [ ] 增加受限 validation 标签预算实验，或使用训练子集 CV；另报总标注成本。
- [ ] 补 MiniRocket 等强时序分类基线，明确变长预处理及其影响。
- [ ] 索取脱敏设备/站点/时间元数据，追加设备分组或时间外推测试；现有三个 CSV 字段不足，不能从唯一 sample ID 猜分组。
- [ ] 前缀 25%/50%/75% 分类（离线进度比例）与固定时间点在线早期诊断分开命名；前缀归一化只读取已观察数据，不读取完整长度或未来真值。
- [ ] 噪声、缺失、降采样实验对所有方法使用一致扰动，并只依据训练/validation 确定强度与补齐方式。
- [ ] 对电池漏检、高置信误报进行业务人工复盘，不凭模型特征重要性作因果解释。
- [ ] 校准、拒识与人工复核比例分析，明确不存在“不漏检、少误报”的现成部署保证。

## 7. E06：模型互补与工程闭环

- [ ] 用 validation-only 权重比较 TCN/MOMENT 概率融合；若训练 stacking，只用训练 OOF 特征。
- [ ] 共同错误率仅作互补性诊断，oracle 上界不可当作可部署准确率。
- [ ] 若教师有效，尝试蒸馏到轻量 TCN；禁止用测试预测训练学生。
- [ ] 统一 `predict`：单条序列、批量 CSV、非法/过短输入、配置与标签编码器加载。
- [ ] 展示曲线、类别、校准置信度、相似历史记录与统计特征；输出脱敏。
- [ ] 同硬件测 batch=1 和批量延迟、吞吐、模型大小、峰值显存；区分训练成本与部署成本。
- [ ] 清理 README 失效链接、更新旧 TODO 状态，增加轻量 CI 与报告自动生成。

## 8. 执行记录（持续追加，以日志和状态文件为准）

### 2026-10-17：新服务器检查

- [x] `ssh autodl` 连接成功。
- [x] GPU：Tesla V100-PCIE-32GB，检查时空闲；驱动 580.65.06。
- [x] 容器内存限制约 25 GiB，数据盘约 50 GB；不能根据宿主机显示的内存直接扩大并发。
- [x] 当前为新容器，`/root/autodl-tmp` 尚无项目、模型及旧产物。系统自带 Python 3.12，不符合项目 3.11 协议；非交互 SSH 的 PATH 也未包含 conda。
- [x] 部署到 `/root/autodl-tmp/ElecClassification`；`uv sync --frozen` 成功，Python 3.11.17、MOMENT 0.1.4、PyTorch 2.12.1+cu126、NumPy 1.25.2，CUDA 可用。
- [x] 环境检查通过；全套 pytest **64 passed**；新增代码 Ruff 与 bash 语法检查通过。
- [x] 1% 标签、seed 42 的 stats RF/SVM validation-only 端到端冒烟完成；只验证流程，不作为正式效果结论。
- [x] MOMENT 单批 CUDA 前向通过，logits shape 为 `[1, 3]`；seed 42 的 FP32 train/validation 特征已完整提取并通过有限值、ID 和哈希校验，分别耗时 198.98/28.41 秒，缓存可复用。
- [x] 服务器访问 `https://hf-mirror.com` 直连/加速均超时；使用 AutoDL 网络加速访问官方源完成下载，后续设置镜像端点但以 `HF_HUB_OFFLINE=1` 读取已校验缓存，避免每个 seed 再请求网络。
- [x] 权重固定为官方 revision `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`；1,385,501,256 bytes，SHA-256 `a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572`，与本地已有官方缓存一致。
- [x] E01 validation-only pilot 已通过 nohup 启动：套件 `e01_20261017_v2`，启动 PID **2907**，2026-10-17 14:19（Asia/Shanghai）；首次检查套件和 pilot 均为 `running`、阶段 `pilot_seed42`，64 项 preflight 测试通过。PID 仅作启动记录，后续状态以 JSON 和日志为准。
- 技术重启留档：首个 `e01_20261017_v1` 在特征提取初期主动停止，没有完成分类器搜索，也没有计算 test。为确保 `python -m` 运行后保存的融合分类器可在新进程重载，将自定义变换器移到独立模块 `feature_transforms.py`，新增跨进程 joblib 重载测试；修复后使用全新 `v2` 套件和缓存，不混用旧源码产物。

#### 第一批运行方式

当前入口：`scripts/run_e01_fast_remote.sh`；加速模块：`src/tcn_moment/run_feature_fusion_fast.py`；原实验定义：`src/tcn_moment/train_feature_fusion.py`；配置：`configs/experiments/improvements/feature_fusion.yaml`。原始慢速入口 `scripts/run_e01_remote.sh` 保留作历史复现，不再用于继续本轮实验。

默认套件名：`e01_20261017_v2`。脚本先跑 seed 42 的 6 种模型 × 5 比例，再固化 validation 门控；4 种基线不论胜负扩展到 seed 43–46，只有通过门控的融合方案才随之扩展。所有阶段均不计算 test 指标。门控比较统计 RF、统计 SVM、优化 MOMENT 中较强者，比只对比两个 SVM 更保守。

```bash
ssh autodl
cd /root/autodl-tmp/ElecClassification
# 长任务使用 nohup；不要重复启动同名套件，脚本使用 flock 防重入。
nohup bash scripts/run_e01_fast_remote.sh > artifacts/logs/e01_20261017_v2.fast_launcher.log 2>&1 < /dev/null &
tail -f artifacts/logs/e01_20261017_v2.log
```

主要产物：

- 套件状态：`artifacts/improvements/e01_20261017_v2_suite_status.json`
- pilot：`artifacts/improvements/e01/e01_20261017_v2_pilot_seed42/`
- 特征缓存：`artifacts/improvements/e01_20261017_v2_features/seed<seed>/`（FP32，校验模型权重/源码/数据/split/ID/文件哈希）
- 扩展计划：`artifacts/improvements/e01_20261017_v2_expansion_plan.json`
- 扩展运行：`artifacts/improvements/e01/e01_20261017_v2_expansion_seed<seed>/`
- 测试日志：`artifacts/logs/e01_pytest.log`
- 环境快照：`artifacts/logs/e01_environment.json`
- 首次镜像失败与后续官方下载日志均保留，不将下载失败记作模型效果失败。

当前没有新测试集效果或涨点结论。正式评估须另行固定方案，再显式使用 `--evaluate-test`。

### 2026-10-17 16:23–16:32：定位慢速并切换等价加速

- [x] 瓶颈不是 MOMENT GPU 提取：特征约 227 秒即完成；后续为 CPU 上重复计算高维 RBF 核的五折网格搜索，原并行只用 6 vCPU 中的 2 核。
- [x] 40% 条件下，原论文 SVM 搜索耗时 1,531 秒，优化 MOMENT SVM 耗时 2,391 秒。此为运行时间，不是效果指标。
- [x] 新增 `fast_svm.py`：每个 CV 训练折独立拟合预处理，再以 **CUDA float64** 计算距离矩阵；各 C 共用相同 gamma 的核矩阵，CPU 5 进程求解。逐折释放临时核，避免将全部折的矩阵同时放进 25 GiB 内存。
- [x] C/gamma 候选、CV 折、训练样本、打分函数不变；最终重拟合并保存的仍是普通 RBF-SVM/Pipeline，不需要部署时提供预计算核矩阵。
- [x] 在已完成的 **10% / seed 42 / fusion_stats_svm** 上做真实数据影子对照：15 候选 × 5 折全部分数一致到舍入精度，最大差 `5.55e-17`；最优参数及 3,510 条 validation 预测完全一致。未读取 test 指标。
- [x] 同一组原耗时 **121.39 秒**，加速后 **9.44 秒**，该次对照约 **12.86 倍**；基线为此前测量，影子实验时另有旧任务占用 CPU，不能将这一倍率直接外推所有数据规模。
- [x] 全套新增/既有测试 **67 passed，6 subtests passed**，包括 CPU/CUDA 等价性与普通模型预测；Ruff 和 Bash 语法检查通过。
- [x] 切换前已完成 **28/30** 条件；保留其 112 个分类器/CV/指标/预测文件并记录 SHA-256。只中断尚未完成的单组搜索，不删除旧结果，不重新提取特征；未完成单组的内部 CV 没有 checkpoint，需用快后端重新拟合。
- [x] 新协调进程通过 nohup 启动，PID **6559**；仍使用同一个 `e01_20261017_v2` 套件、日志和缓存，状态为 `pilot_accelerated_resume`，`n_jobs=5`。原 PID 2907 已停止，其他用户进程不受影响。
- [x] 两个剩余 pilot 条件已完成，按原 validation 门控完成 seed 43–46 的四基线扩展；全部 110 条件完成，测试集保持关闭。

新增留档：

- 等价/耗时报告：`artifacts/analysis/e01_fast_svm_equivalence_20261017.json`
- 切换前指标与文件哈希：`artifacts/analysis/e01_acceleration_switch_20261017/`
- 后端测试日志：`artifacts/logs/e01_fast_full_tests.log`
- 每次恢复在源运行内新增 `acceleration_<timestamp>/`，保留后端源码、验证报告与恢复前指标；新 seed 另存 `execution_backend/`。原始特征代码、旧源码快照和缓存协议不变。

### E01 验证阶段结论（2026-10-17）

- [x] 最终审计 PASSED：110 条件、386,100 行 validation 预测；指标重算最大误差 `3.33e-16`。分类器重载后的类别、维数、参数与 CV 选择相符；源码、缓存、训练 ID、嵌套子集和 CV 分数检查通过。
- [x] 四种基线均完成五种子；在 1%/5%/10% 标签下，统计 SVM 的 validation Macro-F1 均值为 **76.90/85.53/86.85%**，原 MOMENT-SVM 为 **67.17/77.54/80.65%**。
- [x] 因此只能继续保留“相对既有 TCN 的低标签优势”，不能概括为 MOMENT 优于所有低标签方法。本轮仅有 validation 结论，不将其当成最终 test 主表。
- [x] seed 42 的全统计融合在 20%/40% 相对统计 SVM 为 **+2.22/+1.37 pp**，仅属探索信号；它不改变 E01 低标签门控失败的结论。
- [x] 已保存实验记录 `docs/experiment_records/e01_feature_fusion_validation_20261017.md`、机器可读汇总和本地约 23 MB 元数据/预测备份。约 1.2 GB 分类器与 558 MB 特征缓存仍在服务器，元数据备份不包含它们。

### E01B：较高标签预算的单独复验（不回改 E01 门控）

冻结协议：`configs/experiments/improvements/e01b_high_label_protocol.json`。

- [x] 依据 seed 42 的探索结果提出新假设，固定方法为“全部 15 维统计融合”，预算为 20%/40%，保留原 C/gamma、五折、分支权重和全部预处理。
- [x] 复验仅使用尚未运行这些融合条件的 seed **43–46**，共 **8 个条件**；seed 42 不进入成功判定。
- [x] 成功规则在复验前固定：**每个预算**相对该 seed 最强单分支的平均 Macro-F1 增量至少 **1 pp**，且 **至少 3/4 seed 严格提升**；两个预算都达标才记整体规则通过。配对区间只作描述，随机划分不独立，不声称外部数据确认。
- [x] 新增独立协议冻结/哈希校验、负结果照常汇总和门控测试；全部 **73 tests，6 subtests** 通过。
- [x] 在首次复验前于 **2026-10-17 18:04:43（Asia/Shanghai）**写入 `frozen_protocol.json`，固定 **115 个文件**的哈希；协议 SHA-256 为 `408734592360a6a4c24fac9e3657e5af1d6d086a2cd55b671098d1812c517bff`。
- [x] E01B 已通过 nohup 启动，协调 PID **10125**，首次核验阶段为 `replication_seed43`。复用已核验的 E01 表征，不重新提取；沿用等价快后端，5 个 CPU worker。
- [x] **2026-10-17 18:18:42（Asia/Shanghai）**全部 8 条件完成，预测、训练 ID/CV 折、模型参数与父实验/记录核对通过；115 个冻结文件哈希未变，已输出 `summary.json`、`summary.md`、`per_seed_metrics.csv`。
- [x] 启动器已禁止自动解锁 test 或继续启动 LoRA/全量微调；复验完成后只按冻结规则汇总，正负结果均保留。

入口：`scripts/run_e01b_remote.sh`；审计：`python -m tcn_moment.analyze_e01b`。

- 运行目录：`artifacts/improvements/e01b_high_label/e01b_high_label_20261017_v1_seed<seed>/`
- 协议、状态与汇总：`artifacts/improvements/e01b_high_label/e01b_high_label_20261017_v1_analysis/`
- 日志：`artifacts/logs/e01b_high_label_20261017_v1.log`
- 通用训练器生成的旧低标签门控会保留并改名为 `legacy_low_label_gate_not_applicable.json`，不参与 E01B 成功判定。

### E01B 结果与后续决策（2026-10-17）

- [x] 复验 seed 43–46 的两个预算均通过冻结规则，8/8 条件相对最强单分支提升；seed 42 不计入这一判定。
- [x] 20% validation Macro-F1：统计 SVM **88.42%** → 全统计融合 **89.68%**，平均 **+1.26 pp**，4/4 seed 提升；描述性 95% 配对区间 **[−0.08,+2.60] pp**，不能称统计显著。
- [x] 40% validation Macro-F1：统计 SVM **89.80%** → 全统计融合 **91.23%**，平均 **+1.43 pp**，4/4 seed 提升；描述性区间 **[+0.74,+2.12] pp**。
- [x] 对应 validation Accuracy 由 **88.43/89.82%** 提高到 **89.52/91.09%**。Macro-F1 与 Accuracy 不混用。
- [x] 增益主要来自标签 1（充电器故障）：F1 提高约 **3.57/3.85 pp**；电池异常 Recall 几乎未改善，不能宣布电池安全检测全面占优。
- [x] 四个运行累计约 **13.29 min**，复用缓存；这不是包含 backbone 前向的线上推理成本。
- [x] 本轮约 **138 MB** 的完整模型/预测/配置备份位于 `artifacts/imports/e01b_validation_20261017/`；8 个分类器 + 4 个标签编码器的 SHA-256 已与服务器逐一核对，12/12 一致。父 E01 的大部分分类器与共享特征缓存仍在服务器。
- [x] 实验记录：`docs/experiment_records/e01b_high_label_validation_20261017.md`。test 仍未评价，GPU 当前空闲，没有自动启动下一训练。
- [ ] 后续优先补归因控制：匹配核带宽/搜索范围的统计基线、同架构随机表征 + 统计融合，区分新增核几何/随机特征与预训练知识的贡献；作为新阶段冻结协议，不修改 E01/E01B 判定。
- [ ] 随后考虑 E02 池化或 E03 LoRA + 统计支路，以验证能否进一步提升完整标签性能。不能据现有 validation 结果宣称已超过 TCN 或历史完全微调 test 结果。

### E01C：核带宽与随机表征归因控制（独立新阶段）

启动前说明：`docs/experiment_records/e01c_attribution_protocol_20261017.md`；机器协议：`configs/experiments/improvements/e01c_attribution_protocol.json`。

- [x] 固定 seed 43–46、20%/40% 两个预算、两个新对照，共 **16 条件**；复用已完成的 E01B 预训练融合参照，不改其数值或门控。
- [x] 对照 1 为仅 15 维统计、但保持融合统计分支的变换和**精确参照带宽**；对照 2 为同架构随机初始化、冻结的 MOMENT + 原统计分支。
- [x] 不仅匹配候选个数：每个 CV 训练折独立拟合预训练参照的 BranchScaler，计算原 SVC 的实际 scale gamma，再与 0.05、5.0 一起交给两对照。最终 refit 同样只使用选定训练子集估算参照 gamma，不能使用 validation 或全无标签训练池。
- [x] 随机模型每个 data seed 一个初始化，FP32、相同 mask-aware pooling、独立缓存；提取时阻止 MOMENTPipeline/T5Model/T5EncoderModel 的 from_pretrained，并记录防护与缓存哈希。
- [x] 真实随机模型合成输入冒烟通过：341,243,395 参数，1,024 维有限输出，抽查张量与公共预训练权重不同；公共张量只用于比较，不载入随机模型。
- [x] 参照等价性检查通过：seed 43、20% 的 15 候选 × 5 折评分、最优参数、全部 validation 预测及最终 scaler 与 E01B 一致，评分最大差 `5.55e-17`。
- [x] 全套 **81 tests、6 subtests** 通过；新增测试覆盖精确统计分支几何、折外参照不影响训练 scaler、统计模型预测不需参照、随机输入列不混入预训练列、权重加载防护。
- [x] 在 **2026-10-17 19:15:14（Asia/Shanghai）**首次新控制前冻结 **124 个文件**与依赖版本；协议 SHA-256 为 `067ea4070711381b9d517197ed8628e0031ebae9dc3eb0efdfdc8561ceaf2777`。套件通过 nohup 启动，协调 PID **12595**，首次核验为 `controls_seed43`；test 关闭。
- [x] **2026-10-17 19:55:28（Asia/Shanghai）**全部 16 条件完成并通过审计；124 个冻结文件/依赖、随机加载防护、缓存、共享 gamma、训练 ID/CV 折、模型和预测核验通过，没有变更实验范围，test 仍未评价。

两项主张分开判断，每个预算都必须达到相应规则：

1. **超出更强统计对照**：预训练融合比“原统计 RF、原统计 SVM、新匹配带宽统计”的 validation 最强者平均高至少 **1 pp**，至少 **3/4 seed** 正向；两个预算都达标。
2. **预训练权重贡献**：预训练融合比随机表征融合平均高至少 **0.5 pp**，至少 **3/4 seed** 正向；两个预算都达标。

原参照结果已知、随机划分重叠、每 seed 仅一次随机初始化；本轮是归因消融，不是外部独立确认。合并的统计参照具有更大有效搜索预算，属于保守比较。即使只有一项通过，也必须如实分别报告。

- 入口：`scripts/run_e01c_remote.sh`
- 状态/汇总：`artifacts/improvements/e01c_attribution/e01c_attribution_20261017_v1_analysis/`
- 运行：`artifacts/improvements/e01c_attribution/e01c_attribution_20261017_v1_seed<seed>/`
- 随机缓存：`artifacts/improvements/e01c_random_features_v1/seed<seed>/`
- 日志：`artifacts/logs/e01c_attribution_20261017_v1.log`
- 完成后只汇总控制结论，不自动解锁 test 或启动 LoRA。

### E01C 结果与决策（2026-10-17）

- [x] 预训练融合相对随机融合的 validation Macro-F1 增量为 **+2.50/+2.63 pp**（20%/40%），两个预算均 **4/4 seed** 提升，预训练权重贡献规则通过。
- [x] 匹配核带宽后，统计控制 Macro-F1 达到 **89.04/90.19%**；预训练融合仍为 **89.68/91.23%**，实际超出统计的增量缩小为 **+0.64/+1.05 pp**。
- [x] 实用增量规则：20% 未通过（3/4 正向但均值不足 1 pp，描述性区间 [−0.29,+1.58]）；40% 通过（4/4 正向，区间 [+0.22,+1.87]）。由于原规则要求两个预算都通过，**“两个预算均超过更强统计基线”的整体主张未通过**，不能把局部通过改写成整体成功。
- [x] 随机融合本身低于统计控制，因此不能把相对随机的 2.50/2.63 pp 写成相对统计基线的净增益。原 E01B 数值不撤销，但需保留更严格控制下优势变小的边界。
- [x] 4 个运行累计约 **39.30 min**；数据 split 重叠、每 seed 仅一次随机初始化，所有区间只作本实验描述，非独立外部确认。
- [x] 本地备份 `artifacts/imports/e01c_validation_20261017/`：16 分类器 + 4 标签编码器 **20/20 哈希一致**；约 446 MB 随机表征缓存的 **8/8 NPZ 哈希一致**、4 个 manifest 防护记录一致。
- [x] 记录：`docs/experiment_records/e01c_attribution_results_20261017.md`。截至 E01C 结束 GPU 空闲，没有自动开始新训练或 test。
- [x] 用户再次授权“继续”后进入 **E03 LoRA/领域适配 + 统计支路**的独立协议；保留同头冻结参照和仅 MOMENT 消融，不再盲目扩冻结 SVM；工程与冒烟通过，启动状态见下节，不能预先承诺超过 TCN 或达到某个 test 准确率。

### E03P：LoRA 配对消融 pilot（2026-10-17）

启动前设计：`docs/experiment_records/e03_peft_pilot_protocol_20261017.md`；机器协议：`configs/experiments/improvements/e03_peft_pilot_protocol.json`。

- [x] 固定 **seed43、40% 标签（9,827 train + 3,510 validation）**，七个条件：统计 MLP、冻结 MOMENT MLP、冻结融合 MLP、rank8/16 各自的仅 MOMENT/融合。原参照验证结果已知，不能称为独立确认。
- [x] 相同 MLP 头拓扑/初值，缺失支路置零；只拟合选定训练样本的统计标准化。固定原 mask-aware mean；只在 encoder 24 个 block 的 48 个 q/v 投影加 LoRA。
- [x] 基座全部冻结并 eval，只有 adapter/head dropout 训练；原 E01/B/C 受保护文件不修改。NN 与旧 SVM 是不同管线，不宣称搜索预算完全一致。
- [x] V100 真模型合成冒烟通过：零适配器输出误差 0、全部冻结参数哈希不变、A/B 梯度有效、重载输出误差 0。
- [x] AMP FP16、batch32 为 82.5 samples/s / 约10.3 GiB，batch64 为 **111.4 samples/s / 约18.6 GiB**；选 batch64、累积至 effective128，关闭 gradient checkpointing。数据驻 GPU，冻结头复用专属 AMP 缓存，不混用旧 FP32 缓存。
- [x] rank8 可训练 **921,987**（adapter786,432 + head135,555），约0.27%；rank16 可训练约1,708,419。参数少不代表推理便宜或实际训练一定更快。
- [x] 固定 LR head1e-3 / adapter1e-4、weight decay0.01、10% warmup + cosine、最多30epoch/至少10epoch、patience7；min_delta只影响耐心，不丢弃更高分 checkpoint。
- [x] **93 tests、6 subtests** 通过，新增 GPU 集成覆盖 epoch 中断恢复、optimizer/RNG 恢复和未提交 best/history 被覆盖后的恢复；Ruff 与 shell 语法检查通过。
- [x] **2026-10-17 20:44:38（Asia/Shanghai）**首次新验证结果前冻结 **136 个文件**及依赖版本；协议 SHA-256 `8c20bc1050d24887940211db347eaeaa8320770b7535092a39f872f3a0276e12`。七条件套件已通过 nohup 启动，协调 PID **15646**；首次核验阶段为 `statistics_mlp`，测试集锁定。
- [x] **2026-10-17 23:24:42（Asia/Shanghai）**七条件全部完成；2026-10-18 复核 136 个冻结输入/依赖、24,570 行 validation 预测/概率、训练 ID、历史最优点、共享初值、adapter 数目和冻结参数通过。七个运行均无跳过的 optimizer step，未变更范围，test 未评价。

仅当某个 rank 的融合同时满足以下条件，才列为**建议后续复制**的候选：相对同头冻结融合至少 **+0.5 pp**；相对新旧非适配强参照的最高分至少 **+0.5 pp**；相对同 rank 仅 MOMENT LoRA 至少 **+0.25 pp**。合格候选取 validation 最高，差≤0.01 pp 优先低 rank。单分支正结果也保留，但不能冒充融合门控通过。

- 套件：`e03_peft_pilot_20261017_v1`；入口：`scripts/run_e03_remote.sh`。
- 状态/汇总：`artifacts/improvements/e03_peft_pilot/e03_peft_pilot_20261017_v1_analysis/`。
- AMP 缓存：`artifacts/improvements/e03_amp_features_v1/seed43_fraction40/`。
- 日志：`artifacts/logs/e03_peft_pilot_20261017_v1.log`。
- 实际从协议冻结到结束约 **2 h 40 min**，七运行累计 wall time158.69min；已按约定只汇总，不自动复制、不全量微调、不测 test。

### E03P 结果与下一步（2026-10-18 复核）

- [x] seed43、40% validation Macro-F1：统计 MLP **75.49%**；冻结 MOMENT MLP **77.29%**；冻结融合 MLP **83.85%**；LoRA8 仅 MOMENT/融合 **88.47/88.97%**；LoRA16 仅 MOMENT/融合 **88.14/88.98%**。
- [x] 两个融合相对同头冻结各提高约 **5.13 pp**，统计支路在 rank8/16 分别带来 **+0.50/+0.84 pp**，但都低于既有融合 SVM **91.15%** 约 **2.17 pp**，也略低于匹配统计 SVM **89.48%**。两 rank 扩展门槛均失败，`selected_rank_for_proposed_replication=null`，不选择候选复制。
- [x] rank16 融合仅比 rank8 高约 **0.0031 pp**；不能据单 seed 宣称显著优劣，也不能套用“合格者平局优先低 rank”去选择未合格候选。
- [x] 事后曲线诊断（不改变门控）：三个冻结头的最优点均在30epoch边界，需检查头的收敛与缩放；LoRA 后期 train CE 下降而 validation CE 上升，不宜直接继续加 rank/训练时长。未将观察推断为确定原因或 LoRA 普遍无效。
- [x] 本地完整备份 `artifacts/imports/e03_pilot_validation_20261017/`，约 **176 MiB**：7 best + 7 last + 7 标签编码器 **21/21 哈希一致**，2 个 AMP 缓存 NPZ 哈希一致。重建仍依赖已校验公共基座权重和源码。
- [x] 结果记录：`docs/experiment_records/e03_peft_pilot_results_20261017.md`；历史检查：`derived_history_diagnostics.json`。GPU 当前空闲，未开始新训练或 test。
- [x] 用户再次授权后进入独立 **E03D 廉价冻结头预算/尺度诊断**，范围见下节；暂不使用适配表征、不扩多 seed LoRA。对监督适配后的表征做头部 CV 会含有 held-out 标签已参与 encoder 训练的问题，未来若开展需固定探针参数、隔离调参数据或折内重训 encoder，不能冒充无泄漏端到端 CV。

### E03D：缓存分类头的预算与输入尺度诊断（2026-10-18）

启动前协议：`docs/experiment_records/e03d_head_diagnostic_protocol_20261018.md`；机器协议：`configs/experiments/improvements/e03d_head_diagnostic_protocol.json`。

- [x] 固定 seed43、40% 标签、统计/MOMENT/融合三支路；**6 次头训练、12 个预算单元**，不执行 Transformer、不训练 LoRA、不使用 test。
- [x] legacy_tail：继承原第30轮全部模型/optimizer/scaler/CPU-CUDA-shuffle RNG，继续到200；不拉长cosine、不重置学习率，固定在原终点1e-4。
- [x] fanin_gain：相同参数初值，拼接输入乘固定 `sqrt(1039/有效支路数)`，仅改变全局幅度，不改支路比例或添加列标准化；第1–30轮保持原LR轨迹，后续固定1e-4至200。
- [x] 每支路单独比较最好分与30/200端点：额外预算至少+1pp且端点至少+0.5pp才标记有帮助；gain的30轮最好分至少+1pp才标记早期尺度敏感。最佳点在最后10轮时仍标记边界问题，不能直接断言已收敛或找到唯一原因。
- [x] **99 tests、6 subtests** 通过：gain1与父头完全等价、增益保留支路比例、固定30轮调度前缀、状态克隆不改父文件、拒绝adapter状态、GPU训练钩子与30轮快照；Ruff与shell语法检查通过。
- [x] **2026-10-18 10:10:22（Asia/Shanghai）**首次新诊断结果前冻结 **167 个文件**及依赖版本；协议SHA-256 `b2d5a2141c6f089d5f58ab22c7801f83f065f2fbc3b9bed98460ffb88375f812`。六条件套件通过nohup启动，协调PID **1622**；首次核验为准备缓存/原始统计，编码器和test均未执行。
- [x] 一次数据准备及训练集无标签探针完成；初始FC激活RMS约0.024–0.029，加增益后约0.49–0.55。v1三个legacy200完成，但第4项statistics_fanin_gain在132轮梯度溢出，套件按规则失败；已提交至131轮，动态loss scale已升到32768。v1不算完整六条件，不删改事故现场。
- [x] 同一131轮模型/optimizer/RNG，仅重放训练集132轮：scale32768在第50窗口复现非有限梯度，scale1024全部77窗口有限；不计算validation/test，重放权重丢弃。记录 `artifacts/analysis/e03d_fixed_scale_replay_20261018.json`。
- [x] 技术修订协议 `configs/experiments/improvements/e03d_head_diagnostic_stable_protocol.json`；记录 `docs/experiment_records/e03d_numerical_revision_20261018.md`。v2六组全部从同一初值重跑，固定loss scale1024；模式改为legacy_fresh/fanin_gain，其他模型/数据/预算/门槛不变，不混用v1结果。
- [x] 修订后 **101 tests、6 subtests** 通过，包含固定scale不增长、拒绝错误旧scale恢复；旧E01/B/C/E03P/v1受保护文件不改。
- [x] **2026-10-18 10:23:24（Asia/Shanghai）**v2新结果前冻结 **176 个文件**及依赖；协议SHA-256 `5611b044ad6f0c32ca2e7348197636a4153fb63f66c1ad7ece2968703752626b`。协调PID **2308**，六组从初值重跑，已核验固定scale1024。
- [x] **2026-10-18 10:32:09（Asia/Shanghai）**v2六条件、十二预算单元全部完成并审计通过；176冻结输入/依赖、42,120行预测/概率、历史最优、端点和模型哈希核验一致。独立检查1,200个epoch均固定scale1024、无跳过optimizer step；未执行编码器、LoRA或test。

融合候选仅当最好200相对旧融合SVM/旧匹配统计SVM/全部新统计及仅MOMENT最好200的最强者至少 **+0.5pp**，且比同模式仅MOMENT至少 **+0.25pp**，才可建议后续复制。不会自动复制或重跑LoRA；原E03P失败不改写，validation反复复用的选择偏差必须披露。

- 入口：`scripts/run_e03d_remote.sh`；套件：`e03d_head_diagnostic_20261018_v1`。
- 状态/汇总：`artifacts/improvements/e03d_head_diagnostic/e03d_head_diagnostic_20261018_v1_analysis/`。
- 日志：`artifacts/logs/e03d_head_diagnostic_20261018_v1.log`。
- 预计主要为分钟量级头训练，完成后只总结，不自动开始下个阶段。

### E03D v2 结果与决策（2026-10-18）

- [x] 原尺度30→200轮 validation Macro-F1：统计 **75.49→79.65%**，MOMENT **77.28→79.72%**，融合 **83.85→86.94%**；对应端点差也全部正向并过诊断幅度门槛。
- [x] 增益尺度30/200轮：统计 **86.51/88.45%**，MOMENT **81.99/83.76%**，融合 **90.43/91.72%**。同30轮的尺度提升为+11.02/+4.71/+6.58pp，支持原冻结头训练/幅度设置有明显改进空间，不是唯一机制证明。
- [x] 增益融合epoch169取得 **91.7225% Macro-F1、91.5954% Accuracy**；比同seed/预算原融合SVM **91.1459%**高 **+0.5766pp**、比同模式仅MOMENT高+7.9649pp，达到本轮建议复制门槛。余量很小，单seed且validation复用，不宣称显著或稳健领先。
- [x] 200轮精确端点为 **91.5515%**，相对原SVM仅+0.4056pp，未达到0.5pp；候选通过基于原定预算内最好模型，不能改写为任意后期模型都过门槛。
- [x] 充分训练/改尺度的冻结融合已高于旧LoRA融合约2.75pp；旧E03P失败不撤销，旧+5.13pp不能当作相对充分优化冻结头的稳健收益。本轮未在新头设置下重测LoRA，不断言LoRA为负贡献。
- [x] v2约8分45秒结束，六运行wall合计7.03min、epoch计时合计6.44min；不等于完整线上管线耗时。101 tests、6 subtests通过。
- [x] 本地备份 `artifacts/imports/e03d_head_validation_20261018/`：v2约96MiB，37/37模型/快照/标签编码器哈希一致、2/2 prepared NPZ哈希一致；另存v1事故后约176MiB。当前GPU空闲，无自动下一训练。
- [x] 记录 `docs/experiment_records/e03d_head_diagnostic_results_20261018.md`；图 `docs/figures/e03d_head_diagnostic_20261018.svg`；v1事故与修订记录单独保留。
- [x] 用户再次授权后进入独立 E03E 四种子复制方案，包含统计NN、仅MOMENT及既有强SVM参照；不直接重跑长LoRA。方案及启动状态见下节，不据单seed通过解锁test。

### E03E：增益融合头的四种子复制（2026-10-18）

启动前协议：`docs/experiment_records/e03e_gain_replication_protocol_20261018.md`；机器协议：`configs/experiments/improvements/e03e_gain_replication_protocol.json`。

- [x] 固定 **seed42、44、45、46，40%标签**，排除选方案的seed43。每seed统计/仅MOMENT/融合三增益头，共 **12次NN训练**；同数据重叠划分及旧SVM结果已知，seed42曾用于E01探索，不称外部独立确认。
- [x] 固定原GainFusionHead、解析输入增益、200epoch、30轮cosine前缀+恒定尾部、batch128、LR/weight decay/dropout、AMP及scale1024，不再搜索。头/训练seed分别10000+/20000+data seed，按seed三支路初值核验一致。
- [x] 每seed只对9,827选定train和3,510validation提取一次未适配AMP特征；原始数据加载一次，按有效前缀调用原normalize_sequence，测试与原prepare_sequence逐元素相同。测试集不编码，骨干不监督训练。
- [x] seed42补齐缺失的匹配核带宽统计SVM，其他三seed复用E01C。新控制仍用折内未适配FP32参照、5C×3gamma×5折，不将弱NN作唯一对比。
- [x] **106 tests、6 subtests** 通过，覆盖pilot排除、12条件范围、有效前缀归一化精确等价、seed间不同/同seed支路相同初值、seed42正确参照和固定平均/方向门槛；Ruff与shell语法检查通过。
- [x] **2026-10-18 10:46:53（Asia/Shanghai）**新NN结果前冻结 **208 个文件**及依赖；协议SHA-256 `f34f3fbedfbd910be32752e694640b2edd8d96c9dcf03ea7b659b591e357963d`。nohup协调PID **3388**；首次核验为准备seed42数据及补强统计控制，test仍锁定。
- [ ] 完成全部12NN和缺失SVM控制，审计42,120行NN预测、参照、2,400个epoch的固定scale/无跳步、历史最优/端点和模型哈希。

复制成功只按四个新seed：融合相对“本轮统计NN、仅MOMENT NN、父统计RF/SVM、匹配统计SVM、父融合SVM”的逐seed最强参照，平均至少 **+0.5pp**，至少 **3/4** 严格正向；相对仅MOMENT平均至少 **+0.25pp**，至少 **3/4** 正向。两项都过才记复制成功。pilot43不加入平均，区间仅描述，不自动test、LoRA或扩预算。

- 入口：`scripts/run_e03e_remote.sh`；套件：`e03e_gain_replication_20261018_v1`。
- 状态/汇总：`artifacts/improvements/e03e_gain_replication/e03e_gain_replication_20261018_v1_analysis/`。
- 日志：`artifacts/logs/e03e_gain_replication_20261018_v1.log`。
- 本轮需要未适配骨干前向提取，其后训练廉价缓存头；不是完全不执行Transformer，也不是LoRA训练。

### 结果判定提醒

计划中所有未完成项都不是已验证结论。只有 `status.json=completed` 且完整性检查通过的结果才能汇总；“进程已启动”“模型已下载”“单 epoch 冒烟通过”都不能写成实验完成。
