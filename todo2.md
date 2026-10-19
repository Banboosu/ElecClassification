# MOMENT 改进实验备忘与执行台账

> 建立日期：2026-10-17。训练在 `ssh autodl` 的远程 CUDA 环境执行。
> 目标：在不夸大已有结论的前提下，提高 MOMENT 的分类效果，验证其低标签价值，并形成“方法—消融—成本—演示”的毕业设计闭环。
> 本文件是新阶段计划；旧正式结果及 `todo.md` 保留，不覆盖、不混合汇总。

## 当前进度（2026-10-19）

- **已完成**：E01统计强基线/融合及E01B/C复验归因；E02P池化/长度诊断；E03P LoRA pilot（失败门控保留）；E03D/E03E增益头诊断与四seed复制；E03T一次固定正式test；E03F强头LoRA pilot；E05A历史同预算TCN补审及E05B-TCN结构/早停/padding pilot。
- **当前状态**：N1/E05C已完成36条件、1800epoch、379080行validation预测，原审计与本地标准库独立复核通过；521文件/248784612 bytes完整保全且SHA零失败。检查时GPU无计算进程；pi-wake停用，不自动进入N2或test。备份实例已变更，原冻结记录与本地副本一致，但不同实例/GPU计时限制另行披露。
- **接下来优先做**：TCN结构/池化的多seed复核与长度依赖诊断 → 有界优化及受限validation标签对照 → 同硬件成本/工程闭环。强头LoRA四seed复制单列，排在TCN强参照之后；详细范围见下方优先清单。
- **仍可用现有数据推进**：统计置换解释、保存模型的错误分析、前缀/噪声/缺失诊断和统一predict；E02P池化/长度分层已完成，不重复列作必做。每阶段先冻结方案，不自动运行整张清单。
- **仅设备/时间外推被材料阻塞**：需要真实设备/时间映射或新增批次；不因此暂停其他可做项目。
- **结论边界**：融合test91.516%超过固定融合SVM90.952%，但历史同预算TCN均值92.614%更高；不宣称普遍胜过TCN，不在已消费test上反复追分。

## 接下来要做的实验（2026-10-19更新，N1已完成）

先解决参照是否充分的问题，再决定MOMENT是否值得增加适配成本。N1固定复核和已有权重诊断已完成，支持legacy masked作为后续主要TCN参照，unmasked保留为池化对照。下一步建议N2有界优化与1%/5%/10%曲线，N3–N6仍未启动；可选length旁路暂缓。不使用pi-wake、不启动test或自动连续扩展。

| 顺序 | 工作 | 要回答的问题 | 优先级 |
|---|---|---|---|
| N1 | TCN跨seed复核 + 长度/padding诊断 | legacy优势能否复现，未掩码池化依赖什么信息？ | 已完成，主参照建议legacy masked |
| N2 | 有界TCN优化 + 低标签学习曲线 | 合理优化后，MOMENT的低标签优势还剩多少？ | 下一阶段，尚未启动 |
| N3 | 受限validation标签 | 将验证标签也计入成本，低标签结论是否成立？ | 若保留标签效率主张则必做 |
| N4 | 同硬件成本与统一predict | 准确率、训练成本和部署成本如何取舍？ | 必做，可与分析工作并行 |
| N5 | 强头LoRA四seed复制 | +0.727pp是否在多个seed上保留，是否值得额外成本？ | 条件执行，接N1的40%强参照 |
| N6 | 解释、错误与鲁棒性 | 模型依赖什么信息，在哪些输入上失效？ | 先完成低成本解释/错误分析，再选一种扰动 |

### N1：先复核TCN，再解释padding收益

- [x] seed42/44/45/46 × 10%/40%/100% × 当前残差masked/legacy masked/unmasked共36次、1800epoch完成；pilot43排除，current轨迹回放early7，没有另训。40%为后续MOMENT同预算比较提供强参照。
- [x] 复核保持E05B raw/padding1024/AdamW及选模规则，统一每seed数据/嵌套子集，selected/真峰值/端点、各类/电池指标与配对差均保存；379080行预测及1800epoch独立核验通过。三预算legacy masked均4/4优于当前结构，平均+24.4453/+5.5631/+1.7592pp；重叠split区间仅描述，不称外部确认。
- [x] 长度诊断已复用seed43已有模型/预测：统计train/validation的长度与类别分布，边界仅由train确定；完成同预算“仅长度/15维统计”浅树控制。完整输出和77220行预测已独立核验，见E05C执行记录及诊断报告；该控制不是最强统计SVM，不代表因果机制。
- [x] 固定已有权重、有效观测不变，完成padding1024/1280/1536比较并记录logit/预测漂移。masked输出不变，unmasked敏感；不依据新padding分数挑部署长度。
- **暂缓可选旁路消融**：若以后仍需解释length机制，再单独冻结seed43/10%及100%、当前masked与legacy masked显式length旁路，最多4×50epoch，scaler仅train拟合。本轮没有执行；unmasked优势未在四seed稳定复现，无需为完成N1自动增加此训练。
- [x] 后续建议legacy masked为主要TCN参照，unmasked继续保留。masked均值93.6758/97.7790/98.3959%，unmasked−masked均值+0.1403/−0.0142/−0.1805pp，各区间跨0，不能推广seed43的巨大unmasked增益。本轮validation-only，不重测旧test/回改E03T；报告`docs/experiment_records/e05c_tcn_replication_results_20261019.md`。

### N2：有界优化后补齐低标签对照

- [ ] N1先定位结构与池化，随后再建立独立优化协议。建议只在seed43、10%训练预算做一轮小搜索：学习率 `{3e-4, 1e-3, 3e-3}` × dropout `{0.1, 0.3}`，每个预先保留的结构最多6候选、最多2结构、合计 **12次×50epoch**；weight decay、batch和scheduler固定。有限搜索完成即停，不扩到rank/通道/归一化全组合。
- [ ] 优化候选只用训练内部固定分层划分选择；留出数据不参与任何训练/标准化，随后用选定训练子集重拟合。不得在全部train上先监督适配encoder再只对分类头做CV。TCN、统计与MOMENT的历史/新增候选数、选择次数、最大更新数和GPU/CPU耗时分开记录；候选个数接近不等于算力或有效复杂度相等。
- [ ] 锁定选定配置后，在seed42/44/45/46的 **1%/5%/10%** 嵌套子集上补TCN学习曲线；最多2个事前保留结构，共 **最多24次×50epoch**。对照包含现有统计RF/SVM、原/优化MOMENT-SVM，不只比较弱TCN。若选定配置与N1的10%运行逐项相同，核验后复用该运行，扣除重复预算；保存模型与validation预测先校验ID/配置，不重复拟合已有基线。
- [ ] 最佳点若在预算边界，报告训练未充分的可能性；不直接把失败解读为架构上限，也不自动追加epoch。新增调参后，历史1%/5%/10% TCN与MOMENT的test差仅保留为旧配置结果，新表使用同预算validation，不混成新test主表。

### N3：单独核算验证标签的成本

- [ ] 在N2配置冻结后，从原validation按固定分层规则取一个受限子集用于选择/早停，剩余validation只用于最终探索性报告；所有方法使用相同样本ID与标签预算。执行前冻结各训练比例的验证样本数，报告 `训练标签数 + 选择所用验证标签数`，不再将244 train配3510 validation称为只需244标签。
- [ ] 受限标签实验从头训练/选择模型；禁止复用已经用全部3510 validation选出的warm-start头、超参或校准器，再宣称全流程只用了少量验证标签。若先复用既定超参，只能称固定配置下的标签预算敏感性，披露此前方案开发标签成本。
- [ ] 优先比较选定TCN、统计SVM与冻结MOMENT-SVM；暂不加入昂贵LoRA。剩余validation此前在历史研究中已被看过，不能称新的独立test。若改用train内部CV替代validation选择，也需统一总标签预算并明确报告集角色。

### N4：同机成本与交付

- [ ] 固定同一硬件/软件环境，先测N1/N2保留的TCN、统计SVM和冻结MOMENT融合；若执行N5，再补LoRA。测batch1和同一批量的端到端延迟/吞吐，预热、重复次数和CUDA同步事前固定，含预处理与统计旁路，缓存分类头计时单列。
- [ ] 同时报总参数/可训练参数、模型与公共依赖大小、峰值显存/内存、特征提取/搜索/训练耗时。历史V100和RTX训练计时不拼接为同机成本；LoRA合并权重推理与未合并推理若都测，分别命名。
- [ ] 统一predict入口并核对保存模型重载预测一致性；覆盖单条/批量、过短/非法输入、标签编码、mask与有效长度。演示中显示类别/曲线/统计信息；未经校准的softmax不标成可信概率保证。

### N5：有条件复制强头LoRA信号

- [ ] 只有需要验证“MOMENT适配增量”时执行，顺序在N1之后。固定seed42/44/45/46、40%标签、E03E各支路best200强头为起点；每seed frozen/lora8 × fusion/moment四条件，统一追加30epoch，共 **16次、480epoch**，排除pilot43，不再搜rank或延长训练。
- [ ] 保留E03F同支路相同头/缓冲、epoch0可选、两处理均重置optimizer/scheduler、在线encoder前向和相同训练预算；软件/硬件迁移先过技术一致性检查。不能复用只在缓存上训练的冻结控制计时来冒充相同在线训练成本。
- [ ] 启动前独立冻结复制判定：建议融合相对继续冻结融合、逐seed最强非适配参照分别平均≥0.5pp、至少3/4正向，相对同rank仅MOMENT平均≥0.25pp、至少3/4正向。非适配集合含本轮冻结控制、父统计/仅MOMENT/融合NN及既有SVM；**N1的同40% TCN**另外固定列为整体方法参照，不根据结果临时换比较对象。实用收益阈值和成本判定写入新协议后才执行，不改E03F原pilot规则，也不把超过冻结控制等同于超过TCN。
- [ ] 若复制不稳定或收益不足以抵消成本，保留单seed信号及失败复制，停止追加适配实验。无论通过与否，不自动打开旧test。

### N6：解释和一种受控鲁棒性实验

- [ ] 先做保存预测的类别/长度分层、共同错误与高置信错误复盘，以及固定统计模型的置换重要性（固定重复次数/随机数）。相关特征用分组置换补充，不把重要性作因果解释；缺少业务材料的错误只作数据层描述。
- [ ] N1/N2确定模型后，再从前缀、噪声、缺失中选择**一种**先做；建议优先离线25%/50%/75%前缀。先固定是完整序列训练模型的截短敏感性，还是各前缀重新训练的任务，不能混表；前缀归一化、统计与mask只用已观察样本，长度占比本身依赖完整长度，不能称固定时间在线预警。
- [ ] 所有方法使用相同样本/扰动位置，原始统计同步按扰动后的可见观测重算；强度、补齐和缺失mask规则在train上确定。不额外根据validation最优扰动挑结论，报告未扰动基线及全部预设强度。

### 暂缓与结束条件

- 暂缓注意力池化、更多LoRA投影/rank、末层/全量微调网格、自监督适配、概率融合与蒸馏。E02P简单池化没有通过门槛，不据此顺延注意力池化；N1/N2强参照和N4成本未补齐前，不扩大MOMENT搜索。
- MiniRocket保留为可选强基线，待TCN复核后再决定是否需要补；设备/站点/时间外推继续等待真实元数据或新采集数据，不伪造分组。
- **最低收尾集合**：N1 + N2 + N4 + N6的解释/错误分析；若论文保留低标签价值主张，增加N3。N5与鲁棒性训练是明确选择的扩展，不要求把所有待办跑完。
- 完成最低集合即可更新方法比较与论文结论；若MOMENT收益有限，报告精度/成本取舍即可，不为得到领先结果继续追分。新旧test均不在本计划授权范围内，不使用pi-wake；不自动启动下一阶段。

## 0. 当前证据与问题

- 完整标签：TCN Macro-F1 为 **95.99% ± 0.73%**，MOMENT 完全微调为 **95.37% ± 0.39%**；差距只有 0.62 个百分点，但后者训练成本明显更高。
- 1%/5%/10% 训练标签：历史冻结 MOMENT-SVM Macro-F1 为 **68.05/77.40/80.77%**，比当时TCN配置高 **6.43/10.86/12.55** 个百分点；只是旧配置的test对照。E01已补低标签统计强基线，validation整体高于仅MOMENT；E05B又显示legacy TCN有明显潜力，不能继续据旧差值概括低标签总体优势。
- E05B-TCN单seed43 validation：10%标签当前fixed50 **64.9645%**、legacy masked **84.8182%**、unmasked **93.6000%**；100%标签对应 **96.9684/98.1412/98.1048%**。延长当前模型在10%没有收益，需先复核结构与长度信息；这不是原Keras逐位复现，也不是新的test结论。
- E03F强头LoRA单seed43、40% validation **92.4751%**，相对继续冻结融合 **+0.7272pp**，原pilot三门槛通过；尚缺四seed复制及同预算强化TCN上下文。E02P多段/mean+max池化未通过门槛，不自动扩展。
- 同架构预训练与随机初始化对照已支持预训练贡献；不必重复做成新的“创新点”。
- 统计特征对电池异常很有效，`diff_std`、`max_abs_diff` 排名靠前。不能将总体三分类优势外推为电池安全优势。
- `momentfm==0.1.4` 的 `embed()` 内部执行 RevIN；仅修改外部 `normalize: none` 不能保证分类器获得原始幅值信息。优先尝试原始统计信息旁路，而不是盲目移除内部归一化。
- 零样本插补未胜过线性插值，检索也未全面胜过统计特征；保留负结果，避免只挑对 MOMENT 有利的任务。

## 1. 全阶段必须遵守的实验协议

以下为持续适用的执行约束，未勾选表示每个新阶段仍需核对，不表示此前实验全部未完成。

- [x] 远程确认 GPU、驱动、磁盘、内存、Python 与锁定依赖；记录实际配置，不静默替换环境。
- [x] 同步代码与原始数据，校验数据 SHA-256；正式数据哈希应为 `5615a96a7894caed5d14463c77167af8098bdc1e1ebf32a33a89a12c3c5cf6e6`。
- [ ] 新实验使用 `artifacts/improvements/` 等独立目录；旧划分、旧运行和旧模型不可覆盖。
- [ ] 同 seed、同标签预算的模型使用相同 split manifest、训练样本 ID 与类别计数。
- [ ] 预处理参数（标准化/PCA等）只在训练数据拟合；交叉验证时放在每折Pipeline内。分支权重/超参数/epoch只按事前声明的train CV或validation选择，不能使用test。
- [ ] 新探索只计算训练CV/validation指标；本计划不追加旧test评价。只有另有独立数据/协议及明确授权时，才考虑新的最终测试，不把“冻结候选”当作反复使用已消费test的许可。
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
- [ ] 统计特征置换重要性与完整标签方法的后续解释性分析（N6；优先复用保存模型/预测，不重训整套基线）。

## 3. E02：池化与时序局部信息

- [x] 比较 masked mean、mean+max、有效时间轴三段池化；E02P六条件已完整审计/保全。候选thirds/fusion 90.9733%，低于mean融合0.1726pp、低于既有最强0.7493pp，原门槛未通过，不扩展。
- [x] E02P单测及真实MOMENT合成输入覆盖短/常量、非整patch尾部、512/1024 padding和掩码外junk；padding长度最大差约4.17e-7，junk差0。不把完整模型的尾部作用与局部池化性质混同。
- **持续约束**：不直接将完整patch判定`.all()`改为`.any()`；未来若保留尾部，必须同时处理内部patch embedding、attention mask和pooling一致性，此次不启动。
- **暂缓**：注意力池化及容量匹配MLP新对照。简单池化没有通过原门槛，当前优先完成N1/N2。
- [x] E02P已按train长度分位点单独报告短/中/长三组；跨模型的长度依赖与padding敏感性另列N1，不能把E02P结果当作TCN诊断已完成。

## 4. E03：参数高效领域适配

- **暂缓**：最后2/8/12层与完全微调网格；已有冻结头和历史完全微调只作现有管线上下文，环境/协议差异保留。
- [x] encoder q/v原生LoRA工程与冻结/零适配/恢复检查完成；E03P rank8/16门控失败保留，E03F强头rank8 pilot通过，不自动追加rank4。
- **暂缓**：扩展q/k/v/o、LayerNorm/patch embedder；先完成N5四seed固定rank8复制，再决定是否还有必要。
- [ ] 比较验证效果、可训练参数、总参数、峰值显存、训练耗时与推理延迟（N4）；E03F已报告在线训练成本，但不能替代同机端到端推理基准。
- [x] E03F已使用同起点/相同追加预算的frozen与LoRA、moment与fusion四条件，完成强头下逐项消融与统一warmup/cosine；不声称验证了分层学习率衰减。
- [ ] 强头LoRA四seed复制（N5，条件执行）；额外调度/分层学习率搜索暂缓。

## 5. E04：自监督领域适配（暂缓，非当前必做）

以下保留为未来设计要求；N1–N4未收尾前不启动，不因清单尚未勾选而自动补跑。

- [ ] 只在 training split 做遮挡重构适配，验证重构仅用于选轮数；test 不参与自监督训练。
- [ ] 明确两种数据预算：仅低标签样本本身 vs. 全部训练样本去标签；不可混称同一种 few-shot。
- [ ] 对比原始预训练、目标域适配、同架构随机初始化；适配后先冻结 + 同一分类器，再考虑端到端微调。
- [ ] 给 TCN 配置公平的无标签学习对照（或明确结论仅为管线比较）。
- [ ] 同时检查重构提升是否转化为分类提升，避免重构更好但判别更差。

## 6. E05：公平性、泛化、早期诊断与鲁棒性

- [x] E05B-TCN已补单seed结构/池化/早停诊断；没有完成有界超参搜索或跨seed复制。
- [x] N1跨seed结构/池化及长度诊断已完成，补齐新的40%/完整标签TCN强参照；历史E05A只是保存预测审计，二者不混合。
- [ ] 下一步N2：以legacy masked为主，预先确定是否保留当前残差作为优化对照；小范围学习率/dropout搜索后锁配置，补1%/5%/10%学习曲线，不据N1结果追加旧test。
- [ ] 受限validation标签/训练内CV及总标注成本（N3）；若继续写低标签价值主张则必须补。
- **可选**：MiniRocket强时序基线，变长预处理、length/padding信息与训练预算必须披露，N1/N2完成后再决定。
- **材料阻塞**：设备/站点/时间外推等待真实元数据或新增批次，已有材料检查完成；不从唯一sample ID猜分组。
- [ ] 前缀25%/50%/75%或噪声/缺失选一项先做（N6）；离线进度比例与固定时间在线诊断分开，不能读取未来信息或以完整序列统计代替可见前缀统计。
- [ ] 保存预测的漏检/误报与长度分层复盘（N6）；业务原因须有真实业务材料，特征重要性不能证明因果。
- **可选**：校准、拒识与人工复核比例；必须有独立的校准/选择角色及同等标签预算，不声称现成安全保证。

## 7. E06：模型互补与工程闭环

- **暂缓**：TCN/MOMENT概率融合与stacking；先核对共同错误是否有互补性。未来若做，权重选择只用声明的选择集，stacking只用train OOF，不能用test预测训练。
- [ ] 共同错误率诊断并入N6；oracle上界只作描述，不是可部署准确率。
- **暂缓**：蒸馏到轻量TCN；先有充分优化的独立TCN和教师净收益/成本证据，禁止使用test预测训练学生。
- [ ] 统一predict、输入边界与保存模型重载一致性（N4）。
- [ ] 演示曲线/类别/统计信息，输出脱敏；校准置信度和相似记录仅在实现与验证后展示。
- [ ] 同硬件端到端延迟、吞吐、模型/依赖大小、峰值显存/内存（N4），分类头缓存计时单列。
- [ ] README失效链接、旧TODO状态和轻量CI/报告生成收尾；pi-wake已停用，不作为运行依赖。

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
- [x] 当时提出的归因控制已在后续E01C完成：匹配核带宽/搜索范围的统计基线、同架构随机表征+统计融合。预训练贡献通过，两预算都超过强统计的整体主张未通过；原门槛不修改。
- [x] 当时提出的E02池化与E03 LoRA+统计支路pilot已完成（E02P未过门槛、E03F强头pilot通过）。这些是40%validation探索，不能据此宣称超过完整标签TCN或历史完全微调test；下一步以本文件开头N1–N6为准。

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
- [x] 2026-10-18 11:10:45全部12NN和缺失SVM控制完成；11:12再次执行verify/summarize，208冻结输入/依赖、42,120行NN预测、参照、2,400个epoch的固定scale1024/无跳步、历史最优/端点和模型哈希通过。

复制成功只按四个新seed：融合相对“本轮统计NN、仅MOMENT NN、父统计RF/SVM、匹配统计SVM、父融合SVM”的逐seed最强参照，平均至少 **+0.5pp**，至少 **3/4** 严格正向；相对仅MOMENT平均至少 **+0.25pp**，至少 **3/4** 正向。两项都过才记复制成功。pilot43不加入平均，区间仅描述，不自动test、LoRA或扩预算。

- 入口：`scripts/run_e03e_remote.sh`；套件：`e03e_gain_replication_20261018_v1`。
- 状态/汇总：`artifacts/improvements/e03e_gain_replication/e03e_gain_replication_20261018_v1_analysis/`。
- 日志：`artifacts/logs/e03e_gain_replication_20261018_v1.log`。
- 本轮需要未适配骨干前向提取，其后训练廉价缓存头；不是完全不执行Transformer，也不是LoRA训练。

### 无人值守授权、pi-wake 与空闲关机（2026-10-18）

用户在2026-10-18曾授权用pi-wake自主完成目标，只有重要需干预的决定才暂停，AutoDL空闲即shutdown。以下为历史操作记录；2026-10-19用户已明确卸载并停用pi-wake，工作区残留已清理，不再创建闹钟或接管会话。历史操作台账：`docs/autonomous_runbook_20261018.md`。

- [x] 已核验pi-wake0.2.5 daemon/presence。原同owner外部API timer到期仍active：存活session从零闹钟启动时没有scheduler tick，daemon会避开live owner。该timer已暂停保留，不能伪称已递送。
- [x] 通过Pi公开SessionManager创建独立持久无人值守接管会话（parent指向原会话，复制用户授权并附台账）；pi-wake timer `98tcn-autonomy-handoff-20261018-1` 指向此offline owner，于11:08:18触发。原交互会话交接后不同时写项目；无两个LLM并发写同session。owner及任务边界见`.pi/autonomy-handoff.json`与runbook。
- [x] 启动后台保全 `scripts/backup_autodl_preservation.sh`，先备份静态artifact（暂排除运行中E03E），再dereference复制公共基座并校验a569...指纹。补齐旧E01完整模型、FP32缓存与公共基座，不能只备份新头。
- [x] 唤醒后确认E03E全部完成并复跑审计；只按四seed汇总，pilot43未计入，门槛不变，test仍锁定。
- [x] 最终差量及remote/local SHA核验完成，2,727文件、4,714,447,031 bytes全部一致。再次确认无训练/提取/SVM/kernel或其他作业，GPU compute为空；11:19:06执行**远程AutoDL** `sync; /usr/bin/shutdown`返回0，11:19:21一次SSH复验Connection refused。本机未关机，未用GPU短时0%判空闲。
- [x] 已保全并在空闲时远程关机；未启动无限探索。下一步仅询问是否冻结最终方法/预算/比较清单后明确批准一次正式test，当前仍锁定test，等待这一重要决定。

### 接管巡检 2026-10-18 11:08

- [x] 专用owner收到真实pi-wake递送；E03E已到seed46_moment，协调器/训练仍活动，暂不关机。
- [x] 初始静态备份 `static_backup_status.json=copied`，公共基座dereference和大小/SHA检查通过；最终E03E差量及remote/local全清单仍待完成。
- [x] 3分钟timer于11:11:47实际递送，确认套件完成，转入最终保全；未启动新探索。

### E03E 复制结果与最终保全（2026-10-18 11:14）

- [x] 四seed融合validation Macro-F1 **92.04% ± 0.53%**（样本SD），Accuracy **91.90% ± 0.54%**；相对逐seed最强参照平均 **+0.656pp**，4/4正向，描述性95%配对区间 **[+0.213,+1.099]pp**；相对仅MOMENT平均 **+7.387pp**，4/4正向，**两项冻结规则均通过**。
- [x] 200轮固定端点相对最强参照仅 **+0.359pp**、3/4正向，不替代预算内最好模型的主判定；重复validation和重叠划分偏差保留，不能与历史TCN test直接排名。
- [x] 输出 `docs/experiment_records/e03e_gain_replication_results_20261018.md`、`docs/figures/e03e_gain_replication_20261018.svg`、`artifacts/analysis/e03e_replication_20261018/`；含逐运行Accuracy/Macro-F1、类别、电池AP/Recall/FPR、耗时和配对结果。
- [x] 最终差量/源码配置数据快照与公共基座全部保全；11:15:47完成2,727文件SHA逐项核验，零失败。首次manifest的PATH技术错误及attempt1日志保留，改用绝对`.venv/bin/python`重试成功；未改科学代码或结果。11:18timer递送后完成空闲检查和远程关机。
- [x] 报告12组电池AP与远程sklearn交叉检查，最大差1.85e-13个百分点；合成AP/SVG解析检查通过。seed44/46末10轮最好点的收敛边界已披露，不自动延长预算。

### 最终关机与交付（2026-10-18 11:19）

- [x] 证据：保全目录 `artifacts/imports/autodl_preservation_20261018/` 中的 `preservation_verified.json`、remote/local SHA清单、`shutdown_preflight.json`、`shutdown_command.json`、`shutdown_verification.json`。关机命令返回0，随后SSH Connection refused；没有独立平台电源/计费API读数，按证据边界记录。
- [x] 最终报告/曲线/派生结果、辅助脚本、runbook/todo另存`completion_bundle`并列SHA；没有commit、回滚用户改动、自动test或自动扩展。所有本轮timer已触发，不再续订。
- [x] **待用户重要决定（2026-10-19已确认）**：用户选择“1：冻结最终方案与40%预算对照，执行一次正式test”。旧validation阶段不回改，新授权仅在独立E03T生效，见下节；不授权按test调参或自动追加实验。

### 新服务器恢复与环境配置（2026-10-19）

- [x] 用户更换服务器但SSH别名仍为autodl；新主机 `autodl-container-f0c94cb49a-92794602`，RTX4080SUPER（驱动显示32760MiB）、SM8.9、16核CPU配额、62GiB内存；已挂载约50GiB数据盘。
- [x] 原保全目录只读保留；恢复原路径 `/root/autodl-tmp/ElecClassification`，**2727文件 / 4,714,447,031 bytes全部SHA一致**，含旧E01全模型/缓存、失败现场、E03E和实体公共checkpoint。历史环境/指标/冻结协议不改写。
- [x] 单独uv venv安装 **Python3.11.17、Torch2.12.1+cu126、MOMENT0.1.4、NumPy1.25.2**，pyproject/uv.lock不改；预装Python3.12/cu130保留。Python默认下载入口慢，改官方GitHub镜像下载同版本成功，首次日志保留。
- [x] HF缓存按原revision重建，公共权重大小1385501256/SHA a569...一致，补取同revision config；offline模式真实模型加载通过。
- [x] 新卡 **106 tests、6 subtests** 全过；合成FP32/AMP前向、48个q/v adapter梯度、全部冻结权重不变、零适配器/重载误差0、CUDA float64距离对CPU误差约5.68e-14通过。只用合成输入，没有业务数据训练或test评价。
- [x] 只读检查E01B/E01C/E03P/E03D-v1/E03D-v2/E03E六套冻结输入/依赖全部通过；未调用summarize覆盖历史recorded_at。
- [x] 归档完整性通过后叠加当前新文稿/脚本及todo2，不同步本地 `.pi/`、`.git/`、venv或敏感env文件。只增加新部署记录，不修改旧部署provenance。
- [x] 入口 `source scripts/autodl_env.sh`；环境、归档核验、pytest和合成检查证据在 `artifacts/deployments/autodl_20261019/`，本地保留副本。说明 `docs/experiment_records/server_redeployment_20261019.md`。
- [x] 未启动新实验、重算旧结果、解锁test或执行关机。新GPU耗时/浮点路径不能直接与旧V100合并，未来新实验/缓存需独立环境标识；RTX FP64核性能也不保证优于V100。

### E03T：冻结方案后的正式test（2026-10-19）

- [x] 用户明确选择选项1。固定 **seed42/44/45/46、40%标签、9827 train、3510 validation、7020 test**；不补pilot43，不拿历史全标签TCN/完全微调test混做同预算排名。
- [x] **7方法×4seed=28项推理，0重训/搜索**：E03E增益融合/统计/仅MOMENT NN，父融合SVM，匹配统计SVM，原统计SVM/RF。NN只用已有validation选定model_best，SVM/RF只用已有classifier.joblib，不合并train+val或换端点。
- [x] 新协议/模块/测试/启动器独立，**116 tests、13 subtests**，Ruff及shell检查通过。新GPU上四seed先全部通过validation-only恢复/256行新特征probe关卡，再允许任何test编码；容差预先固定，失败停止，不自动放宽。
- [x] **2026-10-19 10:22:27 Asia/Shanghai**在任何test结果前冻结 **394文件、28模型**及包/新硬件；协议SHA **48509b15a6360f1bd713274a9ba774e5612dcbd3e9e7a064bb3ee4e9472f146d**。融合best epoch固定为42→181、44→200、45→173、46→191。
- [x] 启动协调PID **6231**，入口`scripts/run_e03t_remote.sh`，套件`e03t_final_test_20261019_v1`；启动冻结清单本地副本`artifacts/analysis/e03t_startup_20261019/`。
- [x] **10:25:14**四seed技术关卡全部通过（所有检查预测一致，完整旧AMP缓存NN Macro-F1恢复差0），随后才生成test特征；**10:31:30**全部28项完成。196560行预测、原CSV真值/顺序、全部嵌套指标和模型/cache哈希通过；独立stdlib混淆指标最大误差1.11e-16，16项原生概率AP独立重算最大差2.22e-15。
- [x] 固定test确认规则全部通过：融合NN−**预先固定父融合SVM**平均 **+0.5636pp、4/4正向**，描述性95%配对区间[+0.5185,+0.6086]pp；融合NN−仅MOMENT **+7.0551pp、4/4正向**。完整报告其余固定配对差，不按test换参照/seed/阈值。
- [x] 协议`docs/experiment_records/e03t_final_test_protocol_20261019.md`；机器协议`configs/experiments/improvements/e03t_final_test_protocol.json`。输出`artifacts/improvements/e03t_final_test/e03t_final_test_20261019_v1_analysis/`；日志`artifacts/logs/e03t_final_test_20261019_v1.log`。
- [x] 原冻结阶段和V100结果保持不变。已有历史研究用过此数据test，不能称全项目首次触碰的全新数据；同数据重叠split/validation复用限制保留。完成后无论正负均停止，不自动LoRA/追加预算/关机。

### E03T 结果与保全（2026-10-19 10:33）

- [x] 正式test融合NN **Macro-F1 91.5160%±0.3758%，Accuracy 91.3711%±0.4011%**。父融合SVM 90.9524%±0.3516%，匹配统计SVM 90.1335%±0.3788%，统计NN 88.7578%，仅MOMENT 84.4608%，原统计SVM 89.6133%，原统计RF 87.8081%。四seed全部保留，预先固定两条门槛通过，不追加比较选择。
- [x] 相对匹配带宽统计SVM净增 **+1.3825pp、4/4正向**，描述性区间[+0.3159,+2.4490]pp。联合系统有效，不是MOMENT单独优于统计；相对父融合SVM+0.5636pp为小幅改进。正常/充电器故障/电池异常F1均值90.5048/93.6516/90.3915%；电池默认Recall87.9580%、FPR3.8965%、AP96.4991%，无阈值搜索或安全保证。
- [x] freeze→全部完成约9分03秒；两种精度路径test编码计时共3.89分钟，不含CSV解析/关卡/全部审计，非线上延迟，不与V100训练计时混用。
- [x] **173文件、242,935,231 bytes（约232MiB）**远程/本地SHA全部通过，28/28模型和公共基座指纹与此前完整保全对应，零失败。备份`artifacts/imports/e03t_final_test_20261019/`，含test NPZ、全部CSV/metrics、冻结源码快照及独立审计。`preservation_verified.json`及manifestSHA a1617f...为证据。
- [x] 报告`docs/experiment_records/e03t_final_test_results_20261019.md`；生成器`scripts/report_e03t_results.py`，独立审计`scripts/audit_e03t_predictions.py`。原E03E validation成功、E03P/E01C失败及数值事故记录均不回改。
- [x] 本阶段test已消费，只报告这次固定评价，不按本次test继续调参/重选模型。未自动启动新实验、追加seed/预算/LoRA或关机；新机保持开机。后续若要研究外部泛化或公平完整标签TCN，需要独立方案和授权，不能把本轮直接写成历史完整标签排名。

### E05A：历史同预算TCN可比性补审（2026-10-19）

- [x] 用户继续后优先查找已有产物，发现20260728已完成40%TCN；不重复训练或test推理。独立协议明确**事后审计、原结果已知、无新确认性门槛**，E03T原7方法及通过结果不回改。
- [x] 历史压缩包SHA b8367081...匹配；12个TCN metrics/config/status与原tar逐字节一致。四seed42/44/45/46的数据SHA、类别、过滤/split参数、9827训练样本指纹全部相同；历史TCN与融合各42120行validation/test预测的显式ID顺序、真值、指标与AP通过核对。
- [x] test **历史TCN92.6136%±2.7260%，融合91.5160%±0.3758%**；融合−TCN平均 **−1.0976pp，仅1/4正向**，描述性配对区间[−5.6285,+3.4333]pp。validation差−0.9812pp。必须保留这个限制，不能只报相对SVM的优势或声称普遍胜过TCN。
- [x] 电池默认Recall融合87.958%高于TCN85.056%，但FPR融合3.897%高于TCN0.706%，AP融合96.499%低于TCN97.907%；不挑单项指标作安全优势。
- [x] 审计为同标签预算**历史管线**比较，非相同归一化/训练选择预算/硬件对照；旧TCN train ID数组及checkpoint/完整环境字节未随metadata包保全，明确只核对已有指纹与输出，不虚称权重重现。
- [x] `122 tests、19 subtests`全部通过，Ruff通过；独立指标最大差1.11e-16，历史AP2.55e-15。测试helper有一次C408纯lint修订，初始capture/旧文件及维护记录保留；科学代码与结果不变，不伪装成前瞻冻结。
- [x] 报告`docs/experiment_records/e05a_historical_tcn_audit_20261019.md`；总览`docs/experiment_records/final_evidence_synthesis_20261019.md`。数据/输入指纹`artifacts/analysis/e05a_historical_tcn_20261019_v1/`，模块`analyze_historical_tcn.py`只读取保存预测，无训练、模型推理或阈值选择。
- [x] 本地及远程镜像备份`artifacts/imports/e05a_historical_tcn_20261019/`：**57文件、8,930,672 bytes**全SHA通过，包含47个输入路径的内容/映射及原archive、逐行预测、metadata、事后capture与报告。来源/结果全部保留，不改此前E03T保全包。
- [ ] 后续重要方向：是否有可脱敏的设备/站点/会话时间元数据用于独立设备或时间外推？目前三字段不能猜分组。可以继续train/val探索，但不可继续依赖这份已消费test追分并称独立确认；没有自动启动下一训练或关机。

### 独立设备/时间验证的材料检查（2026-10-19）

- [x] 用户继续后检查data目录、字段字典及可能的metadata文件。唯一raw CSV仍为43,289,123 bytes/SHA5615...，仅`charging_station_id / InsertedColumn / charging_powers_str`；没有可核实设备/时间映射或新增采集批次。**当前被数据材料阻塞，不是已完成分组/时间实验。**
- [x] `charging_station_id`仍只按样本唯一ID处理，未猜组、未生成伪元数据、未新训练/新增test推理。
- [x] 准备说明`docs/external_validation_data_requirements.md`与仅含表头的模板`configs/data_templates/external_metadata.csv`；检查证据`artifacts/analysis/e05b_external_validation_readiness_20261019/readiness.json`。
- [ ] 等待真实样本→充电设备/站点/电池ID或会话时间映射，或者真正新采集的带业务标签批次。旧样本补元数据只能支持回溯性分组研究；旧模型若见过留出组须重训，不能将改split冒充全新外部test。

### E03F：强头起点的LoRA对照（2026-10-19，已完成）

- [x] 用户选择继续validation探索，而非本轮解锁test；先前E03T报告已存在并被看到，明确标为post-test探索，不称独立确认，不再评价test。
- [x] 固定seed43、40%标签，父E03D-v2 gain头best200 warm-start；同支路同一头参数/buffers。四条件frozen/lora8×fusion/moment，各追加30epoch，无早停、无rank/seed搜索，epoch0可选；报告best及末轮端点。
- [x] 固定所有条件在线encoder前向、物理64/有效128；head/adapter LR均1e-4、全新AdamW及两处理统一重置30轮warmup/cosine、AMP-FP16固定scale1024。最强非适配参照、继续冻结控制、rank匹配MOMENT三门槛分别+0.5/+0.5/+0.25pp，全部通过才建议另立复制，不自动扩展。
- [x] v1预检128tests/19subtests、强头batch64合成数值检查通过；218文件冻结后首个在线rank0控制触发旧训练器把backbone视作adapter的接口错误（int.sqrt），首个optimizer.step之前退出，0个已提交epoch。v1原协议/源码/状态/epoch0/日志保留，不改成成功。
- [x] 新独立兼容层`strong_head_lora_online.py`将rank0真实冻结encoder注册为online_frozen_encoder，仍逐批在线前向；adapter角色为空，head物理batch显式64。原train_e03与v1源文件不修改、不伪造梯度。
- [x] **132 tests、19 subtests**通过，新增在线包装逐位等价、参数/训练模式、GPU真实rank0训练微批64/64/2回归；两支路实际强头batch64的零适配/重载误差0、A/B梯度有限及冻结参数不变通过。
- [x] **E03F-v2于2026-10-19 10:57:22（Asia/Shanghai）冻结228文件**；协议SHA `ae918b1536461892a36bf7fe5a6e72f75462076c3dacfeb2b6e6d324600338bb`。协调器PID7972，套件`e03f_strong_head_lora_20261019_v2`。四条件新目录重启，不混用v1轨迹。
- [x] 首个frozen fusion epoch0迁移：预测0差异、Macro-F1漂移0；启动时前2轮每轮约36秒，scale1024/零跳步。此为历史启动快照；终态结果及保全见下方完成项。
- [x] v1失败目录及v2冻结协议/两版smoke初步保全属于历史启动记录；最终v2模型/预测/数据已完成备份与SHA核验，见下方终态记录。
- [x] 四条件/120epoch完成；原始审计及本地独立核验42120行初始/最佳/端点预测、模型SHA、同支路初值、固定scale1024/零跳步和epoch0选择规则通过。LoRA融合92.4751%，继续冻结91.7479%，+0.7272pp；对最强非适配亦+0.7272pp，对同rank仅MOMENT+3.8524pp，三条原门槛均通过。仅seed43 validation探索，不自动复制或test。
- [x] 本地只备份watcher（原PID5104）的启动/等待为历史运行记录；套件终态后已完成整个E03F根复制与逐SHA校验并退出。此脚本不是pi-wake；当前没有监控进程，不再设置唤醒任务。
- [x] 终态watcher备份已核对：109文件/274137701 bytes、SHA零不匹配；修复watcher将`st_size`属性误当函数后重试成功，失败日志保留。报告`docs/experiment_records/e03f_strong_head_lora_results_20261019.md`；v1技术失败和v2全部结果均保留，没有test/复制/关机。
- 入口：`scripts/run_e03f_online_remote.sh`；协议/修订说明：`docs/experiment_records/e03f_strong_head_lora_protocol_20261019.md`、`e03f_online_interface_revision_20261019.md`。
- 运行根：`artifacts/improvements/e03f_strong_head_lora/`；日志：`artifacts/logs/e03f_strong_head_lora_20261019_v2.log`。

### E02P：继续现有数据可做的池化诊断（2026-10-19）

- [x] 用户明确“只有这个数据，继续todo2未完成项目”；调整优先级，设备/时间研究仍被材料阻塞，但不再阻塞E02、解释性/鲁棒性/工程等任务。当前按单阶段推进，不自动连续扩展。
- [x] 独立`pooling.py`/`run_pooling_pilot.py`、机器/文字协议和启动器；FP32提取mean1024、mean+max2048、thirds3072，同原15统计旁路。有效patch仍`.all()`，三段按有效patch时间排名分段，空段补零，不改编码器/注意力。
- [x] 同train subset/5折/5C×3gamma的6个SVM条件；mean两条件须先精确复现旧best params/validation预测及75个fold分数，否则停止。候选要比mean融合≥0.5pp、比包含优化增益头的既有最强管线≥0.25pp才建议另立复制；同时报告历史TCN validation差。
- [x] **143 tests、19 subtests**及Ruff/shell检查通过；合成真实模型padding/短常量检查通过。长度分层1/3、2/3分位点只由选定训练子集计算，不据validation/test定阈值。
- [x] **2026-10-19 11:03:33 Asia/Shanghai**、新真实池化结果前冻结**240输入文件**；协议SHA **4728d78d3b90efde875b6f9c37cfabf6e4696bda0e03ce87828820bb5824541b**。nohup协调PID **8554**；套件`e02_pooling_pilot_20261019_v1`，本地启动快照`artifacts/analysis/e02_pooling_startup_20261019/`。
- [x] 6条件完整完成；独立复核21060行validation预测、450个CV fold分数、train-derived长度分层，最大指标误差1.11e-16；终态117文件/608811803 bytes逐SHA核验通过。候选thirds/fusion未过固定门槛，完整结果见`docs/experiment_records/e02_pooling_pilot_results_20261019.md`。
- [x] 入口`scripts/run_e02_pooling_remote.sh`；输出`artifacts/improvements/e02_pooling_pilot/e02_pooling_pilot_20261019_v1_analysis/`；日志`artifacts/logs/e02_pooling_pilot_20261019_v1.log`；协议`docs/experiment_records/e02_pooling_pilot_protocol_20261019.md`。
- [x] 这是已知历史test之后的train/validation探索，不声称独立确认；不碰test、不训练encoder，不自动复制/LoRA/扩预算/关机。旧E03T/E05A输出和结论不回改。

### TCN legacy结构与训练长度诊断（E05B-TCN独立目录，2026-10-19）

- [x] 用户再次要求继续，新增受控真实训练pilot；与旧`e05b_external_validation_readiness`材料检查分目录保留，不覆盖旧检查。固定seed43、10%/100%标签各四条件：当前early7/当前fixed50/legacy masked50/legacy unmasked50，共最多400epoch。
- [x] 新入口只构造train/validation，统一raw/padding1024、AdamW/LR0.001/WD0.0001、scheduler F1、FP32且关闭TF32；legacy为六层128通道因果卷积。原Keras初始化/Adam/val_loss调度差异明确披露，不声称逐位复现或已穷尽性能。
- [x] 5项单测覆盖因果性、masked padding、配对初始化、heldout行不解析及checkpoint恢复dropout/optimizer/打乱状态；远端通过后冻结并于2026-10-19 13:29启动，PID13790。8小时硬预算，不自动扩seed/test/关机。
- [x] 八条件共355epoch完成；共享轨迹前缀、ID/真值、模型/预测SHA、更新数及84240行选择/峰值/端点validation预测审计通过。终态100文件/52906572 bytes已保全，本轮清理前重新逐文件SHA核验通过。
- [x] 10%标签early7/fixed50/legacy masked/unmasked选择F1为64.9645/64.9645/84.8182/93.6000%；100%为96.7023/96.9684/98.1412/98.1048%。单seed重复validation探索，不升级为test或总体优越性证明。结果：`docs/experiment_records/e05b_tcn_legacy_pilot_results_20261019.md`。
- [x] 终态观察器PID4028于13:56完成保全/同步并退出，`watcher_status.json=completed_verified`；2026-10-19用户明确计算结束并卸载pi-wake，已核对无训练/监控进程，清理插件会话残留、缓存及逐SHA确认的重复中间备份。冻结源码、模型、预测和历史运行记录保留，不再设置唤醒任务。
- 协议：`docs/experiment_records/e05b_tcn_legacy_pilot_protocol_20261019.md`。入口：`scripts/run_e05b_remote.sh`。独立根：`artifacts/improvements/e05b_tcn_legacy_pilot/e05b_tcn_legacy_pilot_20261019_v1/`。

### E05C / N1：TCN四seed复核与长度诊断（2026-10-19，已完成）

- [x] 用户要求继续后执行N1；独立机器/文字协议与`replicate_tcn.py`复用未修改的E05B训练实现。seed42/44/45/46，10%/40%/100%预算，current_fixed50/legacy masked/unmasked共36次训练，固定50轮/最多1800epoch，8小时硬预算。
- [x] 8项新增单测及5项E05B回归通过，覆盖范围/seed43排除、训练配置不变、split漂移/重叠拒绝、预测ID/概率、固定三类指标与电池索引、early7回放及train长度分层。没有先试新真实seed效果。
- [x] 92个输入在新诊断/训练前冻结；启动器PID16358，套件`e05c_tcn_replication_20261019_v1`。先执行seed43两预算×三模型×三padding的固定权重诊断，再按seed顺序训练；没有pi-wake、额外观察器、自动下一阶段或test入口。
- [x] seed43两个预算的长度分布、4个浅树控制与18组padding诊断完成；1024checkpoint恢复argmax零差异，原selected模型/ID核验通过。本地保存完整诊断输出并独立核验77220行预测、文件SHA、混淆指标及改变数，最大F1误差1.11e-16；报告`docs/experiment_records/e05c_length_padding_diagnostics_20261019.md`。
- [x] 仅length浅树10%/100% F1为43.2441/36.3936%，同预算15统计浅树75.5304/76.4047%。masked在1024/1280/1536输出不变；unmasked改到1536时预测分别改变100/33条。说明padding依赖，不能据此断言长度是唯一机制、泄漏或因果关系；本轮不加入length旁路训练。
- [x] 15:42:45（Asia/Shanghai）36条件完成并审计1800epoch/379080行预测；本地标准库独立核对原CSV真值、原split有序ID、概率/混淆/类别/电池指标、AP、选模与汇总通过，最大指标误差2.22e-16、AP误差2.22e-15。
- [x] 四seed10%/40%/100% F1均值±SD：当前69.2305±1.6101/92.2159±2.4400/96.6367±0.5722%；legacy masked93.6758±1.9260/97.7790±0.2277/98.3959±0.1729%；unmasked93.8161±1.0299/97.7648±0.3034/98.2154±0.1131%。masked相对当前各预算均4/4正向；unmasked相对masked无稳定优势，下一步建议masked主参照。
- [x] 完整终态521文件/248784612 bytes逐SHA通过，原始冻结与此前本地副本逐字节相同。备份时SSH实例变为`autodl-container-de774f9d7d-2319c4b0`/RTX4080，原冻结为RTX4080SUPER；首次hostname保护拒绝，显式核对并记录后再备份。迁移时刻不明，不认证全套计时为同一物理硬件；没有重推理/重训。
- [x] 报告/图：`docs/experiment_records/e05c_tcn_replication_results_20261019.md`、`docs/figures/e05c_tcn_replication_20261019.svg`；终态`artifacts/imports/e05c_tcn_replication_20261019/terminal_artifacts/`，独立审计及实例观察记录在其父目录。GPU检查无计算进程，未自动N2/LoRA/test/关机。
- 协议：`docs/experiment_records/e05c_tcn_replication_protocol_20261019.md`；入口：`scripts/run_e05c_remote.sh`。
- 状态：`artifacts/improvements/e05c_tcn_replication/e05c_tcn_replication_20261019_v1/suite_status.json`；日志：`artifacts/logs/e05c_tcn_replication_20261019_v1.log`；本地冻结副本：`artifacts/imports/e05c_tcn_replication_20261019/startup/`。

### TCN执行加速修订（2026-10-19）

- [x] 用户要求优化TCN计算速度；新增独立`ResidentLoader`与async epoch实现，只改数据搬运/同步，不启用AMP/TF32/compile、不增batch、不改模型/优化器/预算/选模，旧冻结源码不改。
- [x] 3结构×合成4099行×3epoch，权重、AdamW状态、每epoch train/validation指标、概率、shuffle/CPU/CUDA RNG共7项逐位一致；热身后速度比约1.17/1.29/1.31倍。5项新增CPU/CUDA测试通过，包含旧checkpoint续训一致及非有限fail-stop。
- [x] 用已完成seed42/40% checkpoint各回放1轮9827行train，首259训练行仅作诊断指标；不使用validation/test。三结构7项全部一致，原checkpoint SHA不变、回放权重丢弃；单次速度比1.292/1.335/1.307倍，非全套/线上倍率保证。
- [x] 合成/回放测速时短暂停原进程并finally恢复，3次暂停记录已保留；相关epoch_seconds含操作等待，不用作纯计算测速。切换前SIGINT确认原PID16763退出，保留中断现场与epoch恢复状态，不继续并发旧训练。
- [x] 独立执行修订冻结后，通过`python -m tcn_moment.replicate_tcn_fast run`恢复原36条件套件，新PID18768，原8小时墙钟预算不重置；已完成条件不重跑，未提交epoch按checkpoint重放。源/evidence SHA与逐case`execution_backend.json`均保留。
- [x] 首个加速完成条件seed44/10% current_fixed50已通过原审计器：50epoch、训练ID/保存模型SHA、逐行预测概率/真值、selected/peak/endpoint规则均通过；加速执行修订10个冻结输入的本地快照SHA再次核验一致。
- [x] 原N1全套训练/审计和终态SHA保全完成；加速执行记录/等价证据均在归档。基准暂停计入少数旧epoch_seconds、整套约87.2分钟包含诊断/恢复/审计/写盘，不能当纯计算计时；另有备份实例/GPU变化，报告披露，未据此重算速度倍率。
- 说明：`docs/experiment_records/tcn_speed_revision_20261019.md`；证据：`artifacts/analysis/tcn_resident_speed_20261019/`；执行修订：原E05C根下`execution_revisions/resident_async_v1/`；新日志：`artifacts/logs/e05c_tcn_replication_20261019_v1.fast.log`。无pi-wake/额外观察器/test/自动下一阶段。

### 结果判定提醒

计划中所有未完成项都不是已验证结论。只有 `status.json=completed` 且完整性检查通过的结果才能汇总；“进程已启动”“模型已下载”“单 epoch 冒烟通过”都不能写成实验完成。
