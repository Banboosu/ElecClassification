# 面向充电功率时序故障分类的专用网络与基础模型比较：MOMENT 的标签效率、迁移边界与安全关键评估

> 稿件状态：期刊论文初稿，2026-08-05。
>
> 使用说明：本文只把当前项目中已有五随机种子或完整零样本协议支持的结果写成事实。所有
> `【待补】` 项均需在投稿前完成或从正文中删除；对应实验编号见
> [论文证据与补实验台账](paper_experiment_backlog_20260805.md)。目标期刊确定后，还需按其模板
> 调整篇幅、图表和参考文献格式。

**作者：**【待补】  
**单位：**【待补】  
**通信作者：**【待补】  
**基金项目：**【待补/无】

## 中文摘要

充电功率曲线能够在不增加额外传感器的条件下反映充电过程状态，但内部真实运行场景中的故障
样本积累成本较高，且“正常—充电器故障—电池异常”分类同时受到变长输入、类别相关的序列长度
和安全关键误报—漏检权衡影响。为检验通用时序预训练能否在该场景中优于专用模型，本文在真实
内部运行资源中的 35,099 条有效充电功率序列上，对多数类、统计特征逻辑回归、随机森林、
1D-CNN、时序卷积网络（TCN）和
MOMENT-1-large 进行统一比较。每个随机种子采用 70%/10%/20% 分层训练、验证和测试划分，所有
模型共享样本清单；变长序列通过显式掩码排除填充值。MOMENT 分别采用线性探测、最后两层微调、
冻结表征 RBF-SVM 和完全微调，并在 1%、5%、10%、20% 和 40% 标签比例下与从头训练的 TCN
进行配对比较。完整标签条件下，TCN 获得 95.99% ± 0.73% Macro-F1，完全微调 MOMENT 为
95.37% ± 0.39%，但后者的训练时间、峰值显存和参数量约为前者的 31.5、58.5 和 1,560 倍。
相反，在 1%、5% 和 10% 标签下，冻结 MOMENT 表征配合 RBF-SVM 分别领先 TCN 6.43、10.86
和 12.55 个 Macro-F1 百分点，且随机种子间标准差更小；40% 标签时 TCN 反超 7.54 个百分点。
在固定架构、样本、预处理、池化和 RBF-SVM 的归因消融中，预训练表征相对随机初始化表征在
1%、5% 和 10% 标签下进一步领先 28.60、22.58 和 22.01 个百分点，三个配对 95% 区间均高于 0。
在无监督检索中，MOMENT 的 Macro-Precision@10 为 69.12%，高于原始曲线的 65.99%，但略低于
人工统计特征的 70.08%；在零样本插补的 8 个条件中，MOMENT 均未超过线性插值。电池异常专项
评估进一步表明，完整标签 TCN 具有更好的召回—误报权衡，少样本 MOMENT 的相对收益尚不足以
支持安全部署。结果说明，MOMENT 在该任务中的可复现优势不是完整标签精度或通用零样本能力，
而是低标签三分类中的标签效率与稳定性；标签充足且强调部署成本时，专用 TCN 更合适。

**关键词：** 充电功率时序；故障分类；时序基础模型；MOMENT；时序卷积网络；少样本学习；
安全关键评估

## English title

**Specialized Temporal Networks versus a Time-Series Foundation Model for Charging-Power Fault
Classification: Label Efficiency, Transfer Boundaries, and Safety-Critical Evaluation of MOMENT**

## Abstract

Charging-power curves provide a non-intrusive signal for characterizing charging states. In real
internal operations, fault samples are costly to accumulate, and the classification of normal,
charger-fault, and battery-abnormal sessions
is complicated by variable-length inputs, class-correlated sequence length, and the safety-critical
trade-off between false alarms and missed detections. This study evaluates whether generic time-series
pretraining offers a practical advantage over task-specific models on 35,099 valid charging-power
sequences from a real internal operational resource. Majority prediction, statistical-feature logistic
regression and random forest, a 1D-CNN,
a temporal convolutional network (TCN), and MOMENT-1-large were compared under paired stratified
70%/10%/20% train/validation/test splits. Padding was excluded through explicit masks. MOMENT was
evaluated with a linear probe, last-two-layer fine-tuning, frozen embeddings plus an RBF-SVM, and full
fine-tuning. Label-efficiency experiments used nested 1%, 5%, 10%, 20%, and 40% subsets. With all
labels, TCN achieved a Macro-F1 of 95.99% ± 0.73%, compared with 95.37% ± 0.39% for fully fine-tuned
MOMENT, while the latter required approximately 31.5 times the training time, 58.5 times the peak GPU
memory, and 1,560 times the parameters. With 1%, 5%, and 10% labels, however, frozen MOMENT
embeddings plus an RBF-SVM exceeded TCN by 6.43, 10.86, and 12.55 Macro-F1 percentage points and
showed lower variability across seeds. TCN regained a 7.54-point advantage at 40% labels. An
architecture-matched ablation further showed that pretrained embeddings exceeded randomly
initialized embeddings by 28.60, 22.58, and 22.01 points at the three low-label budgets; all paired
95% intervals were above zero. Frozen MOMENT embeddings also improved unsupervised
Macro-Precision@10 over raw resampled curves
(69.12% versus 65.99%) but remained below handcrafted statistical features (70.08%). MOMENT did not
outperform linear interpolation in any of eight zero-shot imputation conditions. Safety-critical
battery-abnormality evaluation favored fully supervised TCN in recall–false-positive trade-offs and
did not support deployment claims for few-label methods. The reproducible advantage of MOMENT in
this dataset is therefore label efficiency and stability in low-label multiclass classification, rather
than superior full-label accuracy or universal zero-shot transfer.

**Keywords:** charging-power time series; fault classification; time-series foundation model;
MOMENT; temporal convolutional network; label efficiency; safety-critical evaluation

## 1 引言

充电设备故障与电池异常会降低充电设施可用性，并可能带来维护和安全风险。相比增加专用传感器，
直接分析运行过程中已有的电压、电流或功率曲线具有非侵入和易部署的特点。已有研究已将自编码器、
传统机器学习和时空特征融合用于充电设施状态监测与故障识别[1-2]。然而，工业场景中故障样本的
获取和确认通常比正常样本困难，数据驱动方法既需要处理有限标注，也必须报告漏检和误报，而不能
只报告总体准确率。

TCN 通过扩张因果卷积和残差连接建模序列，具有结构清晰、并行计算方便等特点[3]。ROCKET 等
方法则表明，随机卷积特征配合简单分类器也可以成为准确且高效的时序分类基线[4]。近年来，
TS2Vec 等自监督表征方法[5]以及 MOMENT、TimesFM 和 Chronos 等时序基础模型[6-8]尝试利用跨数据集
预训练获得可迁移表示。其中，MOMENT 面向分类、插补、异常检测等多类任务，并强调有限监督下的
统一评估[6]。这使其看似适合标注昂贵的充电故障识别，但预训练收益是否能够跨越领域差异、是否
优于强专用模型，以及这种收益需要多少计算代价，仍需在目标数据上验证。

现有应用研究常在单一训练规模下比较最终精度，容易混合三个不同问题：模型容量上限、标签效率
和工程成本。对于电池异常，还存在第四个问题：默认分类阈值下的 Macro-F1 并不等价于可接受的
漏检—误报权衡。因此，本文不以“基础模型替代专用模型”为先验结论，而围绕以下研究问题展开：

1. 完整标签条件下，统计模型、CNN、TCN 和 MOMENT 的效果—成本权衡如何？
2. MOMENT 的不同适配策略能否释放预训练表示，冻结表示是否具有非线性可分性？
3. 当仅使用 1%–40% 标签时，MOMENT 是否比从头训练的 TCN 更准确、更稳定？
4. MOMENT 表征能否零训练复用于检索和插补，优势边界在哪里？
5. 对电池异常这一安全关键类别，模型在给定召回目标下会产生多少误报？

本文当前版本的主要贡献如下。

1. 建立了面向变长充电功率序列的统一实验协议，固定样本清单、显式 padding mask、验证集模型
   选择和五随机种子配对比较，并把早期不可比结果排除在正式主表之外。
2. 系统比较 MOMENT 的线性探测、浅层微调、论文协议对齐的冻结表征 RBF-SVM 和完全微调，说明
   下游适配方式会实质改变对基础模型能力的判断。
3. 通过嵌套少标签子集证明，MOMENT 当前最明确的优势是 1%–10% 标签下的标签效率与稳定性，
   同时给出其在标签充足时被 TCN 反超的转折区间。
4. 将无监督检索、安全关键阈值评估和零样本插补纳入同一证据框架，既报告正面迁移结果，也保留
   强基线胜出和迁移失败的边界。

## 2 相关工作

### 2.1 充电设施与电池故障的数据驱动识别

充电过程的功率或电流曲线包含设备控制、车辆需求和电池状态共同作用下的动态信息。Sakwa 等[1]
使用真实充电功率曲线和自编码器开展充电设备早期异常检测；Duan 等[2]通过时域、频域和空间特征
融合识别充电桩故障。这类研究说明，运行曲线能够支持非侵入式状态监测，也表明人工特征和传统
机器学习仍是不可忽略的强基线。与其不同，本文关注单条变长功率曲线的三分类，并将低标签迁移、
基础模型适配、安全阈值和计算成本作为同等重要的评价维度。

### 2.2 专用时序分类模型

TCN 使用扩张卷积扩大感受野，并通过残差结构改善深层训练[3]。对于规模有限、标签相对充分的
单领域任务，TCN 的参数规模和推理路径通常比大规模 Transformer 更紧凑。另一方面，ROCKET
使用大量随机卷积核生成特征，再训练线性分类器，在广泛的时序分类基准上展示了较高精度和较低
训练成本[4]。当前实验已包含统计特征、1D-CNN 和 TCN，但尚未完成 ROCKET/MiniROCKET、
InceptionTime 等通用时序分类强基线；因此本文现阶段不使用“达到该领域最优”之类表述。

### 2.3 自监督表征与时序基础模型

TS2Vec 通过分层对比学习获得时间戳和子序列级通用表示[5]。MOMENT 则在大规模异构时序集合上
进行掩码重构预训练，提供面向多种下游任务的开放模型族[6]。TimesFM[7]和 Chronos[8]主要展示
时序基础模型在零样本预测中的能力。基础模型的价值不应只由一个完整标签任务的最终精度判断：
若预训练表示在低标签条件下达到更高性能，可能节省更昂贵的标注成本；反之，若需要全量微调且
资源开销显著增加，其工程收益可能很小。本文据此将“标签效率”“适配深度”“跨任务复用”和
“计算成本”分开评价。

## 3 数据与方法

### 3.1 任务定义与数据清洗

原始数据为真实内部运行资源，包含唯一记录 ID、类别标签和以逗号分隔的单变量充电功率序列。
项目当前采用的标签约定为：0 表示正常、1 表示充电器故障、2 表示电池异常、5 表示无效或不完整
记录。该映射已记录在
`docs/data_dictionary.md`，并得到历史代码的交叉支持：`legacy/ceshi.py` 明确写出 0/1/2 的
语义和诊断名称，旧 TCN 脚本直接读取 `InsertedColumn` 进行编码，没有重新定义类别。因此本文
直接沿用“正常、充电器故障、电池异常”三个现行业务标签。旧版 TCN 代码与模型已经投入内部
实际使用，本文在相同业务定义上比较其改进实现、传统基线与 MOMENT。

表 1 给出清洗统计。首先过滤标签 5，再过滤长度小于 18 的序列，最终保留 35,099 条三分类样本。
有效序列最大长度为 816；模型输入上限设置为 1,024，因此正式数据没有因上限而截断。

**表 1 数据清洗与类别分布**

| 项目 | 数量 |
|---|---:|
| 原始记录 | 46,535 |
| 原始标签 0 / 1 / 2 / 5 | 12,115 / 10,020 / 12,965 / 11,435 |
| 长度小于 18 的记录 | 10,614 |
| 有效三分类记录 | 35,099 |
| 有效标签 0 / 1 / 2 | 12,115 / 10,020 / 12,964 |
| 原始序列最短 / 最长长度 | 1 / 816 |

标签 5 中有 10,613 条序列短于 18 个点，中位长度为 1，且 9,101 条为恒定序列。按照内部既定
处理协议，将其与有效三分类数据分离。过滤后的有效数据未发现完全重复序列或冲突标签，仅有
4 条恒定序列。

### 3.2 数据划分与泄漏控制

实验使用随机种子 42、43、44、45 和 46。每个种子均按类别分层划分 70% 训练集、10% 验证集和
20% 测试集，对应 24,569、3,510 和 7,020 条样本；同一随机种子下所有模型共享完全相同的样本
ID 清单。三部分样本 ID 无交集，其并集等于全部有效记录。验证集用于早停、模型选择和安全阈值
选择，测试集仅用于最终评价。

### 3.3 变长序列、归一化与掩码

所有序列在自身有效区间完成预处理，再补零至 1,024 点，并保存同形状的布尔有效位掩码。TCN
和 CNN 的全局平均池化只汇聚有效时间点；MOMENT 将时间点掩码转为完整 patch 掩码，只汇聚
所有输入点均有效的 patch，从而避免 padding 污染表征。MOMENT 的 patch 长度和步长均为 8。

TCN 比较三种输入处理：保持原始尺度（none）、逐序列 z-score 和逐序列 min-max。统计特征模型
使用原始尺度；MOMENT 分类使用逐序列 z-score。少样本比较沿用各模型在完整实验中的既定协议：
TCN 使用原始尺度，MOMENT 使用 z-score。因此该比较回答的是“各自已确定协议下的实际标签效率”，
不是完全相同预处理下的单因素模型消融。

### 3.4 对照模型

统计特征包括序列长度、均值、标准差、最小值、最大值、10%/25%/50%/75%/90% 分位数、首尾差、
线性斜率、一阶差分均值、一阶差分标准差和最大绝对突变，共 15 维。基于这些特征训练逻辑回归和
300 棵树的随机森林[10]，并加入多数类和仅长度决策树作为机会水平与捷径审计。

1D-CNN 使用 32、64 和 128 个通道的三层卷积，卷积核大小为 5，dropout 为 0.3。TCN 使用
64、64、128 和 128 个通道的四个残差块；每个残差块包含两层因果卷积，卷积核大小为 3，扩张率
依次为 1、2、4 和 8，dropout 为 0.3。两者均使用掩码平均池化和线性三分类头。

### 3.5 MOMENT 适配策略

本文使用 341,243,395 参数的 MOMENT-1-large。其编码器以 FLAN-T5-large 为骨干，包含 24 层、
1,024 维隐藏表示和 16 个注意力头。比较以下四种适配策略：

1. **线性探测：**冻结 embedder 和 encoder，只训练 3,075 参数的线性分类头；
2. **最后两层微调：**解冻 24 层编码器中的最后 2 层及分类头；
3. **冻结表征 + RBF-SVM：**提取 1,024 维掩码池化表示，使用 RBF 核 SVM[9]；只在训练集中
   分层抽取最多 10,000 条样本，以五折交叉验证从
   \(C\in\{10^{-4},10^{-3},\ldots,10^4\}\) 中选值，\(\gamma\) 取 `scale`；
4. **完全微调：**更新全部参数，分类头学习率为 \(10^{-4}\)，backbone 学习率为 \(10^{-5}\)。

第三种方法与 MOMENT 原论文的冻结表示分类思路对齐[6]，但本文对变长序列增加了掩码池化，且
目标数据不是 UCR 基准，故应称为“下游协议适配”，而非对原论文结果的严格复现。

### 3.6 少标签与跨任务协议

少标签实验在每个随机种子的完整训练集内按类别分层抽取 1%、5%、10%、20% 和 40% 的嵌套子集，
实际样本数分别为 244、1,227、2,456、4,913 和 9,827；验证集和测试集保持完整。TCN 从头训练，
MOMENT 骨干保持冻结，每个种子只提取一次表征，再为各标签比例独立执行训练集内部的 SVM 交叉
验证。相同种子、相同比例的两种方法使用完全一致的样本 ID。

预训练归因消融只保留 1%、5% 和 10% 三个核心低标签比例。条件 A 使用预训练
MOMENT-1-large；条件 B 直接构造参数量相同的随机初始化 MOMENT-1-large，不加载 MOMENT 或
FLAN-T5 checkpoint。两条件均冻结 encoder，并使用相同样本 ID、逐序列 z-score、mask-aware
pooling、1,024 维表示和 RBF-SVM 搜索范围。随机 encoder 的初始化种子与对应数据划分 seed
一致。该对照只改变预训练权重，用于判断预训练相对同架构随机表征的贡献。

无监督检索以训练集 24,569 条序列作为 gallery、测试集 7,020 条作为 query，比较冻结 MOMENT
表示、128 维重采样原始曲线、18 维检索统计特征和仅长度特征；近邻排序不读取标签，标签只用于
事后计算类别一致性。零样本插补在测试集上设置随机 patch 和连续区块两种缺失模式，缺失率为
10%、25%、40% 和 60%，比较 MOMENT、线性插值、前向填充和可见值均值。

### 3.7 电池异常安全关键复评

将标签 2 视为正类，分别计算 Precision、Recall、假阴性率（FNR）、假阳性率（FPR）、F2 和
PR-AUC。每个模型先在验证集连续分数上选择最大 F2 阈值或达到 95%/98%/99% Recall 的阈值，
然后原样应用于测试集；测试标签不参与阈值选择。PR 曲线适合展示正类识别的排序质量和阈值权衡
[11]。除三分类模型复评外，还比较专用 one-vs-rest MOMENT-SVM、逻辑回归和随机森林。

### 3.8 评价指标与统计分析

三分类主指标为 Macro-F1：

\[
\mathrm{Macro\mbox{-}F1}=\frac{1}{K}\sum_{k=1}^{K}
\frac{2P_kR_k}{P_k+R_k},\quad K=3.
\]

同时报告 Accuracy、Balanced Accuracy 和 Weighted-F1。五随机种子结果报告均值 ± 样本标准差；
模型差值按相同种子配对，并给出未校正的双侧 95% 配对 t 区间。检索的干净查询比较对 7,020 条
query 执行 2,000 次类别分层 bootstrap。由于随机种子数仅为 5，且多个消融未进行多重比较校正，
正文优先解释效应量、区间和跨种子一致性，不单独依赖“显著/不显著”措辞。

### 3.9 实现与计算环境

实验使用 Python 3.11、PyTorch 2.12.1+cu126 和 MOMENT 0.1.4，在 NVIDIA Tesla V100 32 GB
服务器运行。TCN/CNN 的训练 batch size 为 32，最多 50 epochs，以验证集 Macro-F1 早停；TCN
正式实验使用 FP32。MOMENT 使用 AMP，并在检测到非有限表示时以 FP32 重算。所有正式运行保存
配置快照、环境信息、数据哈希、样本清单、状态文件和机器可读指标。

## 4 结果

### 4.1 完整标签三分类

**表 2 完整标签测试集结果（%，五随机种子均值 ± 标准差）**

| 模型或策略 | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 多数类 | 36.94 ± 0.00 | 33.33 ± 0.00 | 17.98 ± 0.00 |
| 逻辑回归（统计特征） | 74.13 ± 0.59 | 73.68 ± 0.59 | 73.75 ± 0.60 |
| 随机森林（统计特征） | 88.85 ± 0.23 | 88.90 ± 0.24 | 88.89 ± 0.23 |
| 1D-CNN（z-score） | 94.29 ± 1.27 | 94.45 ± 1.21 | 94.38 ± 1.27 |
| TCN（min-max） | 91.79 ± 1.37 | 92.18 ± 1.33 | 92.00 ± 1.37 |
| TCN（z-score） | 94.92 ± 1.44 | 95.06 ± 1.46 | 95.02 ± 1.43 |
| **TCN（原始尺度）** | **95.94 ± 0.74** | **96.09 ± 0.67** | **95.99 ± 0.73** |
| MOMENT 线性探测 | 77.02 ± 0.88 | — | 77.41 ± 0.89 |
| MOMENT 最后两层微调 | 81.08 ± 1.13 | — | 81.50 ± 1.06 |
| MOMENT 冻结表征 + RBF-SVM | 84.77 ± 1.04 | 85.26 ± 1.03 | 85.17 ± 1.03 |
| **MOMENT 完全微调** | **95.26 ± 0.41** | **95.52 ± 0.39** | **95.37 ± 0.39** |

随机森林已达到 88.89% Macro-F1，说明幅值、分位数、趋势和突变等人工统计量包含很强的类别
信息。原始尺度 TCN 相对随机森林提高 7.09 个百分点，配对 95% 区间为 [6.21, 7.98]，五个
种子均有提升。TCN 比 CNN 平均高 1.61 个百分点，但区间为 [−0.38, 3.61]，当前证据不足以将
其写成稳定优于 CNN。

完全微调 MOMENT 与 TCN 的 Macro-F1 相差 0.62 个百分点，相同种子的配对区间跨 0。结果说明
MOMENT 容量足以接近本任务的专用模型上限，但没有证据表明其在完整标签平均效果上优于 TCN。

### 4.2 输入归一化

**表 3 TCN 输入处理消融（%）**

| 输入处理 | 最佳验证 Macro-F1 | 测试 Macro-F1 |
|---|---:|---:|
| 原始尺度 | **96.08 ± 0.70** | **95.99 ± 0.73** |
| 逐序列 z-score | 95.29 ± 1.25 | 95.02 ± 1.43 |
| 逐序列 min-max | 92.38 ± 1.45 | 92.00 ± 1.37 |

原始尺度减 z-score 为 +0.97 个百分点，95% CI 为 [−0.55, 2.49]；z-score 减 min-max 为
+3.01 个百分点，95% CI 为 [0.86, 5.16]。逐序列归一化会移除样本间绝对功率幅值，因此结果
提示绝对功率幅值是现行业务任务中的有效判别信号。实验 E06 将通过训练集全局标准化进一步
量化保留幅值信息与逐序列归一化之间的影响。

### 4.3 MOMENT 适配深度

**表 4 MOMENT 适配策略消融**

| 策略 | 可训练参数 | Macro-F1 | 相对上一项变化 |
|---|---:|---:|---:|
| 冻结 backbone + 线性头 | 3,075 | 77.41 ± 0.89 | — |
| 解冻最后 2/24 层 | 25,697,283 | 81.50 ± 1.06 | +4.09 pp |
| 冻结表示 + RBF-SVM | backbone 0 | 85.17 ± 1.03 | +3.67 pp |
| 完全微调 | 341,243,395 | 95.37 ± 0.39 | +10.20 pp |

RBF-SVM 比线性探测高 7.75 个百分点，也比最后两层微调高 3.67 个百分点，说明冻结表征具有
明显的非线性可分结构。最后两层结果只代表本文测试的 2/24 层、学习率和训练协议，不能解释为
“MOMENT 部分微调的上限”。完全微调相对冻结 RBF-SVM 再提高 10.20 个百分点，表明目标域深层
适配对完整标签性能至关重要。

### 4.4 少标签分类：MOMENT 的核心优势

**表 5 少标签测试 Macro-F1（%，五随机种子均值 ± 标准差）**

| 标签比例（样本数） | TCN | MOMENT + RBF-SVM | MOMENT − TCN |
|---:|---:|---:|---:|
| 1%（244） | 61.62 ± 4.12 | **68.05 ± 1.59** | **+6.43 pp** |
| 5%（1,227） | 66.54 ± 2.54 | **77.40 ± 0.93** | **+10.86 pp** |
| 10%（2,456） | 68.22 ± 7.15 | **80.77 ± 0.59** | **+12.55 pp** |
| 20%（4,913） | 81.34 ± 8.80 | 83.21 ± 0.63 | +1.87 pp |
| 40%（9,827） | **92.57 ± 2.36** | 85.03 ± 0.58 | **−7.54 pp** |

![图 1 少标签条件下的 Macro-F1 学习曲线](figures/few_shot_macro_f1_20260728.png)

1%、5% 和 10% 标签下，MOMENT − TCN 的配对 95% 区间分别为 [0.22, 12.63]、
[8.31, 13.41] 和 [3.80, 21.29] 个百分点；20% 标签时区间为 [−9.53, 13.27]，40% 标签时为
[−10.59, −4.50]。MOMENT 在所有标签比例下的标准差为 0.58–1.59 个百分点，TCN 为
2.36–8.80 个百分点。

因此，MOMENT 的核心正面结论是低标签下的标签效率和跨种子稳定性，而不是完整标签性能。
冻结表示在约 85% Macro-F1 附近趋于饱和；标签增加后，TCN 能继续学习目标域判别特征，并在
40% 标签时接近其完整数据上限。MOMENT 与 TCN 的差异仍是各自既定协议下的管线比较；下一节
使用同架构、同分类器对照进一步隔离预训练权重的作用。

### 4.5 同架构预训练归因

**表 6 预训练与随机初始化冻结表征的低标签 Macro-F1（%，五随机种子）**

| 标签比例 | 预训练 MOMENT | 随机初始化 MOMENT | 配对差 | 95% 配对区间 |
|---:|---:|---:|---:|---:|
| 1% | **68.05 ± 1.59** | 39.44 ± 5.00 | **+28.60** | **[+22.24, +34.97]** |
| 5% | **77.40 ± 0.93** | 54.81 ± 0.95 | **+22.58** | **[+20.89, +24.28]** |
| 10% | **80.77 ± 0.59** | 58.76 ± 1.26 | **+22.01** | **[+20.48, +23.53]** |

三个比例的 15 个逐 seed 差值全部为正。两条件的数据哈希、split manifest、样本 ID、预处理、
池化、1,024 维特征、341,243,395 参数和 SVM 搜索范围均通过自动一致性检查；随机条件明确记录
未加载预训练 checkpoint。结果表明，在当前低标签任务和固定 RBF-SVM 协议下，预训练权重相对
同架构随机表征带来稳定的大幅收益。该消融支持“预训练提高当前任务的低标签分类性能”，但不
支持将 MOMENT 外推为所有数据集或所有通用时序表征中的最优方法。

### 4.6 电池异常的召回—误报权衡

**表 7 完整标签电池异常测试结果（%，阈值仅由验证集选择）**

| 运行点 | 模型 | Recall | Precision | FPR | F2 | 平均 FN / FP |
|---|---|---:|---:|---:|---:|---:|
| Argmax | TCN | **92.66 ± 1.41** | **98.99 ± 0.29** | **0.56 ± 0.16** | **93.86 ± 1.14** | 190.2 / 24.6 |
| Argmax | MOMENT 完全微调 | 92.12 ± 0.94 | 97.22 ± 0.17 | 1.54 ± 0.10 | 93.09 ± 0.76 | 204.4 / 68.2 |
| Max-F2 | TCN | **97.41 ± 0.56** | **95.95 ± 0.85** | **2.41 ± 0.52** | **97.11 ± 0.48** | 67.2 / 106.8 |
| Max-F2 | MOMENT 完全微调 | 95.56 ± 0.29 | 93.25 ± 1.91 | 4.07 ± 1.25 | 95.08 ± 0.46 | 115.2 / 180.4 |
| 99% Recall 目标 | TCN | **99.07 ± 0.18** | **78.45 ± 6.87** | **16.41 ± 6.73** | **94.01 ± 2.01** | 24.2 / 726.6 |
| 99% Recall 目标 | MOMENT 完全微调 | 98.97 ± 0.43 | 54.78 ± 4.55 | 48.42 ± 8.38 | 85.12 ± 1.88 | 26.6 / 2,143.6 |

TCN 的 Battery PR-AUC 为 99.14% ± 0.17%，完全微调 MOMENT 为 97.91% ± 0.42%，配对差的
95% 区间为 [0.65, 1.81] 个百分点。在约 99% Recall 时，两者平均漏检数接近，但 TCN 的 FPR
低 32.01 个百分点。完整标签条件下，TCN 的安全关键排序与阈值权衡均更好。

![图 2 完整标签电池异常的 Recall–FPR 权衡](figures/battery_safety_threshold_tradeoff_20260804.png)

少标签默认 argmax 下，MOMENT 在 1%–10% 标签时具有更高 Battery Recall，但也带来更高 FPR；
统一在验证集选择 95% Recall 后，MOMENT 在 1% 和 5% 标签时相对少样本 TCN 的 FPR 分别低
8.91 和 10.60 个百分点，但绝对 FPR 仍为 77.80% 和 65.62%。专用二分类 MOMENT-SVM 能进一步
改善三分类 MOMENT 的 PR-AUC 和 FPR，但 1%–10% 标签下随机森林更强。故安全结果不能作为
MOMENT 的主要优势，也不能支持上线报警。

![图 3 少标签电池异常评估](figures/battery_safety_few_shot_20260804.png)

![图 4 专用电池检测器比较](figures/battery_binary_comparison_20260804.png)

### 4.7 无监督相似序列检索

**表 8 干净查询的无监督检索结果（%）**

| 特征 | Macro-P@1 | Macro-P@10 | mAP@10 | Top-10 长度相对误差 |
|---|---:|---:|---:|---:|
| MOMENT | **75.58** | 69.12 | 60.23 | 3.96 |
| 原始曲线重采样 | 72.04 | 65.99 | 57.04 | 38.90 |
| 人工统计特征 | 75.57 | **70.08** | **61.26** | 12.17 |
| 仅长度 | 38.01 ± 0.31 | 38.44 ± 0.28 | 23.26 ± 0.21 | 0.01 |

MOMENT 相对原始曲线的 Macro-P@10 提高 3.14 个百分点，分层 bootstrap 95% CI 为
[2.53, 3.75]；相对统计特征低 0.95 个百分点，区间为 [−1.67, −0.26]。MOMENT 在不更新参数、
不训练检索器的前提下获得了具有类别相关性的近邻结构，但人工统计特征总体略好。

MOMENT 检索明显受序列长度影响，但仅长度基线远低于 MOMENT，说明长度不能解释其大部分检索
纯度。40% 遮挡下，MOMENT 保持最高的干净—遮挡查询特征余弦，却没有取得最高 Macro-P@10 或
最稳定的 Top-10 邻居集合。故该结果支持“可零训练复用的强表征”，不支持“最准确或最鲁棒的
检索方法”。

![图 5 无监督检索结果](figures/moment_retrieval_metrics_20260803.png)

### 4.8 零样本插补的负结果

**表 9 零样本插补 Macro-NRMSE（越低越好）**

| 缺失模式 | 比例 | Linear | Forward Fill | MOMENT | Visible Mean |
|---|---:|---:|---:|---:|---:|
| 随机 patch | 10% | **0.0973** | 0.1638 | 0.2101 | 0.9734 |
| 随机 patch | 25% | **0.1254** | 0.2175 | 0.3071 | 1.0016 |
| 随机 patch | 40% | **0.1496** | 0.2695 | 0.4287 | 1.0137 |
| 随机 patch | 60% | **0.1951** | 0.3674 | 0.6320 | 1.0296 |
| 连续区块 | 10% | **0.1231** | 0.2452 | 0.4916 | 0.9629 |
| 连续区块 | 25% | **0.2691** | 0.4987 | 0.9085 | 1.1198 |
| 连续区块 | 40% | **0.4487** | 0.7520 | 1.1424 | 1.2691 |
| 连续区块 | 60% | **0.7564** | 1.0974 | 1.3864 | 1.4458 |

![图 6 零样本插补结果](figures/moment_imputation_macro_nrmse_20260803.png)

线性插值在 8/8 个条件中最优。MOMENT 始终优于可见值均值，但没有超过线性插值或前向填充。
这说明“冻结表征有利于低标签分类”不等于“通用重构头可零样本迁移到充电曲线插补”。该负结果
界定了预训练能力的任务边界，也避免只选择对 MOMENT 有利的任务报告。

### 4.9 效果—资源权衡

**表 10 正式模型训练资源**

| 模型/协议 | 平均训练或拟合时间 | 峰值显存 | 参数量或训练规模 |
|---|---:|---:|---:|
| 1D-CNN | 136.3 ± 51.7 s | 63 MiB | 51,971 参数 |
| TCN（原始尺度） | 345.5 ± 79.4 s | 267 MiB | 218,691 参数 |
| MOMENT RBF-SVM | 15.62 ± 0.27 min | 1,881 MiB | 骨干冻结；SVM 最多 10,000 样本 |
| MOMENT 最后两层微调 | 约 0.67 h | 约 2,692 MiB | 25,697,283 可训练参数 |
| MOMENT 完全微调 | 3.02 ± 0.52 h | 15,613 MiB | 341,243,395 可训练参数 |

完全微调 MOMENT 相对 TCN 的平均训练时间、峰值显存和参数量约为 31.5、58.5 和 1,560 倍。
因此，在完整标签且只需完成当前三分类任务时，TCN 的性能—资源权衡明显更好。冻结 MOMENT 的
价值主要来自减少标签需求，而不是比 TCN 节省计算；少标签实验中，MOMENT 的 GPU 表征提取后还
需要 CPU SVM 网格搜索。

## 5 讨论

### 5.1 MOMENT 的优势究竟是什么

现有实验能够支持的最强结论是：MOMENT 在本数据集 1%–10% 标签三分类中具有更高 Macro-F1
和更低随机种子方差。这一结论在相同样本 ID、相同验证/测试集合和配对随机种子下成立，效应量为
6.43–12.55 个百分点。它符合跨数据集预训练应在有限监督条件下提供先验表示的预期[6]。

同架构消融进一步表明，在固定 RBF-SVM、样本、预处理和池化后，预训练表征在 1%–10% 标签下
比随机初始化表征高 22.01–28.60 个百分点，三个配对区间均完全高于 0。这排除了“3.41 亿参数
随机特征加 RBF-SVM 已足以解释结果”的替代解释，支持预训练权重提高当前任务的低标签分类性能。
不过 MOMENT 与 TCN 仍使用不同分类器、预处理和调参方式，且尚缺 MiniROCKET/TS2Vec 等强表示
基线。因此，对 TCN 的结论仍应表述为既定管线下的经验优势，不能扩展成 MOMENT 相对所有专用或
通用方法的普遍优越性。

无监督检索提供次级证据：MOMENT 不需要任务训练即可组织具有类别相关性的邻域，并显著超过原始
曲线和仅长度基线。但统计特征略优，遮挡后的邻居身份稳定性也较差，因此检索应被定位为可复用性
证据，而不是核心优越性证据。

### 5.2 为什么 TCN 在标签充足时更合适

完整标签下，TCN 与完全微调 MOMENT 的 Macro-F1 接近，但 TCN 的参数量和显存需求低两个至三个
数量级。原始尺度 TCN 优于逐序列归一化版本，说明绝对功率及其局部变化是重要线索；随机森林的
强结果以及电池二分类中 `diff_std`、`max_abs_diff` 的高重要性也与此一致。专用卷积网络可以直接
利用这些领域信号，而冻结 MOMENT 表征约在 85% Macro-F1 饱和；只有昂贵的全量微调才能释放
接近 TCN 的性能。

这并不意味着基础模型无价值，而是表明模型选择取决于瓶颈：若主要成本是人工标注，冻结 MOMENT
可用于低标签冷启动；若已有近万条以上可靠标签且部署资源有限，TCN 更合理；若需要快速构建跨任务
原型，可尝试复用 MOMENT 表征，但必须为每个任务设置强基线。

### 5.3 安全关键类别不能由总体指标替代

三分类 Macro-F1 接近并不保证电池异常的 Recall–FPR 曲线接近。完整标签 TCN 在 PR-AUC、最大
F2 和高召回运行点均优于完全微调 MOMENT。低标签 MOMENT 的默认 Recall 较高，却伴随很高 FPR；
专用二分类和阈值调整可以减少部分误报，但仍明显弱于相同标签下的随机森林或完整标签 TCN。

因此，本文将安全专项结果视为模型选择约束，而不是 MOMENT 优势。进一步部署应明确漏检/误报
成本，完成前缀早期检测、概率校准、线上回放和持续监控。目前结果用于比较候选方案，不能替代
既有生产验收流程。

### 5.4 迁移失败也是模型边界

MOMENT 的零样本插补落后于简单线性插值，表明目标曲线的局部平滑归纳偏置比通用重构先验更匹配。
该结果与低标签分类优势并不矛盾：分类使用 encoder 表征和经过训练的下游分类器，而插补直接复用
预训练重构头。未来可以测试目标域自监督遮挡适配，但适配后的插补必须与当前 zero-shot 结果分开
报告。

### 5.5 局限性

1. 当前统计比较基于五个随机种子，配对区间未进行多重比较校正。
2. 同架构消融隔离了预训练与随机权重，但 MOMENT—TCN 少标签比较仍使用各自既定最佳协议，
   预处理、分类器和调参预算并不完全相同。
3. 缺少 ROCKET/MiniROCKET、TS2Vec 和其他时序基础模型等强通用表征基线。
4. 尚未完成通用三分类错误样例、噪声/重采样/截断鲁棒性、概率校准和按序列长度分层分析。
5. CNN 五随机种子来自 AMP 配置，尚需用最终 FP32 协议低成本复跑以统一数值设置。
6. 尚未测量各模型在统一 batch 和硬件下的推理延迟、吞吐与能耗。
7. 检索以类别一致性代理业务相关性，尚缺少面向运维用途的相似度评价。

## 6 结论

本文在统一变长序列协议下比较了专用 TCN 与 MOMENT 时序基础模型。完整标签条件下，TCN 以
95.99% ± 0.73% Macro-F1 获得最高均值，并以远低于完全微调 MOMENT 的计算成本达到接近的分类
性能。MOMENT 的主要价值出现在低标签区间：冻结表征 + RBF-SVM 在 1%、5% 和 10% 标签下领先
从头训练 TCN 6.43–12.55 个百分点，并具有更低跨种子方差；随着标签增加，该优势消失，40% 标签
时 TCN 明显反超。同架构随机初始化对照进一步显示，预训练权重在三个低标签比例带来
22.01–28.60 个百分点的稳定收益。无监督检索显示 MOMENT 表征具有零训练复用价值，但人工统计特征仍略优；
零样本插补和电池安全评估进一步限定了其适用范围。

因此，本研究支持的模型选择原则是：低标签冷启动优先评估冻结 MOMENT 表征，标签充足且强调
效率时优先 TCN，安全报警和跨任务迁移必须独立优化与验证。后续补充强表示基线、误差与鲁棒性
分析、效率测量及线上场景回放，可进一步完善模型选型与部署结论。

## 数据与代码可用性

本文数据为真实内部运行资源，受内部数据管理与保密要求约束，不公开原始记录。投稿版本将依据
内部许可披露可公开的实验代码、配置、聚合指标、数据哈希和正式实验产物清单；任何披露内容均不
包含可反向识别内部设备、站点或业务主体的信息。

## 伦理声明

本文使用受内部管理制度约束的运行数据，论文仅报告匿名化后的聚合实验结果，不披露原始业务
记录。本文模型评估用于研究和方案比较，不构成安全认证或自动控制依据。

## 作者贡献

【待补：按 CRediT taxonomy 填写 Conceptualization、Methodology、Software、Validation、
Writing 等。】

## 利益冲突

【待补：作者声明不存在利益冲突/列出相关关系。】

## 致谢

【待补】

## 参考文献

[1] Sakwa M, Nespoli A, Matrone S, et al. Electric vehicle supply equipment monitoring and early
fault detection through autoencoders[J]. *Sustainable Energy, Grids and Networks*, 2024, 40:
101497. <https://doi.org/10.1016/j.segan.2024.101497>.

[2] Duan Y, Shu S, Zhao Y, et al. Machine learning-based spatiotemporal fusion method for
non-intrusive charging pile fault identification[J]. *Frontiers in Electronics*, 2024, 5: 1490939.
<https://doi.org/10.3389/felec.2024.1490939>.

[3] Bai S, Kolter J Z, Koltun V. An empirical evaluation of generic convolutional and recurrent
networks for sequence modeling[EB/OL]. arXiv:1803.01271, 2018.
<https://arxiv.org/abs/1803.01271>.

[4] Dempster A, Petitjean F, Webb G I. ROCKET: exceptionally fast and accurate time series
classification using random convolutional kernels[J]. *Data Mining and Knowledge Discovery*,
2020, 34: 1454-1495. <https://doi.org/10.1007/s10618-020-00701-z>.

[5] Yue Z, Wang Y, Duan J, et al. TS2Vec: Towards universal representation of time series[C]//
Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(8): 8980-8987.
<https://doi.org/10.1609/aaai.v36i8.20881>.

[6] Goswami M, Szafer K, Choudhry A, et al. MOMENT: A family of open time-series foundation
models[C]//*Proceedings of the 41st International Conference on Machine Learning*. PMLR, 2024,
235: 16115-16152. <https://proceedings.mlr.press/v235/goswami24a.html>.

[7] Das A, Kong W, Sen R, Zhou Y. A decoder-only foundation model for time-series forecasting[C]//
*Proceedings of the 41st International Conference on Machine Learning*. PMLR, 2024, 235:
10148-10167. <https://proceedings.mlr.press/v235/das24c.html>.

[8] Ansari A F, Stella L, Turkmen C, et al. Chronos: Learning the language of time series[J].
*Transactions on Machine Learning Research*, 2024.
<https://arxiv.org/abs/2403.07815>.

[9] Cortes C, Vapnik V. Support-vector networks[J]. *Machine Learning*, 1995, 20: 273-297.
<https://doi.org/10.1007/BF00994018>.

[10] Breiman L. Random forests[J]. *Machine Learning*, 2001, 45: 5-32.
<https://doi.org/10.1023/A:1010933404324>.

[11] Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when
evaluating binary classifiers on imbalanced datasets[J]. *PLOS ONE*, 2015, 10(3): e0118432.
<https://doi.org/10.1371/journal.pone.0118432>.

## 附录 A 当前结果的复现索引

- 综合实验依据：[experiment_report_20260804.md](experiment_report_20260804.md)
- 少标签实验：[experiment_records/few_shot_experiment_20260728.md](experiment_records/few_shot_experiment_20260728.md)
- 电池安全评估：[experiment_records/battery_safety_experiment_20260804.md](experiment_records/battery_safety_experiment_20260804.md)
- 电池专用二分类：[experiment_records/battery_binary_experiment_20260804.md](experiment_records/battery_binary_experiment_20260804.md)
- 零样本插补：[experiment_records/moment_imputation_20260803.md](experiment_records/moment_imputation_20260803.md)
- 无监督检索：[experiment_records/moment_retrieval_20260803.md](experiment_records/moment_retrieval_20260803.md)
- 数据质量：[experiment_records/data_quality_findings.md](experiment_records/data_quality_findings.md)
- 补实验台账：[paper_experiment_backlog_20260805.md](paper_experiment_backlog_20260805.md)
