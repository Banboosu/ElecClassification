# MOMENT 无监督表征与相似序列检索实验（2026-08-03）

## 1. 实验目的

评价冻结 MOMENT-1-large 表征能否在不训练分类器、不更新参数的情况下，将形态或语义相近的
充电功率序列组织到相邻区域。本实验不把“相似序列检索”简化成另一项监督分类训练，而是直接
对冻结特征执行精确余弦近邻搜索，并用类别一致性作为事后定量评价。

实验同时加入缺失查询鲁棒性：人工隐藏 40% 的有效完整 patch，检验同一个查询的近邻集合是否
保持稳定，以及检索到的序列是否仍具有较高类别一致性。

## 2. 数据角色与防泄漏协议

- 沿用 seed 42 的统一固定划分和数据 SHA-256。
- 训练集 24,569 条序列作为 gallery，测试集 7,020 条序列作为 query；两者样本 ID 完全不重叠。
- validation 3,510 条序列不参与特征变换、参数选择或结果计算。
- 继续沿用既有的分层固定划分，因此“创建划分”曾使用类别标签；但冻结表征提取、基线特征
  变换、标准化和近邻搜索均不读取标签。
- gallery/test 标签只在搜索结束后用于计算类别纯度，不参与近邻排序。
- 模型参数更新为 0；不训练分类头、SVM、度量学习器或投影头。
- 输入使用原始功率尺度（`normalize: none`），避免外部归一化统计量读取人工隐藏的真值。
- 遮挡查询传入 MOMENT 的输入值在隐藏位置置 0，同时 observation mask 标记为不可见；
  mask-aware pooling 只汇聚完整可见 patch。
- 原始曲线基线仅使用可见点做线性补齐，再进行单序列 z-score 和定长重采样；统计特征也只从
  可见点计算。测试已验证修改隐藏真值不会改变这两类基线特征。

## 3. 检索方法

| 方法 | 特征维度 | 标签/训练 | 说明 |
|---|---:|---|---|
| MOMENT | 1024（运行时核验） | 无 | 冻结 encoder，mask-aware patch 平均池化 |
| Raw Resampled | 128 | 无 | 可见点插值、单序列 z-score、归一化时间轴重采样 |
| Statistical | 18 | 无 | 长度、可见率、幅值分位数、端点、趋势、变化率等；StandardScaler 只拟合 gallery |

三种方法均执行 L2 归一化，并在完整训练 gallery 上进行精确余弦 Top-K 搜索，不使用近似索引。

## 4. 条件与指标

### 4.1 查询条件

- `clean`：完整测试查询。
- `random_patches / 40%`：随机隐藏完整 patch。
- `contiguous_block / 40%`：隐藏连续完整 patch 区块。
- 遮挡种子：42、43、44、45、46。

随机遮挡使用 `SHA-256(mask_seed, sample_id)` 派生每条序列的随机状态，且与零样本插补实验
使用相同的 patch 对齐遮挡函数。所有方法在相同条件中共享完全相同的 observation mask。

### 4.2 主要指标

- `Macro-Precision@10`：先计算每条查询 Top-10 中同类别比例，再对每个类别分别平均，最后对
  类别宏平均；这是主要语义检索指标，降低类别不均衡影响。
- `mAP@K` 与 `NDCG@K`：评价同类别序列是否更靠前。
- `Precision@1/5/10`：全部查询上的微平均类别一致性。
- `Clean-neighbor-overlap@K`：遮挡前后 Top-K gallery ID 的交集比例，作为主要稳定性指标。
- `Clean-query-feature-cosine`：同一序列在干净和遮挡输入下的表征余弦相似度。
- `Mean-length-relative-error@K`：检查模型是否主要按序列长度而非形态/语义检索。

类别标签只是可量化的代理相关性：同类别不保证曲线完全相似，不同类别也可能具有相近局部形态。
因此论文应将类别纯度与近邻曲线可视化结合解释，不能把它等同于人工相关性标注。

## 5. 输出与可复现性

- 正式输出：`artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1/`
- 日志：`artifacts/logs/moment_retrieval_zero_shot_v1_20260803.log`
- 主要文件：
  - `metrics.json`
  - `condition_metrics.csv`
  - `summary.csv`
  - `example_neighbors.csv`
  - `neighbors_clean.npz`
  - `neighbors_<pattern>_rate0p4_seed<seed>.npz`
  - `gallery_sample_ids.npy`、`query_sample_ids.npy`
  - `split_manifest.json`、`status.json`、配置与环境快照

每个 NPZ 保存三种方法的 Top-10 gallery 行索引和余弦分数；遮挡条件还保存 observation mask，
可依据样本 ID 数组完整复核检索结果。

## 6. 远程执行

```bash
tmux new-session -d -s moment_retrieval_20260803 \
  "cd /root/autodl-tmp/ElecClassification && bash scripts/run_retrieval_remote.sh"
```

查看状态：

```bash
tmux capture-pane -pt moment_retrieval_20260803:0.0 -S -120
cat artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1/status.json
```

## 7. 当前状态

- 相关语法检查和 12 个检索/配置/遮挡单元测试已通过。
- 远程命令入口已验证，启动前 V100 显存占用为 0 MiB、利用率为 0%。
- 正式运行 `moment_retrieval_zero_shot_thesis_v1` 已完成，33/33 个方法—条件结果均生成：
  3 个干净查询结果，加上 `2 × 5 × 3 = 30` 个遮挡查询结果。
- 开始时间：2026-08-03 17:39:14；结束时间：17:53:29（Asia/Shanghai）。
- 总 wall time：854.63 秒（14 分 15 秒）；冻结 MOMENT 特征提取累计 719.32 秒
  （11 分 59 秒），33 次精确 GPU Top-K 搜索累计仅 2.03 秒。
- PyTorch 峰值显存：1,947.59 MiB；总参数 341,243,395，可训练参数 0。
- AMP 在干净 gallery 的 batch start 18,688 检测到非有限特征，自动用 FP32 重算并让后续
  提取保持 FP32；所有最终特征和指标均有限，运行状态为 `completed`。
- 实际 patch 遮挡率为 40.0375%；10 个遮挡条件具有不同且已保存的 mask SHA-256。

## 8. 干净查询结果

下表均为百分数。Length Only 的误差为五个与标签无关的并列打破种子之样本标准差；其他干净
方法为单次确定性全量精确检索，不人为制造重复运行。

| 方法 | Macro-P@1 | Macro-P@5 | Macro-P@10 | mAP@10 | NDCG@10 | Top-10 长度相对误差 |
|---|---:|---:|---:|---:|---:|---:|
| MOMENT | **75.58** | 71.55 | 69.12 | 60.23 | 69.56 | 3.96 |
| Raw Resampled | 72.04 | 68.08 | 65.99 | 57.04 | 65.46 | 38.90 |
| Statistical | 75.57 | **72.34** | **70.08** | **61.26** | **70.93** | 12.17 |
| Length Only | 38.01 ± 0.31 | 38.53 ± 0.40 | 38.44 ± 0.28 | 23.26 ± 0.21 | 38.72 ± 0.24 | 0.01 |

随机机会水平为 Micro-Precision 33.71%、Macro-Precision 33.33%。MOMENT 的
Macro-Precision@10 为 69.12%，明显高于机会水平、长度基线和 Raw Resampled，但比
Statistical 低 0.95 个百分点。它的 Macro-P@1 与 Statistical 几乎相同，说明最邻近结果
很强，但扩大到 Top-5/10 后人工统计特征更稳定。

### 8.1 分类别 Precision@10

| 方法 | 类别 0 | 类别 1 | 类别 2 | Macro |
|---|---:|---:|---:|---:|
| MOMENT | 79.78 | 76.31 | 51.28 | 69.12 |
| Raw Resampled | **84.96** | **80.77** | 32.24 | 65.99 |
| Statistical | 73.17 | 73.03 | **64.03** | **70.08** |
| Length Only | 43.76 ± 0.26 | 33.24 ± 0.50 | 38.30 ± 0.28 | 38.44 ± 0.28 |

MOMENT 相比 Raw Resampled 在类别 2 上提高 19.04 个百分点，但在类别 0 和 1 上分别低
5.18 和 4.46 个百分点。因此其宏平均提升来自更均衡的类别 2 表征，而不是所有类别一致领先。
Statistical 在类别 2 上进一步取得 64.03%，使其成为总体最优方法。

### 8.2 配对不确定性分析

| 条件 | 配对差值 | Macro-P@10 差值（百分点） | 95% CI（百分点） |
|---|---|---:|---:|
| Clean | MOMENT − Raw | +3.14 | [2.53, 3.75] |
| Clean | MOMENT − Statistical | −0.95 | [−1.67, −0.26] |
| Random 40% | MOMENT − Raw | −1.11 | [−1.39, −0.82] |
| Random 40% | MOMENT − Statistical | −3.77 | [−4.10, −3.45] |
| Contiguous 40% | MOMENT − Raw | +0.18 | [−0.49, 0.86] |
| Contiguous 40% | MOMENT − Statistical | −3.93 | [−4.33, −3.52] |

Clean 使用 7,020 条查询的 2,000 次类别分层 bootstrap；遮挡条件使用五个相同 mask seed 的
配对 t 区间。干净查询下 MOMENT 相对 Raw 的小幅优势具有稳定性，但 Statistical 的总体优势
也不能解释为抽样波动。连续遮挡下 MOMENT 与 Raw 的区间跨 0，没有证据区分二者。

## 9. 40% 遮挡查询结果

| 遮挡模式 | 方法 | Macro-P@10 | Clean Top-10 overlap | Clean-query feature cosine |
|---|---|---:|---:|---:|
| Random patch | MOMENT | 63.20 ± 0.28 | 24.36 ± 0.06 | **98.91 ± 0.00** |
| Random patch | Raw Resampled | 64.31 ± 0.12 | **65.28 ± 0.35** | 98.57 ± 0.02 |
| Random patch | Statistical | **66.97 ± 0.20** | 31.42 ± 0.34 | 90.11 ± 0.17 |
| Contiguous block | MOMENT | 51.13 ± 0.37 | 4.80 ± 0.09 | **96.58 ± 0.02** |
| Contiguous block | Raw Resampled | 50.94 ± 0.30 | **29.35 ± 0.32** | 92.78 ± 0.13 |
| Contiguous block | Statistical | **55.06 ± 0.32** | 6.64 ± 0.20 | 56.81 ± 0.19 |

MOMENT 在两类遮挡下都保持最高的“同查询干净—遮挡特征余弦”，表明全局嵌入方向具有较强
不变性；但这一性质没有转化为更稳定的 Top-10 身份或更高的类别纯度。随机遮挡时 Raw 保留
65.28% 的干净近邻，MOMENT 仅保留 24.36%；连续遮挡时分别为 29.35% 和 4.80%。因此论文
可以报告 MOMENT 的**特征级遮挡不变性**，但不能称其具有更好的实际近邻检索鲁棒性。

## 10. 长度混杂补充分析

MOMENT 干净 Top-10 的平均长度相对误差只有 3.96%，明显低于 Statistical 的 12.17% 和
Raw 的 38.90%，说明序列长度是其相似度的重要组成部分。训练 gallery 中类别 0、1、2 的
平均长度分别为 265.08、375.63 和 351.86，长度本身确实携带类别信息。

为区分“语义结构”与“仅按长度检索”，追加了完全不使用标签的 Length Only 基线，并用五个
哈希种子随机打破相同长度候选的并列。该基线 Macro-P@10 为 38.44% ± 0.28%，高于 33.33%
机会水平，却远低于 MOMENT 的 69.12%。所以 MOMENT 明显编码长度，但长度不足以解释其大部分
类别纯度；预训练表征仍捕获了额外的幅值变化或曲线结构。

该补充属于实验完成后的混杂审计，输出单独保存在 `derived_analysis.json`、
`length_only_condition_metrics.csv`、`length_summary_by_class.csv` 和
`paired_comparisons.csv`，没有修改正式运行的原始邻居结果。

## 11. 图表与定性检查

- [语义纯度、分类别结果和遮挡稳定性主图](../figures/moment_retrieval_metrics_20260803.png)
- [三个类别的确定性 Top-3 近邻曲线样例](../figures/moment_retrieval_examples_20260803.png)

样例查询按类别和样本 ID 确定性选择，没有按结果好坏挑选。图中同类别近邻使用绿色、不同类别
使用红色。部分不同类别曲线在形态上仍非常相似，说明现有类别标签只是检索相关性的代理指标，
类别纯度不能完全替代业务人员或专门相似度标注。

## 12. 论文结论与表述边界

这组实验适合写入毕业论文，而且可以体现 MOMENT 在分类之外的有限优势：在完全不更新参数、
不训练下游检索器的情况下，冻结表征取得 69.12% 的 Macro-P@10，显著高于随机机会、仅长度
和直接原始曲线检索；相对 Raw 的优势为 3.14 个百分点，并明显改善最困难的类别 2。这说明
MOMENT 表征具有可直接用于相似序列搜索的结构，而不只是一组供监督分类器使用的输入特征。

结论必须同时保留两项限制：人工 Statistical 特征以 70.08% 略高于 MOMENT；在 40% 遮挡
查询下，MOMENT 虽保持最高的特征余弦一致性，却没有获得最佳语义纯度或近邻集合稳定性。
因此推荐定位为“强且无需任务训练的通用检索表征”，而不是“当前数据上最准确或最鲁棒的
检索方法”。

推荐正文表述：

> 在训练集作为 gallery、测试集作为 query 的无监督相似序列检索中，冻结 MOMENT 表征无需
> 任何参数更新即可取得 69.12% 的 Macro-Precision@10，高于原始曲线重采样的 65.99%、
> 仅长度基线的 38.44% 和 33.33% 的随机机会水平，但略低于人工统计特征的 70.08%。配对
> bootstrap 显示 MOMENT 相对原始曲线的提升为 3.14 个百分点，95% CI 为 [2.53, 3.75]。
> 这表明预训练表征能够在零训练条件下组织具有类别相关性的时序邻域，但其优势并非对所有
> 无监督特征工程方案成立。进一步的 40% 遮挡实验中，MOMENT 保持了最高的同查询特征余弦
> 一致性，却未保持最稳定的 Top-10 邻居集合，说明特征级不变性不必然转化为检索级鲁棒性。

## 13. 本地产物

- 完整导入目录：`artifacts/imports/moment_retrieval_zero_shot_thesis_v1/`。
- 原始结果、11 份邻居 NPZ、样本 ID 映射、配置、环境、代码差异、拆分清单和后处理审计文件
  均已下载。
- 分析脚本：`scripts/analyze_retrieval_results.py`。
- 绘图脚本：`scripts/plot_retrieval_results.py`。
