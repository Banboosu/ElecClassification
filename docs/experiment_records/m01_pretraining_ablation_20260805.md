# M01 预训练归因简化消融（2026-08-05）

## 1. 实验目的与结论

本实验在冻结 encoder、RBF-SVM、样本 ID、预处理、掩码池化和参数搜索全部相同的条件下，比较
预训练 MOMENT-1-large 与同架构随机初始化模型，判断低标签收益是否至少部分来自预训练权重。

五随机种子结果一致支持预训练条件。预训练相对随机初始化在 1%、5% 和 10% 标签下的 Macro-F1
分别提高 **28.60、22.58 和 22.01 个百分点**；双侧 95% 配对 t 区间分别为
**[22.24, 34.97]、[20.89, 24.28] 和 [20.48, 23.53] 个百分点**，均完全高于 0。

因此，当前论文可以写“在同架构、同下游分类器和同样本的简化消融中，MOMENT 预训练权重提高了
1%–10% 标签条件下的分类性能”。该结论不等于 MOMENT 在所有数据集或相对所有强时序表征都占优。

## 2. 实验协议

- 条件 A：`AutonLab/MOMENT-1-large` 预训练权重，冻结 encoder，mask-aware pooling，RBF-SVM。
- 条件 B：直接构造相同 MOMENT-1-large 架构，不调用 `from_pretrained`，随机初始化后冻结
  encoder，使用完全相同的 pooling 和 RBF-SVM。
- 标签比例：1%、5%、10%；每个比例分别使用 244、1,227、2,456 条训练样本。
- 随机种子：42、43、44、45、46。
- SVM：RBF kernel，`gamma=scale`，训练集内五折交叉验证，
  `C ∈ {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000}`。
- validation 和 test 不参与 SVM 参数选择；两条件复用相同完整 validation/test。
- 统计：均值、样本标准差和相同 seed 的双侧 95% 配对 Student-t 区间。

随机条件不能只在 `from_pretrained` 中设置随机 backbone 标记，因为模型仓库 mixin 会在构造架构后
继续加载 checkpoint。正式实现直接调用 MOMENT 构造器，并在每个产物中记录
`model_initialization=random`、`pretrained_checkpoint_loaded=false` 和初始化 seed。

## 3. 正式结果

### 3.1 五种子汇总

| 标签比例 | 预训练 Macro-F1 | 随机初始化 Macro-F1 | 预训练 − 随机初始化 | 95% 配对区间 |
|---:|---:|---:|---:|---:|
| 1% | **68.05% ± 1.59%** | 39.44% ± 5.00% | **+28.60 pp** | **[+22.24, +34.97] pp** |
| 5% | **77.40% ± 0.93%** | 54.81% ± 0.95% | **+22.58 pp** | **[+20.89, +24.28] pp** |
| 10% | **80.77% ± 0.59%** | 58.76% ± 1.26% | **+22.01 pp** | **[+20.48, +23.53] pp** |

### 3.2 逐 seed Macro-F1 与配对差

| 标签比例 | seed | 预训练 | 随机初始化 | 配对差 |
|---:|---:|---:|---:|---:|
| 1% | 42 | 70.07% | 35.66% | +34.41 pp |
| 1% | 43 | 66.83% | 43.90% | +22.93 pp |
| 1% | 44 | 66.37% | 35.77% | +30.60 pp |
| 1% | 45 | 69.31% | 45.82% | +23.49 pp |
| 1% | 46 | 67.65% | 36.06% | +31.60 pp |
| 5% | 42 | 78.28% | 54.13% | +24.15 pp |
| 5% | 43 | 77.49% | 56.49% | +21.00 pp |
| 5% | 44 | 78.31% | 54.43% | +23.88 pp |
| 5% | 45 | 76.41% | 54.48% | +21.93 pp |
| 5% | 46 | 76.48% | 54.54% | +21.94 pp |
| 10% | 42 | 81.10% | 57.55% | +23.55 pp |
| 10% | 43 | 81.39% | 60.58% | +20.81 pp |
| 10% | 44 | 81.07% | 58.46% | +22.61 pp |
| 10% | 45 | 80.12% | 57.76% | +22.36 pp |
| 10% | 46 | 80.15% | 59.46% | +20.69 pp |

所有 15 个逐 seed 配对差均为正。

## 4. 完整性与可比性核验

自动分析对每个 seed、每个标签比例逐项检查，全部通过：

- 数据集 SHA-256、split manifest 和训练样本 ID 完全相同；
- 数据预处理、特征提取 batch、AMP 设置和 SVM 搜索范围相同；
- 两条件均为 341,243,395 参数、1,024 维池化表示、patch length/stride 8/8；
- 随机条件五个运行均明确记录未加载预训练 checkpoint；
- 五个随机运行状态均为 `completed`，无 CUDA、NaN/Inf 或协议自动变化。

正式环境为 Python 3.11、MOMENT 0.1.4、PyTorch 2.12.1+cu126、Tesla V100-PCIE-32GB。随机条件
单 seed 墙钟时间为 226.8–240.8 秒，峰值显存均为 1,881.1 MiB。

数据集 SHA-256 为
`5615a96a7894caed5d14463c77167af8098bdc1e1ebf32a33a89a12c3c5cf6e6`。远程基础 commit 为
`175193a996fc`，五个随机条件产物记录的工作树 diff SHA-256 均为
`327bceb9b89d09a6cca36f1a04a83861945f04b1b0a86068b6837393f823010f`；完整 diff 随运行环境快照留档。

## 5. 运行与产物

配置与入口：

```text
configs/experiments/pretraining_ablation/moment_svm_random.yaml
scripts/run_m01_pretraining_ablation_remote.sh
scripts/analyze_m01_pretraining_ablation.py
```

远程结果：

```text
artifacts/moment_svm_few_shot/moment_svm_thesis_few_shot_v1_seed{42..46}
artifacts/moment_svm_pretraining_ablation/moment_svm_random_m01_random_encoder_v1_seed{42..46}
artifacts/analysis/m01_pretraining_ablation/
```

本地结果包：

```text
artifacts/imports/m01_pretraining_ablation_20260805.tar.gz
SHA-256: 69bda9f7294e93a0b1468b187654c96374c817fbf62ed79da360be8c8cabdb9e
```

结果包共 135 个文件，包含两条件的配置、环境、状态、指标、split manifest、训练样本清单、
逐样本预测和最终分析，不包含模型权重或内部原始数据。

## 6. 对论文结论的影响

M01 将原来的“预训练 MOMENT + RBF-SVM 管线在低标签下优于 TCN”进一步拆解为两层证据：

1. 同架构、同 RBF-SVM 的对照表明，低标签性能相对随机 encoder 的大部分差异与预训练权重有关；
2. MOMENT 与 TCN 的比较仍是各自既定协议下的模型选择实验，不能据此声称 MOMENT 全面优于
   专用网络或所有通用时序表征。

这满足学生论文 M01 的最小验收标准；时间打乱、有效秩、类间/类内距离、密集标签比例和 AULC
仍可作为未来机制研究，但不再是当前投稿硬缺口。
