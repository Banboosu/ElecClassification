# 充电功率时序分类实验

本项目用于充电功率时序的三分类实验，统一比较多数类、传统机器学习、1D-CNN、TCN 和
MOMENT 时序基础模型。项目支持固定数据划分、消融实验、多随机种子运行、断点续训及论文指标
汇总。

当前标签约定：

| 标签 | 含义 | 是否参与训练 |
|---|---|---|
| `0` | 正常 | 是 |
| `1` | 充电器故障 | 是 |
| `2` | 电池异常 | 是 |
| `5` | 无效或不完整记录 | 否 |

标签定义及仍需数据提供者确认的信息见 `docs/data_dictionary.md`。

## 一、项目目录

```text
.
├── configs/
│   ├── moment.yaml                  # 统一数据、MOMENT 和 TCN 配置
│   ├── models/                      # 固化的 MOMENT 官方基础配置
│   └── experiments/                 # 归一化、长度、微调策略等消融配置
├── data/
│   └── raw/
│       └── 最新多.csv               # 原始数据，不由 Git 管理
├── docs/                            # 标签说明、实验记录和故障排查
├── src/tcn_moment/
│   ├── data.py                      # 数据解析、固定划分和 mask
│   ├── metrics.py                   # 统一评价指标
│   ├── train_baselines.py           # 多数类、逻辑回归、随机森林
│   ├── train_cnn.py                 # 1D-CNN
│   ├── train_tcn.py                 # TCN
│   └── train_moment.py              # MOMENT
├── artifacts/                       # 模型、指标、日志和图表，不由 Git 管理
└── legacy/                          # 旧版脚本，仅供历史参考
```

## 二、运行环境

项目锁定以下核心版本：

- Python `3.11`
- MOMENT `0.1.4`
- PyTorch `2.12.1`
- NumPy `1.25.2`

正式训练统一在 Linux CUDA 服务器执行。本地只建议进行代码和数据检查，不运行完整模型训练。

## 三、首次部署到远程服务器

### 1. 同步项目与数据

进入项目根目录，例如：

```bash
cd ~/autodl-tmp/ElecClassification
```

如果通过 Git 管理代码，先同步最新代码：

```bash
git pull
```

`data/raw/` 被 Git 忽略，需要单独确认数据文件存在：

```bash
ls -lh data/raw/最新多.csv
```

### 2. 安装锁定环境

```bash
uv sync --frozen
```

### 3. 检查 CUDA 和依赖

```bash
uv run moment-check-environment --require-cuda
```

命令会输出 Python、PyTorch、CUDA、cuDNN、显卡、驱动及核心依赖版本。CUDA 不可用或版本不符
时会直接失败，此时不要开始正式实验。

## 四、检查数据与固定划分

```bash
uv run moment-inspect-data --config configs/moment.yaml
```

默认数据协议：

- 过滤标签 `5`
- 过滤长度小于 `18` 的序列
- 保留标签 `0`、`1`、`2`
- 按 70%/10%/20% 分层划分训练集、验证集和测试集
- 随机种子为 `42`
- 测试集只用于最佳模型的最终评价

首次执行会生成：

```text
artifacts/splits/unified_split.json
```

该文件保存三个集合的样本 ID、类别数量、数据 SHA-256 和过滤协议。TCN、MOMENT 及所有基线
都会复用同一份划分。

只有明确需要重新划分时才执行：

```bash
uv run moment-inspect-data \
  --config configs/moment.yaml \
  --rebuild-split
```

不要在已有正式实验中途随意重建划分。

## 五、正式训练前的模型加载检查

设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

MOMENT 首次运行需要下载约 1.4 GB 的预训练权重。正式训练前先执行一次无训练前向检查：

```bash
uv run moment-check-model --config configs/moment.yaml
```

正常输出应包含：

```json
{
  "status": "ok",
  "device": "cuda",
  "logits_shape": [1, 3]
}
```

该命令会显式加载 `configs/models/moment-1-large.json`、初始化三分类头并进行一次前向传播，
但不会训练模型。

## 六、先运行单种子冒烟实验

在启动五随机种子任务前，先用种子 `42` 验证全部入口。

### 1. 传统机器学习基线

```bash
uv run baseline-train \
  --config configs/moment.yaml \
  --seed 42 \
  --run-name statistical_smoke_seed42
```

该命令会运行多数类、逻辑回归和随机森林。

### 2. 1D-CNN

```bash
uv run cnn-train \
  --config configs/experiments/cnn_baseline.yaml \
  --seed 42 \
  --run-name cnn_smoke_seed42
```

### 3. TCN

```bash
uv run tcn-train \
  --config configs/experiments/normalization_zscore.yaml \
  --seed 42 \
  --run-name tcn_zscore_smoke_seed42
```

### 4. MOMENT 线性探测

```bash
uv run moment-train \
  --config configs/experiments/moment_linear_probe.yaml \
  --seed 42 \
  --run-name moment_linear_smoke_seed42
```

先确认线性探测能够完成，再尝试部分解冻和完全微调。后两者显存占用明显更高。

## 七、正式五随机种子实验

推荐显式设置 `--suite-name`，便于定位结果和避免目录重名。标准随机种子为
`42 43 44 45 46`。

### 1. 传统机器学习基线

```bash
uv run experiment-suite \
  --model baseline \
  --configs configs/moment.yaml \
  --seeds 42 43 44 45 46 \
  --suite-name thesis_baseline_v1
```

### 2. 1D-CNN

```bash
uv run experiment-suite \
  --model cnn \
  --configs configs/experiments/cnn_baseline.yaml \
  --seeds 42 43 44 45 46 \
  --suite-name thesis_cnn_v1
```

### 3. TCN 归一化消融

```bash
uv run experiment-suite \
  --model tcn \
  --configs configs/experiments/normalization_none.yaml \
            configs/experiments/normalization_minmax.yaml \
            configs/experiments/normalization_zscore.yaml \
  --seeds 42 43 44 45 46 \
  --suite-name thesis_tcn_norm_v1
```

### 4. MOMENT 微调策略消融

```bash
uv run experiment-suite \
  --model moment \
  --configs configs/experiments/moment_linear_probe.yaml \
            configs/experiments/moment_partial_finetune.yaml \
            configs/experiments/moment_full_finetune.yaml \
  --seeds 42 43 44 45 46 \
  --suite-name thesis_moment_strategy_v1
```

### 5. MOMENT 分类头学习率消融

```bash
uv run experiment-suite \
  --model moment \
  --configs configs/experiments/moment_head_lr_1e3.yaml \
            configs/experiments/moment_head_lr_1e4.yaml \
            configs/experiments/moment_head_lr_1e5.yaml \
  --seeds 42 43 44 45 46 \
  --suite-name thesis_moment_head_lr_v1
```

### 6. 序列长度消融

以下配置可分别用于 TCN 和 MOMENT：

```bash
uv run experiment-suite \
  --model tcn \
  --configs configs/experiments/length_256.yaml \
            configs/experiments/length_512.yaml \
            configs/experiments/length_816.yaml \
            configs/experiments/length_1024.yaml \
  --seeds 42 43 44 45 46 \
  --suite-name thesis_tcn_length_v1
```

如需运行 MOMENT 长度消融，将 `--model tcn` 改为 `--model moment`，并使用新的
`--suite-name`。

## 八、训练功能与指标

TCN、CNN 和 MOMENT 支持：

- 以验证集 Macro-F1 选择最佳权重
- Early Stopping
- ReduceLROnPlateau
- CUDA AMP 混合精度
- 梯度裁剪
- NaN/Inf 检测
- 断点续训
- 参数量、单轮耗时、总耗时和峰值显存记录

统一输出指标：

- Accuracy
- Balanced Accuracy
- Macro Precision、Recall、F1
- Weighted Precision、Recall、F1
- 混淆矩阵
- 各类别分类报告

论文主指标建议使用 Macro-F1，同时报告 Accuracy 和 Balanced Accuracy。

## 九、实验产物

每次运行使用独立目录：

```text
artifacts/<模型>/<运行名>/
├── checkpoint_latest.pt            # 最近一个完整 epoch，可用于恢复
├── config.yaml                      # 本次配置快照
├── environment.json                 # Python、CUDA、GPU、Git 等环境信息
├── label_encoder.pkl
├── metrics.json                     # 最终测试结果
├── metrics_partial.json             # 已完成 epoch 的历史
├── resolved_config.json             # 继承与覆盖后的实际配置
├── split_manifest.json              # 本次使用的数据划分副本
├── status.json                      # running/completed/interrupted/failed
└── <模型>_classifier_best.pt         # 验证集 Macro-F1 最佳权重
```

批量任务摘要位于：

```text
artifacts/suites/
```

## 十、断点续训

训练被中断后，先查看：

```bash
cat artifacts/moment/<运行名>/status.json
```

若 `resume_available` 为 `true`，可以恢复：

```bash
uv run moment-train \
  --config configs/experiments/moment_linear_probe.yaml \
  --resume artifacts/moment/<运行名>
```

TCN 恢复示例：

```bash
uv run tcn-train \
  --config configs/experiments/normalization_zscore.yaml \
  --resume artifacts/tcn/<运行名>
```

如果需要继续更多 epoch，应先在对应配置中提高 `epochs`。恢复时必须使用与原运行相同的模型和
数据配置。

## 十一、汇总五随机种子结果

以 TCN z-score 实验为例：

```bash
uv run experiment-summarize \
  --runs artifacts/tcn/normalization_zscore_thesis_tcn_norm_v1_seed42 \
         artifacts/tcn/normalization_zscore_thesis_tcn_norm_v1_seed43 \
         artifacts/tcn/normalization_zscore_thesis_tcn_norm_v1_seed44 \
         artifacts/tcn/normalization_zscore_thesis_tcn_norm_v1_seed45 \
         artifacts/tcn/normalization_zscore_thesis_tcn_norm_v1_seed46
```

结果保存到：

```text
artifacts/summaries/
```

汇总文件包含每项指标的均值、标准差和有效运行数量，可用于论文表格。

## 十二、生成数据质量报告

该步骤不训练模型：

```bash
uv run moment-analyze-data --config configs/moment.yaml
```

输出：

```text
artifacts/data_quality/
├── data_quality.json
└── typical_sequences.png
```

当前数据质量结论见 `docs/experiment_records/data_quality_findings.md`。

## 十三、主要消融配置

| 配置 | 作用 |
|---|---|
| `normalization_none.yaml` | 不归一化 |
| `normalization_minmax.yaml` | 单序列 Min-Max 归一化 |
| `normalization_zscore.yaml` | 单序列 z-score 归一化 |
| `length_256.yaml` | 最大长度 256 |
| `length_512.yaml` | 最大长度 512 |
| `length_816.yaml` | 最大长度 816 |
| `length_1024.yaml` | 最大长度 1024 |
| `moment_linear_probe.yaml` | 仅训练分类头 |
| `moment_partial_finetune.yaml` | 解冻最后两个编码层 |
| `moment_full_finetune.yaml` | 完全微调 |
| `moment_head_lr_*.yaml` | 分类头学习率消融 |
| `cnn_baseline.yaml` | 1D-CNN 基线 |

配置文件通过 `extends` 继承 `configs/moment.yaml`，每次运行都会保存展开后的
`resolved_config.json`。

## 十四、常见问题

### 1. `MOMENTPipeline.__init__()` 缺少 `config`

确认已经同步最新代码，并存在：

```text
configs/models/moment-1-large.json
```

然后重新执行：

```bash
uv sync --frozen
uv run moment-check-model --config configs/moment.yaml
```

详细原因见 `docs/troubleshooting.md`。

### 2. 运行目录已经存在

不要覆盖旧结果。换一个 `--run-name`，或者为批量任务设置新的 `--suite-name`。

### 3. CUDA 显存不足

处理顺序：

1. 先确认 `amp: true`；
2. 减小 `batch_size`；
3. 先运行线性探测；
4. 再尝试部分解冻；
5. 最后尝试完全微调。

修改配置前建议复制出新的实验 YAML，避免破坏已有实验条件。

### 4. 数据划分协议不匹配

通常说明 CSV、过滤条件或随机种子发生变化。不要直接覆盖正式划分；确认变化是有意的，再使用
新的 `split_path` 或执行 `--rebuild-split`。

### 5. Hugging Face 下载失败

MOMENT 基础配置已经随项目保存，但预训练权重仍需从 `AutonLab/MOMENT-1-large` 下载。检查远程
网络、代理或已有缓存后，再运行 `moment-check-model`。

## 十五、补充说明

- 输入张量形状为 `[batch, 1, length]`。
- MOMENT 使用 `input_mask`，TCN/CNN 使用 masked pooling 排除 padding。
- 每个随机种子使用独立但可复现的划分清单；相同种子的不同模型共享同一划分。
- `legacy/` 中旧脚本的数据协议与当前正式实验不一致，不应直接用于论文横向比较。
- 初始历史结果见 `docs/experiment_records/initial_baseline_results.md`。
- 后续任务与验收状态见 `todo.md`。
