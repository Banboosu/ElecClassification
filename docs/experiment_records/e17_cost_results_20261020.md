# E17：同硬件端到端成本（validation-only）

**状态：RTX 4080，seed42 / 40% checkpoint；完整在线推理路径；0 test。**

测量包含：raw 预处理（normalize+diff）、DYN 60 维提取、MOMENT 原序列、MOMENT ΔP、TCN 前向、HGB/logistic/meta 头。不是缓存分类头计时，也不是训练计时。

**核心数字：**

- batch=1：**46.3 ms/sample**（p95 67.5；单样本/小 batch 波动较大）；
- batch=32：**10.92 ms/sample**，batch≈349 ms，约 **91.6 samples/s**；
- batch=128：**11.48 ms/sample**，batch≈1,469 ms，约 **87.1 samples/s**；
- GPU 峰值显存 **2,493 MB**；host max RSS 3,815 MB；
- MOMENT **341.24M 参数 / 1,302 MB FP32**；TCN 0.25M / 0.9 MB；HGB pickle 4.18 MB，MD logistic 0.07 MB，full meta 0.001 MB。

| 方法 | batch=1 ms/sample | batch=32 ms/sample | batch=128 ms/sample |
|---|---:|---:|---:|
| DYN+HGB | 11.43 | 1.99 | 1.83 |
| TCN | 0.88 | 0.039 | 0.050 |
| TCN+DYN | 12.60 | 2.09 | 1.93 |
| **Full** | **46.10** | **11.13** | **11.42** |

**读法：**

- Full 相对 TCN+DYN 在 40% 只高 **+0.16pp（3/4）**，但 batch=32 每样本成本约 **5.3×**（11.13 vs 2.09 ms），batch=1 约 3.7×（46.1 vs 12.6）；MOMENT 两条流是主要成本（batch=1 约 33.4 ms，占 72%）。
- Full 相对 TCN 在 40% 高 **+0.56pp（4/4）**，但 batch=32 每样本成本约 **285×**（11.13 vs 0.039 ms）；批量下 TCN 极便宜，MOMENT 是瓶颈。
- 这正好回答“40% 时多 0.16pp 是否值得”：若只追求 clean 精度且可接受更大模型/显存，Full 略优；若成本敏感，TCN+DYN 或 TCN 是更合理的运行点。10% 下 Full 相对 TCN+DYN +0.76pp/4/4，但本轮未单独测 10p 成本（同一在线管线，差异只在 checkpoint）。
- 结果只代表 seed42/40%、单张 RTX 4080、无并发/服务化/磁盘 I/O/网络；不是部署 SLA。

## 单样本组件（batch=1，中位数）

| 组件 | ms |
|---|---:|
| 预处理（normalize+diff） | 0.157 |
| DYN 60 维提取 | 2.009 |
| MOMENT 原序列 | 16.873 |
| MOMENT ΔP | 16.514 |
| TCN 前向 | 0.953 |
| HGB+MD+meta 头 | 10.312 |

## 方法级端到端成本（ms/sample 中位数）

| 方法 | batch=1 | batch=32 | batch=128 |
|---|---:|---:|---:|
| DYN+HGB | 11.430 | 1.990 | 1.825 |
| TCN | 0.882 | 0.039 | 0.050 |
| TCN+DYN | 12.595 | 2.092 | 1.930 |
| Full | 46.103 | 11.128 | 11.421 |

## 在线路径精度核对

- full macro-F1：97.99%；E12 保存参考：97.99%
- full 概率与参考最大差异：2.41e-02

## 边界

- 只测 seed42/40% 的推理，不是多 seed 成本均值；batch=1 受 Python/CPU 开销影响。
- 包含预处理、DYN、双流 MOMENT、TCN 和 heads，不是缓存分类头速度。
- 没有并发、服务化、磁盘 I/O、数据解析、网络或端侧硬件；不是部署 SLA。
- 没有做能耗、模型量化/编译或 TensorRT；GPU 只测了这一张卡。