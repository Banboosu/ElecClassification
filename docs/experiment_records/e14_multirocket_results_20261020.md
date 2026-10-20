# E14：MultiRocket 强基线（validation-only）

**状态：4 seed × 2 预算 × 1 变体 = 8 条件 / 28,080 行；仅 validation，0 test。**

MultiRocket 官方实现（vendored）只用于 `fit`/`transform`；固定 `interp_z`：逐序列 z-score 后线性插值到 256 点；分类头为 `StandardScaler + RidgeClassifierCV`，alpha 只在训练子集内 3 折从 `{100,1000}` 选择。`num_features=2000`，低于论文默认 50k，是固定算力预算。

**核心结论：**

- **10% 标签：MultiRocket 96.36 ± 0.35%，高于完整三路 95.02%（+1.34pp，4/4），也高于 TCN 89.04%（+7.32pp）和 DYN 95.00%（+3.55pp）。**
  这是一个强基线，直接挑战“低标签下 MOMENT/TCN 融合更优”的说法。
- **40% 标签：MultiRocket 97.28 ± 0.40%，与 TCN 97.53% 基本持平（−0.25pp），低于完整三路 98.09%（−0.81pp），高于 DYN 95.00%（+2.28pp）。**
  40% 下完整融合仍最高，但领先幅度不大。
- 电池 40%：MultiRocket recall/FPR **97.15/1.76%**；完整三路 97.38/0.85%。召回接近，FPR 更高；默认 argmax，未调阈值。
- 按用户确定的范围，论文不引入 MultiRocket；本结果作为**内部防守性证据**，用于限制“低标签全面最优/预训练不可替代”的写法。可用结论是条件性的：低标签下经典卷积核基线即可达到或超过当前融合；40% 下融合只有小幅优势。

## Macro-F1 %

| 方法 | 10% | 40% |
|---|---:|---:|
| MultiRocket (interp_z) | 96.36 ± 0.35 | 97.28 ± 0.40 |

## 与 E12b 参照的逐 seed 配对差（pp）

| 预算 | 变体 | vs DYN | vs TCN | vs Full |
|---|---|---:|---:|---:|
| 10p | interp_z | +3.55 (SD 0.40) | +7.32 (SD 0.36) | +1.34 (SD 0.27) |
| 40p | interp_z | +2.28 (SD 0.37) | -0.25 (SD 0.24) | -0.81 (SD 0.36) |

## 电池异常（40%，默认 argmax）

| 变体 | Recall % | FPR % |
|---|---:|---:|
| interp_z | 97.15 | 1.76 |

## 审计

- 8 条件 / 28080 行；Macro-F1 复算最大误差 0；0 test。

## 边界

- MultiRocket 不支持 mask；固定使用 interp_z（z-score + 256 点插值），未为它单独调参。
- 使用 vendored 官方实现，只调用 fit/transform；分类头是 RidgeClassifierCV（{100,1000}，训练内 3 折），不是完整原仓库脚本。
- `num_features=2000` 低于论文默认 50k，是为本数据规模/内存固定的预算；不能据此说 MultiRocket 的上限只有这个水平。
- 四 seed split 重叠、validation 反复使用；没有 test、没有外部数据。
- 这是 clean 输入的强基线；没有做噪声/量化鲁棒性。
## 保全

- 本地：`artifacts/imports/e14_multirocket_20261020/`（40 文件 / 2.65MB / 31,590 行，其中 fast suite 28,080 行；另含 CV-ridge 版 aborted partial 3,510 行）。
- 远端：`artifacts/improvements/e14_multirocket/e14_multirocket_fast_20261020_v1/`；初始 CV-ridge 版 `e14_multirocket_20261020_v1/` 因 CPU 过慢中止，只保留 aborted partial，不用于结论。
- 协议：`configs/experiments/improvements/e14_multirocket_fast_protocol.json`；代码：`src/tcn_moment/run_e14_multirocket.py`；vendor：`vendor/multirocket/`（GPL-3.0）；测试：`tests/test_e14_multirocket.py`。
