# E15：MiniRocket 对照基线（validation-only）

**状态：4 seed × 2 预算 = 8 条件 / 28,080 行；仅 validation，0 test。**

MiniRocket 官方实现（vendored）只调用 `fit`/`transform`；固定 `interp_z`（逐序列 z-score + 256 点线性插值），`num_features=10000`（官方默认），`RidgeClassifierCV` 在训练子集内 3 折从 `{100,1000}` 选 alpha。MiniRocket 只使用原序列，不额外拼接 ΔP。

**核心结论：**

- MiniRocket 10% **92.89 ± 0.34%**、40% **94.42 ± 0.24%**。
- 10%：与 DYN 92.81% 基本持平（+0.08±0.50，四 seed 不稳定）；高于 TCN 89.04%（+3.85），低于完整三路 95.02%（−2.13）。
- 40%：低于 DYN 95.00%（−0.58）、TCN 97.53%（−3.11）、完整三路 98.09%（−3.67）。
- 因此 MiniRocket 单独并不构成比 MultiRocket 更强的挑战；MultiRocket 在 10% 的 96.36% 仍是最强经典基线。MiniRocket 结果说明“经典卷积核基线”内部差异也很大，不能用 MiniRocket 代替 MultiRocket 的结论。

## Macro-F1 %

| 方法 | 10% | 40% |
|---|---:|---:|
| MiniRocket (interp_z) | 92.89 ± 0.34 | 94.42 ± 0.24 |

## 与 E12b 参照的逐 seed 配对差（pp）

| 预算 | 变体 | vs DYN | vs TCN | vs Full |
|---|---|---:|---:|---:|
| 10p | interp_z | +0.08 (SD 0.50) | +3.85 (SD 0.13) | -2.13 (SD 0.21) |
| 40p | interp_z | -0.58 (SD 0.34) | -3.11 (SD 0.25) | -3.67 (SD 0.19) |

## 电池异常（40%，默认 argmax）

| 变体 | Recall % | FPR % |
|---|---:|---:|
| interp_z | 90.68 | 2.54 |

## 审计

- 8 条件 / 28080 行；Macro-F1 复算最大误差 0；0 test。

## 边界

- MiniRocket 不支持 mask；固定使用 interp_z（z-score + 256 点插值），未为它单独调参。
- 使用 vendored 官方实现，只调用 fit/transform；分类头是 RidgeClassifierCV（{100,1000}，训练内 3 折），不是完整原仓库脚本。
- `num_features=10000` 使用官方默认值；输入长度固定为 256 点。
- 四 seed split 重叠、validation 反复使用；没有 test、没有外部数据。
- 这是 clean 输入的强基线；没有做噪声/量化鲁棒性。
## 保全

- 本地：`artifacts/imports/e15_minirocket_20261020/`（29 文件 / 2.28MB / 28,080 行，全部 validation，0 test）。
- 远端：`artifacts/improvements/e15_minirocket/e15_minirocket_20261020_v1/`。
- 协议：`configs/experiments/improvements/e15_minirocket_protocol.json`；代码：`src/tcn_moment/run_e15_minirocket.py`；vendor：`vendor/minirocket/`（GPL-3.0）；测试：`tests/test_e15_minirocket.py`。
