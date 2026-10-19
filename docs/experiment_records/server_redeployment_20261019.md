# 2026-10-19 AutoDL 更换服务器后的恢复记录

## 范围

用户更换算力服务器，SSH别名仍为 `autodl`。本次只恢复项目、锁定环境、实验产物和公共模型，并做环境/合成输入验证。**没有启动新实验、重训历史结果、评价业务test或执行关机。**

旧机保全目录 `artifacts/imports/autodl_preservation_20261018/` 保持不变；新配置与检查放入独立 `artifacts/deployments/autodl_20261019/`。不能对新实例重跑旧finalize备份脚本并覆盖原保全证据。

## 新硬件与隔离环境

| 项目 | 新服务器 |
|---|---|
| 主机 | autodl-container-f0c94cb49a-92794602 |
| GPU | NVIDIA GeForce RTX 4080 SUPER，驱动报告32,760 MiB |
| Compute capability | 8.9 |
| 驱动 | 595.71.05，nvidia-smi显示最高CUDA 13.2 |
| CPU quota | 16核（quota1600000 / period100000），不是宿主机128核 |
| 容器内存限额 | 62 GiB |
| 项目数据盘 | `/root/autodl-tmp`，挂载后约50 GiB |
| 项目 | `/root/autodl-tmp/ElecClassification` |
| 训练Python | 3.11.17，独立uv `.venv` |
| PyTorch | **2.12.1+cu126**，`torch.version.cuda=12.6` |
| MOMENT / NumPy | 0.1.4 / 1.25.2 |

新机原预装Python3.12.3、Torch2.12.1+cu130保留，没有卸载或原地替换。训练仍用原 `pyproject.toml` 和 `uv.lock`，两文件恢复前已与备份比较一致，`uv sync --frozen --python 3.11.17` 安装。

nvidia-smi的CUDA13.2是驱动能力，不等于本项目实际CUDA运行库版本；无需据此改锁定依赖或另外安装系统CUDA12.6。

uv默认Python下载入口吞吐过低，停止该安装尝试并保留日志，改用官方GitHub python-build-standalone镜像下载同版本3.11.17，随后成功。没有更换Python版本或解除锁。网络加速配置只在安装进程读取，不输出其代理凭据。

## 归档与模型恢复

原备份包括旧E01全部分类器与缓存、E01B/C、E03P、失败E03D-v1、E03D-v2、E03E，以及远程源码/配置/文档/测试/数据快照。

1. 先恢复归档源码快照和artifacts，再按原manifest逐文件核对。
2. 新机校验 **2,727 / 2,727文件通过**，共 **4,714,447,031 bytes**，无失败；原manifest SHA-256为 `dfef9bef3187ee315f478b0d7c0ecedc66fda6f2cb2675030743d9f39f5faa40`。
3. 公共权重为实体文件，不依赖旧机失效软链接：
   - `AutonLab/MOMENT-1-large`
   - revision `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`
   - 1,385,501,256 bytes
   - SHA-256 `a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572`
4. 重建 `HF_HOME=/root/autodl-tmp/.huggingface` 的snapshot及refs/main；补取同revision的公共config.json。后续模型加载在offline模式通过。
5. 归档完整性通过后再同步当前工作区的新报告/辅助脚本和台账；当前与归档同名文件的差异仅为todo2.md。冻结实验输入及原环境/指标文件不覆盖、不改写；`.pi/`、`.git/`、本地venv及敏感env文件不上传。

准确的“逐字节恢复”证据对应归档恢复阶段；当前台账/新部署文件是随后另行记录的工作区叠加，不宣称整个工作目录没有新增文件。

## 检查结果

- `tcn_moment.check_environment --require-cuda` 通过。
- 全套pytest：**106 passed，6 subtests passed**。只有已知transformers pytree弃用警告。
- 只读校验全部通过：E01B 115文件、E01C 124、E03P 136、E03D-v1 167、E03D-v2 176、E03E 208。未调用summarize改写旧recorded_at。
- 合成输入覆盖短、常量和不同有效长度序列；FP32 pooled embedding为4×1024且有限，AMP输出有限。
- q/v LoRA 48投影，零初始化输出误差0；合成梯度有效，冻结参数哈希不变，学习状态重载输出误差0。仅合成一步优化，不训练项目样本。
- CUDA float64距离与CPU对照最大绝对误差约 **5.68e-14**，满足1e-12容差。
- `allow_tf32=false`，matmul precision为highest；没有为新卡静默开启TF32、改BF16、改batch或增加worker。

## 新GPU下的实验边界

原结果仍是原V100环境的结果。RTX 4080 SUPER和V100的硬件、驱动、内核选择及浮点舍入可能不同；以上检查不证明跨GPU逐位重现。

- 不把旧V100耗时/显存与新卡结果直接混为同硬件对比。
- RTX的FP64能力不同于V100；原CUDA float64核搜索不保证在新卡上更快，后续若改执行后端需单独验证数值等价并记录。
- 旧缓存作为冻结产物保留；新硬件产生的新实验/新缓存应使用独立目录并记录环境，不向旧冻结运行静默追加或覆盖数据。
- E03E复制通过（融合Macro-F1四seed均值92.04%±0.53%）的既有结论不变。是否正式test仍需单独冻结并批准，本次迁移不构成test授权。

## 使用方法

```bash
ssh autodl
cd /root/autodl-tmp/ElecClassification
source scripts/autodl_env.sh
uv run --frozen python -m tcn_moment.check_environment --require-cuda
```

默认offline模型缓存、单线程BLAS/OpenMP设置与旧实验一致。已有实验启动器仍可使用，但本次没有运行任何训练启动器。

新增辅助脚本：

- `scripts/autodl_env.sh`：显式加载环境，不启动训练。
- `scripts/bootstrap_autodl_locked.sh`：隔离安装原锁，保留基座conda环境。
- `scripts/restore_autodl_backup.sh`：本地发起恢复，先核对确切新hostname，不删除目标、不关机。
- `scripts/verify_autodl_restoration.py`：按原manifest逐文件核验，不执行pickle模型。
- `scripts/check_replacement_server.py`：只用合成输入验证新硬件与离线公共模型。

机器可读证据：`artifacts/deployments/autodl_20261019/` 下的 `host_before_install.json`、`install_status.json`、`environment.json`、`archive_restoration_verified.json`、`frozen_protocol_checks.json`、`pytest.log`、`synthetic_gpu_checks.json` 与最终就绪状态。本地也保留副本。
