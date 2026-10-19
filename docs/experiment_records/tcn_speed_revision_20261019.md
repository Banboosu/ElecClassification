# TCN执行加速修订：resident data + async checks

日期：2026-10-19。用户要求优化TCN计算速度，E05C原源码/冻结协议不改。新增独立执行层并在切换前保存检查证据，不修改数据、模型、batch、FP32、TF32关闭、optimizer/scheduler、epoch、数值失败策略或选模规则。

## 变更与不变项

- train/validation的FP32输入、mask及long标签一次上传显存；仍使用原PyTorch DataLoader的BatchSampler/RandomSampler顺序，保留iter时base_seed消耗和generator epoch结束状态。每epoch合并一次传送采样索引，GPU index_select取batch；尾batch大小不变。
- 每batch原有loss/logit/gradient有限性检查移到同一CUDA stream中的`torch._assert_async`；遇非有限值使进程报错，不跳步、无fallback。失败epoch不会提交；GPU异步断言使CUDA上下文失效，必须退出并从已提交checkpoint重新启动，不能捕获后继续训练。
- loss按原顺序用FP64标量累加，减少每batch `.item()` 的host同步；validation完整概率一次拷回CPU，仍计算原FP32前向和softmax。同一训练张量/梯度更新未使用AMP/TF32/compile/fused Adam。
- 随机种子、模型权重、AdamW及其moment状态、shuffle/CPU/CUDA RNG、每epoch train/validation指标与概率均要求逐位一致；不能只比较预测argmax。

## 切换前证据

- 合成4099行、长度1024、尾batch3，每种结构各3epoch：三种TCN的7项检查全部相同。热身后速度比current **1.168×**、legacy masked **1.291×**、unmasked **1.311×**。首轮CUDA初始化不用于稳态倍率。
- 使用已完成seed42、40%模型checkpoint，按保存optimizer/scheduler/RNG回放**一轮选定训练集9827行**，两实现初值完全相同；只在首259训练行做诊断指标，不用validation或test。三结构7项检查均逐位相同，原checkpoint SHA不变、回放权重丢弃。单次速度比 **1.292× / 1.335× / 1.307×**，不是整套训练或部署延迟保证。
- 5项新增测试通过：多epoch采样/尾batch/RNG、完整epoch/validation逐位相等、旧checkpoint继续训练相等、CPU非有限输入先于step报错、隔离CUDA进程异步断言确实失败。原套件13项针对性回归另行保留。
- 证据根`artifacts/analysis/tcn_resident_speed_20261019/`，含synthetic_equivalence(v1/v2)、real_train_replay及3次暂停恢复记录。为避免GPU争抢，基准期间暂时SIGSTOP原训练进程，finally恢复；受影响epoch_seconds含操作暂停，性能比较须排除，原科学历史不改写。

## 应用与恢复

通过后使用`replicate_tcn_fast.py`独立包装器恢复E05C，只替换本进程的case执行函数，不修改原`replicate_tcn.py`或E05B源码。切换前SIGINT请求旧进程退出，确认已退出/锁释放；未提交epoch从原子checkpoint重放，已完成36范围内的条件不重跑。

新层在首次恢复前独立冻结4个实现模块、测试/基准/本文与等价性报告SHA到`execution_revisions/resident_async_v1/`；旧协议SHA/旧文件不变。保存中断前不完整目录快照、suite状态和新层逐case执行记录。每epoch原子checkpoint仍含原完整模型/optimizer/scheduler/RNG，不沿用未提交内存状态。

FP32模型峰值显存增加约数据常驻大小；完整train+val数据约220MiB（另有激活/模型/优化器），不自动扩大batch。实际峰值以运行记录为准；本硬件可容纳不意味着所有GPU均可容纳。计算目标仍N1原36条件/1800epoch，8小时原墙钟预算不重置，不自动下一实验/test/关机，不使用pi-wake。
