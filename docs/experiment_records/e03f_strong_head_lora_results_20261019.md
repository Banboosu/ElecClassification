# E03F-v2：强头起点的LoRA对照结果（validation-only）

日期：2026-10-19。seed43、40%标签（9827 train、3510 validation）；已知E03T test之后的探索，无新test评价。

四条件从各支路200轮预算内已选择的强gain头出发；同支路两处理初始头与初始概率完全一致，统一追加30轮、重置optimizer/scheduler。epoch0纳入选择。

| 条件 | 初始F1 % | 最佳F1 % | 最佳epoch | 末轮F1 % | 可训练参数 | 训练分钟 |
|---|---:|---:|---:|---:|---:|---:|
| frozen_fusion | 91.7225 | 91.7479 | 15 | 91.5814 | 135555 | 19.52 |
| lora8_fusion | 91.7225 | 92.4751 | 24 | 92.0974 | 921987 | 41.46 |
| frozen_moment | 83.7576 | 83.8440 | 14 | 83.8129 | 135555 | 18.04 |
| lora8_moment | 83.7576 | 88.6227 | 27 | 88.4758 | 921987 | 41.59 |

冻结pilot规则全部通过：

- LoRA融合相对继续冻结融合 **+0.7272pp**，门槛≥0.5pp。
- 相对最强非适配参照 **+0.7272pp**，门槛≥0.5pp。
- 相对同rank仅MOMENT **+3.8524pp**，门槛≥0.25pp。

可建议另立复制协议，但没有自动复制、追加rank/训练预算或test。不能将单seed重复validation上的+0.727pp当作泛化确认，也不据此改写E03T/E05A结果。

本地独立核验120epoch和42120行初始/最佳/端点预测，F1最大误差1.11e-16；逐行ID/真值、支路内初始概率、模型SHA、固定scale1024/零跳步及epoch0选择规则通过。原始训练器完整审计还核对train ID/scaler、适配器梯度及冻结参数不变。

终态保全109文件、274137701bytes，SHA无不匹配；v1在第一个optimizer.step前失败的原目录仍保留。备份watcher的st_size误调用在本轮修复后重试，verified记录明确晚于第一次失败。

可训练参数不代表完整管线规模：冻结融合总参数341378950，LoRA融合342165382，仍依赖完整MOMENT大骨干。各支路父模型分别经过validation选择，追加预算和优化器重置两处理相同，但算力成本与dropout随机消耗不同。原父头V100与本次RTX4080SUPER计时不直接拼接。

本结论只说明：强头warm-start下，固定rank8在该探索seed上仍有增量，旧弱头LoRA失败不能推广为LoRA普遍无效；也不能反向抹去原E03P失败。

证据：`artifacts/imports/e03f_strong_head_lora_20261019/terminal_artifacts/e03f_strong_head_lora_20261019_v2_analysis/summary.json`、`artifacts/imports/e03f_strong_head_lora_20261019/independent_output_audit.json`、`watcher_status.json`和`terminal_remote_manifest.json`。
