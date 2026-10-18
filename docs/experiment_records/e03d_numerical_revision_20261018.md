# E03D 数值修订记录与 v2 启动前补充协议

## v1 事故，不删改失败结果

2026-10-18，`e03d_head_diagnostic_20261018_v1` 在第4个条件 `statistics_fanin_gain` 的第132轮因非有限梯度范数中止。last checkpoint 提交到131轮；动态AMP loss scale已按2000步间隔从1024增长至32768。程序依照原规则报错，没有跳过坏梯度或继续生成成功标记。

三个 legacy200 条件已完成，第四条件只部分完成，v1不能写成完整实验。原协议、源码、状态、历史、模型和已知分数全部保留，不用v2覆盖，也不从中只挑正结果。

## 训练集故障复现

`check_e03d_fixed_scale.py` 从同一个131轮模型、optimizer和CPU/CUDA/shuffle RNG出发，重放132轮；不评价validation/test，重放权重丢弃，不初始化后续候选。

- scale32768：第50个优化窗口出现非有限梯度（此前49个完成）。
- scale1024：全部77个窗口完成，梯度有限。

这支持本次故障与动态loss scale升高有关，而不是以验证分数为依据修改学习率或删除样本。重放报告：`artifacts/analysis/e03d_fixed_scale_replay_20261018.json`。

## v2：单独冻结的数值执行版本

套件：`e03d_head_diagnostic_20261018_v2`。
协议：`configs/experiments/improvements/e03d_head_diagnostic_stable_protocol.json`。
入口：`scripts/run_e03d_stable_remote.sh`。

- 六组**全部从相同参数初值重跑**，不混入v1的已完成结果或部分训练状态。
- 所有头固定loss scale=1024，growth_interval=1000000000（大于本轮总步数），不自动增长；非有限loss/梯度仍立即中止。
- 模式改为 `legacy_fresh` 与 `fanin_gain`。前者是原几何gain1的新鲜训练，不再从动态scale的30轮checkpoint继续。
- 两模式都采用相同固定scale；均比较自身30/200预算，所以预算和增益的主要比较不混用数值策略。
- 原E03P的30轮数值只作已知历史参照，v2不要求与它逐位相同。AMP数值路径变化不等于严格数学等价。
- 输入、缓存、样本、模型拓扑、权重初值、支路相对权重、数据种子、LR/weight decay/dropout、30轮cosine前缀和其后恒定1e-4、200轮上限及诊断/候选门槛均不变。
- 仍然只训练缓存分类头，不执行Transformer、不训练LoRA、不触碰test；结束不自动复制或开展下一阶段。
- 原v1文件与E01/B/C/E03P文件不编辑。修订版使用新runner和固定scale模块，并冻结事故记录、复现脚本/报告以及修订代码。
- 这是在已知部分v1结果后的技术修订，必须在论文/报告中披露，不能称为全新盲法预注册或独立确认；原E03P失败门槛也不改写。

输出目录：`artifacts/improvements/e03d_head_diagnostic_stable/`。
日志：`artifacts/logs/e03d_head_diagnostic_20261018_v2.log`。
