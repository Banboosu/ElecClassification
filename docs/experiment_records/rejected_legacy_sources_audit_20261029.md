# 两个用户否定的legacy文件：引用与代码来源审计（2026-10-29）

## 结论

用户明确说明`legacy/ceshi.py`和`legacy/画曲线测试.py`是错误代码，不得使用。

1. **存在历史文档引用污染。** 归档旧`docs/data_dictionary.md`曾把ceshi注释当作0/1/2业务标签依据，这一依据撤回；`label_mapping_audit_20261024.md`也检查过它，现明确标为错误代码反例而非可信codebook。当前映射依据用户确认，不依据两文件。
2. **未发现正式运行代码的直接依赖或特有算法移植。** 本地检查292个当前源码/脚本/测试/配置文件；未发现命名导入/文件路径引用、ceshi特有API/产物名、非平凡完整函数AST复制，或该测试画图稿的完整连续绘图块进入当前代码。ubuntu当前294个文件的独立名称/API检查亦为0。
3. **确实发现关联旧绘图稿。** `legacy/画曲线.py`与`画曲线测试.py`有5个相同的连续三语句AST块，均读取`tcn_model_metrics.xlsx`、按相同列绘制2×2图；视为相关历史稿，不作为独立来源或正式出图入口。不能据相似性判定谁复制谁。
4. **TCN结构参照不是这两个文件。** E05B/E05C/N2/E12等显式参照的是另一文件`legacy/改进TCN.py`；当前`LegacyTCN`是PyTorch六层因果膨胀卷积的重建，并非ceshi中的趋势编码器/孤立森林。不同文件不能混称为统一可信的“legacy”。
5. **仍有参数来源待核：最短18点。** 当前配置和ceshi都出现18，但仅相同数字不能证明复制方向；现有记录未能提供其独立业务依据。当前只把它视为历史冻结研究筛选值，不能沿用“18点=7.5分钟”“专利规定”或硬件依据。未擅自修改过滤阈值、split或重跑实验。

以上是静态检查的有界结论，**不证明从未人工参考、改名或改写过类似代码**。

## 核查到的具体差异

| 项目 | 被否定的ceshi | 当前正式实现 |
|---|---|---|
| 标签 | 中文字符串映射，含旧错误业务顺序 | `data.py::load_dataset`读取原始数字标签，经LabelEncoder编码；业务名由用户确认的统一语义定义给出 |
| 功率清洗 | 丢弃0及超过550的值 | `parse_power_sequence`保留有限数值，不套用0–550W传感器假设 |
| 输入 | 35点滑窗、min-max、25秒采样、缺失率28%规则 | 保存的记录级序列、显式mask、冻结split；不调用该滑窗/缺失率规则 |
| 模型 | 趋势编码器 + IsolationForest + 手写融合 | DYN/HGB、冻结MOMENT、TCN及OOF Logistic堆叠；无上述特有API或产物依赖 |
| 阈值 | `AdaptiveThreshold`及手写类型阈值 | E10类别1分位数工作点，validation内描述性评价；不移植该动态阈值 |
| 图 | 测试画图稿读取Excel并输出旧PNG | 当前论文图由审计JSON/保存预测或正式训练ID对应的原始曲线生成，不加载该脚本或其PNG |

`画曲线测试.py`本身是Excel的读取者，不产生数字标签；不能仅凭共享Excel文件名就说它导致训练或标签颠倒。`改进TCN.py`也输出同名Excel，旧初始结果报告提到该文件名；这不是调用测试画图稿的证据。

## 搜索范围与证据

- 本地当前代码：`src/`、`scripts/`、`tests/`、`configs/`、`pyproject.toml`。
- 历史文本：5,678个保存文本文件，含122份`frozen_protocol.json`；未发现两拒绝文件被列为冻结输入的引用。历史正文引用命中归档旧数据字典，归档原件不改写。
- 本地证据：`artifacts/analysis/rejected_legacy_sources_20261029.json`。
- ubuntu独立名称/API检查：`artifacts/analysis/rejected_legacy_remote_check_20261029.json`，不声称它也独立复核了全部AST相似性或全部本地档案。
- 只读工具：`scripts/audit_rejected_legacy_sources.py`。仅读取/解析源码，不import、eval、exec或运行拒绝文件；不训练、不推理、不评价test。

拒绝文件SHA-256（检查前后不变，本地与ubuntu一致）：

- ceshi：`cf63c4255f75811c5fa2fcc8b3f58939031948c13c25edb37c0327781a65cf82`。
- 画曲线测试：`708ff081968e31cea11a448ddf0e18ba50df8aacc334e351ba875d9b3d72b897`。

## 已处理

- 新增`configs/legacy_source_policy.json`及`legacy/README.md`，明确两个文件禁用，其他legacy不自动视为可信。
- 当前数据字典、README和旧标签审计撤回ceshi的依据地位；不修改历史快照以掩盖曾经引用过它。
- 两份Markdown期刊稿明确18点阈值来源待核、不换算时间；长稿删除未经独立核实的固定采样间隔/分钟小时解释，长度特征按采样点数报告。
- 新增3项只读回归：检测命名/动态路径引用、原文件字节及模块不变、关联旧绘图稿识别。本地3项通过；ubuntu全量`pytest -q tests`为243 passed，另51 subtests passed。测试日志：`artifacts/analysis/rejected_legacy_tests_20261029.xml`。不执行legacy代码。

## 未解决及使用边界

- 最短18点的独立业务来源仍需用户/数据提供方确认。若需要改变阈值，应另立数据协议与敏感性分析，不能覆盖已有结果或声称新盲测。
- 不以被否定脚本推断采样频率、功率单位、550W量程、专利规则、接触不良/老化等机理。
- 旧PDF/DOCX/PPT未重导出，不能因为Markdown已更正就当作它们也已更新。
- 未据此认定所有数值实验作废；当前静态检查和已有逐样本审计未发现这两文件进入正式计算路径，但这不等同于证明所有人工设计来源均独立。
