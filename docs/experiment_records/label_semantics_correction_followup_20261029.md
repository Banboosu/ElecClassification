# 类别语义纠错续报与论文修订（2026-10-29）

## 状态与原则

正确映射：**0=正常、1=电池异常、2=充电器故障**。依据用户确认，不由曲线形态反推标签；尚未附原始codebook。

本轮完成保存validation预测的扩展复评、类别1的E10描述性工作点、两份期刊稿和简报的主要更正，以及样例/框架/风险图更新。**没有重训、重新推理或重开test**。原始CSV、数字标签、score列、split、旧JSON/CSV、冻结协议和源码快照不覆盖。

## 1. 扩展复评：1,496条件 / 5,250,960行

| 来源 | 条件 |
|---|---:|
| E18 + E18-100 + E12b修正版 | 352 |
| E08 | 88 |
| E09有效原条件 + stack_three修正版 | 60 + 12 |
| E12 | 32 |
| E13压力 + E13b细网格 | 336 + 384 |
| E14 MultiRocket正式fast / E15 MiniRocket正式套件 | 8 + 8 |
| E16 / E16b | 48 + 72 |
| E18b / E18c / E18d | 60 + 20 + 16 |
| 合计 | **1,496** |

逐份核对保全预测及指标SHA、正式seed的validation ID与顺序、原始数字真值、有限分数、预测和Macro-F1；最大Macro-F1复算误差 **1.1102230246251565e-16**。四seed分别计算后汇总，不能把seed42指标当作四seed均值。

- 概率/softmax分数检查范围、行和及argmax。
- E08 SVC输出的是原始OVR `decision_function`，不要求0–1或行和为1；保留`SVC.predict`投票类别，按各类原始分数算AP，不强迫预测等于OVR分数argmax。
- E09旧损坏`stack_three`的12条件不混入当前比较，另用12份修复预测；中止的E14 partial不混入正式四seed汇总。原档均保留，排除清单及理由写入新协议和结果。
- E13两套实验分开汇总，不将重复clean/quantization条件当作额外独立seed。
- 仍为重叠split、反复使用validation的描述性证据，不构成外部确认。

产物：`artifacts/analysis/label_semantics_correction_20261029_v2/`；协议：`configs/experiments/improvements/label_semantics_correction_expanded_20261029.json`；[复评全表](label_semantics_reassessment_expanded_20261029.md)。v1的352条件纠错产物保留。

## 2. 科学解释的实质变化

### 差分流

E09的40%结果，电池类别1：原序列Recall/FPR **89.42/3.81%**，双流 **93.36/2.56%**；电池召回增加约3.94pp。原先77.87→94.31%的大幅提升是**充电器类别2**（约16.44pp），不能写成电池提升。

### 融合与安全

E12b的40% Full电池Recall/FPR **98.23/0.67%**，TCN+DYN **98.18/0.67%**，电池召回净增仅约0.05pp，不是旧误述0.50pp。10% Full电池Recall **95.28%**，略低于TCN+DYN **95.33%**。E18的100% Full/TCN+DYN Macro-F1 **98.53/98.54%**，电池Recall **98.68/98.80%**。因此删除“Full始终最高召回”“预训练始终必需”的解释。

### 噪声风险

E13b的40% Full在0.25/0.5/1% RMS高斯噪声下，真正电池Recall **96.71/73.28/14.10%**，FPR **0.57/0.19/0.04%**：低FPR同时伴随严重漏检。原“电池FPR暴涨11.02/53.21/92.42%”实际属于充电器类别2。预测偏向充电器，不是电池报警能力增强。量化5单位时电池Recall/FPR **68.01/0.25%**。

运行极值分母与平台只作候选机制；`max_runup`依赖running-min，`max_drawdown`依赖running-max，不混写。没有把裁剪或增广写成已验证修复，高斯σ也不等于有界±误差。

## 3. E10：电池类别1的描述性工作点

只读8份已审计E12 Full的validation预测，**不重新拟合温度或模型**。分位数阈值在同一validation上构造并评价，不能称独立校准或部署阈值。

| 训练预算 | 目标FPR 1%处的电池Recall | 实际FPR | 目标Recall 99%处的实际Recall | 对应FPR |
|---|---:|---:|---:|---:|
| 10% | 88.80% | 1.04% | 98.90% | 7.31% |
| 40% | 98.75% | 1.04% | 98.90% | 1.43% |

电池占比从旧误解释的36.92%更正为**28.55%**；固定50%报警时完美排序FPR下界约**30.02%**。这不是“高召回必然大量误报”的证明。

产物：`artifacts/analysis/e10_battery_semantics_v2_20261029/`；脚本：`scripts/reassess_e10_battery_semantics.py`；[完整工作点](e10_battery_semantics_v2_20261029.md)。旧类别2阈值仍为充电器对象，不移作电池阈值。

## 4. 文稿与图表

已修订：

- `docs/manuscript_journal_zh.md`：标签与表1；E09两类Recall/FPR；E12b电池表；100%预算；E18与E12的架构差别；噪声风险及结论。
- `docs/manuscript_journal_zh_full.md`：同步修订中英文摘要、数据定义/分布、曲线描述、粗糙度解释、E09与E12b分项表、100%主表、校准/风险/成本取舍及结论。保留原有其余章节结构与参考文献，不恢复已被用户删除的旧稿。
- `docs/experiment_report_brief.md`：电池主表改为类别1，历史test演进表明确是类别2充电器；旧二分类与检索对象不再解释为电池。另修正错误图片相对路径。
- E08/E09更正版/E12b/E13报告的关键分项与机制文字同步更正；其他历史记录保留并标注语义/归档范围。

新图另存，不覆盖旧导出物：

- `docs/figures/label_semantics_v2/paper/fig1_dataset_overview.{svg,png}`：使用正式seed42训练ID，仍为固定随机抽样，(b)电池1、(c)充电器2；不从样例推断故障原因。
- `docs/figures/label_semantics_v2/paper/fig2_overall_framework.{svg,png}`：输出顺序0正常/1电池/2充电器，对应E12联合MD的9维元特征。
- `docs/figures/label_semantics_v2/followup/e10_battery_alert_v2.{svg,png}`：类别1、描述性报警数量曲线。
- `docs/figures/label_semantics_v2/followup/e13_class_risks_v2.{svg,png}`：电池Recall/FPR与充电器FPR分开画。
- 同目录保留1%–100%的电池/充电器默认argmax曲线。

还发现并修正旧`analyze_e18_deep.py`把seed42长度掩码用于其他seed的问题：新版逐seed读取其训练三分位边界及validation长度，并核对预测ID。新版另存`artifacts/analysis/e18_deep_analysis_semantics_v2/`；旧长度表不再作为跨seed长度结论。逐类/电池AP的seed42描述仍与四seed汇总区分。

## 5. 代码入口与测试

E08/E09/E12/E13/E14/E15/E16的当前runner和E18补充脚本改用统一标签定义，`run_e08._evaluate`拒绝把类别2写成新的电池指标；冻结输入清单纳入语义模块。未来实验须新suite/新freeze，不修改旧冻结产物来绕过指纹。

当前电池安全、二分类、错误分析CLI的默认正类改为1（版本2），拒绝传类别2冒充电池，并要求显式`--allow-test-evaluation`才可执行其历史test路径。**本轮未执行这些CLI的训练/test流程**。旧类别2二分类模型和旧test档案保留为充电器任务。历史原始数值审核器仍需按旧class-2字段核验原档，不据其`battery_*`旧键名当电池证据；未宣称所有历史脚本都已转换为新分析入口。

ubuntu环境通过：`PYTHONPATH=src:vendor .venv/bin/python -m pytest -q tests`，**240 passed，另51 subtests passed**，1条transformers弃用警告。本地标准库纠错回归10项通过。首次不限定目录的pytest误收集了冻结快照内同名测试，出现收集冲突；改为只测`tests/`，没有删除或改写快照。

后续用户明确否定ceshi和测试画图稿后，已完成只读来源审计、撤回旧标签依据，并将18点阈值的独立业务来源列为待核；见[来源审计](rejected_legacy_sources_audit_20261029.md)。新增3项回归后，ubuntu全量为243 passed、另51 subtests passed，原legacy脚本字节保持不变。

文稿表格复核入口：`scripts/audit_corrected_manuscripts.py`；输出`artifacts/analysis/label_semantics_manuscript_check_20261029.json`。覆盖两份稿件的E09两类分项、E12b电池表、E18的1%–100%主表以及引用图片存在性，不是全部引文与全文事实的独立验收。

## 6. 仍需注意

- 旧PDF/DOCX/PPT导出物未重导出，可能保留旧标签；本轮更新的是Markdown源稿与新图。不得把旧PDF当本轮纠正版提交。
- 未重测旧test、未训练类别1专用二分类器、未取得新外部数据/设备/时间元数据。
- E18用原序列/差分两个独立概率块，E12用联合MD概率块，不能把二者差异全归于硬件；新稿已说明。
- 机理与部署可靠性尚需原始标注/运维或独立新数据支撑；本轮只更正语义和已有数据分析。
