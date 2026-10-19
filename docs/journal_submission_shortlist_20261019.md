# EI期刊投稿候选（公开资料核查：2026-10-19）

研究定位：充电功率曲线状态识别/异常分类；预训练时序表示与原尺度统计融合；小标签预算、强基线、归因与成本验证。当前不以“发明新基础模型”或“全面超过TCN”定位。

## 首选候选

| 期刊 | 推荐定位 | EI公开依据 | 时间/费用公开信息 |
|---|---|---|---|
| Engineering Research Express（IOP，ISSN2631-8695） | 英文首选，工程实证与技术严谨性较匹配 | IOP官网Abstracting and indexing明确列EI Compendex | 官网：订阅路线发表免费；可选Gold OA APC£1970/€2235/$2645，税另计；未核到可靠平均录用周期 |
| Journal of Engineering and Applied Science（Springer Nature/开罗大学，eISSN2536-9512） | 英文备选，电气工程/故障诊断应用 | Springer官方期刊信息列EI Compendex | 官网现称APC由Specialized Presidential Council for Education and Scientific Research承担，作者无需出版费；不能保证审稿速度 |
| 计算机工程与应用（ISSN1002-8331） | 中文首选，应用机器学习/时序分类较贴近 | 主办方2024年确认进入EI Compendex，期刊官网有对应公告 | 官网投稿指南称审稿约2个月；初审通过后审理费150元/篇，版面费230元/千字符，图表公式折字，一般总额不超过1万元；需投稿前复核 |
| 电子测量与仪器学报（ISSN1000-7105） | 中文备选，突出真实监测数据、信号处理与故障诊断 | 中国电子学会刊物介绍明确列EI | 官网简介称平均审稿约3个月、发表周期约半年；当前准确版面费需咨询 |
| 系统仿真学报（ISSN1004-731X） | 中文备选，需自然匹配系统建模/仿真/智能诊断，不为投稿虚构仿真 | 官方收录情况列EI | 官网称投稿到录用平均约4个月，录用后知网首发，纸刊还需约10个月；有版面费，金额咨询 |

“相对好投”是基于范围、评审政策与当前稿件适配度的判断，不是录用率统计。英文ERE官网明确注重科学有效性/方法严谨，不主观判断预期影响力或新颖性，较适合严谨工程实证；仍须有原创贡献和完整验证。

以上都有常规投稿渠道，不需要等中介特刊。JEAS官网另列开放专题 AI and Digital Twins for Energy Storage Systems，截止2027-04-23，涉及AI预测维护/BMS/可靠性；我们并非数字孪生工作，不应强行套题，常规稿通常更自然。

## 预警核查边界

- 对照公开2020–2025中科院国际期刊预警表及高校公开的2026 Under Review风险名单，以上五刊未见列入。
- 2026有“新锐/Under Review”等口径，不与往年中科院名单混称。官方历史预警网站本次DNS访问失败，使用高校科技处转载核对，不宣称完成实时官方站全部核验。
- 国际名单未列入≠学校必然认可；校内黑名单、低质量刊物清单、ESCI认可规则尚未核查。
- 保守排除：Computers & Electrical Engineering（2025预警）、Wireless Personal Communications（2025）、Frontiers in Energy Research（2024且2026 Under Review）、IEEE Access、Energies、Applied Sciences、Electronics（均有历史预警记录）。这不等于断言其当前每年都被预警。
- EI依据为出版社/主办方公开说明，不是付费Engineering Village实时来源检索。投稿前按精确刊名/ISSN再核当前Compendex来源，并向学校确认是期刊而非会议、录用/见刊/实际检索哪一项满足要求。
- 不把ERE的ESCI说成SCIE；不把Scopus/CSCD/中文核心等同EI；不混淆开罗大学单数Science刊与近似名称的Sciences刊。

## 对当前研究的投稿准备建议

1. 完成E03E多种子复制，不能只用seed43的0.58pp最好点宣称稳健提升。
2. 最终方法/评价协议冻结后再作正式评价；既有test历史上已被查看，不能称从未接触的全新盲测。
3. 优先补充公开相关时序数据或新增充电数据验证；若可获得真实设备/时间元数据，再做分组/时间外推，不从唯一sample ID猜分组。
4. 主张集中在归一化信息损失、原尺度统计旁路、NN分支幅度与训练预算、公平强基线和成本；不能把简单使用MOMENT或LoRA包装为新基础模型。
5. 自有数据需有授权及数据可用性声明，确认类别标注依据；“充电状态识别/异常分类”比未经物理验证的“电池健康精确诊断”更稳妥。

## 官方与机构来源

- ERE范围、EI、费用：https://publishingsupport.iopscience.iop.org/journals/engineering-research-express/about-engineering-research-express/
- JEAS主页/索引：https://link.springer.com/journal/44147
- JEAS费用：https://link.springer.com/journal/44147/how-to-publish-with-us
- JEAS开放专题：https://link.springer.com/collections/djdhidcbhg
- 计算机工程与应用主办方EI公告：https://www.cetc.com.cn/15/338652/338628/1802048/index.html
- CEA投稿指南：http://cea.ceaj.org/CN/column/column16.shtml
- 电子测量与仪器学报主办方介绍：https://www.cie.org.cn/site/content/4351.html
- JEMI简介：https://jemi.etmchina.com/jemi/site/menu/20150415023343001
- 系统仿真学报EI：https://www.china-simulation.com/CN/column/column1013.shtml
- 系统仿真投稿指南：https://www.china-simulation.com/CN/column/column3.shtml
- 历史预警转载：https://kjc.ahau.edu.cn/info/1031/10329.htm
- 2026 Under Review高校转载：https://kyc.hnuit.edu.cn/info/1005/13451.htm

投稿前仍需用户补充：学校认可中文EI吗？目标是多久录用、见刊还是EI入库？预算上限与学校预警/期刊认定清单是什么？
