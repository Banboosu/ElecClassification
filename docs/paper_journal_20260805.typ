// Journal-formatted manuscript generated from paper_draft_20260805.md.
// Regenerate with: python3 scripts/build_typst_paper.py

#import "paper_journal_body_20260805.typ": paper-body

#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 16mm, right: 16mm),
  header: align(center)[
    #text(size: 7.5pt, fill: luma(95))[充电功率时序故障分类：TCN 与 MOMENT 比较]
  ],
  footer: context align(center)[
    #text(size: 8pt, fill: luma(90))[#counter(page).display("1")]
  ],
)

#set text(
  font: ("Times New Roman", "SimSun"),
  fallback: true,
  size: 9pt,
  lang: "zh",
)
#set par(justify: true, leading: 0.55em, first-line-indent: 2em)
#set list(indent: 1.25em, body-indent: 0.45em, spacing: 0.25em)
#set enum(indent: 1.25em, body-indent: 0.45em, spacing: 0.25em)
#set heading(numbering: none)

#show heading.where(level: 1): it => block(above: 1.05em, below: 0.55em)[
  #set text(font: ("SimHei", "Arial"), size: 11pt, weight: "bold")
  #set par(first-line-indent: 0em)
  #it.body
]
#show heading.where(level: 2): it => block(above: 0.8em, below: 0.35em)[
  #set text(font: ("SimHei", "Arial"), size: 9.5pt, weight: "bold")
  #set par(first-line-indent: 0em)
  #it.body
]
#show figure.caption: set text(size: 7.5pt)
#show figure.caption: set par(justify: false, first-line-indent: 0em, leading: 0.35em)

#let journal-figure(path, caption, wide: false) = {
  let item = figure(
    image(path, width: 100%),
    caption: caption,
    supplement: none,
    numbering: none,
  )
  if wide {
    place(top + center, scope: "parent", float: true, clearance: 0.8em, item)
  } else {
    align(center, item)
  }
}

#let journal-table(caption: none, columns: 0, header: (), cells: (), wide: false) = {
  let item = figure(
    table(
      columns: (1fr,) * columns,
      align: (x, y) => if x == 0 { left } else { center },
      inset: (x: 2.4pt, y: 2pt),
      fill: (x, y) => if y == 0 { luma(235) } else { none },
      stroke: (x, y) => (
        top: if y == 0 { 0.8pt + luma(70) } else { none },
        bottom: if y == 0 { 0.55pt + luma(90) } else { 0.25pt + luma(190) },
      ),
      table.header(..header),
      ..cells,
    ),
    caption: caption,
    supplement: none,
    numbering: none,
  )
  if wide {
    place(top + center, scope: "parent", float: true, clearance: 0.8em, item)
  } else {
    align(center, item)
  }
}

#align(center)[
  #set par(first-line-indent: 0em, justify: false)
  #text(
    font: ("SimHei", "Arial"),
    size: 16pt,
    weight: "bold",
  )[面向充电功率时序故障分类的专用网络与基础模型比较：MOMENT 的标签效率、迁移边界与安全关键评估]

  #v(0.9em)
  #text(size: 10pt, weight: "bold")[【待补：作者姓名】]
  #linebreak()
  #text(size: 8.5pt)[【待补：作者单位、城市、邮编】]
  #linebreak()
  #text(size: 8pt)[通信作者：【待补】　E-mail：【待补】]
]

#v(0.7em)
#block(fill: luma(246), stroke: 0.35pt + luma(190), inset: 8pt, radius: 2pt)[
  #set par(first-line-indent: 2em, leading: 0.45em)
  #align(center)[#text(font: ("SimHei", "Arial"), weight: "bold", size: 9.5pt)[摘要]]
  充电功率曲线可以直接反映充电过程状态，无需增加额外传感器。真实业务运行中的故障样本积累较慢，“正常—充电器故障—电池异常”分类还要同时处理变长输入、与类别相关的序列长度，以及漏检和误报之间的取舍。本文使用真实内部运行资源中的 35,099 条有效充电功率序列，比较多数类、统计特征逻辑回归、随机森林、1D-CNN、时序卷积网络（TCN）和MOMENT-1-large。每个随机种子采用 70%/10%/20% 分层训练、验证和测试划分，所有模型共享样本清单；变长序列通过显式掩码排除填充值。MOMENT 分别采用线性探测、最后两层微调、冻结表征 RBF-SVM 和完全微调，并在 1%、5%、10%、20% 和 40% 标签比例下与从头训练的 TCN进行配对比较。完整标签条件下，TCN 获得 95.99% ± 0.73% Macro-F1，完全微调 MOMENT 为95.37% ± 0.39%，但后者的训练时间、峰值显存和参数量约为前者的 31.5、58.5 和 1,560 倍。在 1%、5% 和 10% 标签下，冻结 MOMENT 表征配合 RBF-SVM 分别领先 TCN 6.43、10.86和 12.55 个 Macro-F1 百分点，且随机种子间标准差更小；40% 标签时 TCN 反超 7.54 个百分点。在固定架构、样本、预处理、池化和 RBF-SVM 的归因消融中，预训练表征相对随机初始化表征在1%、5% 和 10% 标签下进一步领先 28.60、22.58 和 22.01 个百分点，三个配对 95% 区间均高于 0。在无监督检索中，MOMENT 的 Macro-Precision\@10 为 69.12%，高于原始曲线的 65.99%，但略低于人工统计特征的 70.08%；在零样本插补的 8 个条件中，MOMENT 均未超过线性插值。电池异常专项评估中，完整标签 TCN 的召回—误报权衡更好，少样本 MOMENT 的收益还不足以支持安全部署。综合来看，MOMENT 在该任务中的可复现优势集中于低标签三分类的标签效率和稳定性，并未延伸到完整标签精度或通用零样本能力。标签充足且关注已测训练侧资源时，专用 TCN 更合适。

  #set par(first-line-indent: 0em)
  *关键词：*  充电功率时序；故障分类；时序基础模型；MOMENT；时序卷积网络；少样本学习；安全关键评估
]

#v(0.8em)
#align(center)[
  #set par(first-line-indent: 0em, justify: false)
  #text(
    size: 12pt,
    weight: "bold",
  )[Specialized Temporal Networks versus a Time-Series Foundation Model for Charging-Power Fault Classification: Label Efficiency, Transfer Boundaries, and Safety-Critical Evaluation of MOMENT]
  #linebreak()
  #text(size: 8.5pt)[【Authors and affiliations to be completed】]
]

#v(0.35em)
#block(inset: (x: 5pt, y: 3pt))[
  #set text(size: 8pt, lang: "en")
  #set par(first-line-indent: 0em, leading: 0.42em)
  *Abstract:* Charging-power curves provide a non-intrusive signal for characterizing charging states. In real internal operations, fault samples are costly to accumulate, and the classification of normal, charger-fault, and battery-abnormal sessions is complicated by variable-length inputs, class-correlated sequence length, and the safety-critical trade-off between false alarms and missed detections. This study evaluates whether generic time-series pretraining offers a practical advantage over task-specific models on 35,099 valid charging-power sequences from a real internal operational resource. Majority prediction, statistical-feature logistic regression and random forest, a 1D-CNN, a temporal convolutional network (TCN), and MOMENT-1-large were compared under paired stratified 70%/10%/20% train/validation/test splits. Padding was excluded through explicit masks. MOMENT was evaluated with a linear probe, last-two-layer fine-tuning, frozen embeddings plus an RBF-SVM, and full fine-tuning. Label-efficiency experiments used nested 1%, 5%, 10%, 20%, and 40% subsets. With all labels, TCN achieved a Macro-F1 of 95.99% ± 0.73%, compared with 95.37% ± 0.39% for fully fine-tuned MOMENT, while the latter required approximately 31.5 times the training time, 58.5 times the peak GPU memory, and 1,560 times the parameters. With 1%, 5%, and 10% labels, however, frozen MOMENT embeddings plus an RBF-SVM exceeded TCN by 6.43, 10.86, and 12.55 Macro-F1 percentage points and showed lower variability across seeds. TCN regained a 7.54-point advantage at 40% labels. An architecture-matched ablation further showed that pretrained embeddings exceeded randomly initialized embeddings by 28.60, 22.58, and 22.01 points at the three low-label budgets; all paired 95% intervals were above zero. Frozen MOMENT embeddings also improved unsupervised Macro-Precision\@10 over raw resampled curves (69.12% versus 65.99%) but remained below handcrafted statistical features (70.08%). MOMENT did not outperform linear interpolation in any of eight zero-shot imputation conditions. Safety-critical battery-abnormality evaluation favored fully supervised TCN in recall–false-positive trade-offs and did not support deployment claims for few-label methods. The reproducible advantage of MOMENT in this dataset is therefore label efficiency and stability in low-label multiclass classification, rather than superior full-label accuracy or universal zero-shot transfer. When labels are sufficient and measured training-side resources matter, TCN is the more appropriate choice under the tested setup.

  *Keywords:*  charging-power time series; fault classification; time-series foundation model; MOMENT; temporal convolutional network; label efficiency; safety-critical evaluation
]

#v(0.45em)
#line(length: 100%, stroke: 0.5pt + luma(120))
#v(0.25em)

#columns(2, gutter: 7mm)[
  #paper-body(journal-table, journal-figure)
]
