import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（根据系统调整）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 增大全局字体大小和线条样式（比原版本更大）
plt.rcParams.update({
    'font.size': 14,          # 基础字体大小（原12）
    'axes.labelsize': 16,     # 坐标轴标签字体大小（原14）
    'axes.titlesize': 18,     # 子图标题字体大小（原16）
    'xtick.labelsize': 14,    # x轴刻度字体大小（原12）
    'ytick.labelsize': 14,    # y轴刻度字体大小（原12）
    'legend.fontsize': 14,    # 图例字体大小（原12，保留但不生效）
    'figure.titlesize': 20,   # 总标题字体大小
    'lines.linewidth': 2.5,   # 曲线加粗
    'figure.figsize': (16, 12) # 画布尺寸
})

# 读取Excel文件
df = pd.read_excel('tcn_model_metrics.xlsx')

# 提取需要绘制的数据
epochs = df['Epoch'].values
val_accuracy = df['Validation Accuracy'].values
precision = df['Precision'].values
recall = df['Recall'].values
f1_score = df['F1 Score'].values

# 创建2行2列的子图布局
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# 将2x2的axes数组展平，方便逐个操作
ax1, ax2, ax3, ax4 = axes.flatten()

# 子图1：验证集准确率（移除图例）
ax1.plot(epochs, val_accuracy, color='#1f77b4', marker='o', markersize=4)
ax1.set_xlabel('训练轮数 (Epoch)')
ax1.set_ylabel('准确率')
ax1.set_title('验证集准确率变化曲线')
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, linestyle='--')

# 子图2：精确率（移除图例）
ax2.plot(epochs, precision, color='#ff7f0e', marker='s', markersize=4)
ax2.set_xlabel('训练轮数 (Epoch)')
ax2.set_ylabel('精确率')
ax2.set_title('精确率变化曲线')
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3, linestyle='--')

# 子图3：召回率（移除图例）
ax3.plot(epochs, recall, color='#2ca02c', marker='^', markersize=4)
ax3.set_xlabel('训练轮数 (Epoch)')
ax3.set_ylabel('召回率')
ax3.set_title('召回率变化曲线')
ax3.set_ylim(0, 1.05)
ax3.grid(True, alpha=0.3, linestyle='--')

# 子图4：F1分数（移除图例）
ax4.plot(epochs, f1_score, color='#d62728', marker='d', markersize=4)
ax4.set_xlabel('训练轮数 (Epoch)')
ax4.set_ylabel('F1分数')
ax4.set_title('F1分数变化曲线')
ax4.set_ylim(0, 1.05)
ax4.grid(True, alpha=0.3, linestyle='--')

# 设置整个画布的总标题
fig.suptitle('TCN模型训练关键指标变化曲线', fontsize=20, y=0.98)  # 总标题字体放大

# 调整子图间距，防止重叠
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片（高分辨率）
plt.savefig('tcn_training_metrics_subplots.png', dpi=300, bbox_inches='tight')

# 显示图片
plt.show()