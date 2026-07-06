import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import joblib
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


# 更严格的数据预处理函数
def preprocess_data(row):
    charging_powers_str = row['charging_powers_str']
    charging_powers = []
    if isinstance(charging_powers_str, str):
        charging_powers_str_cleaned = charging_powers_str.replace('"', '').strip()
        power_str_list = charging_powers_str_cleaned.split(',')
        for power_str in power_str_list:
            try:
                power = float(power_str)
                charging_powers.append(power)
            except ValueError:
                print(f"跳过无法转换为数值的数据: {power_str}")
    return charging_powers


# 读取CSV文件
data = pd.read_csv('最新多.csv', delimiter=' ')  # 假设数据按照空格分隔

# 数据预处理
data['charging_powers'] = data.apply(preprocess_data, axis=1)

# 提取标签并进行编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['InsertedColumn'].values)
y_categorical = to_categorical(y)

# 保存 LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# 数据划分 - 修复：过滤超长序列时同步过滤标签，保持数据匹配
max_allowed_len = 5000
# 创建有效数据的掩码
valid_mask = [len(seq) <= max_allowed_len for seq in data['charging_powers'].tolist()]
valid_X = [seq for seq, valid in zip(data['charging_powers'].tolist(), valid_mask) if valid]
valid_y = y_categorical[valid_mask]

if len(valid_X) < len(data['charging_powers']):
    print(f"删除了{len(data['charging_powers']) - len(valid_X)}条长度过长的数据")

# 确保X和y长度一致后再划分
X_train_seq, X_test_seq, y_train, y_test = train_test_split(valid_X, valid_y, test_size=0.2, random_state=42)

# 对每个序列进行截断
safe_maxlen = 1000
X_train_seq = [seq[:safe_maxlen] for seq in X_train_seq]
X_test_seq = [seq[:safe_maxlen] for seq in X_test_seq]

# 填充序列
maxlen = max(len(seq) for seq in X_train_seq) if X_train_seq else 1  # 防止空列表
X_train_seq_padded = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', value=0.0)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', value=0.0)

print(f"训练集输入形状: {X_train_seq_padded.shape}")
print(f"测试集输入形状: {X_test_seq_padded.shape}")


# 自定义回调类，增强数据收集功能
class MetricsCallback(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.precision = []
        self.recall = []
        self.f1 = []
        self.val_data = val_data
        self.epoch_metrics = {}  # 存储每个epoch的指标

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        X_val, y_val = self.val_data
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        # 计算加权指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes,
            average='weighted',
            zero_division=1
        )
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)

        # 记录每个epoch的完整指标
        self.epoch_metrics[epoch] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'val_accuracy': logs.get('val_accuracy', 0),
            'accuracy': logs.get('accuracy', 0),
            'val_loss': logs.get('val_loss', 0),
            'loss': logs.get('loss', 0)
        }


# 构建TCN模型
def build_tcn(input_shape, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    dilation_bases = [1, 2, 4, 8, 16, 32]
    num_filters = 128
    kernel_size = 3
    dropout_rate = 0.3

    for dilation in dilation_bases:
        model.add(Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation,
            activation='relu'
        ))
        model.add(Dropout(dropout_rate))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_classes, activation='softmax'))

    return model


# 构建并编译模型
input_shape = (maxlen, 1)
num_classes = y_categorical.shape[1]
model = build_tcn(input_shape, num_classes)

# 学习率调整回调
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 重塑输入数据以适应Conv1D
X_train_reshaped = X_train_seq_padded.reshape((X_train_seq_padded.shape[0], X_train_seq_padded.shape[1], 1))
X_test_reshaped = X_test_seq_padded.reshape((X_test_seq_padded.shape[0], X_test_seq_padded.shape[1], 1))

# 训练模型
metrics_callback = MetricsCallback(val_data=(X_test_reshaped, y_test))
history = model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    verbose=1,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[metrics_callback, reduce_lr]
)

# 预测和评估
y_pred = model.predict(X_test_reshaped, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 计算最终评估指标
final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(
    y_true_classes, y_pred_classes,
    average='weighted',
    zero_division=1
)

# 保存模型
model.save('charging_power_model_tcn_optimized.h5')

# 打印分类报告
labels = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
target_names = label_encoder.classes_[labels].astype(str)
print("\n分类报告:")
print(classification_report(
    y_true_classes, y_pred_classes,
    labels=labels,
    target_names=target_names,
    zero_division=1
))


# 保存指标数据到Excel
def save_metrics_to_excel(history, metrics_callback, smoothing_window=15):
    # 准备数据
    epochs = range(1, len(history.history.get('val_accuracy', [])) + 1)

    # 确保所有指标长度一致
    max_epoch = len(epochs)
    precision = metrics_callback.precision[:max_epoch]
    recall = metrics_callback.recall[:max_epoch]
    f1 = metrics_callback.f1[:max_epoch]
    val_accuracy = history.history.get('val_accuracy', [])[:max_epoch]

    # 创建DataFrame存储原始指标
    df = pd.DataFrame({
        'Epoch': epochs,
        'Training Accuracy': history.history.get('accuracy', [])[:max_epoch],
        'Validation Accuracy': val_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    # 计算smoothed值并添加到DataFrame
    for metric in df.columns[1:]:  # 跳过Epoch列
        if len(df[metric]) >= smoothing_window:
            smoothed = df[metric].rolling(window=smoothing_window, center=True).mean()
        else:
            smoothed = df[metric]  # 如果数据不足，不做平滑
        df[f'Smoothed {metric}'] = smoothed

    # 创建Excel工作簿并写入数据
    wb = Workbook()
    ws = wb.active
    ws.title = "Model Metrics"
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)
    wb.save('tcn_model_metrics.xlsx')
    print("\n指标数据已保存到 tcn_model_metrics.xlsx")


# 调用函数保存数据
save_metrics_to_excel(history, metrics_callback)

# 绘制指标变化曲线 - 修复：确保每个指标使用正确的数据
plt.figure(figsize=(14, 10))

# 定义要绘制的指标和对应的数据来源
metrics_config = [
    ('Training Accuracy', history.history.get('accuracy', [])),
    ('Validation Accuracy', history.history.get('val_accuracy', [])),
    ('Precision', metrics_callback.precision),
    ('Recall', metrics_callback.recall),
    ('F1 Score', metrics_callback.f1)
]

# 调整子图布局（2行3列，最后一个子图留空）
for i, (metric_name, values) in enumerate(metrics_config):
    plt.subplot(2, 3, i + 1)
    # 确保values长度和epochs一致
    epochs = range(1, len(values) + 1)
    values = values[:len(epochs)]

    # 绘制原始值
    plt.plot(epochs, values, label=metric_name, color='blue', alpha=0.7)

    # 绘制平滑值
    if len(values) >= 15:
        smoothed = pd.Series(values).rolling(window=15, center=True).mean()
        plt.plot(epochs, smoothed, label=f'Smoothed {metric_name}', linestyle='--', color='orange')

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Epochs')
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 打印最终指标汇总
print(f"\n最终评估指标:")
print(f"精确率 (Precision): {final_precision:.4f}")
print(f"召回率 (Recall): {final_recall:.4f}")
print(f"F1分数 (F1 Score): {final_f1:.4f}")
print(f"验证集准确率 (Val Accuracy): {history.history['val_accuracy'][-1]:.4f}")