import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import warnings

warnings.filterwarnings('ignore')
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


# -------------------------- 1. 专利步骤2：数据预处理（无效筛选+标准化+窗口分割） --------------------------
def preprocess_data_patent(row, L_min=18, W=35, sample_interval=25):
    """
    功能：实现专利无效数据筛选（长度+缺失率）、min-max标准化、滑动窗口分割
    参数：
        L_min: 有效序列最短采样次数（18次=7.5分钟，专利定义）
        W: 窗口大小（35次=14.6分钟，专利定义）
        sample_interval: 采样间隔（25秒，专利定义）
    返回：有效窗口列表、标签列表、处理状态
    """
    # 1.1 功率数据提取与清洗（过滤0值与超量程数据，适配0-550W传感器）
    charging_powers_str = row['charging_powers_str']
    charging_powers = []
    if isinstance(charging_powers_str, str):
        power_str_list = charging_powers_str.replace('"', '').strip().split(',')
        for power_str in power_str_list:
            try:
                power = float(power_str)
                if 0 < power <= 550:  # 过滤无效功率值
                    charging_powers.append(power)
            except ValueError:
                print(f"跳过无效功率数据: {power_str}")

    # 1.2 无效数据筛选（专利步骤2.1）
    # 序列长度判定：小于18次采样视为无效
    if len(charging_powers) < L_min:
        return None, None, "无效（序列过短）"
    # 缺失率判定：基于充电时长计算，缺失率≥28%视为无效
    try:
        charging_duration = float(row.get('charging_duration_seconds', 0))
        theoretical_len = max(1, int(charging_duration / sample_interval))  # 理论采样数
        missing_rate = 1 - len(charging_powers) / theoretical_len
    except:
        missing_rate = 0.3  # 无法计算时长时默认高缺失率
    if missing_rate >= 0.28:
        return None, None, "无效（缺失率过高）"

    # 1.3 min-max标准化（专利步骤2.3，映射到[0,1]区间）
    min_p, max_p = np.min(charging_powers), np.max(charging_powers)
    if max_p - min_p < 1e-6:  # 避免功率恒定时除以0
        charging_powers_norm = [0.5] * len(charging_powers)
    else:
        charging_powers_norm = [(p - min_p) / (max_p - min_p) for p in charging_powers]

    # 1.4 滑动窗口分割（专利步骤2.4，步长=1）
    windows = []
    for i in range(len(charging_powers_norm) - W + 1):
        windows.append(charging_powers_norm[i:i + W])

    # 1.5 标签映射（专利步骤2.5：0=正常，1=充电器故障，2=电池异常）
    label_map = {'正常': 0, '充电器问题': 1, '电池问题': 2}
    raw_label = str(row['InsertedColumn']).strip()
    label = label_map.get(raw_label, -1)
    if label == -1:
        return None, None, f"无效（未定义标签: {raw_label}）"

    return windows, [label] * len(windows), "有效"


# -------------------------- 2. 专利步骤3.1：自主设计功率时序趋势特征编码器 --------------------------
def build_trend_encoder(W=35):
    """
    功能：构建专利定义的四层架构编码器（时序平滑→趋势提取→特征压缩→异常量化）
    输入：(None, W, 1) 单功率时序窗口
    输出1：(None, 1) 趋势异常评分（0-1，越高越异常）
    输出2：(None, 12) 核心趋势特征（用于计算正常基准）
    """
    # 输入层：单功率时序窗口
    inputs = Input(shape=(W, 1), name='power_sequence_input')

    # 2.1 时序平滑层（专利定义：1D平均池化，核5，步1）
    smooth_layer = AveragePooling1D(
        pool_size=5,
        strides=1,
        padding='same',
        name='time_smoothing_layer'
    )(inputs)
    # 作用：过滤±5W内的正常短期波动，保留长期趋势

    # 2.2 趋势特征提取层（专利定义：1D卷积，16核，核7，LeakyReLU）
    conv_layer = Conv1D(
        filters=16,
        kernel_size=7,
        strides=1,
        padding='same',
        activation=lambda x: K.relu(x, alpha=0.01),  # LeakyReLU(α=0.01)
        name='trend_conv_layer'
    )(smooth_layer)
    conv_layer = Dropout(0.1, name='conv_dropout')(conv_layer)  # 正则化防止过拟合
    # 作用：捕捉175秒（7个采样点）内的中期趋势片段

    # 2.3 特征压缩层（专利定义：全连接，12神经元，LeakyReLU）
    flatten_layer = Flatten(name='flatten_layer')(conv_layer)  # 展平为35*16=560维
    compress_layer = Dense(
        12,
        activation=lambda x: K.relu(x, alpha=0.01),
        name='feature_compress_layer'
    )(flatten_layer)
    # 作用：压缩冗余特征，保留“趋势速率、稳定性”等核心信息

    # 2.4 异常量化层（专利定义：欧氏距离+Sigmoid评分）
    # 正常趋势特征基准（训练时动态更新）
    norm_feature = K.variable(np.zeros(12), name='normal_trend_feature')
    # 计算核心特征与正常基准的欧氏距离
    dist_layer = Lambda(
        lambda x: K.sqrt(K.sum(K.square(x - norm_feature), axis=1, keepdims=True)),
        name='euclidean_distance_layer'
    )(compress_layer)
    # Sigmoid标准化评分（专利参数：a=2.5, b=-0.6）
    score_layer = Lambda(
        lambda x: K.sigmoid(2.5 * x - 0.6),
        name='trend_anomaly_score_layer'
    )(dist_layer)

    # 构建多输出模型
    model = Model(inputs=inputs, outputs=[score_layer, compress_layer], name='power_trend_encoder')
    return model, norm_feature


# -------------------------- 3. 专利步骤3.1.2：趋势偏差损失函数 --------------------------
def trend_loss(w1=0.7, w2=0.3):
    """
    功能：实现专利定义的损失函数（w1*评分偏差损失 + w2*特征偏差损失）
    参数：
        w1: 评分偏差权重（0.7，确保正常样本评分趋近0）
        w2: 特征偏差权重（0.3，确保正常样本特征趋近基准）
    """

    def loss_fn(y_true, y_pred):
        # y_pred[0]：趋势评分，y_pred[1]：核心特征
        score_pred, feat_pred = y_pred[0], y_pred[1]
        # 评分偏差损失（MSE）
        score_loss = K.mean(K.square(score_pred - y_true))
        # 特征偏差损失（MAE，正常基准从模型中获取）
        norm_feat = K.get_value(score_pred._keras_history.model.get_layer('normal_trend_feature').variables[0])
        feat_loss = K.mean(K.abs(feat_pred - norm_feat))
        # 总损失
        return w1 * score_loss + w2 * feat_loss

    return loss_fn


# -------------------------- 4. 专利步骤3.2：孤立森林（突变异常检测） --------------------------
def train_isolation_forest(X_normal_flat):
    """
    功能：训练专利定义的孤立森林模型（检测充电器故障的功率突变）
    参数：X_normal_flat：正常样本特征（展平为2D：(样本数, 35)）
    返回：训练好的孤立森林模型
    """
    if_model = IsolationForest(
        n_estimators=120,  # 专利定义：120棵树
        contamination=0.06,  # 专利定义：异常占比6%
        random_state=36,  # 固定随机种子确保可复现
        max_samples=256,  # 专利定义：每棵树采样256个样本
        verbose=0
    )
    if_model.fit(X_normal_flat)
    return if_model


def calculate_if_score(if_model, X):
    """
    功能：计算专利定义的孤立森林异常评分（0-1，越高越异常）
    """
    # decision_function输出[-1,1]，转换为[0,1]区间
    if_decision = if_model.decision_function(X)
    if_score = (1 - if_decision) / 2  # 反转：原1=正常→0，原-1=异常→1
    return if_score.reshape(-1, 1)


# -------------------------- 5. 专利步骤4：自适应阈值学习模块 --------------------------
class AdaptiveThreshold:
    def __init__(self, init_Th=None):
        """
        功能：实现专利定义的动态阈值学习（基于异常占比调整）
        参数：init_Th：初始阈值（默认基于正常样本94%置信区间）
        """
        self.Th = init_Th if init_Th is not None else 0.2  # 默认初始阈值
        self.R_max = 0.06  # 专利定义：异常占比上限6%
        self.R_min = 0.015  # 专利定义：异常占比下限1.5%
        self.beta = 0.045  # 专利定义：调整步长4.5%
        self.window_size = 12  # 专利定义：基于最近12个窗口
        self.score_history = []  # 存储最近12个融合评分

    def update(self, current_total_score):
        """
        功能：更新阈值（输入当前窗口融合评分，返回更新后阈值）
        """
        # 1. 维护评分历史
        self.score_history.append(current_total_score)
        if len(self.score_history) > self.window_size:
            self.score_history.pop(0)
        # 2. 历史不足时不更新
        if len(self.score_history) < self.window_size:
            return self.Th
        # 3. 计算异常占比R
        R = sum(1 for s in self.score_history if s > self.Th) / self.window_size
        # 4. 调整阈值
        if R > self.R_max:
            new_Th = self.Th * (1 - self.beta)
        elif R < self.R_min:
            new_Th = self.Th * (1 + self.beta)
        else:
            new_Th = self.Th
        # 5. 阈值范围约束（0.05-0.8，避免极端值）
        self.Th = max(0.05, min(0.8, new_Th))
        return self.Th


# -------------------------- 6. 专利适配：训练回调（更新正常趋势特征基准） --------------------------
class NormalFeatureUpdateCallback(Callback):
    def __init__(self, norm_feature, X_normal, update_epoch=10):
        """
        功能：每10轮更新正常趋势特征基准（专利步骤3.1.2）
        参数：
            norm_feature：模型中的正常基准变量
            X_normal：正常样本数据
            update_epoch：更新间隔（10轮，专利建议）
        """
        super().__init__()
        self.norm_feature = norm_feature
        self.X_normal = X_normal
        self.update_epoch = update_epoch

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.update_epoch == 0:
            # 提取所有正常样本的核心特征
            _, feat_output = self.model.predict(self.X_normal, verbose=0)
            # 更新基准为特征均值
            new_norm_feat = np.mean(feat_output, axis=0)
            K.set_value(self.norm_feature, new_norm_feat)
            print(f"\nEpoch {epoch + 1}: 正常趋势特征基准更新（前3维：{new_norm_feat[:3].round(4)}）")


# -------------------------- 7. 专利步骤5：故障诊断分类逻辑 --------------------------
def diagnose_fault(trend_score, if_score, total_score, current_Th):
    """
    功能：实现专利定义的故障分类逻辑（正常→异常→细分类型）
    参数：
        trend_score：编码器趋势评分（电池异常主导）
        if_score：孤立森林突变评分（充电器故障主导）
        total_score：融合评分（0.52*trend + 0.48*if，专利权重）
        current_Th：动态正常/异常阈值
    返回：故障类型（0=正常，1=充电器故障，2=电池异常）、诊断结果
    """
    # 正常/异常判定
    if total_score <= current_Th:
        return 0, "正常充电（功率趋势稳定，无突变）"
    # 异常细分（专利阈值Th_type=0.53）
    Th_type = 0.53
    if total_score > Th_type:
        if trend_score > if_score:
            return 2, "电池异常（功率缓慢衰减，符合电池老化特征）"
        else:
            return 1, "充电器故障（功率突然波动，符合接触不良/电压不稳特征）"
    else:
        # 轻度异常修正（专利步骤5.4）
        if (if_score - trend_score) > 0.1:
            return 1, "充电器故障（轻度接触不良，功率小幅波动）"
        else:
            return 2, "电池异常（轻度容量衰减，功率缓慢下降）"


# -------------------------- 8. 专利步骤6：自定义指标回调（保留原功能） --------------------------
class MetricsCallback(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.precision = []
        self.recall = []
        self.f1 = []
        self.val_data = val_data  # (X_val, y_val)
        self.epoch_metrics = {}  # 存储每个epoch的完整指标

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_data
        # 预测编码器评分（仅用趋势评分评估电池异常检测效果）
        trend_score_val, _ = self.model.predict(X_val, verbose=0)
        # 二分类评估（0=正常，1=异常）
        y_val_binary = np.where(y_val == 0, 0, 1)  # 正常=0，异常（1+2）=1
        trend_score_binary = np.where(trend_score_val > 0.5, 1, 0)  # 评分>0.5视为异常
        # 计算精确率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_binary, trend_score_binary, average='weighted', zero_division=1
        )
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)
        # 记录完整指标
        self.epoch_metrics[epoch] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'val_loss': logs.get('val_loss', 0),
            'val_mae': logs.get('val_mae', 0)
        }


# -------------------------- 9. 专利步骤7：指标保存与可视化（保留原功能） --------------------------
def save_metrics_to_excel(history, metrics_callback, smoothing_window=15):
    """
    功能：保存训练指标到Excel（含平滑处理，保留原代码实用功能）
    参数：
        history：模型训练历史
        metrics_callback：自定义指标回调实例
        smoothing_window：平滑窗口大小（15，原代码参数）
    """
    # 准备epoch序列
    epochs = range(1, len(history.history['loss']) + 1)

    # 构建指标DataFrame
    df_metrics = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': history.history['loss'],
        'Val Loss': history.history['val_loss'],
        'Val MAE': history.history['val_mae'],
        'Precision': metrics_callback.precision,
        'Recall': metrics_callback.recall,
        'F1 Score': metrics_callback.f1
    })

    # 计算平滑指标（减少波动，便于分析）
    for col in ['Train Loss', 'Val Loss', 'Val MAE', 'Precision', 'Recall', 'F1 Score']:
        df_metrics[f'Smoothed {col}'] = df_metrics[col].rolling(window=smoothing_window, min_periods=1).mean()

    # 写入Excel
    wb = Workbook()
    ws = wb.active
    ws.title = 'Training Metrics'
    # 写入表头
    ws.append(df_metrics.columns.tolist())
    # 写入数据行
    for _, row in df_metrics.iterrows():
        ws.append(row.round(4).tolist())  # 保留4位小数，提升可读性
    wb.save('patent_model_metrics.xlsx')
    print(f"\n指标已保存到：patent_model_metrics.xlsx")
    return df_metrics


def plot_training_metrics(df_metrics):
    """
    功能：绘制训练指标可视化图表（保留原代码2x2子图布局）
    参数：df_metrics：含平滑指标的DataFrame
    """
    plt.rcParams['font.sans-serif'] = ['Arial']  # 解决中文显示问题
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Patent-Matched Model Training Metrics', fontsize=16, fontweight='bold')

    # 1. 损失曲线（左上）
    axes[0, 0].plot(df_metrics['Epoch'], df_metrics['Train Loss'], label='Train Loss', color='#1f77b4')
    axes[0, 0].plot(df_metrics['Epoch'], df_metrics['Val Loss'], label='Val Loss', color='#ff7f0e')
    axes[0, 0].plot(df_metrics['Epoch'], df_metrics['Smoothed Val Loss'],
                    label='Smoothed Val Loss', linestyle='--', color='#ff7f0e')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. 验证MAE曲线（右上）
    axes[0, 1].plot(df_metrics['Epoch'], df_metrics['Val MAE'], label='Val MAE', color='#2ca02c')
    axes[0, 1].plot(df_metrics['Epoch'], df_metrics['Smoothed Val MAE'],
                    label='Smoothed Val MAE', linestyle='--', color='#2ca02c')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. 精确率曲线（左下）
    axes[1, 0].plot(df_metrics['Epoch'], df_metrics['Precision'], label='Precision', color='#d62728')
    axes[1, 0].plot(df_metrics['Epoch'], df_metrics['Smoothed Precision'],
                    label='Smoothed Precision', linestyle='--', color='#d62728')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. F1分数曲线（右下）
    axes[1, 1].plot(df_metrics['Epoch'], df_metrics['F1 Score'], label='F1 Score', color='#9467bd')
    axes[1, 1].plot(df_metrics['Epoch'], df_metrics['Smoothed F1 Score'],
                    label='Smoothed F1 Score', linestyle='--', color='#9467bd')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('patent_model_metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"指标图表已保存到：patent_model_metrics_plot.png")


# -------------------------- 10. 主流程：数据加载→模型训练→故障诊断→结果评估（核心） --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("专利适配：电动自行车充电故障诊断系统（完整流程）")
    print("=" * 60)

    # -------------------------- 10.1 数据加载与预处理（专利步骤2） --------------------------
    print("\n【步骤1/6】数据加载与预处理...")
    # 读取数据（需确保CSV包含：charging_powers_str, charging_duration_seconds, InsertedColumn）
    try:
        data = pd.read_csv('最新多.csv', delimiter=' ', encoding='utf-8')
        print(f"成功加载原始数据：{len(data)}条记录")
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        print("请检查CSV文件路径、分隔符（当前为空格）及列名是否正确")
        exit()

    # 预处理：生成有效窗口与标签
    all_windows = []
    all_labels = []
    invalid_log = []
    for idx, row in data.iterrows():
        windows, labels, status = preprocess_data_patent(row)
        if windows is not None:
            all_windows.extend(windows)
            all_labels.extend(labels)
        else:
            invalid_log.append(f"第{idx}行：{status}")

    # 数据有效性检查
    if len(all_windows) == 0:
        print("错误：无有效数据，无法继续训练")
        print("无效数据详情：", invalid_log[:10])  # 打印前10条无效原因
        exit()

    # 转换为模型输入格式
    X = np.array(all_windows).reshape(-1, 35, 1)  # (样本数, 窗口大小35, 1通道)
    y = np.array(all_labels)
    print(
        f"预处理完成：有效窗口{len(X)}个，标签分布→正常(0):{sum(y == 0)}, 充电器故障(1):{sum(y == 1)}, 电池异常(2):{sum(y == 2)}")
    if invalid_log:
        print(f"无效数据{len(invalid_log)}条（示例：{invalid_log[:3]}）")

    # -------------------------- 10.2 数据划分（专利要求：仅用正常样本训练双模型） --------------------------
    print("\n【步骤2/6】数据划分...")
    # 分离正常样本（用于训练编码器和孤立森林）
    normal_mask = y == 0
    X_normal = X[normal_mask]
    if len(X_normal) < 100:  # 确保正常样本数量足够训练
        print(f"警告：正常样本仅{len(X_normal)}个，可能影响模型性能（建议≥100个）")

    # 正常样本划分：训练集80%，验证集20%
    X_norm_train, X_norm_val, _, _ = train_test_split(
        X_normal, X_normal, test_size=0.2, random_state=42, shuffle=True
    )
    # 全量数据划分：训练集80%，测试集20%（用于最终故障诊断评估）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y  # 分层抽样保持标签分布
    )
    print(f"正常样本划分→训练：{len(X_norm_train)}个，验证：{len(X_norm_val)}个")
    print(f"全量数据划分→训练：{len(X_train)}个，测试：{len(X_test)}个")

    # 准备孤立森林输入（展平为2D：(样本数, 35)）
    X_norm_train_flat = X_norm_train.reshape(-1, 35)
    X_test_flat = X_test.reshape(-1, 35)

    # -------------------------- 10.3 训练趋势编码器（专利步骤3.1） --------------------------
    print("\n【步骤3/6】训练功率时序趋势特征编码器...")
    # 构建编码器
    encoder, norm_feature = build_trend_encoder(W=35)
    # 初始化正常趋势特征基准（用前50个正常样本的特征均值）
    init_feat = encoder.predict(X_norm_train[:50], verbose=0)[1]
    K.set_value(norm_feature, np.mean(init_feat, axis=0))

    # 配置训练参数（专利步骤3.1.2）
    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)  # 带权重衰减的优化器
    encoder.compile(
        optimizer=optimizer,
        loss=trend_loss(w1=0.7, w2=0.3),  # 专利定义的趋势偏差损失
        metrics=['mae']  # 监控平均绝对误差
    )

    # 训练回调（早停+学习率衰减+正常基准更新）
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),  # 早停防止过拟合
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=2.5e-4, verbose=1),  # 学习率衰减
        NormalFeatureUpdateCallback(norm_feature, X_norm_train, update_epoch=10),  # 正常基准每10轮更新
        MetricsCallback(val_data=(X_norm_val, y[normal_mask][len(X_norm_train):]))  # 自定义指标监控
    ]

    # 训练编码器（仅用正常样本，标签为0：正常样本评分目标趋近0）
    train_history = encoder.fit(
        X_norm_train,
        [np.zeros(len(X_norm_train)), np.zeros((len(X_norm_train), 12))],  # 多输出标签：评分目标0，特征占位
        validation_data=(
            X_norm_val,
            [np.zeros(len(X_norm_val)), np.zeros((len(X_norm_val), 12))]
        ),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=callbacks
    )

    # 保存编码器与正常基准
    encoder.save('patent_trend_encoder.h5')
    joblib.dump(K.get_value(norm_feature), 'normal_trend_feature_norm.pkl')
    print(f"编码器保存完成→模型：patent_trend_encoder.h5，正常基准：normal_trend_feature_norm.pkl")

    # -------------------------- 10.4 训练孤立森林（专利步骤3.2） --------------------------
    print("\n【步骤4/6】训练孤立森林模型...")
    if_model = train_isolation_forest(X_norm_train_flat)
    # 保存孤立森林
    joblib.dump(if_model, 'patent_isolation_forest.pkl')
    print(f"孤立森林保存完成→模型：patent_isolation_forest.pkl")

    # -------------------------- 10.5 故障诊断（专利步骤5：双模型融合+动态阈值） --------------------------
    print("\n【步骤5/6】故障诊断（双模型融合+动态阈值）...")
    # 加载模型（实际部署时可直接加载，无需重复训练）
    # encoder = load_model('patent_trend_encoder.h5', custom_objects={'loss_fn': trend_loss(), 'K': K})
    # if_model = joblib.load('patent_isolation_forest.pkl')

    # 1. 计算测试集评分
    # 编码器：趋势评分（电池异常主导）
    trend_score_test, _ = encoder.predict(X_test, verbose=0)
    # 孤立森林：突变评分（充电器故障主导）
    if_score_test = calculate_if_score(if_model, X_test_flat)
    # 融合评分（专利权重：0.52*趋势 + 0.48*突变）
    total_score_test = 0.52 * trend_score_test + 0.48 * if_score_test

    # 2. 动态阈值学习（专利步骤4）
    adaptive_th = AdaptiveThreshold(init_Th=0.2)  # 初始阈值0.2（专利默认）
    th_history = []  # 记录阈值更新过程
    y_pred_test = []  # 记录诊断结果
    diag_results = []  # 记录诊断描述

    for i in range(len(total_score_test)):
        # 更新动态阈值
        current_th = adaptive_th.update(total_score_test[i][0])
        th_history.append(current_th)
        # 故障诊断
        pred_label, pred_desc = diagnose_fault(
            trend_score=trend_score_test[i][0],
            if_score=if_score_test[i][0],
            total_score=total_score_test[i][0],
            current_Th=current_th
        )
        y_pred_test.append(pred_label)
        diag_results.append(pred_desc)

    # 转换为numpy数组便于评估
    y_pred_test = np.array(y_pred_test)
    th_history = np.array(th_history)
    print(f"动态阈值范围：{th_history.min():.3f} ~ {th_history.max():.3f}（最终阈值：{th_history[-1]:.3f}）")

    # -------------------------- 10.6 结果评估与保存（保留原代码分类报告功能） --------------------------
    print("\n【步骤6/6】结果评估与保存...")
    # 1. 计算分类评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted', zero_division=1
    )
    accuracy = np.mean(y_test == y_pred_test)
    print(f"\n=== 故障诊断评估结果 ===")
    print(f"准确率（Accuracy）：{accuracy:.4f}")
    print(f"加权精确率（Precision）：{precision:.4f}")
    print(f"加权召回率（Recall）：{recall:.4f}")
    print(f"加权F1分数：{f1:.4f}")

    # 打印详细分类报告（保留原代码格式）
    print(f"\n=== 详细分类报告 ===")
    label_names = ['正常', '充电器故障', '电池异常']
    print(classification_report(
        y_test, y_pred_test,
        labels=[0, 1, 2],
        target_names=label_names,
        zero_division=1
    ))

    # 2. 保存诊断结果到CSV
    df_diag = pd.DataFrame({
        'Test Sample Index': range(len(X_test)),
        'True Label': y_test,
        'Pred Label': y_pred_test,
        'Trend Score': trend_score_test.flatten().round(4),
        'IF Score': if_score_test.flatten().round(4),
        'Total Score': total_score_test.flatten().round(4),
        'Current Threshold': th_history.round(4),
        'Diagnosis Result': diag_results
    })
    df_diag.to_csv('patent_fault_diagnosis_results.csv', index=False, encoding='utf-8-sig')
    print(f"诊断结果保存到：patent_fault_diagnosis_results.csv")

    # 3. 保存训练指标与可视化（补充完整）
    # 从训练回调列表中提取MetricsCallback实例（之前定义的指标监控回调）
    metrics_callback = [cb for cb in callbacks if isinstance(cb, MetricsCallback)][0]
    # 保存指标到Excel
    df_metrics = save_metrics_to_excel(train_history, metrics_callback, smoothing_window=15)
    # 绘制指标可视化图表
    plot_training_metrics(df_metrics)

    # 4. 保存动态阈值历史到CSV（便于分析阈值调整效果）
    df_th_history = pd.DataFrame({
        'Sample Index': range(len(th_history)),
        'Total Score': total_score_test.flatten().round(4),
        'Dynamic Threshold': th_history.round(4),
        'True Label': y_test,
        'Pred Label': y_pred_test
    })
    df_th_history.to_csv('patent_dynamic_threshold_history.csv', index=False, encoding='utf-8-sig')
    print(f"动态阈值历史保存到：patent_dynamic_threshold_history.csv")

    # 5. 打印流程完成提示
    print("\n" + "=" * 60)
    print("专利适配：电动自行车充电故障诊断系统 完整流程执行完成！")
    print("=" * 60)
    print("生成文件清单：")
    print("1. 趋势编码器模型：patent_trend_encoder.h5")
    print("2. 正常趋势特征基准：normal_trend_feature_norm.pkl")
    print("3. 孤立森林模型：patent_isolation_forest.pkl")
    print("4. 故障诊断结果：patent_fault_diagnosis_results.csv")
    print("5. 训练指标数据：patent_model_metrics.xlsx")
    print("6. 指标可视化图表：patent_model_metrics_plot.png")
    print("7. 动态阈值历史：patent_dynamic_threshold_history.csv")
    print("=" * 60)
