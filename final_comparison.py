"""
最终对比实验：5种不确定性量化方法
1. Baseline Point Prediction (无不确定性)
2. Quantile Regression (直接分位数预测)
3. GP Approximation (高斯过程近似)
4. NPKDE (传统核密度估计)
5. Adaptive CAUN (我们的创新方法) ⭐
"""

from TCN import TCNBlock
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 全局参数
window = 40
length_size = 1
batch_size = 32
epochs_baseline = 5  # 基准模型训练轮数
epochs_stage1 = 20    # CAUN第一阶段
epochs_stage2 = 20    # CAUN第二阶段

n_filters = 32
n_filters1 = 50
filter_size = 4
dropout_rate = 0.1


# ==================== 统一编码器 ====================
class TCNBiLSTMEncoder(nn.Module):
    """统一的TCN-BiLSTM编码器"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size, dropout_rate=0.1):
        super().__init__()
        self.tcn = TCNBlock(n_features=n_features, n_filters=n_filters, filter_size=filter_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bilstm = nn.LSTM(
            input_size=n_filters,
            hidden_size=n_filters1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        tcn_out = self.tcn(x)
        tcn_out = self.dropout(tcn_out)
        lstm_out, _ = self.bilstm(tcn_out)
        return lstm_out
    
    def get_last_hidden(self, x):
        lstm_out = self.forward(x)
        return lstm_out[:, -1, :]


# ==================== 方法1: Baseline Point Prediction ====================
class BaselinePointModel(nn.Module):
    """基础点预测模型（无不确定性量化）"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size):
        super().__init__()
        self.encoder = TCNBiLSTMEncoder(n_features, n_filters, n_filters1, filter_size)
        self.fc = nn.Linear(n_filters1 * 2, 1)
        
    def forward(self, x):
        hidden = self.encoder.get_last_hidden(x)
        return self.fc(hidden).squeeze(-1)
    
    def get_features(self, x):
        """供Adaptive CAUN使用"""
        return self.encoder(x)


# ==================== 方法2: Quantile Regression ====================
class QuantileRegressionModel(nn.Module):
    """直接预测三个分位数"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size):
        super().__init__()
        self.encoder = TCNBiLSTMEncoder(n_features, n_filters, n_filters1, filter_size)
        self.fc_lower = nn.Linear(n_filters1 * 2, 1)
        self.fc_median = nn.Linear(n_filters1 * 2, 1)
        self.fc_upper = nn.Linear(n_filters1 * 2, 1)
        
    def forward(self, x):
        hidden = self.encoder.get_last_hidden(x)
        lower = self.fc_lower(hidden).squeeze(-1)
        median = self.fc_median(hidden).squeeze(-1)
        upper = self.fc_upper(hidden).squeeze(-1)
        return torch.stack([lower, median, upper], dim=1)


def quantile_loss(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """分位数损失函数"""
    targets = targets.unsqueeze(1).expand_as(predictions)
    errors = targets - predictions
    
    loss = 0
    for i, q in enumerate(quantiles):
        loss += torch.max(q * errors[:, i], (q - 1) * errors[:, i]).mean()
    
    return loss


# ==================== 方法3: GP Approximation ====================
class GPApproximationModel(nn.Module):
    """高斯过程近似（神经网络实现）"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size):
        super().__init__()
        self.encoder = TCNBiLSTMEncoder(n_features, n_filters, n_filters1, filter_size)
        self.fc_mean = nn.Linear(n_filters1 * 2, 1)
        self.fc_var = nn.Sequential(
            nn.Linear(n_filters1 * 2, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        hidden = self.encoder.get_last_hidden(x)
        mean = self.fc_mean(hidden).squeeze(-1)
        var = self.fc_var(hidden).squeeze(-1)
        return mean, var
    
    def predict_with_uncertainty(self, x):
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(x)
            std = torch.sqrt(var)
            lower = mean - 1.28 * std  # 10%分位数
            upper = mean + 1.28 * std  # 90%分位数
        return lower, mean, upper


def gaussian_nll_loss(pred_mean, pred_var, target):
    """高斯负对数似然损失"""
    return 0.5 * (torch.log(pred_var) + (target - pred_mean)**2 / pred_var).mean()


# ==================== 方法4: NPKDE ====================
class NPKDEUncertainty:
    """非参数核密度估计"""
    def __init__(self, bandwidth=0.05, n_samples=100):
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        self.residual_distribution = None
        
    def fit(self, residuals):
        self.residual_distribution = KernelDensity(
            kernel='gaussian',
            bandwidth=self.bandwidth
        ).fit(residuals.reshape(-1, 1))
        print(f"  NPKDE拟合完成 (带宽={self.bandwidth}, 样本数={len(residuals)})")
        
    def predict_interval(self, point_predictions, confidence=0.8):
        samples = self.residual_distribution.sample(self.n_samples * len(point_predictions))
        samples = samples.reshape(len(point_predictions), self.n_samples)
        
        alpha = (1 - confidence) / 2
        predicted_distributions = point_predictions.reshape(-1, 1) + samples
        
        lower = np.quantile(predicted_distributions, alpha, axis=1)
        upper = np.quantile(predicted_distributions, 1-alpha, axis=1)
        median = np.median(predicted_distributions, axis=1)
        
        return lower, median, upper


# ==================== 方法5: Adaptive CAUN (创新方法) ⭐ ====================
class SelfSupervisedUncertaintyLabeler(nn.Module):
    """
    自监督不确定性标记器
    完全让模型自己学习不确定性模式，不预设类别
    """
    def __init__(self, base_encoder, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = base_encoder
        self.latent_dim = latent_dim
        
        # 特征编码器：时序特征 -> 特征嵌入
        self.feature_projection = nn.Sequential(
            nn.Linear(n_filters1 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 不确定性编码器：特征+误差 -> 潜在表示（learned tags）
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for error
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 标记预测头：从潜在空间输出标记
        self.tag_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # 不确定性预测头：从tag预测误差（用于自监督）
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # 温度参数（用于对比学习）
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, x, errors=None):
        """
        输入：
        - x: 时序特征 [batch, seq_len, features]
        - errors: 预测误差 [batch] (训练时提供，预测时为None)
        
        输出：
        - learned_tags: 学习到的标记 [batch, latent_dim]
        - predicted_uncertainty: 预测的不确定性（用于自监督）
        """
        # 编码特征
        feat_emb = self.encoder.get_features(x)  # [batch, seq_len, n_filters1*2]
        feat_pooled = feat_emb.mean(dim=1)  # [batch, n_filters1*2]
        feat_proj = self.feature_projection(feat_pooled)  # [batch, hidden_dim]
        
        # 编码不确定性模式
        if errors is not None:
            # 训练时：使用实际误差作为自监督信号
            error_expanded = errors.unsqueeze(-1)  # [batch, 1]
            uncertainty_input = torch.cat([feat_proj, error_expanded], dim=-1)
        else:
            # 预测时：只用特征（需要估计误差）
            # 使用特征本身的方差作为误差估计
            error_estimate = feat_emb.std(dim=1).mean(dim=-1).unsqueeze(-1)
            uncertainty_input = torch.cat([feat_proj, error_estimate], dim=-1)
        
        # 生成潜在表示（learned tags）
        latent = self.uncertainty_encoder(uncertainty_input)  # [batch, latent_dim]
        learned_tags = self.tag_predictor(latent)  # [batch, latent_dim]
        
        # 从tag预测不确定性（用于自监督学习）
        predicted_uncertainty = self.uncertainty_predictor(learned_tags).squeeze(-1)
        
        return learned_tags, predicted_uncertainty
    
    def compute_contrastive_loss(self, learned_tags, errors, margin=1.0):
        """
        对比学习损失：相似误差模式 -> 相似tag
        """
        batch_size = learned_tags.size(0)
        
        # 计算误差相似度矩阵
        error_sim = torch.abs(errors.unsqueeze(0) - errors.unsqueeze(1))  # [batch, batch]
        error_sim_normalized = 1.0 / (1.0 + error_sim)  # 转换为相似度 [0, 1]
        
        # 计算tag相似度矩阵（余弦相似度）
        learned_tags_norm = F.normalize(learned_tags, p=2, dim=1)
        tag_sim = torch.mm(learned_tags_norm, learned_tags_norm.t())  # [batch, batch]
        
        # 对比损失：误差相似度应该与tag相似度一致
        contrastive_loss = F.mse_loss(tag_sim, error_sim_normalized)
        
        # 多样性约束：避免所有tag相同
        tag_std = learned_tags.std(dim=0).mean()
        diversity_loss = -tag_std  # 鼓励多样性
        
        return contrastive_loss, diversity_loss
    
    def compute_reconstruction_loss(self, predicted_uncertainty, true_errors):
        """
        重构损失：从tag预测误差
        """
        return F.mse_loss(predicted_uncertainty, true_errors)


class UncertaintyDetector:
    """识别高不确定性区域"""
    def __init__(self, threshold_percentile=70):
        self.threshold_percentile = threshold_percentile
        self.uncertainty_scores = None
        
    def detect(self, y_true, y_pred):
        residuals = np.abs(y_true - y_pred)
        
        # 方法1: 残差大小
        residual_scores = residuals / (np.std(residuals) + 1e-8)
        
        # 方法2: 局部波动性
        window_size = 5
        local_volatility = np.array([
            np.std(residuals[max(0, i-window_size):min(len(residuals), i+window_size)])
            for i in range(len(residuals))
        ])
        volatility_scores = local_volatility / (np.std(local_volatility) + 1e-8)
        
        # 方法3: 预测变化率
        pred_diff = np.abs(np.diff(y_pred, prepend=y_pred[0]))
        pred_diff_scores = pred_diff / (np.std(pred_diff) + 1e-8)
        
        # 综合不确定性分数
        self.uncertainty_scores = (
            0.5 * residual_scores +
            0.3 * volatility_scores +
            0.2 * pred_diff_scores
        )
        
        threshold = np.percentile(self.uncertainty_scores, self.threshold_percentile)
        high_uncertainty_mask = self.uncertainty_scores >= threshold
        
        print(f"  识别高不确定性区域: {np.sum(high_uncertainty_mask)}/{len(y_true)} "
              f"({np.mean(high_uncertainty_mask)*100:.1f}%)")
        
        return high_uncertainty_mask, self.uncertainty_scores


class AdaptiveCAUN(nn.Module):
    """
    Adaptive Cross-Attention Uncertainty Network with Self-Supervised Learned Tags
    创新点：
    1. 基于Transformer Decoder的自适应不确定性量化
    2. 使用自监督学习到的tags（learned tags）而不是固定类别
    3. 基于tag相似度的条件KTO fine-tuning
    """
    def __init__(self, base_encoder, labeler, decoder_dim=100, num_heads=5, num_decoder_layers=2):
        super().__init__()
        
        self.encoder = base_encoder
        self.labeler = labeler  # 自监督标记器
        self.encoder_projection = nn.Linear(n_filters1 * 2, decoder_dim)
        
        # Tag嵌入投影（将learned tags映射到decoder空间）
        self.tag_dim = labeler.latent_dim if labeler is not None else 16
        self.tag_projection = nn.Sequential(
            nn.Linear(self.tag_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 区间宽度调制器（基于learned tags）
        self.width_modulator = nn.Sequential(
            nn.Linear(decoder_dim + self.tag_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Query tokens (3个：lower, median, upper)
        self.query_tokens = nn.Parameter(torch.randn(1, 3, decoder_dim))
        
        # 分位数预测头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(decoder_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, learned_tags=None, errors=None):
        """
        输入：
        - x: 时序特征
        - learned_tags: 学习到的标记 [batch, latent_dim] (可选，如果None则自动生成)
        - errors: 预测误差 [batch] (用于生成tags，可选)
        """
        # 编码
        lstm_out = self.encoder.get_features(x)
        memory = self.encoder_projection(lstm_out)
        
        batch_size = memory.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # 如果有learned tags，将其融合到query中
        if learned_tags is None:
            # 如果没有提供tags，使用labeler生成（预测模式）
            if self.labeler is not None:
                with torch.no_grad():
                    learned_tags, _ = self.labeler(x, errors=errors)
            else:
                # 如果没有labeler，使用零向量
                learned_tags = torch.zeros(batch_size, 16, device=x.device)
        
        # 将tags投影到decoder空间
        tag_proj = self.tag_projection(learned_tags)  # [batch, decoder_dim]
        
        # 将tag信息融合到queries中
        queries = queries + tag_proj.unsqueeze(1)  # [batch, 3, decoder_dim]
        
        # Transformer Decoder解码
        decoder_out = self.transformer_decoder(queries, memory)
        
        # 预测中位数（点预测）
        median_raw = self.uncertainty_head(decoder_out[:, 1, :]).squeeze(-1)
        
        # 基于learned tags动态调整区间宽度
        if learned_tags is not None:
            pooled = memory.mean(dim=1)  # [batch, decoder_dim]
            width_input = torch.cat([pooled, learned_tags], dim=-1)  # [batch, decoder_dim + tag_dim]
            
            # 宽度调制（基于tag的相似度）
            width_scale = self.width_modulator(width_input).squeeze(-1)  # [batch]
            width_scale = 0.5 + 2.5 * width_scale  # 映射到[0.5, 3.0]
            
            # 基础区间宽度（基于中位数的相对值）
            interval_raw = self.uncertainty_head(decoder_out[:, 0, :]).squeeze(-1)
            interval_raw = 0.5 + 1.5 * interval_raw  # 映射到[0.5, 2.0]
            
            # 动态区间宽度：基于中位数和learned tags
            base_std = torch.abs(median_raw) * 0.3
            base_width = base_std * interval_raw * width_scale
            
            lower = median_raw - base_width
            upper = median_raw + base_width
            
            return torch.stack([lower, median_raw, upper], dim=1), learned_tags
        
        # 传统方法：固定区间宽度（如果没有tags）
        lower_raw = self.uncertainty_head(decoder_out[:, 0, :]).squeeze(-1)
        upper_raw = self.uncertainty_head(decoder_out[:, 2, :]).squeeze(-1)
        
        return torch.stack([lower_raw, median_raw, upper_raw], dim=1), None


def analyze_difficulty(data_train, model, device, window):
    """分析训练集的困难度"""
    model.eval()
    X_train, y_train = [], []
    for i in range(len(data_train) - window):
        X_train.append(data_train[i:i + window, :])
        y_train.append(data_train[i + window, -1])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_tensor = torch.FloatTensor(X_train).to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor)
        if isinstance(predictions, tuple):
            pred_values = predictions[0][:, 1].cpu().numpy()
        else:
            pred_values = predictions[:, 1].cpu().numpy()
    
    residuals = np.abs(y_train - pred_values)
    
    # 4维困难度向量
    change_rate = np.zeros(len(y_train))
    for i in range(1, len(y_train)):
        change_rate[i] = abs(y_train[i] - y_train[i-1])
    change_rate = (change_rate - change_rate.mean()) / (change_rate.std() + 1e-8)
    
    peak_threshold = np.percentile(y_train, 90)
    peak_proximity = np.abs(y_train - peak_threshold) / (peak_threshold + 1e-8)
    peak_proximity = 1 - np.clip(peak_proximity, 0, 1)
    
    window_size = 10
    local_volatility = np.zeros(len(y_train))
    for i in range(window_size, len(y_train)):
        local_window = y_train[i-window_size:i]
        local_volatility[i] = np.std(local_window)
    local_volatility = (local_volatility - local_volatility.mean()) / (local_volatility.std() + 1e-8)
    
    hist_error = (residuals - residuals.mean()) / (residuals.std() + 1e-8)
    
    difficulty_vectors = np.stack([change_rate, peak_proximity, local_volatility, hist_error], axis=1)
    difficulty_vectors = np.clip(difficulty_vectors, -3, 3)
    
    difficulty_scores = np.mean(np.abs(difficulty_vectors), axis=1)
    difficulty_scores = (difficulty_scores - difficulty_scores.min()) / (difficulty_scores.max() - difficulty_scores.min() + 1e-8)
    
    return difficulty_vectors, difficulty_scores


def kto_loss_function_with_tags(predictions, targets, learned_tags, errors,
                                 lambda_std=0.7, lambda_tag=0.2, temperature=0.1):
    """
    KTO损失函数（基于learned tags的相似度）
    相似tag的样本使用相似的策略，不同tag的样本使用不同策略
    """
    if isinstance(predictions, tuple):
        predictions_tensor = predictions[0]
    else:
        predictions_tensor = predictions
    
    lower = predictions_tensor[:, 0]
    median = predictions_tensor[:, 1]
    upper = predictions_tensor[:, 2]
    
    # 标准分位数损失
    q_loss = quantile_loss(predictions_tensor, targets)
    
    # 区间合理性约束
    order_loss = F.relu(lower - median).mean() + F.relu(median - upper).mean()
    
    # 基于tag相似度的条件损失
    tag_loss = torch.tensor(0.0).to(targets.device)
    if learned_tags is not None and learned_tags.size(0) > 1:
        # 计算tag相似度矩阵
        learned_tags_norm = F.normalize(learned_tags, p=2, dim=1)
        tag_sim = torch.mm(learned_tags_norm, learned_tags_norm.t())  # [batch, batch]
        
        # 计算误差相似度矩阵
        errors_tensor = errors if isinstance(errors, torch.Tensor) else torch.tensor(errors, device=targets.device)
        error_sim = 1.0 / (1.0 + torch.abs(errors_tensor.unsqueeze(0) - errors_tensor.unsqueeze(1)))
        
        # 计算区间宽度
        width = upper - lower
        width_matrix = torch.abs(width.unsqueeze(0) - width.unsqueeze(1))
        
        # Tag相似度高 -> 宽度应该相似
        # Tag相似度低 -> 宽度可以不同
        tag_loss = F.mse_loss(width_matrix, (1.0 - tag_sim) * width_matrix.max())
    
    # 宽度惩罚（基于误差，误差大允许更宽）
    width = upper - lower
    errors_tensor = errors if isinstance(errors, torch.Tensor) else torch.tensor(errors, dtype=torch.float32, device=targets.device)
    normalized_errors = (errors_tensor - errors_tensor.min()) / (errors_tensor.max() - errors_tensor.min() + 1e-8)
    width_penalty = (width * (1 - 0.4 * normalized_errors)).mean()
    
    loss_standard = q_loss + 0.1 * order_loss + 0.01 * width_penalty
    loss_total = lambda_std * loss_standard + lambda_tag * tag_loss
    
    return loss_total, {
        'loss_standard': loss_standard.item(),
        'loss_tag': tag_loss.item() if isinstance(tag_loss, torch.Tensor) else 0,
        'loss_total': loss_total.item()
    }


def adaptive_loss(predictions, targets, uncertainty_scores=None, alpha=0.1):
    """自适应损失函数"""
    if isinstance(predictions, tuple):
        predictions_tensor = predictions[0]
    else:
        predictions_tensor = predictions
    
    q_loss = quantile_loss(predictions_tensor, targets)
    
    # 区间合理性约束
    lower, median, upper = predictions_tensor[:, 0], predictions_tensor[:, 1], predictions_tensor[:, 2]
    order_loss = F.relu(lower - median).mean() + F.relu(median - upper).mean()
    
    # 宽度惩罚
    width = upper - lower
    if uncertainty_scores is not None:
        unc_tensor = torch.tensor(uncertainty_scores, dtype=torch.float32, device=predictions_tensor.device)
        width_penalty = (width * (1 - unc_tensor)).mean()
    else:
        width_penalty = width.mean()
    
    return q_loss + alpha * order_loss + 0.01 * width_penalty


# ==================== 数据加载 ====================
def load_data():
    # 使用power.xlsx（包含Total solar irradiance特征）
    data = pd.read_excel('power.xlsx')
    
    # 提取Total solar irradiance和Power
    irradiance_col = 'Total solar irradiance (W/m2)'
    power_col = data.columns[-1]
    
    # 组合特征：Total solar irradiance + Power（作为历史值）
    data_selected = data[[irradiance_col, power_col]]
    data_selected = data_selected.apply(pd.to_numeric, errors='coerce')
    data_selected = data_selected.fillna(0)
    
    data_dim = len(data_selected.columns)  # 2 features
    
    # 为完整数据创建scaler（用于特征）
    scaler_full = preprocessing.MinMaxScaler()
    data_scaled = scaler_full.fit_transform(data_selected.values)
    
    # 为power列单独创建scaler（用于反标准化预测值）
    scaler_power = preprocessing.MinMaxScaler()
    scaler_power.fit(data_selected[power_col].values.reshape(-1, 1))
    
    data_length = len(data_scaled)
    train_set = 0.975
    
    data_train = data_scaled[:int(train_set * data_length), :]
    data_test = data_scaled[int(train_set * data_length):, :]
    
    print(f"  特征: {irradiance_col} + {power_col}")
    print(f"  特征维度: {data_dim}")
    
    return data_train, data_test, scaler_full, scaler_power, data_dim


def create_dataloader(data, window, length_size, batch_size, data_dim):
    seq_len = window
    sequence_length = seq_len + length_size
    result = []
    
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    result = np.array(result)
    x_data = result[:, :-length_size]
    y_data = result[:, -length_size:, -1]
    
    X = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], data_dim))
    y = np.reshape(y_data, (y_data.shape[0], -1))
    
    X_tensor = torch.tensor(X).to(torch.float32)
    y_tensor = torch.tensor(y).to(torch.float32)
    
    ds = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    return dataloader, X_tensor, y_tensor


# ==================== 评估指标 ====================
def calculate_metrics(y_true, lower, median, upper, target_coverage=0.8):
    """计算综合评估指标"""
    picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    mpiw = np.mean(upper - lower)
    mae = np.mean(np.abs(y_true - median))
    rmse = np.sqrt(np.mean((y_true - median)**2))
    
    # CWC (Coverage Width Criterion)
    if picp < target_coverage * 100:
        penalty = 50 * (target_coverage - picp/100)
        cwc = mpiw * (1 + penalty)
    else:
        cwc = mpiw
    
    # Interval Score
    alpha = 1 - target_coverage
    interval_score = mpiw + (2/alpha) * np.mean(
        np.maximum(0, lower - y_true) + np.maximum(0, y_true - upper)
    )
    
    return {
        'PICP': picp,
        'MPIW': mpiw,
        'MAE': mae,
        'RMSE': rmse,
        'CWC': cwc,
        'IS': interval_score
    }


# ==================== 主函数 ====================
def main():
    print("="*80)
    print("不确定性量化方法最终对比实验")
    print("="*80)
    print("\n对比方法:")
    print("  1. Baseline Point Prediction (无不确定性)")
    print("  2. Quantile Regression (直接分位数预测)")
    print("  3. GP Approximation (高斯过程近似)")
    print("  4. NPKDE (传统核密度估计)")
    print("  5. Adaptive CAUN + Self-Supervised Learned Tags (创新方法) ⭐")
    print("      - 多特征输入 (Total solar irradiance + Power)")
    print("      - 自监督标记器：让模型自动发现不确定性模式")
    print("      - 基于learned tags的条件KTO fine-tuning")
    print("      - 完全无预设类别，模型自己学习标记")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n[数据准备]")
    data_train, data_test, scaler_full, scaler_power, data_dim = load_data()
    scaler = scaler_power  # 用于反标准化预测值
    
    dataloader_train, X_train, y_train = create_dataloader(
        data_train, window, length_size, batch_size, data_dim
    )
    _, X_test, y_test = create_dataloader(
        data_test, window, length_size, batch_size, data_dim
    )
    
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    all_results = {}
    
    # ==================== 方法1: Baseline ====================
    print("\n" + "="*80)
    print("方法1: Baseline Point Prediction")
    print("="*80)
    
    baseline_model = BaselinePointModel(data_dim, n_filters, n_filters1, filter_size).to(device)
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    baseline_model.train()
    for epoch in range(epochs_baseline):
        epoch_loss = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            optimizer.zero_grad()
            pred = baseline_model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs_baseline}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    baseline_model.eval()
    with torch.no_grad():
        baseline_pred = baseline_model(X_test.to(device)).cpu().numpy()
    
    baseline_pred_original = scaler.inverse_transform(baseline_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
    
    baseline_mae = np.mean(np.abs(y_test_original - baseline_pred_original))
    print(f"✓ Baseline MAE: {baseline_mae:.4f}")
    
    all_results['Baseline'] = {
        'median': baseline_pred_original,
        'MAE': baseline_mae
    }
    
    # 保存baseline模型供后续使用
    torch.save(baseline_model.state_dict(), 'checkpoint/baseline_final.pt')
    
    # ==================== 方法2: Quantile Regression ====================
    print("\n" + "="*80)
    print("方法2: Quantile Regression")
    print("="*80)
    
    qr_model = QuantileRegressionModel(data_dim, n_filters, n_filters1, filter_size).to(device)
    optimizer = optim.Adam(qr_model.parameters(), lr=0.001)
    
    qr_model.train()
    for epoch in range(epochs_baseline):
        epoch_loss = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            optimizer.zero_grad()
            pred = qr_model(data)
            loss = quantile_loss(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs_baseline}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    qr_model.eval()
    with torch.no_grad():
        qr_pred = qr_model(X_test.to(device))
    
    qr_lower_original = scaler.inverse_transform(qr_pred[:, 0].cpu().numpy().reshape(-1, 1)).flatten()
    qr_median_original = scaler.inverse_transform(qr_pred[:, 1].cpu().numpy().reshape(-1, 1)).flatten()
    qr_upper_original = scaler.inverse_transform(qr_pred[:, 2].cpu().numpy().reshape(-1, 1)).flatten()
    
    # 应用物理约束
    qr_lower_original = np.maximum(qr_lower_original, 0)
    qr_median_original = np.maximum(qr_median_original, 0)
    qr_upper_original = np.maximum(qr_upper_original, 0)
    
    qr_metrics = calculate_metrics(y_test_original, qr_lower_original, qr_median_original, qr_upper_original)
    print(f"✓ Quantile Regression - PICP: {qr_metrics['PICP']:.2f}%, MPIW: {qr_metrics['MPIW']:.4f}, MAE: {qr_metrics['MAE']:.4f}")
    
    all_results['Quantile_Regression'] = {
        'lower': qr_lower_original,
        'median': qr_median_original,
        'upper': qr_upper_original,
        **qr_metrics
    }
    
    # ==================== 方法3: GP Approximation ====================
    print("\n" + "="*80)
    print("方法3: GP Approximation")
    print("="*80)
    
    gp_model = GPApproximationModel(data_dim, n_filters, n_filters1, filter_size).to(device)
    optimizer = optim.Adam(gp_model.parameters(), lr=0.001)
    
    gp_model.train()
    for epoch in range(epochs_baseline):
        epoch_loss = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            optimizer.zero_grad()
            mean, var = gp_model(data)
            loss = gaussian_nll_loss(mean, var, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs_baseline}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    gp_lower, gp_median, gp_upper = gp_model.predict_with_uncertainty(X_test.to(device))
    
    gp_lower_original = scaler.inverse_transform(gp_lower.cpu().numpy().reshape(-1, 1)).flatten()
    gp_median_original = scaler.inverse_transform(gp_median.cpu().numpy().reshape(-1, 1)).flatten()
    gp_upper_original = scaler.inverse_transform(gp_upper.cpu().numpy().reshape(-1, 1)).flatten()
    
    gp_lower_original = np.maximum(gp_lower_original, 0)
    gp_median_original = np.maximum(gp_median_original, 0)
    gp_upper_original = np.maximum(gp_upper_original, 0)
    
    gp_metrics = calculate_metrics(y_test_original, gp_lower_original, gp_median_original, gp_upper_original)
    print(f"✓ GP Approximation - PICP: {gp_metrics['PICP']:.2f}%, MPIW: {gp_metrics['MPIW']:.4f}, MAE: {gp_metrics['MAE']:.4f}")
    
    all_results['GP_Approximation'] = {
        'lower': gp_lower_original,
        'median': gp_median_original,
        'upper': gp_upper_original,
        **gp_metrics
    }
    
    # ==================== 方法4: NPKDE ====================
    print("\n" + "="*80)
    print("方法4: NPKDE")
    print("="*80)
    
    # 使用baseline模型在训练集上的残差
    baseline_model.eval()
    with torch.no_grad():
        train_pred = baseline_model(X_train.to(device)).cpu().numpy()
    
    train_residuals = y_train.cpu().numpy().flatten() - train_pred
    
    npkde = NPKDEUncertainty(bandwidth=0.05, n_samples=100)
    npkde.fit(train_residuals)
    
    npkde_lower, npkde_median, npkde_upper = npkde.predict_interval(baseline_pred, confidence=0.8)
    
    npkde_lower_original = scaler.inverse_transform(npkde_lower.reshape(-1, 1)).flatten()
    npkde_median_original = scaler.inverse_transform(npkde_median.reshape(-1, 1)).flatten()
    npkde_upper_original = scaler.inverse_transform(npkde_upper.reshape(-1, 1)).flatten()
    
    npkde_lower_original = np.maximum(npkde_lower_original, 0)
    npkde_median_original = np.maximum(npkde_median_original, 0)
    npkde_upper_original = np.maximum(npkde_upper_original, 0)
    
    npkde_metrics = calculate_metrics(y_test_original, npkde_lower_original, npkde_median_original, npkde_upper_original)
    print(f"✓ NPKDE - PICP: {npkde_metrics['PICP']:.2f}%, MPIW: {npkde_metrics['MPIW']:.4f}, MAE: {npkde_metrics['MAE']:.4f}")
    
    all_results['NPKDE'] = {
        'lower': npkde_lower_original,
        'median': npkde_median_original,
        'upper': npkde_upper_original,
        **npkde_metrics
    }
    
    # ==================== 方法5: Adaptive CAUN with Self-Supervised Learned Tags ⭐ ====================
    print("\n" + "="*80)
    print("方法5: Adaptive CAUN + Self-Supervised Learned Tags (完全学习型标记) ⭐")
    print("="*80)
    
    # 阶段1: 基础CAUN训练（无标记器）
    print("\n[阶段1: 基础CAUN训练]")
    adaptive_caun_basic = AdaptiveCAUN(
        base_encoder=baseline_model,
        labeler=None,  # 暂时不使用标记器
        decoder_dim=100,
        num_heads=5,
        num_decoder_layers=2
    ).to(device)
    
    optimizer_caun_basic = optim.Adam(adaptive_caun_basic.parameters(), lr=0.001)
    
    adaptive_caun_basic.train()
    for epoch in range(epochs_stage1):
        epoch_loss = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            optimizer_caun_basic.zero_grad()
            predictions, _ = adaptive_caun_basic(data)
            loss = adaptive_loss(predictions, target)
            loss.backward()
            optimizer_caun_basic.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs_stage1}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    # 阶段2: 自监督标记器训练
    print("\n[阶段2: 自监督标记器训练 - 让模型自动发现不确定性模式]")
    
    # 使用基础CAUN在训练集上生成预测和误差
    adaptive_caun_basic.eval()
    train_predictions, train_errors = [], []
    with torch.no_grad():
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            pred, _ = adaptive_caun_basic(data)
            pred_values = pred[:, 1]  # 中位数
            errors = torch.abs(target - pred_values)
            train_predictions.append(pred_values.cpu())
            train_errors.append(errors.cpu())
    
    # 创建标记器
    labeler = SelfSupervisedUncertaintyLabeler(
        base_encoder=baseline_model,
        latent_dim=16,
        hidden_dim=64
    ).to(device)
    
    optimizer_labeler = optim.Adam(labeler.parameters(), lr=0.001)
    
    # 准备训练数据（包含误差）
    # 先将train_errors展平
    all_train_errors = torch.cat(train_errors, dim=0).numpy()
    
    X_train_list, y_train_list, error_list = [], [], []
    for i in range(len(data_train) - window):
        X_train_list.append(data_train[i:i + window, :])
        y_train_list.append(data_train[i + window, -1])
        # 使用对应的误差值
        if i < len(all_train_errors):
            error_list.append(float(all_train_errors[i]))
        else:
            error_list.append(0.01)
    
    X_train_tag = torch.FloatTensor(np.array(X_train_list)).to(device)
    y_train_tag = torch.FloatTensor(np.array(y_train_list)).to(device)
    errors_train_tag = torch.FloatTensor(np.array(error_list)).to(device)
    
    train_dataset_tag = TensorDataset(X_train_tag, y_train_tag, errors_train_tag)
    train_loader_tag = DataLoader(train_dataset_tag, batch_size=batch_size, shuffle=True)
    
    # 训练标记器
    labeler.train()
    epochs_labeler = 5
    for epoch in range(epochs_labeler):
        epoch_loss = 0
        for X_batch, y_batch, errors_batch in train_loader_tag:
            X_batch = X_batch.to(device)
            errors_batch = errors_batch.to(device)
            
            optimizer_labeler.zero_grad()
            learned_tags, predicted_uncertainty = labeler(X_batch, errors=errors_batch)
            
            # 自监督损失
            recon_loss = labeler.compute_reconstruction_loss(predicted_uncertainty, errors_batch)
            contrastive_loss, diversity_loss = labeler.compute_contrastive_loss(learned_tags, errors_batch)
            
            total_loss = recon_loss + 0.5 * contrastive_loss + 0.1 * diversity_loss
            total_loss.backward()
            optimizer_labeler.step()
            epoch_loss += total_loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs_labeler}, Loss: {epoch_loss/len(train_loader_tag):.6f}")
    
    print("  标记器训练完成：模型已自动学习不确定性模式")
    
    # 阶段3: 使用标记器的Adaptive CAUN训练
    print("\n[阶段3: 基于Learned Tags的条件KTO Fine-tuning]")
    
    # 创建带标记器的Adaptive CAUN
    adaptive_caun = AdaptiveCAUN(
        base_encoder=baseline_model,
        labeler=labeler,
        decoder_dim=100,
        num_heads=5,
        num_decoder_layers=2
    ).to(device)
    
    # 复制基础CAUN的参数（除了标记器相关部分）
    adaptive_caun.encoder.load_state_dict(adaptive_caun_basic.encoder.state_dict())
    adaptive_caun.encoder_projection.load_state_dict(adaptive_caun_basic.encoder_projection.state_dict())
    adaptive_caun.transformer_decoder.load_state_dict(adaptive_caun_basic.transformer_decoder.state_dict())
    adaptive_caun.query_tokens.data = adaptive_caun_basic.query_tokens.data
    adaptive_caun.uncertainty_head.load_state_dict(adaptive_caun_basic.uncertainty_head.state_dict())
    
    optimizer_caun = optim.Adam(adaptive_caun.parameters(), lr=0.0005)
    
    adaptive_caun.train()
    labeler.eval()  # 标记器冻结
    
    for epoch in range(epochs_stage2):
        epoch_loss = 0
        for X_batch, y_batch, errors_batch in train_loader_tag:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            errors_batch = errors_batch.to(device)
            
            optimizer_caun.zero_grad()
            
            # 先进行一次预测以获取当前误差
            with torch.no_grad():
                temp_predictions, _ = adaptive_caun(X_batch, learned_tags=None)
                point_pred_temp = temp_predictions[:, 1]
                current_errors = torch.abs(y_batch - point_pred_temp)
            
            # 使用当前误差生成learned tags
            with torch.no_grad():
                learned_tags, _ = labeler(X_batch, errors=current_errors)
            
            # 使用learned tags进行预测
            predictions, output_tags = adaptive_caun(X_batch, learned_tags=learned_tags)
            
            # 重新计算误差（用于损失）
            point_pred = predictions[:, 1]
            final_errors = torch.abs(y_batch - point_pred).detach()
            
            # KTO损失（基于learned tags）
            loss, loss_dict = kto_loss_function_with_tags(
                predictions, y_batch, learned_tags, final_errors.cpu().numpy()
            )
            loss.backward()
            optimizer_caun.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs_stage2}, Loss: {epoch_loss/len(train_loader_tag):.6f}")
    
    torch.save(adaptive_caun.state_dict(), 'checkpoint/adaptive_caun_learned_tags.pt')
    torch.save(labeler.state_dict(), 'checkpoint/uncertainty_labeler.pt')
    
    # 测试Adaptive CAUN with Learned Tags
    print("\n[测试阶段]")
    
    adaptive_caun.eval()
    labeler.eval()
    
    test_predictions_list = []
    test_tags_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size].to(device)
            
            # 生成learned tags（预测模式，不需要真实误差）
            learned_tags, _ = labeler(X_batch, errors=None)
            
            # 预测
            predictions, _ = adaptive_caun(X_batch, learned_tags=learned_tags)
            
            test_predictions_list.append(predictions.cpu())
            test_tags_list.append(learned_tags.cpu())
    
    caun_predictions = torch.cat(test_predictions_list, dim=0)
    caun_tags = torch.cat(test_tags_list, dim=0)
    
    caun_lower = scaler.inverse_transform(caun_predictions[:, 0].cpu().numpy().reshape(-1, 1)).flatten()
    caun_median = scaler.inverse_transform(caun_predictions[:, 1].cpu().numpy().reshape(-1, 1)).flatten()
    caun_upper = scaler.inverse_transform(caun_predictions[:, 2].cpu().numpy().reshape(-1, 1)).flatten()
    
    caun_lower = np.maximum(caun_lower, 0)
    caun_median = np.maximum(caun_median, 0)
    caun_upper = np.maximum(caun_upper, 0)
    
    caun_metrics = calculate_metrics(y_test_original, caun_lower, caun_median, caun_upper)
    
    # 分析learned tags的多样性
    tag_diversity = caun_tags.std(dim=0).mean().item()
    print(f"✓ Adaptive CAUN + Learned Tags - PICP: {caun_metrics['PICP']:.2f}%, "
          f"MPIW: {caun_metrics['MPIW']:.4f}, MAE: {caun_metrics['MAE']:.4f}")
    print(f"  Learned Tags多样性: {tag_diversity:.4f} (越高表示模式越丰富)")
    
    all_results['Adaptive_CAUN_Learned_Tags'] = {
        'lower': caun_lower,
        'median': caun_median,
        'upper': caun_upper,
        'learned_tags': caun_tags.numpy(),
        **caun_metrics
    }
    
    # ==================== 保存结果 ====================
    print("\n" + "="*80)
    print("保存对比结果")
    print("="*80)
    
    # 保存指标对比（基于全部测试集）
    metrics_comparison = []
    for method_name, results in all_results.items():
        if 'PICP' in results:
            row = {'Method': method_name}
            row.update({k: v for k, v in results.items() 
                       if k not in ['lower', 'median', 'upper', 'risk_levels', 'learned_tags']})
            metrics_comparison.append(row)
    
    metrics_df = pd.DataFrame(metrics_comparison)
    metrics_df.to_csv('final_methods_comparison.csv', index=False)
    print("✓ 指标对比: final_methods_comparison.csv")
    
    # 只保存测试集50-150点的数据（用于可视化）
    start_idx, end_idx = 50, 150
    if len(y_test_original) < end_idx:
        end_idx = len(y_test_original)
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = min(100, len(y_test_original))
    
    print(f"\n保存测试集[{start_idx}:{end_idx}]点的数据（共{end_idx-start_idx}个点）")
    
    results_dict = {'True_Value': y_test_original[start_idx:end_idx]}
    for method_name, results in all_results.items():
        if 'median' in results:
            results_dict[f'{method_name}_Median'] = results['median'][start_idx:end_idx]
        if 'lower' in results:
            results_dict[f'{method_name}_Lower'] = results['lower'][start_idx:end_idx]
            results_dict[f'{method_name}_Upper'] = results['upper'][start_idx:end_idx]
        if 'risk_levels' in results:
            results_dict[f'{method_name}_RiskLevel'] = results['risk_levels'][start_idx:end_idx]
    
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv('final_methods_predictions_50_150.csv', index=False)
    print("✓ 预测结果（50-150点）: final_methods_predictions_50_150.csv")
    
    # 绘制可视化图像（50-150点）
    print("\n生成可视化图像（测试集50-150点）...")
    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 4 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]
    
    x_range = np.arange(start_idx, end_idx)
    
    for idx, (method_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        true_values = y_test_original[start_idx:end_idx]
        
        if 'median' in results:
            median = results['median'][start_idx:end_idx]
            ax.plot(x_range, true_values, 'k-', label='True Value', linewidth=2)
            ax.plot(x_range, median, 'b--', label='Predicted Median', linewidth=1.5)
            
            if 'lower' in results and 'upper' in results:
                lower = results['lower'][start_idx:end_idx]
                upper = results['upper'][start_idx:end_idx]
                ax.fill_between(x_range, lower, upper, alpha=0.3, color='blue', label='80% Prediction Interval')
            
            ax.set_xlabel('Test Sample Index', fontsize=12)
            ax.set_ylabel('Power (kW)', fontsize=12)
            ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            # Baseline点预测
            predicted = results['median'][start_idx:end_idx]
            ax.plot(x_range, true_values, 'k-', label='True Value', linewidth=2)
            ax.plot(x_range, predicted, 'r--', label='Predicted', linewidth=1.5)
            ax.set_xlabel('Test Sample Index', fontsize=12)
            ax.set_ylabel('Power (kW)', fontsize=12)
            ax.set_title(f'{method_name} (Point Prediction)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_comparison_50_150.png', dpi=300, bbox_inches='tight')
    print("✓ 可视化图像: final_comparison_50_150.png")
    
    # ==================== 结果展示 ====================
    print("\n" + "="*80)
    print("实验结果总结")
    print("="*80)
    
    print("\n【按CWC排名 (越小越好)】:")
    metrics_df_sorted = metrics_df.sort_values('CWC')
    for idx, row in enumerate(metrics_df_sorted.itertuples(), 1):
        symbol = "⭐" if 'Learned_Tags' in row.Method or 'CAUN' in row.Method else "  "
        print(f"  {symbol}{idx}. {row.Method:25s} - CWC: {row.CWC:.4f}, PICP: {row.PICP:.2f}%, "
              f"MPIW: {row.MPIW:.4f}, MAE: {row.MAE:.4f}")
    
    print("\n【按PICP最接近80%排名】:")
    metrics_df['PICP_Error'] = np.abs(metrics_df['PICP'] - 80)
    metrics_df_picp = metrics_df.sort_values('PICP_Error')
    for idx, row in enumerate(metrics_df_picp.itertuples(), 1):
        symbol = "⭐" if 'Learned_Tags' in row.Method or 'CAUN' in row.Method else "  "
        print(f"  {symbol}{idx}. {row.Method:25s} - PICP: {row.PICP:.2f}% (误差: {row.PICP_Error:.2f}%), "
              f"MPIW: {row.MPIW:.4f}")
    
    print("\n【综合指标对比表】:")
    print(metrics_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    
    return all_results, metrics_df


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # 创建checkpoint目录
    os.makedirs('checkpoint', exist_ok=True)
    
    all_results, metrics_df = main()

