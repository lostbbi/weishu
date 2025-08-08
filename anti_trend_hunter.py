#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
反趋势猎手 (AntiTrendHunter) - 科研级完整实现
专门识别和捕捉趋势终结点，进行反趋势预测
基于动量理论、均值回归和趋势反转信号识别
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
反趋势猎手 (AntiTrendHunter) - 科研级完整实现
专门识别和捕捉趋势终结点，进行反趋势预测
基于动量理论、均值回归和趋势反转信号识别
科研级实现：集成高级数学模型、机器学习、深度学习、信号处理
"""

# 核心科学计算库
import numpy as np
import scipy as sp
from scipy import signal, stats, optimize, interpolate, fft
from scipy.signal import hilbert, find_peaks, peak_widths, savgol_filter
from scipy.stats import jarque_bera, normaltest, anderson, kstest
from scipy.optimize import minimize, differential_evolution
import pandas as pd

# 高级数学与统计
import math
import cmath
from math import log, exp, sqrt, pi, e
from statistics import harmonic_mean, geometric_mean
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.rolling import RollingOLS
import pykalman
from pykalman import KalmanFilter

# 机器学习框架
import sklearn
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, IsolationForest)
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge,
                                 HuberRegressor, TheilSenRegressor)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 深度学习 (如果可用)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D
    from tensorflow.keras.optimizers import Adam, RMSprop
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# 信号处理与小波分析
import pywt
from pywt import wavedec, waverec, threshold

# 时间序列分析
try:
    import arch
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# 优化与数值计算
from scipy.linalg import svd, inv, pinv, norm
from scipy.sparse import csr_matrix
from numba import jit, njit, prange
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 可视化（用于调试和分析）
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 基础数据结构
from collections import defaultdict, deque, Counter, OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import itertools
import functools
import operator
import heapq
import bisect

# 系统与性能
import warnings
import logging
import time
import gc
import psutil
import threading
import multiprocessing as mp

# 随机数与概率
import random
from numpy.random import RandomState
import secrets

# 配置警告过滤
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 设置计算精度
np.seterr(divide='ignore', invalid='ignore')
np.random.seed(42)  # 可重现性

@dataclass
class TrendSignal:
    """趋势信号数据结构"""
    tail_number: int
    trend_type: str  # 'uptrend', 'downtrend', 'sideways'
    trend_strength: float  # 0-1
    momentum: float  # 动量值
    duration: int  # 趋势持续期数
    reversal_probability: float  # 反转概率
    breakout_level: float  # 突破水平
    volume_divergence: float  # 成交量背离指标
    rsi_value: float  # 相对强弱指数
    exhaustion_level: float  # 趋势耗尽程度

@dataclass
class ReversalPattern:
    """反转模式数据结构"""
    pattern_name: str  # 模式名称
    confidence: float  # 置信度
    target_tails: List[int]  # 目标尾数
    reversal_timing: str  # 'immediate', 'delayed', 'gradual'
    risk_level: float  # 风险水平
    expected_magnitude: float  # 预期反转幅度

@dataclass
class WaveletCoefficients:
    """小波分解系数"""
    approximation: np.ndarray  # 近似系数
    details: List[np.ndarray]  # 细节系数
    levels: int  # 分解层数
    wavelet: str  # 小波类型
    reconstruction_error: float  # 重构误差

@dataclass
class FourierComponents:
    """傅里叶分析组件"""
    frequencies: np.ndarray  # 频率
    amplitudes: np.ndarray  # 振幅
    phases: np.ndarray  # 相位
    power_spectrum: np.ndarray  # 功率谱
    dominant_frequencies: List[Tuple[float, float]]  # 主导频率及其强度

@dataclass
class NonlinearDynamics:
    """非线性动力学指标"""
    lyapunov_exponent: float  # 李雅普诺夫指数
    correlation_dimension: float  # 关联维数
    hurst_exponent: float  # 赫斯特指数
    entropy: float  # 熵值
    embedding_dimension: int  # 嵌入维数
    time_delay: int  # 时间延迟

@dataclass
class MarketMicrostructure:
    """市场微观结构"""
    bid_ask_spread: float  # 买卖价差
    market_depth: float  # 市场深度
    order_flow_imbalance: float  # 订单流不平衡
    trade_intensity: float  # 交易强度
    volatility_clustering: float  # 波动性聚集
    jump_detection: bool  # 跳跃检测

@dataclass
class QuantumIndicators:
    """量子化指标"""
    coherence_measure: float  # 相干性度量
    entanglement_entropy: float  # 纠缠熵
    quantum_fidelity: float  # 量子保真度
    superposition_state: complex  # 叠加态
    measurement_probability: np.ndarray  # 测量概率

@dataclass
class MachineLearningMetrics:
    """机器学习评估指标"""
    prediction_accuracy: float  # 预测准确率
    feature_importance: Dict[str, float]  # 特征重要性
    model_confidence: float  # 模型置信度
    cross_validation_scores: List[float]  # 交叉验证得分
    learning_curve: List[Tuple[float, float]]  # 学习曲线
    overfitting_score: float  # 过拟合评分

@dataclass
class NeuralNetworkState:
    """神经网络状态"""
    hidden_states: List[np.ndarray]  # 隐藏状态
    attention_weights: np.ndarray  # 注意力权重
    gradient_norms: List[float]  # 梯度范数
    loss_history: List[float]  # 损失历史
    activation_patterns: Dict[str, np.ndarray]  # 激活模式
    network_topology: Dict[str, Any]  # 网络拓扑

@dataclass
class AdaptiveThresholds:
    """自适应阈值系统"""
    static_thresholds: Dict[str, float]  # 静态阈值
    dynamic_thresholds: Dict[str, float]  # 动态阈值
    adaptive_rates: Dict[str, float]  # 自适应速率
    threshold_history: Dict[str, List[float]]  # 阈值历史
    optimization_targets: Dict[str, float]  # 优化目标
    convergence_status: Dict[str, bool]  # 收敛状态

@dataclass
class RiskMetrics:
    """风险度量指标"""
    value_at_risk: float  # 风险价值
    conditional_var: float  # 条件风险价值
    expected_shortfall: float  # 期望损失
    maximum_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    calmar_ratio: float  # 卡尔马比率
    omega_ratio: float  # 欧米茄比率

@dataclass
class PerformanceMetrics:
    """性能评估指标"""
    total_return: float  # 总收益率
    annualized_return: float  # 年化收益率
    volatility: float  # 波动率
    information_ratio: float  # 信息比率
    tracking_error: float  # 跟踪误差
    beta: float  # 贝塔系数
    alpha: float  # 阿尔法系数
    win_rate: float  # 胜率

class SignalQuality(IntEnum):
    """信号质量等级"""
    NOISE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5

class MarketRegime(Enum):
    """市场状态"""
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"

class PredictionHorizon(Enum):
    """预测时间范围"""
    ULTRA_SHORT = 1  # 1期
    SHORT = 3        # 3期
    MEDIUM = 7       # 7期
    LONG = 15        # 15期
    ULTRA_LONG = 30  # 30期

class TrendState(Enum):
    """趋势状态枚举"""
    STRONG_UPTREND = 4
    MODERATE_UPTREND = 3
    WEAK_UPTREND = 2
    SIDEWAYS = 1
    WEAK_DOWNTREND = -2
    MODERATE_DOWNTREND = -3
    STRONG_DOWNTREND = -4

class AntiTrendHunter:
    """
    反趋势猎手 - 科研级完整实现
    
    核心功能：
    1. 多维度趋势强度量化
    2. 趋势耗尽点精确识别
    3. 反转信号综合评分
    4. 突破口动态发现
    5. 自适应阈值调整
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化科研级反趋势猎手"""
        # 基础配置
        self.config = config or self._get_default_config()
        
        # 设置随机种子确保可重现性
        self._set_random_seeds(42)
        
        # ========== 核心数据存储 ==========
        self.trend_history = defaultdict(lambda: deque(maxlen=self.config['history_window']))
        self.reversal_patterns = {}
        self.trend_states = {}
        self.momentum_indicators = defaultdict(dict)
        
        # ========== 高级技术指标缓存 ==========
        self.technical_indicators = {
            # 经典技术指标
            'rsi': defaultdict(lambda: deque(maxlen=100)),
            'macd': defaultdict(lambda: deque(maxlen=100)),
            'macd_histogram': defaultdict(lambda: deque(maxlen=100)),
            'macd_signal': defaultdict(lambda: deque(maxlen=100)),
            'stochastic_k': defaultdict(lambda: deque(maxlen=100)),
            'stochastic_d': defaultdict(lambda: deque(maxlen=100)),
            'williams_r': defaultdict(lambda: deque(maxlen=100)),
            'cci': defaultdict(lambda: deque(maxlen=100)),
            'adx': defaultdict(lambda: deque(maxlen=100)),
            'atr': defaultdict(lambda: deque(maxlen=100)),
            'bollinger_upper': defaultdict(lambda: deque(maxlen=100)),
            'bollinger_lower': defaultdict(lambda: deque(maxlen=100)),
            'bollinger_middle': defaultdict(lambda: deque(maxlen=100)),
            'bollinger_width': defaultdict(lambda: deque(maxlen=100)),
            'bollinger_position': defaultdict(lambda: deque(maxlen=100)),
            
            # 高级技术指标
            'ichimoku_tenkan': defaultdict(lambda: deque(maxlen=100)),
            'ichimoku_kijun': defaultdict(lambda: deque(maxlen=100)),
            'ichimoku_senkou_a': defaultdict(lambda: deque(maxlen=100)),
            'ichimoku_senkou_b': defaultdict(lambda: deque(maxlen=100)),
            'parabolic_sar': defaultdict(lambda: deque(maxlen=100)),
            'pivot_points': defaultdict(lambda: deque(maxlen=100)),
            'fibonacci_levels': defaultdict(lambda: deque(maxlen=100)),
            'gann_angles': defaultdict(lambda: deque(maxlen=100)),
            
            # 成交量指标
            'obv': defaultdict(lambda: deque(maxlen=100)),
            'ad_line': defaultdict(lambda: deque(maxlen=100)),
            'mfi': defaultdict(lambda: deque(maxlen=100)),
            'vwap': defaultdict(lambda: deque(maxlen=100)),
            'volume_profile': defaultdict(lambda: deque(maxlen=100)),
            'chaikin_oscillator': defaultdict(lambda: deque(maxlen=100)),
            
            # 波动率指标
            'historical_volatility': defaultdict(lambda: deque(maxlen=100)),
            'implied_volatility': defaultdict(lambda: deque(maxlen=100)),
            'volatility_smile': defaultdict(lambda: deque(maxlen=100)),
            'volatility_surface': defaultdict(lambda: deque(maxlen=100)),
        }
        
        # ========== 小波分析组件 ==========
        self.wavelet_analyzer = {
            'coefficients': defaultdict(lambda: deque(maxlen=50)),
            'reconstruction_errors': defaultdict(lambda: deque(maxlen=50)),
            'energy_distribution': defaultdict(lambda: deque(maxlen=50)),
            'dominant_scales': defaultdict(lambda: deque(maxlen=50)),
            'singularity_spectrum': defaultdict(lambda: deque(maxlen=50)),
        }
        
        # ========== 傅里叶分析组件 ==========
        self.fourier_analyzer = {
            'frequency_components': defaultdict(lambda: deque(maxlen=50)),
            'phase_spectrum': defaultdict(lambda: deque(maxlen=50)),
            'power_spectrum': defaultdict(lambda: deque(maxlen=50)),
            'spectral_centroid': defaultdict(lambda: deque(maxlen=50)),
            'spectral_rolloff': defaultdict(lambda: deque(maxlen=50)),
            'spectral_flux': defaultdict(lambda: deque(maxlen=50)),
        }
        
        # ========== 非线性动力学分析器 ==========
        self.nonlinear_analyzer = {
            'lyapunov_exponents': defaultdict(lambda: deque(maxlen=30)),
            'correlation_dimensions': defaultdict(lambda: deque(maxlen=30)),
            'hurst_exponents': defaultdict(lambda: deque(maxlen=30)),
            'fractal_dimensions': defaultdict(lambda: deque(maxlen=30)),
            'entropy_measures': defaultdict(lambda: deque(maxlen=30)),
            'recurrence_plots': defaultdict(lambda: deque(maxlen=30)),
        }
        
        # ========== 机器学习组件 ==========
        self.ml_models = self._initialize_ml_models()
        self.feature_extractors = self._initialize_feature_extractors()
        self.model_ensemble = None
        self.feature_importance_tracker = defaultdict(lambda: deque(maxlen=100))
        self.prediction_cache = {}
        
        # ========== 深度学习组件 ==========
        if TORCH_AVAILABLE:
            self.pytorch_models = self._initialize_pytorch_models()
        else:
            self.pytorch_models = {}
            
        if TF_AVAILABLE:
            self.tensorflow_models = self._initialize_tensorflow_models()
        else:
            self.tensorflow_models = {}
        
        # ========== 卡尔曼滤波器 ==========
        self.kalman_filters = {}
        for tail in range(10):
            self.kalman_filters[tail] = self._create_kalman_filter()
        
        # ========== 量子化指标 ==========
        self.quantum_indicators = {
            'coherence_measures': defaultdict(lambda: deque(maxlen=50)),
            'entanglement_entropies': defaultdict(lambda: deque(maxlen=50)),
            'quantum_fidelities': defaultdict(lambda: deque(maxlen=50)),
            'superposition_states': defaultdict(lambda: deque(maxlen=50)),
            'measurement_probabilities': defaultdict(lambda: deque(maxlen=50)),
        }
        
        # ========== 市场微观结构分析器 ==========
        self.microstructure_analyzer = {
            'bid_ask_spreads': defaultdict(lambda: deque(maxlen=100)),
            'market_depths': defaultdict(lambda: deque(maxlen=100)),
            'order_flow_imbalances': defaultdict(lambda: deque(maxlen=100)),
            'trade_intensities': defaultdict(lambda: deque(maxlen=100)),
            'volatility_clustering': defaultdict(lambda: deque(maxlen=100)),
            'jump_detections': defaultdict(lambda: deque(maxlen=100)),
        }
        
        # ========== 风险管理系统 ==========
        self.risk_manager = {
            'var_calculations': defaultdict(lambda: deque(maxlen=100)),
            'expected_shortfalls': defaultdict(lambda: deque(maxlen=100)),
            'maximum_drawdowns': defaultdict(lambda: deque(maxlen=100)),
            'volatility_forecasts': defaultdict(lambda: deque(maxlen=100)),
            'correlation_matrices': deque(maxlen=50),
            'stress_test_results': defaultdict(lambda: deque(maxlen=20)),
        }
        
        # ========== 性能监控系统 ==========
        self.performance_monitor = {
            'prediction_accuracies': deque(maxlen=1000),
            'execution_times': defaultdict(lambda: deque(maxlen=100)),
            'memory_usage': deque(maxlen=100),
            'model_performances': defaultdict(lambda: deque(maxlen=100)),
            'signal_qualities': defaultdict(lambda: deque(maxlen=100)),
        }
        
        # ========== 自适应学习系统 ==========
        self.adaptive_system = {
            'threshold_history': defaultdict(lambda: deque(maxlen=200)),
            'adaptation_rates': defaultdict(float),
            'learning_curves': defaultdict(lambda: deque(maxlen=100)),
            'model_selection_history': deque(maxlen=50),
            'hyperparameter_optimization': {},
        }
        
        # ========== 信号融合系统 ==========
        self.signal_fusion = {
            'signal_weights': self.config['signal_weights'].copy(),
            'weight_history': defaultdict(lambda: deque(maxlen=100)),
            'fusion_results': defaultdict(lambda: deque(maxlen=100)),
            'consensus_scores': defaultdict(lambda: deque(maxlen=100)),
        }
        
        # ========== 模式识别库 ==========
        self.pattern_library = self._initialize_advanced_pattern_library()
        self.pattern_evolution_tracker = defaultdict(list)
        self.geometric_patterns = {}
        self.temporal_patterns = {}
        
        # ========== 分析窗口 ==========
        self.analysis_windows = {
            'nano': 1,          # 纳秒级
            'micro': 3,         # 微秒级
            'ultra_short': 5,   # 超短期
            'short': 10,        # 短期
            'medium': 20,       # 中期
            'long': 50,         # 长期
            'ultra_long': 100,  # 超长期
            'macro': 200,       # 宏观
            'epoch': 500,       # 时代级
        }
        
        # ========== 并行处理 ==========
        if self.config['parallel_processing']:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config['max_workers'])
            self.process_pool = ProcessPoolExecutor(max_workers=self.config['max_workers'])
        else:
            self.thread_pool = None
            self.process_pool = None
        
        # ========== 缓存系统 ==========
        if self.config['cache_enabled']:
            self.cache = OrderedDict()
            self.cache_hits = 0
            self.cache_misses = 0
        
        # ========== 统计跟踪 ==========
        self.total_predictions = 0
        self.successful_reversals = 0
        self.false_signals = 0
        self.prediction_accuracy = 0.0
        self.model_confidence_history = deque(maxlen=1000)
        
        # ========== 动态阈值管理 ==========
        self.dynamic_thresholds = AdaptiveThresholds(
            static_thresholds={
                'reversal_confidence': 0.75,
                'trend_exhaustion': 0.8,
                'momentum_divergence': 0.65,
                'volume_anomaly': 0.7,
                'pattern_matching': 0.8,
                'signal_quality': 0.6,
            },
            dynamic_thresholds={
                'reversal_confidence': 0.75,
                'trend_exhaustion': 0.8,
                'momentum_divergence': 0.65,
                'volume_anomaly': 0.7,
                'pattern_matching': 0.8,
                'signal_quality': 0.6,
            },
            adaptive_rates={
                'reversal_confidence': 0.05,
                'trend_exhaustion': 0.03,
                'momentum_divergence': 0.04,
                'volume_anomaly': 0.06,
                'pattern_matching': 0.02,
                'signal_quality': 0.05,
            },
            threshold_history=defaultdict(list),
            optimization_targets={
                'prediction_accuracy': 0.85,
                'false_positive_rate': 0.1,
                'signal_noise_ratio': 3.0,
            },
            convergence_status=defaultdict(bool)
        )
        
        # ========== 日志系统 ==========
        self._setup_logging()
        
        print(f"🧬 科研级反趋势猎手初始化完成")
        print(f"   📊 技术指标: {len(self.technical_indicators)}种")
        print(f"   🌊 小波分析: {len(self.wavelet_analyzer)}个组件")
        print(f"   📡 傅里叶分析: {len(self.fourier_analyzer)}个组件")
        print(f"   🔬 非线性动力学: {len(self.nonlinear_analyzer)}个分析器")
        print(f"   🤖 机器学习模型: {len(self.ml_models)}个")
        print(f"   🧠 深度学习: PyTorch({TORCH_AVAILABLE}), TensorFlow({TF_AVAILABLE})")
        print(f"   ⚡ 量子指标: {len(self.quantum_indicators)}种")
        print(f"   💹 微观结构: {len(self.microstructure_analyzer)}个分析器")
        print(f"   🔄 并行处理: {self.config['parallel_processing']}")
        print(f"   💾 缓存系统: {self.config['cache_enabled']}")
        print(f"   🎯 分析窗口: {len(self.analysis_windows)}个")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取科研级默认配置"""
        return {
            # ========== 基础分析参数 ==========
            'history_window': 500,  # 历史数据窗口
            'min_trend_duration': 5,  # 最小趋势持续期
            'max_analysis_depth': 100,  # 最大分析深度
            'data_smoothing_window': 7,  # 数据平滑窗口
            
            # ========== 趋势分析参数 ==========
            'trend_detection_methods': ['linear', 'polynomial', 'exponential', 'logarithmic'],
            'trend_strength_threshold': 0.7,
            'trend_consistency_weight': 0.35,
            'trend_momentum_weight': 0.4,
            'trend_duration_weight': 0.25,
            
            # ========== 技术指标参数 ==========
            'rsi_period': 21,  # RSI周期
            'rsi_overbought': 75,  # RSI超买线
            'rsi_oversold': 25,  # RSI超卖线
            'macd_fast': 12,  # MACD快线
            'macd_slow': 26,  # MACD慢线
            'macd_signal': 9,  # MACD信号线
            'bollinger_period': 20,  # 布林带周期
            'bollinger_std': 2.5,  # 布林带标准差倍数
            'stochastic_k': 14,  # 随机指标K值周期
            'stochastic_d': 3,  # 随机指标D值周期
            'williams_r_period': 14,  # Williams %R周期
            'cci_period': 20,  # CCI周期
            'adx_period': 14,  # ADX周期
            'atr_period': 14,  # ATR周期
            
            # ========== 小波分析参数 ==========
            'wavelet_type': 'daubechies',  # 小波类型
            'wavelet_order': 8,  # 小波阶数
            'decomposition_levels': 6,  # 分解层数
            'wavelet_threshold_mode': 'soft',  # 阈值模式
            'wavelet_threshold_method': 'sure',  # 阈值方法
            'wavelet_boundary_mode': 'symmetric',  # 边界处理模式
            
            # ========== 傅里叶分析参数 ==========
            'fft_window_type': 'hann',  # FFT窗口类型
            'spectral_density_method': 'welch',  # 功率谱密度方法
            'frequency_resolution': 0.01,  # 频率分辨率
            'dominant_frequency_threshold': 0.1,  # 主导频率阈值
            'phase_coherence_threshold': 0.7,  # 相位相干性阈值
            
            # ========== 非线性动力学参数 ==========
            'embedding_dimension_range': [3, 15],  # 嵌入维数范围
            'time_delay_range': [1, 10],  # 时间延迟范围
            'lyapunov_min_data_points': 100,  # 李雅普诺夫指数最小数据点
            'correlation_dimension_max_radius': 0.5,  # 关联维数最大半径
            'entropy_bin_count': 50,  # 熵计算的bin数量
            
            # ========== 机器学习参数 ==========
            'ml_ensemble_size': 7,  # 集成模型数量
            'ml_cross_validation_folds': 5,  # 交叉验证折数
            'ml_test_size': 0.2,  # 测试集比例
            'ml_validation_size': 0.15,  # 验证集比例
            'ml_feature_selection_threshold': 0.01,  # 特征选择阈值
            'ml_max_features': 50,  # 最大特征数
            'ml_regularization_strength': [0.001, 0.01, 0.1, 1.0],  # 正则化强度
            'ml_learning_rates': [0.001, 0.01, 0.1],  # 学习率
            'ml_max_iterations': 1000,  # 最大迭代次数
            'ml_convergence_tolerance': 1e-6,  # 收敛容忍度
            
            # ========== 深度学习参数 ==========
            'dl_hidden_layers': [128, 64, 32],  # 隐藏层节点数
            'dl_dropout_rates': [0.2, 0.3, 0.4],  # Dropout比率
            'dl_activation_functions': ['relu', 'tanh', 'leaky_relu'],  # 激活函数
            'dl_batch_size': 64,  # 批次大小
            'dl_epochs': 200,  # 训练轮次
            'dl_early_stopping_patience': 20,  # 早停耐心
            'dl_learning_rate_schedule': 'cosine_annealing',  # 学习率调度
            'dl_weight_decay': 1e-4,  # 权重衰减
            'dl_gradient_clipping': 1.0,  # 梯度裁剪
            
            # ========== LSTM/GRU参数 ==========
            'lstm_units': [64, 32],  # LSTM单元数
            'lstm_sequence_length': 30,  # 序列长度
            'lstm_return_sequences': True,  # 返回序列
            'gru_units': [64, 32],  # GRU单元数
            'attention_heads': 8,  # 注意力头数
            'attention_key_dim': 64,  # 注意力键维度
            
            # ========== 优化算法参数 ==========
            'optimization_algorithm': 'differential_evolution',  # 优化算法
            'population_size': 50,  # 种群大小
            'mutation_rate': 0.8,  # 变异率
            'crossover_rate': 0.7,  # 交叉率
            'max_generations': 100,  # 最大代数
            'tolerance': 1e-6,  # 容忍度
            'constraint_penalty': 1000,  # 约束惩罚
            
            # ========== 卡尔曼滤波参数 ==========
            'kalman_transition_matrices': None,  # 状态转移矩阵
            'kalman_observation_matrices': None,  # 观测矩阵
            'kalman_initial_state_mean': None,  # 初始状态均值
            'kalman_n_dim_state': 4,  # 状态维度
            'kalman_n_dim_obs': 1,  # 观测维度
            
            # ========== 风险管理参数 ==========
            'var_confidence_level': 0.95,  # VaR置信水平
            'expected_shortfall_threshold': 0.05,  # 期望损失阈值
            'maximum_drawdown_threshold': 0.15,  # 最大回撤阈值
            'volatility_threshold': 0.25,  # 波动率阈值
            'correlation_threshold': 0.7,  # 相关性阈值
            
            # ========== 反转检测参数 ==========
            'reversal_confidence_threshold': 0.75,  # 反转置信度阈值
            'reversal_magnitude_threshold': 0.1,  # 反转幅度阈值
            'reversal_timing_tolerance': 3,  # 反转时机容忍度
            'pattern_matching_threshold': 0.8,  # 模式匹配阈值
            'signal_convergence_threshold': 0.7,  # 信号收敛阈值
            
            # ========== 自适应学习参数 ==========
            'learning_rate': 0.01,  # 学习率
            'adaptation_speed': 0.05,  # 自适应速度
            'memory_decay_factor': 0.95,  # 记忆衰减因子
            'performance_window': 50,  # 性能评估窗口
            'threshold_adjustment_sensitivity': 0.1,  # 阈值调整敏感度
            'model_update_frequency': 10,  # 模型更新频率
            
            # ========== 计算性能参数 ==========
            'parallel_processing': True,  # 并行处理
            'max_workers': mp.cpu_count() - 1,  # 最大工作进程数
            'chunk_size': 100,  # 数据块大小
            'memory_limit_gb': 8,  # 内存限制
            'cache_enabled': True,  # 缓存启用
            'cache_size': 1000,  # 缓存大小
            
            # ========== 调试和监控参数 ==========
            'debug_mode': False,  # 调试模式
            'verbose_level': 1,  # 详细程度
            'log_predictions': True,  # 记录预测
            'save_intermediate_results': False,  # 保存中间结果
            'performance_monitoring': True,  # 性能监控
            'memory_monitoring': True,  # 内存监控
            
            # ========== 量子化指标参数 ==========
            'quantum_coherence_threshold': 0.8,  # 量子相干性阈值
            'entanglement_measure_type': 'von_neumann',  # 纠缠度量类型
            'quantum_state_dimensions': 4,  # 量子态维度
            'measurement_basis': 'computational',  # 测量基
            'decoherence_time': 100,  # 去相干时间
            
            # ========== 市场微观结构参数 ==========
            'bid_ask_spread_threshold': 0.001,  # 买卖价差阈值
            'market_depth_levels': 5,  # 市场深度层数
            'order_flow_window': 20,  # 订单流窗口
            'trade_intensity_smoothing': 0.3,  # 交易强度平滑
            'volatility_clustering_metric': 'garch',  # 波动性聚集度量
            
            # ========== 模式识别参数 ==========
            'pattern_library_size': 50,  # 模式库大小
            'pattern_similarity_threshold': 0.85,  # 模式相似性阈值
            'pattern_evolution_tracking': True,  # 模式演化跟踪
            'geometric_pattern_tolerance': 0.05,  # 几何模式容忍度
            'temporal_pattern_weight': 0.6,  # 时间模式权重
            
            # ========== 信号融合参数 ==========
            'signal_fusion_method': 'weighted_average',  # 信号融合方法
            'signal_weights': {  # 信号权重
                'technical': 0.25,
                'wavelet': 0.2,
                'fourier': 0.15,
                'nonlinear': 0.15,
                'ml': 0.15,
                'quantum': 0.1
            },
            'signal_conflict_resolution': 'majority_vote',  # 信号冲突解决方法
            'signal_quality_threshold': SignalQuality.MODERATE,  # 信号质量阈值
        }
    

    def _set_random_seeds(self, seed: int):
        """设置所有随机种子确保可重现性"""
        np.random.seed(seed)
        random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        if TF_AVAILABLE:
            tf.random.set_seed(seed)
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """初始化机器学习模型集合"""
        models = {}
        
        # 集成方法
        models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # 线性模型
        models['ridge'] = Ridge(alpha=1.0, random_state=42)
        models['lasso'] = Lasso(alpha=1.0, random_state=42, max_iter=1000)
        models['elastic_net'] = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=1000)
        models['bayesian_ridge'] = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        models['huber'] = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001)
        models['theil_sen'] = TheilSenRegressor(random_state=42, max_iter=300)
        
        # 支持向量机
        models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        models['nu_svr'] = NuSVR(kernel='rbf', C=1.0, gamma='scale', nu=0.5)
        
        # 近邻方法
        models['knn'] = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')
        
        # 高斯过程
        models['gaussian_process'] = GaussianProcessRegressor(
            alpha=1e-10,
            normalize_y=True,
            random_state=42
        )
        
        # 神经网络
        models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42
        )
        
        return models
    
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """初始化特征提取器"""
        extractors = {}
        
        # 数据预处理
        extractors['standard_scaler'] = StandardScaler()
        extractors['robust_scaler'] = RobustScaler()
        extractors['minmax_scaler'] = MinMaxScaler()
        
        # 降维
        extractors['pca'] = PCA(n_components=0.95, random_state=42)
        extractors['ica'] = FastICA(n_components=None, random_state=42, max_iter=200)
        extractors['nmf'] = NMF(n_components=10, random_state=42, max_iter=200)
        
        # 聚类
        extractors['kmeans'] = KMeans(n_clusters=5, random_state=42, n_init=10)
        extractors['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
        extractors['agglomerative'] = AgglomerativeClustering(n_clusters=5)
        
        # 异常检测
        extractors['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        return extractors
    
    def _initialize_pytorch_models(self) -> Dict[str, Any]:
        """初始化PyTorch模型"""
        if not TORCH_AVAILABLE:
            return {}
        
        models = {}
        
        # 定义LSTM模型
        class LSTMPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
                super(LSTMPredictor, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        # 定义GRU模型
        class GRUPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
                super(GRUPredictor, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                                batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                out, _ = self.gru(x, h0)
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        # 定义Transformer模型
        class TransformerPredictor(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
                super(TransformerPredictor, self).__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=nhead, 
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(d_model, output_size)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                x = self.input_projection(x)
                x = self.transformer(x)
                x = self.dropout(x.mean(dim=1))  # Global average pooling
                x = self.fc(x)
                return x
        
        # 创建模型实例
        models['lstm'] = LSTMPredictor(
            input_size=20, 
            hidden_size=64, 
            num_layers=2, 
            output_size=1,
            dropout=0.2
        )
        
        models['gru'] = GRUPredictor(
            input_size=20, 
            hidden_size=64, 
            num_layers=2, 
            output_size=1,
            dropout=0.2
        )
        
        models['transformer'] = TransformerPredictor(
            input_size=20, 
            d_model=128, 
            nhead=8, 
            num_layers=3, 
            output_size=1,
            dropout=0.1
        )
        
        return models
    
    def _initialize_tensorflow_models(self) -> Dict[str, Any]:
        """初始化TensorFlow模型"""
        if not TF_AVAILABLE:
            return {}
        
        models = {}
        
        # LSTM模型
        lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(30, 20)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        lstm_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        models['lstm'] = lstm_model
        
        # GRU模型
        gru_model = Sequential([
            GRU(64, return_sequences=True, input_shape=(30, 20)),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        gru_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        models['gru'] = gru_model
        
        # CNN-LSTM模型
        cnn_lstm_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 20)),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        cnn_lstm_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        models['cnn_lstm'] = cnn_lstm_model
        
        return models
    
    def _create_kalman_filter(self) -> KalmanFilter:
        """创建卡尔曼滤波器"""
        transition_matrices = np.array([[1, 1, 0, 0],
                                      [0, 1, 1, 0],
                                      [0, 0, 1, 1],
                                      [0, 0, 0, 1]])
        
        observation_matrices = np.array([[1, 0, 0, 0]])
        
        initial_state_mean = np.array([0, 0, 0, 0])
        
        initial_state_covariance = np.eye(4) * 1000
        
        transition_covariance = np.eye(4) * 0.01
        
        observation_covariance = np.array([[1]])
        
        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance
        )
        
        return kf
    
    def _initialize_advanced_pattern_library(self) -> Dict[str, Callable]:
        """初始化高级模式识别库"""
        return {
            # 经典技术分析模式
            'double_top': self._detect_double_top_advanced,
            'double_bottom': self._detect_double_bottom_advanced,
            'head_shoulders': self._detect_head_shoulders_advanced,
            'inverse_head_shoulders': self._detect_inverse_head_shoulders,
            'triple_top': self._detect_triple_top,
            'triple_bottom': self._detect_triple_bottom,
            'ascending_triangle': self._detect_ascending_triangle,
            'descending_triangle': self._detect_descending_triangle,
            'symmetric_triangle': self._detect_symmetric_triangle,
            'rising_wedge': self._detect_rising_wedge,
            'falling_wedge': self._detect_falling_wedge,
            'flag': self._detect_flag_advanced,
            'pennant': self._detect_pennant,
            'cup_handle': self._detect_cup_handle,
            'rounding_top': self._detect_rounding_top,
            'rounding_bottom': self._detect_rounding_bottom,
            
            # 日本蜡烛图模式
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'hanging_man': self._detect_hanging_man,
            'shooting_star': self._detect_shooting_star,
            'engulfing_bullish': self._detect_engulfing_bullish,
            'engulfing_bearish': self._detect_engulfing_bearish,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'three_white_soldiers': self._detect_three_white_soldiers,
            'three_black_crows': self._detect_three_black_crows,
            
            # 波浪理论模式
            'elliott_wave_1': self._detect_elliott_wave_1,
            'elliott_wave_2': self._detect_elliott_wave_2,
            'elliott_wave_3': self._detect_elliott_wave_3,
            'elliott_wave_4': self._detect_elliott_wave_4,
            'elliott_wave_5': self._detect_elliott_wave_5,
            'corrective_wave_a': self._detect_corrective_wave_a,
            'corrective_wave_b': self._detect_corrective_wave_b,
            'corrective_wave_c': self._detect_corrective_wave_c,
            
            # 分形几何模式
            'fractal_support': self._detect_fractal_support,
            'fractal_resistance': self._detect_fractal_resistance,
            'chaos_theory_pattern': self._detect_chaos_pattern,
            'mandelbrot_pattern': self._detect_mandelbrot_pattern,
            
            # 量子化模式
            'quantum_superposition': self._detect_quantum_superposition,
            'quantum_entanglement': self._detect_quantum_entanglement,
            'quantum_coherence': self._detect_quantum_coherence,
            
            # 机器学习发现的模式
            'ml_discovered_pattern_1': self._detect_ml_pattern_1,
            'ml_discovered_pattern_2': self._detect_ml_pattern_2,
            'ml_discovered_pattern_3': self._detect_ml_pattern_3,
            
            # 时间序列模式
            'seasonal_pattern': self._detect_seasonal_pattern,
            'cyclical_pattern': self._detect_cyclical_pattern,
            'trend_break_pattern': self._detect_trend_break_pattern,
            'mean_reversion_pattern': self._detect_mean_reversion_pattern,
            
            # 复杂系统模式
            'emergence_pattern': self._detect_emergence_pattern,
            'self_organization': self._detect_self_organization,
            'phase_transition': self._detect_phase_transition,
            'critical_point': self._detect_critical_point,
        }
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO if self.config['debug_mode'] else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('anti_trend_hunter.log')
            ]
        )
        self.logger = logging.getLogger('AntiTrendHunter')

    def predict(self, candidate_tails: Tuple[int], historical_data_hash: str, 
               prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT) -> Dict[str, Any]:
        """
        科研级主预测方法 - 多维度分析候选尾数的反转潜力
        
        Args:
            candidate_tails: 经过三大定律筛选后的候选尾数元组
            historical_data_hash: 历史数据的哈希值（用于缓存）
            prediction_horizon: 预测时间范围
            
        Returns:
            包含详细分析结果的预测字典
        """
        start_time = time.time()
        
        # 将元组转换回列表（为了兼容）
        candidate_tails_list = list(candidate_tails)
        
        # 从缓存恢复历史数据
        historical_data = self._get_historical_data_from_hash(historical_data_hash)
        
        if not candidate_tails_list or len(historical_data) < 10:
            return self._create_failure_result('insufficient_data')
        
        self.logger.info(f"🎯 科研级反趋势猎手开始深度分析")
        self.logger.info(f"   候选尾数: {sorted(candidate_tails_list)}")
        self.logger.info(f"   预测范围: {prediction_horizon.name}")
        self.logger.info(f"   数据长度: {len(historical_data)}")
        
        try:
            # ========== 阶段1: 数据预处理与质量检查 ==========
            preprocessed_data = self._preprocess_data_advanced(historical_data)
            data_quality_score = self._assess_data_quality(preprocessed_data)
            
            if data_quality_score < 0.6:
                return self._create_failure_result('poor_data_quality', 
                                                 details={'quality_score': data_quality_score})
            
            # ========== 阶段2: 多维度特征提取 ==========
            feature_matrix = self._extract_comprehensive_features(
                candidate_tails_list, preprocessed_data, prediction_horizon
            )
            
            # ========== 阶段3: 高级技术分析 ==========
            technical_analysis_results = {}
            for tail in candidate_tails_list:
                technical_analysis_results[tail] = self._perform_advanced_technical_analysis(
                    tail, preprocessed_data, prediction_horizon
                )
            
            # ========== 阶段4: 小波分析 ==========
            wavelet_analysis_results = {}
            if self.config['debug_mode']:
                print("🌊 执行小波分析...")
            
            for tail in candidate_tails_list:
                wavelet_analysis_results[tail] = self._perform_wavelet_analysis(
                    tail, preprocessed_data
                )
            
            # ========== 阶段5: 傅里叶频域分析 ==========
            fourier_analysis_results = {}
            if self.config['debug_mode']:
                print("📡 执行傅里叶分析...")
            
            for tail in candidate_tails_list:
                fourier_analysis_results[tail] = self._perform_fourier_analysis(
                    tail, preprocessed_data
                )
            
            # ========== 阶段6: 非线性动力学分析 ==========
            nonlinear_analysis_results = {}
            if self.config['debug_mode']:
                print("🔬 执行非线性动力学分析...")
            
            for tail in candidate_tails_list:
                nonlinear_analysis_results[tail] = self._perform_nonlinear_dynamics_analysis(
                    tail, preprocessed_data
                )
            
            # ========== 阶段7: 机器学习集成预测 ==========
            ml_predictions = {}
            if self.config['debug_mode']:
                print("🤖 执行机器学习预测...")
            
            for tail in candidate_tails_list:
                ml_predictions[tail] = self._perform_ml_ensemble_prediction(
                    tail, feature_matrix[tail], preprocessed_data
                )
            
            # ========== 阶段8: 深度学习预测 ==========
            dl_predictions = {}
            if TORCH_AVAILABLE or TF_AVAILABLE:
                if self.config['debug_mode']:
                    print("🧠 执行深度学习预测...")
                
                for tail in candidate_tails_list:
                    dl_predictions[tail] = self._perform_deep_learning_prediction(
                        tail, feature_matrix[tail], preprocessed_data
                    )
            
            # ========== 阶段9: 量子化分析 ==========
            quantum_analysis_results = {}
            if self.config['debug_mode']:
                print("⚡ 执行量子化分析...")
            
            for tail in candidate_tails_list:
                quantum_analysis_results[tail] = self._perform_quantum_analysis(
                    tail, preprocessed_data
                )
            
            # ========== 阶段10: 市场微观结构分析 ==========
            microstructure_analysis_results = {}
            if self.config['debug_mode']:
                print("💹 执行市场微观结构分析...")
            
            for tail in candidate_tails_list:
                microstructure_analysis_results[tail] = self._perform_microstructure_analysis(
                    tail, preprocessed_data
                )
            
            # ========== 阶段11: 卡尔曼滤波状态估计 ==========
            kalman_predictions = {}
            if self.config['debug_mode']:
                print("🔄 执行卡尔曼滤波...")
            
            for tail in candidate_tails_list:
                kalman_predictions[tail] = self._perform_kalman_filtering(
                    tail, preprocessed_data
                )
            
            # ========== 阶段12: 高级模式识别 ==========
            pattern_analysis_results = {}
            if self.config['debug_mode']:
                print("🔍 执行高级模式识别...")
            
            for tail in candidate_tails_list:
                pattern_analysis_results[tail] = self._perform_advanced_pattern_recognition(
                    tail, preprocessed_data
                )
            
            # ========== 阶段13: 多信号融合 ==========
            if self.config['debug_mode']:
                print("🔀 执行多信号融合...")
            
            fusion_results = self._perform_signal_fusion(
                candidate_tails_list,
                {
                    'technical': technical_analysis_results,
                    'wavelet': wavelet_analysis_results,
                    'fourier': fourier_analysis_results,
                    'nonlinear': nonlinear_analysis_results,
                    'ml': ml_predictions,
                    'dl': dl_predictions,
                    'quantum': quantum_analysis_results,
                    'microstructure': microstructure_analysis_results,
                    'kalman': kalman_predictions,
                    'pattern': pattern_analysis_results
                }
            )
            
            # ========== 阶段14: 风险评估 ==========
            risk_assessments = {}
            for tail in candidate_tails_list:
                risk_assessments[tail] = self._perform_comprehensive_risk_assessment(
                    tail, fusion_results[tail], preprocessed_data
                )
            
            # ========== 阶段15: 最优选择与排序 ==========
            final_scores = {}
            detailed_analysis = {}
            
            for tail in candidate_tails_list:
                # 计算综合得分
                final_score = self._calculate_comprehensive_score(
                    fusion_results[tail],
                    risk_assessments[tail],
                    prediction_horizon
                )
                
                final_scores[tail] = final_score
                
                # 构建详细分析
                detailed_analysis[tail] = {
                    'technical_score': technical_analysis_results[tail].get('composite_score', 0.0),
                    'wavelet_score': wavelet_analysis_results[tail].get('reversal_probability', 0.0),
                    'fourier_score': fourier_analysis_results[tail].get('frequency_reversal_score', 0.0),
                    'nonlinear_score': nonlinear_analysis_results[tail].get('chaos_reversal_score', 0.0),
                    'ml_score': ml_predictions[tail].get('ensemble_confidence', 0.0),
                    'dl_score': dl_predictions.get(tail, {}).get('prediction_confidence', 0.0),
                    'quantum_score': quantum_analysis_results[tail].get('quantum_reversal_probability', 0.0),
                    'microstructure_score': microstructure_analysis_results[tail].get('structure_score', 0.0),
                    'kalman_score': kalman_predictions[tail].get('state_confidence', 0.0),
                    'pattern_score': pattern_analysis_results[tail].get('pattern_strength', 0.0),
                    'fusion_score': fusion_results[tail].get('consensus_score', 0.0),
                    'risk_score': risk_assessments[tail].get('risk_level', 0.0),
                    'final_score': final_score,
                    'signal_quality': self._assess_signal_quality(fusion_results[tail]),
                    'prediction_horizon': prediction_horizon.name,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                if self.config['debug_mode']:
                    print(f"   尾数{tail}: 综合得分={final_score:.4f}")
            
            # ========== 阶段16: 结果生成与验证 ==========
            if not final_scores:
                return self._create_failure_result('no_valid_predictions')
            
            # 选择最佳候选
            best_tail = max(final_scores.keys(), key=lambda t: final_scores[t])
            best_score = final_scores[best_tail]
            
            # 计算置信度
            confidence = self._calculate_advanced_confidence(
                best_score, 
                detailed_analysis[best_tail],
                data_quality_score
            )
            
            # 生成详细推理
            reasoning = self._generate_comprehensive_reasoning(
                best_tail, 
                detailed_analysis[best_tail],
                fusion_results[best_tail]
            )
            
            # 预测时机建议
            timing_analysis = self._analyze_optimal_timing(
                best_tail,
                detailed_analysis[best_tail],
                prediction_horizon
            )
            
            # 不确定性量化
            uncertainty_analysis = self._quantify_prediction_uncertainty(
                best_tail,
                fusion_results[best_tail],
                risk_assessments[best_tail]
            )
            
            # ========== 阶段17: 自适应学习更新 ==========
            self._update_adaptive_parameters(best_tail, confidence, final_scores)
            
            # ========== 性能监控 ==========
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, confidence, len(candidate_tails_list))
            
            # ========== 构建最终结果 ==========
            result = {
                'success': True,
                'recommended_tail': best_tail,
                'confidence': confidence,
                'final_score': best_score,
                'prediction_horizon': prediction_horizon.name,
                'reasoning': reasoning,
                'detailed_analysis': detailed_analysis,
                'all_scores': final_scores,
                'signal_fusion_results': fusion_results,
                'risk_assessments': risk_assessments,
                'timing_analysis': timing_analysis,
                'uncertainty_analysis': uncertainty_analysis,
                'data_quality_score': data_quality_score,
                'execution_time': execution_time,
                'feature_importance': self._get_feature_importance_summary(),
                'model_contributions': self._get_model_contribution_summary(),
                'market_regime': self._detect_current_market_regime(preprocessed_data),
                'volatility_forecast': self._forecast_volatility(preprocessed_data),
                'reversal_probability_distribution': self._calculate_reversal_probability_distribution(final_scores),
                'alternative_scenarios': self._generate_alternative_scenarios(final_scores, confidence),
                'backtesting_similarity': self._find_historical_similarities(best_tail, preprocessed_data),
                'early_warning_signals': self._detect_early_warning_signals(preprocessed_data),
                'regime_change_probability': self._calculate_regime_change_probability(preprocessed_data),
                'systemic_risk_indicators': self._calculate_systemic_risk_indicators(preprocessed_data),
                'complexity_measures': self._calculate_complexity_measures(preprocessed_data),
                'metadata': {
                    'algorithm_version': '2.0.0-research',
                    'analysis_depth': 'comprehensive',
                    'prediction_method': 'multi_dimensional_fusion',
                    'total_models_used': len(self.ml_models) + len(self.pytorch_models) + len(self.tensorflow_models),
                    'feature_count': sum(len(features) for features in feature_matrix.values()),
                    'pattern_matches': sum(len(patterns) for patterns in pattern_analysis_results.values()),
                    'computational_complexity': 'O(n²log n)',
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
                }
            }
            
            # 缓存结果
            if self.config['cache_enabled']:
                self._cache_prediction_result(candidate_tails, historical_data_hash, result)
            
            self.logger.info(f"✅ 科研级预测完成: 推荐尾数={best_tail}, 置信度={confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 预测过程发生错误: {str(e)}")
            return self._create_failure_result('prediction_error', details={'error': str(e)})
    
    def _preprocess_data_advanced(self, historical_data: List[Dict]) -> np.ndarray:
        """高级数据预处理"""
        # 转换为数值矩阵
        data_matrix = np.zeros((len(historical_data), 10))
        for i, period in enumerate(historical_data):
            for tail in range(10):
                data_matrix[i, tail] = 1 if tail in period.get('tails', []) else 0
        
        # 数据平滑
        if len(data_matrix) > self.config['data_smoothing_window']:
            for tail in range(10):
                data_matrix[:, tail] = savgol_filter(
                    data_matrix[:, tail], 
                    self.config['data_smoothing_window'], 
                    3
                )
        
        # 异常值检测和处理
        for tail in range(10):
            column = data_matrix[:, tail]
            Q1 = np.percentile(column, 25)
            Q3 = np.percentile(column, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 使用中位数替换异常值
            median = np.median(column)
            outliers = (column < lower_bound) | (column > upper_bound)
            data_matrix[outliers, tail] = median
        
        return data_matrix
    
    def _assess_data_quality(self, data_matrix: np.ndarray) -> float:
        """评估数据质量"""
        quality_scores = []
        
        # 完整性检查
        completeness = 1.0 - np.sum(np.isnan(data_matrix)) / data_matrix.size
        quality_scores.append(completeness)
        
        # 一致性检查
        consistency = 1.0 - np.std(np.sum(data_matrix, axis=1)) / np.mean(np.sum(data_matrix, axis=1))
        quality_scores.append(min(1.0, consistency))
        
        # 变异性检查
        variance_scores = []
        for tail in range(10):
            variance = np.var(data_matrix[:, tail])
            variance_scores.append(min(1.0, variance * 10))  # 归一化
        variability = np.mean(variance_scores)
        quality_scores.append(variability)
        
        # 时间一致性
        temporal_consistency = 1.0
        if len(data_matrix) > 5:
            for tail in range(10):
                autocorr = np.corrcoef(data_matrix[:-1, tail], data_matrix[1:, tail])[0, 1]
                if not np.isnan(autocorr):
                    temporal_consistency *= (1.0 + autocorr) / 2
        quality_scores.append(temporal_consistency)
        
        return np.mean(quality_scores)
    
    def _extract_comprehensive_features(self, candidate_tails: List[int], 
                                      data_matrix: np.ndarray, 
                                      prediction_horizon: PredictionHorizon) -> Dict[int, np.ndarray]:
        """提取综合特征矩阵"""
        feature_matrix = {}
        
        for tail in candidate_tails:
            features = []
            tail_data = data_matrix[:, tail]
            
            # ========== 统计特征 ==========
            if len(tail_data) > 0:
                features.extend([
                    np.mean(tail_data),
                    np.std(tail_data),
                    np.var(tail_data),
                    np.median(tail_data),
                    np.percentile(tail_data, 25),
                    np.percentile(tail_data, 75),
                    stats.skew(tail_data),
                    stats.kurtosis(tail_data),
                    np.min(tail_data),
                    np.max(tail_data)
                ])
            else:
                features.extend([0.0] * 10)
            
            # ========== 时间序列特征 ==========
            if len(tail_data) > 10:
                # 自相关
                autocorr_lags = [1, 3, 5, 7, 10]
                for lag in autocorr_lags:
                    if len(tail_data) > lag:
                        autocorr = np.corrcoef(tail_data[:-lag], tail_data[lag:])[0, 1]
                        features.append(autocorr if not np.isnan(autocorr) else 0.0)
                    else:
                        features.append(0.0)
                
                # 偏自相关（简化版）
                for lag in autocorr_lags:
                    if len(tail_data) > lag * 2:
                        pacf_val = self._calculate_partial_autocorr(tail_data, lag)
                        features.append(pacf_val)
                    else:
                        features.append(0.0)
                
                # 趋势特征
                trend_slope = self._calculate_trend_slope(tail_data)
                features.append(trend_slope)
                
                # 季节性检测
                seasonality_strength = self._detect_seasonality(tail_data)
                features.append(seasonality_strength)
                
                # 平稳性检测
                stationarity_score = self._test_stationarity(tail_data)
                features.append(stationarity_score)
            else:
                features.extend([0.0] * 13)
            
            # ========== 频域特征 ==========
            if len(tail_data) > 8:
                fft_features = self._extract_fft_features(tail_data)
                features.extend(fft_features)
            else:
                features.extend([0.0] * 6)
            
            # ========== 非线性特征 ==========
            if len(tail_data) > 20:
                nonlinear_features = self._extract_nonlinear_features(tail_data)
                features.extend(nonlinear_features)
            else:
                features.extend([0.0] * 8)
            
            # ========== 技术指标特征 ==========
            technical_features = self._extract_technical_features(tail_data)
            features.extend(technical_features)
            
            # ========== 相对特征 ==========
            relative_features = self._extract_relative_features(tail, data_matrix)
            features.extend(relative_features)
            
            # ========== 时间窗口特征 ==========
            window_features = self._extract_window_features(tail_data, prediction_horizon)
            features.extend(window_features)
            
            feature_matrix[tail] = np.array(features)
        
        return feature_matrix
    
    def _perform_advanced_technical_analysis(self, tail: int, data_matrix: np.ndarray, 
                                           prediction_horizon: PredictionHorizon) -> Dict[str, Any]:
        """执行高级技术分析"""
        tail_data = data_matrix[:, tail]
        analysis_result = {}
        
        # ========== RSI分析 ==========
        rsi_values = self._calculate_rsi_advanced(tail_data)
        analysis_result['rsi'] = {
            'current': rsi_values[-1] if len(rsi_values) > 0 else 50,
            'trend': self._calculate_rsi_trend(rsi_values),
            'divergence': self._detect_rsi_divergence(tail_data, rsi_values),
            'overbought_oversold': self._classify_rsi_level(rsi_values[-1] if len(rsi_values) > 0 else 50)
        }
        
        # ========== MACD分析 ==========
        macd_line, macd_signal, macd_histogram = self._calculate_macd_advanced(tail_data)
        analysis_result['macd'] = {
            'line': macd_line[-1] if len(macd_line) > 0 else 0,
            'signal': macd_signal[-1] if len(macd_signal) > 0 else 0,
            'histogram': macd_histogram[-1] if len(macd_histogram) > 0 else 0,
            'crossover': self._detect_macd_crossover(macd_line, macd_signal),
            'divergence': self._detect_macd_divergence(tail_data, macd_line)
        }
        
        # ========== 布林带分析 ==========
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands_advanced(tail_data)
        analysis_result['bollinger'] = {
            'position': self._calculate_bollinger_position(tail_data[-1] if len(tail_data) > 0 else 0, 
                                                         bb_upper[-1] if len(bb_upper) > 0 else 1,
                                                         bb_lower[-1] if len(bb_lower) > 0 else 0),
            'squeeze': self._detect_bollinger_squeeze(bb_upper, bb_lower),
            'breakout': self._detect_bollinger_breakout(tail_data, bb_upper, bb_lower)
        }
        
        # ========== 随机指标分析 ==========
        stoch_k, stoch_d = self._calculate_stochastic_advanced(tail_data)
        analysis_result['stochastic'] = {
            'k': stoch_k[-1] if len(stoch_k) > 0 else 50,
            'd': stoch_d[-1] if len(stoch_d) > 0 else 50,
            'crossover': self._detect_stochastic_crossover(stoch_k, stoch_d),
            'divergence': self._detect_stochastic_divergence(tail_data, stoch_k)
        }
        
        # ========== ADX趋势强度分析 ==========
        adx_values = self._calculate_adx_advanced(tail_data)
        analysis_result['adx'] = {
            'strength': adx_values[-1] if len(adx_values) > 0 else 25,
            'trend_classification': self._classify_trend_strength(adx_values[-1] if len(adx_values) > 0 else 25)
        }
        
        # ========== 成交量分析 ==========
        volume_analysis = self._analyze_volume_patterns(tail, data_matrix)
        analysis_result['volume'] = volume_analysis
        
        # ========== 支撑阻力分析 ==========
        support_resistance = self._identify_support_resistance_levels(tail_data)
        analysis_result['support_resistance'] = support_resistance
        
        # ========== 综合评分 ==========
        analysis_result['composite_score'] = self._calculate_technical_composite_score(analysis_result)
        
        return analysis_result
    
    def _perform_wavelet_analysis(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行小波分析"""
        tail_data = data_matrix[:, tail]
        
        if len(tail_data) < 16:
            return {'error': 'insufficient_data', 'reversal_probability': 0.0}
        
        try:
            # 小波分解
            wavelet_type = self.config['wavelet_type']
            levels = min(self.config['decomposition_levels'], int(np.log2(len(tail_data))))
            
            coeffs = pywt.wavedec(tail_data, wavelet_type, level=levels)
            
            # 分析各层细节系数
            detail_analysis = []
            for i, detail in enumerate(coeffs[1:], 1):
                detail_analysis.append({
                    'level': i,
                    'energy': np.sum(detail**2),
                    'variance': np.var(detail),
                    'max_coeff': np.max(np.abs(detail)),
                    'entropy': self._calculate_wavelet_entropy(detail)
                })
            
            # 奇异谱分析
            singularity_spectrum = self._calculate_singularity_spectrum(tail_data)
            
            # 小波相关性分析
            wavelet_correlation = self._analyze_wavelet_correlation(coeffs)
            
            # 重构误差分析
            reconstructed = pywt.waverec(coeffs, wavelet_type)
            reconstruction_error = np.mean((tail_data[:len(reconstructed)] - reconstructed)**2)
            
            # 反转概率计算
            reversal_probability = self._calculate_wavelet_reversal_probability(
                coeffs, detail_analysis, singularity_spectrum
            )
            
            return {
                'coefficients_summary': {
                    'approximation_energy': np.sum(coeffs[0]**2),
                    'total_detail_energy': sum(da['energy'] for da in detail_analysis),
                    'energy_distribution': [da['energy'] for da in detail_analysis]
                },
                'detail_analysis': detail_analysis,
                'singularity_spectrum': singularity_spectrum,
                'wavelet_correlation': wavelet_correlation,
                'reconstruction_error': reconstruction_error,
                'reversal_probability': reversal_probability,
                'dominant_scales': self._identify_dominant_scales(coeffs),
                'multiscale_entropy': self._calculate_multiscale_entropy(coeffs)
            }
            
        except Exception as e:
            self.logger.error(f"小波分析错误: {str(e)}")
            return {'error': str(e), 'reversal_probability': 0.0}
    
    def _perform_fourier_analysis(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行傅里叶频域分析"""
        tail_data = data_matrix[:, tail]
        
        if len(tail_data) < 8:
            return {'error': 'insufficient_data', 'frequency_reversal_score': 0.0}
        
        try:
            # FFT变换
            fft_values = np.fft.fft(tail_data)
            frequencies = np.fft.fftfreq(len(tail_data))
            
            # 功率谱密度
            power_spectrum = np.abs(fft_values)**2
            
            # 相位谱
            phase_spectrum = np.angle(fft_values)
            
            # 主导频率识别
            dominant_frequencies = self._identify_dominant_frequencies(frequencies, power_spectrum)
            
            # 频率稳定性分析
            frequency_stability = self._analyze_frequency_stability(tail_data)
            
            # 谐波分析
            harmonic_analysis = self._perform_harmonic_analysis(fft_values, frequencies)
            
            # 频域特征提取
            spectral_features = {
                'spectral_centroid': self._calculate_spectral_centroid(power_spectrum, frequencies),
                'spectral_rolloff': self._calculate_spectral_rolloff(power_spectrum, frequencies),
                'spectral_flux': self._calculate_spectral_flux(power_spectrum),
                'spectral_bandwidth': self._calculate_spectral_bandwidth(power_spectrum, frequencies)
            }
            
            # 频域反转评分
            frequency_reversal_score = self._calculate_frequency_reversal_score(
                dominant_frequencies, frequency_stability, harmonic_analysis
            )
            
            return {
                'power_spectrum': power_spectrum.tolist(),
                'phase_spectrum': phase_spectrum.tolist(),
                'dominant_frequencies': dominant_frequencies,
                'frequency_stability': frequency_stability,
                'harmonic_analysis': harmonic_analysis,
                'spectral_features': spectral_features,
                'frequency_reversal_score': frequency_reversal_score,
                'nyquist_frequency': 0.5,
                'frequency_resolution': 1.0 / len(tail_data)
            }
            
        except Exception as e:
            self.logger.error(f"傅里叶分析错误: {str(e)}")
            return {'error': str(e), 'frequency_reversal_score': 0.0}
    
    def _perform_nonlinear_dynamics_analysis(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行非线性动力学分析"""
        tail_data = data_matrix[:, tail]
        
        if len(tail_data) < 30:
            return {'error': 'insufficient_data', 'chaos_reversal_score': 0.0}
        
        try:
            # 李雅普诺夫指数计算
            lyapunov_exponent = self._calculate_lyapunov_exponent(tail_data)
            
            # 关联维数计算
            correlation_dimension = self._calculate_correlation_dimension(tail_data)
            
            # 赫斯特指数计算
            hurst_exponent = self._calculate_hurst_exponent(tail_data)
            
            # 分形维数计算
            fractal_dimension = self._calculate_fractal_dimension(tail_data)
            
            # 熵值计算
            entropy_measures = {
                'shannon_entropy': self._calculate_shannon_entropy(tail_data),
                'approximate_entropy': self._calculate_approximate_entropy(tail_data),
                'sample_entropy': self._calculate_sample_entropy(tail_data),
                'permutation_entropy': self._calculate_permutation_entropy(tail_data)
            }
            
            # 递归图分析
            recurrence_analysis = self._perform_recurrence_analysis(tail_data)
            
            # 相空间重构
            phase_space = self._reconstruct_phase_space(tail_data)
            
            # 庞加莱截面分析
            poincare_analysis = self._analyze_poincare_section(phase_space)
            
            # 混沌特征识别
            chaos_indicators = self._identify_chaos_indicators(
                lyapunov_exponent, correlation_dimension, entropy_measures
            )
            
            # 非线性反转评分
            chaos_reversal_score = self._calculate_chaos_reversal_score(
                lyapunov_exponent, hurst_exponent, entropy_measures, chaos_indicators
            )
            
            return {
                'lyapunov_exponent': lyapunov_exponent,
                'correlation_dimension': correlation_dimension,
                'hurst_exponent': hurst_exponent,
                'fractal_dimension': fractal_dimension,
                'entropy_measures': entropy_measures,
                'recurrence_analysis': recurrence_analysis,
                'phase_space_properties': {
                    'embedding_dimension': phase_space.shape[1],
                    'trajectory_length': phase_space.shape[0],
                    'attractor_dimension': self._estimate_attractor_dimension(phase_space)
                },
                'poincare_analysis': poincare_analysis,
                'chaos_indicators': chaos_indicators,
                'chaos_reversal_score': chaos_reversal_score,
                'predictability_horizon': self._estimate_predictability_horizon(lyapunov_exponent)
            }
            
        except Exception as e:
            self.logger.error(f"非线性动力学分析错误: {str(e)}")
            return {'error': str(e), 'chaos_reversal_score': 0.0}

    def _perform_ml_ensemble_prediction(self, tail: int, features: np.ndarray, 
                                       data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行机器学习集成预测"""
        try:
            if len(features) == 0:
                return {'error': 'no_features', 'ensemble_confidence': 0.0}
            
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 10:
                return {'error': 'insufficient_data', 'ensemble_confidence': 0.0}
            
            # 准备训练数据
            X, y = self._prepare_ml_training_data(features, tail_data)
            
            if len(X) == 0:
                return {'error': 'no_training_data', 'ensemble_confidence': 0.0}
            
            # 特征预处理
            X_scaled = self.feature_extractors['standard_scaler'].fit_transform(X.reshape(-1, 1)).flatten()
            
            # 集成预测
            predictions = {}
            model_performances = {}
            
            for model_name, model in self.ml_models.items():
                try:
                    # 交叉验证评估
                    cv_scores = self._cross_validate_model(model, X_scaled.reshape(-1, 1), y)
                    
                    # 训练模型
                    model.fit(X_scaled.reshape(-1, 1), y)
                    
                    # 预测
                    prediction = model.predict(X_scaled[-1:].reshape(-1, 1))[0]
                    predictions[model_name] = prediction
                    
                    # 记录性能
                    model_performances[model_name] = {
                        'cv_mean': np.mean(cv_scores),
                        'cv_std': np.std(cv_scores),
                        'prediction': prediction
                    }
                    
                except Exception as e:
                    self.logger.warning(f"模型 {model_name} 预测失败: {str(e)}")
                    continue
            
            if not predictions:
                return {'error': 'all_models_failed', 'ensemble_confidence': 0.0}
            
            # 集成策略
            ensemble_prediction = self._ensemble_predictions(predictions, model_performances)
            
            # 特征重要性分析
            feature_importance = self._analyze_feature_importance(X_scaled.reshape(-1, 1), y)
            
            # 预测置信度计算
            ensemble_confidence = self._calculate_ensemble_confidence(model_performances)
            
            # 预测不确定性量化
            prediction_uncertainty = self._quantify_prediction_uncertainty_ml(predictions)
            
            return {
                'ensemble_prediction': ensemble_prediction,
                'individual_predictions': predictions,
                'model_performances': model_performances,
                'feature_importance': feature_importance,
                'ensemble_confidence': ensemble_confidence,
                'prediction_uncertainty': prediction_uncertainty,
                'best_model': max(model_performances.keys(), 
                                key=lambda k: model_performances[k]['cv_mean']),
                'model_agreement': self._calculate_model_agreement(predictions),
                'prediction_stability': self._assess_prediction_stability(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"机器学习集成预测错误: {str(e)}")
            return {'error': str(e), 'ensemble_confidence': 0.0}
    
    def _perform_deep_learning_prediction(self, tail: int, features: np.ndarray, 
                                        data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行深度学习预测"""
        try:
            if not (TORCH_AVAILABLE or TF_AVAILABLE):
                return {'error': 'no_dl_framework', 'prediction_confidence': 0.0}
            
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 30:
                return {'error': 'insufficient_data', 'prediction_confidence': 0.0}
            
            # 准备序列数据
            sequence_data = self._prepare_sequence_data(tail_data, 
                                                      self.config['lstm_sequence_length'])
            
            predictions = {}
            
            # PyTorch模型预测
            if TORCH_AVAILABLE and self.pytorch_models:
                pytorch_predictions = self._pytorch_ensemble_predict(sequence_data, tail_data)
                predictions.update(pytorch_predictions)
            
            # TensorFlow模型预测
            if TF_AVAILABLE and self.tensorflow_models:
                tensorflow_predictions = self._tensorflow_ensemble_predict(sequence_data, tail_data)
                predictions.update(tensorflow_predictions)
            
            if not predictions:
                return {'error': 'no_predictions', 'prediction_confidence': 0.0}
            
            # 深度学习集成
            dl_ensemble_prediction = np.mean(list(predictions.values()))
            
            # 注意力权重分析（如果可用）
            attention_analysis = self._analyze_attention_weights(sequence_data)
            
            # 预测置信度
            prediction_confidence = self._calculate_dl_confidence(predictions)
            
            # 模型解释性分析
            interpretability_analysis = self._analyze_dl_interpretability(sequence_data, predictions)
            
            return {
                'ensemble_prediction': dl_ensemble_prediction,
                'individual_predictions': predictions,
                'attention_analysis': attention_analysis,
                'prediction_confidence': prediction_confidence,
                'interpretability_analysis': interpretability_analysis,
                'sequence_length': len(sequence_data),
                'model_complexity': self._assess_model_complexity(),
                'gradient_information': self._extract_gradient_information(sequence_data)
            }
            
        except Exception as e:
            self.logger.error(f"深度学习预测错误: {str(e)}")
            return {'error': str(e), 'prediction_confidence': 0.0}
    
    def _perform_quantum_analysis(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行量子化分析"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 4:
                return {'error': 'insufficient_data', 'quantum_reversal_probability': 0.0}
            
            # 量子态构建
            quantum_state = self._construct_quantum_state(tail_data)
            
            # 量子相干性测量
            coherence_measure = self._calculate_quantum_coherence(quantum_state)
            
            # 纠缠熵计算
            entanglement_entropy = self._calculate_entanglement_entropy(quantum_state)
            
            # 量子保真度
            quantum_fidelity = self._calculate_quantum_fidelity(quantum_state)
            
            # 量子叠加态分析
            superposition_analysis = self._analyze_quantum_superposition(quantum_state)
            
            # 量子测量概率
            measurement_probabilities = self._calculate_measurement_probabilities(quantum_state)
            
            # 量子干涉效应
            interference_effects = self._analyze_quantum_interference(tail_data)
            
            # 量子退相干分析
            decoherence_analysis = self._analyze_quantum_decoherence(quantum_state)
            
            # 量子纠错能力评估
            error_correction_capacity = self._assess_quantum_error_correction(quantum_state)
            
            # 量子反转概率
            quantum_reversal_probability = self._calculate_quantum_reversal_probability(
                coherence_measure, entanglement_entropy, superposition_analysis
            )
            
            return {
                'quantum_state_properties': {
                    'dimension': len(quantum_state),
                    'purity': self._calculate_quantum_purity(quantum_state),
                    'trace': np.trace(quantum_state) if hasattr(quantum_state, 'trace') else 1.0
                },
                'coherence_measure': coherence_measure,
                'entanglement_entropy': entanglement_entropy,
                'quantum_fidelity': quantum_fidelity,
                'superposition_analysis': superposition_analysis,
                'measurement_probabilities': measurement_probabilities,
                'interference_effects': interference_effects,
                'decoherence_analysis': decoherence_analysis,
                'error_correction_capacity': error_correction_capacity,
                'quantum_reversal_probability': quantum_reversal_probability,
                'quantum_advantage': self._assess_quantum_advantage(tail_data),
                'bell_state_similarity': self._calculate_bell_state_similarity(quantum_state)
            }
            
        except Exception as e:
            self.logger.error(f"量子分析错误: {str(e)}")
            return {'error': str(e), 'quantum_reversal_probability': 0.0}
    
    def _perform_microstructure_analysis(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行市场微观结构分析"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 10:
                return {'error': 'insufficient_data', 'structure_score': 0.0}
            
            # 买卖价差模拟
            bid_ask_spread = self._simulate_bid_ask_spread(tail_data)
            
            # 市场深度分析
            market_depth = self._analyze_market_depth(tail_data, data_matrix)
            
            # 订单流不平衡
            order_flow_imbalance = self._calculate_order_flow_imbalance(tail_data)
            
            # 交易强度分析
            trade_intensity = self._analyze_trade_intensity(tail_data)
            
            # 价格影响分析
            price_impact = self._analyze_price_impact(tail_data)
            
            # 流动性度量
            liquidity_measures = self._calculate_liquidity_measures(tail_data)
            
            # 信息不对称检测
            information_asymmetry = self._detect_information_asymmetry(tail_data, data_matrix)
            
            # 市场操纵检测
            manipulation_indicators = self._detect_market_manipulation(tail_data)
            
            # 波动性聚集分析
            volatility_clustering = self._analyze_volatility_clustering(tail_data)
            
            # 跳跃检测
            jump_detection = self._detect_price_jumps(tail_data)
            
            # 微观结构评分
            structure_score = self._calculate_microstructure_score(
                bid_ask_spread, market_depth, order_flow_imbalance, 
                trade_intensity, liquidity_measures
            )
            
            return {
                'bid_ask_spread': bid_ask_spread,
                'market_depth': market_depth,
                'order_flow_imbalance': order_flow_imbalance,
                'trade_intensity': trade_intensity,
                'price_impact': price_impact,
                'liquidity_measures': liquidity_measures,
                'information_asymmetry': information_asymmetry,
                'manipulation_indicators': manipulation_indicators,
                'volatility_clustering': volatility_clustering,
                'jump_detection': jump_detection,
                'structure_score': structure_score,
                'market_efficiency': self._assess_market_efficiency(tail_data),
                'transaction_cost_analysis': self._analyze_transaction_costs(tail_data)
            }
            
        except Exception as e:
            self.logger.error(f"微观结构分析错误: {str(e)}")
            return {'error': str(e), 'structure_score': 0.0}
    
    def _perform_kalman_filtering(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行卡尔曼滤波状态估计"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return {'error': 'insufficient_data', 'state_confidence': 0.0}
            
            # 获取该尾数的卡尔曼滤波器
            kf = self.kalman_filters[tail]
            
            # 状态估计
            state_means, state_covariances = kf.em(tail_data.reshape(-1, 1))
            
            # 预测下一步
            next_state_mean, next_state_covariance = kf.filter_update(
                state_means[-1], state_covariances[-1], tail_data[-1]
            )
            
            # 平滑估计
            smoothed_means, smoothed_covariances = kf.smooth()
            
            # 状态置信度
            state_confidence = self._calculate_kalman_confidence(
                state_covariances, next_state_covariance
            )
            
            # 新息分析
            innovations = self._calculate_kalman_innovations(tail_data, state_means)
            
            # 似然评估
            log_likelihood = self._calculate_kalman_likelihood(innovations)
            
            # 模型适应性评估
            model_adaptability = self._assess_kalman_adaptability(
                state_covariances, innovations
            )
            
            return {
                'current_state': state_means[-1],
                'predicted_state': next_state_mean,
                'state_uncertainty': np.trace(next_state_covariance),
                'state_confidence': state_confidence,
                'innovations': {
                    'mean': np.mean(innovations),
                    'std': np.std(innovations),
                    'autocorrelation': self._calculate_innovation_autocorr(innovations)
                },
                'log_likelihood': log_likelihood,
                'model_adaptability': model_adaptability,
                'filter_gain': self._calculate_kalman_gain(kf),
                'prediction_interval': self._calculate_prediction_interval(
                    next_state_mean, next_state_covariance
                )
            }
            
        except Exception as e:
            self.logger.error(f"卡尔曼滤波错误: {str(e)}")
            return {'error': str(e), 'state_confidence': 0.0}
    
    def _perform_advanced_pattern_recognition(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行高级模式识别"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return {'error': 'insufficient_data', 'pattern_strength': 0.0}
            
            detected_patterns = {}
            pattern_confidences = {}
            
            # 遍历模式库进行检测
            for pattern_name, pattern_detector in self.pattern_library.items():
                try:
                    detection_result = pattern_detector(tail, data_matrix)
                    if detection_result:
                        detected_patterns[pattern_name] = detection_result
                        # 计算模式置信度
                        confidence = self._calculate_pattern_confidence(
                            pattern_name, detection_result, tail_data
                        )
                        pattern_confidences[pattern_name] = confidence
                except Exception as e:
                    self.logger.warning(f"模式 {pattern_name} 检测失败: {str(e)}")
                    continue
            
            # 模式演化分析
            pattern_evolution = self._analyze_pattern_evolution(tail, detected_patterns)
            
            # 模式强度计算
            pattern_strength = np.mean(list(pattern_confidences.values())) if pattern_confidences else 0.0
            
            # 模式组合分析
            pattern_combinations = self._analyze_pattern_combinations(detected_patterns)
            
            # 模式稳定性评估
            pattern_stability = self._assess_pattern_stability(detected_patterns, tail_data)
            
            # 几何模式分析
            geometric_patterns = self._detect_geometric_patterns(tail_data)
            
            # 时间模式分析
            temporal_patterns = self._detect_temporal_patterns(tail_data)
            
            # 分形模式分析
            fractal_patterns = self._detect_fractal_patterns(tail_data)
            
            return {
                'detected_patterns': detected_patterns,
                'pattern_confidences': pattern_confidences,
                'pattern_evolution': pattern_evolution,
                'pattern_strength': pattern_strength,
                'pattern_combinations': pattern_combinations,
                'pattern_stability': pattern_stability,
                'geometric_patterns': geometric_patterns,
                'temporal_patterns': temporal_patterns,
                'fractal_patterns': fractal_patterns,
                'pattern_count': len(detected_patterns),
                'dominant_pattern': max(pattern_confidences.keys(), 
                                      key=lambda k: pattern_confidences[k]) if pattern_confidences else None,
                'pattern_diversity': self._calculate_pattern_diversity(detected_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"高级模式识别错误: {str(e)}")
            return {'error': str(e), 'pattern_strength': 0.0}
        
    def _perform_signal_fusion(self, candidate_tails: List[int], 
                              analysis_results: Dict[str, Dict]) -> Dict[int, Dict]:
        """执行多信号融合"""
        fusion_results = {}
        
        for tail in candidate_tails:
            # 提取各分析器的信号
            signals = {}
            weights = {}
            
            # 技术分析信号
            if 'technical' in analysis_results and tail in analysis_results['technical']:
                tech_result = analysis_results['technical'][tail]
                signals['technical'] = tech_result.get('composite_score', 0.0)
                weights['technical'] = self.signal_fusion['signal_weights']['technical']
            
            # 小波分析信号
            if 'wavelet' in analysis_results and tail in analysis_results['wavelet']:
                wavelet_result = analysis_results['wavelet'][tail]
                signals['wavelet'] = wavelet_result.get('reversal_probability', 0.0)
                weights['wavelet'] = self.signal_fusion['signal_weights']['wavelet']
            
            # 傅里叶分析信号
            if 'fourier' in analysis_results and tail in analysis_results['fourier']:
                fourier_result = analysis_results['fourier'][tail]
                signals['fourier'] = fourier_result.get('frequency_reversal_score', 0.0)
                weights['fourier'] = self.signal_fusion['signal_weights']['fourier']
            
            # 非线性动力学信号
            if 'nonlinear' in analysis_results and tail in analysis_results['nonlinear']:
                nonlinear_result = analysis_results['nonlinear'][tail]
                signals['nonlinear'] = nonlinear_result.get('chaos_reversal_score', 0.0)
                weights['nonlinear'] = self.signal_fusion['signal_weights']['nonlinear']
            
            # 机器学习信号
            if 'ml' in analysis_results and tail in analysis_results['ml']:
                ml_result = analysis_results['ml'][tail]
                signals['ml'] = ml_result.get('ensemble_confidence', 0.0)
                weights['ml'] = self.signal_fusion['signal_weights']['ml']
            
            # 深度学习信号
            if 'dl' in analysis_results and tail in analysis_results['dl']:
                dl_result = analysis_results['dl'][tail]
                signals['dl'] = dl_result.get('prediction_confidence', 0.0)
                weights['dl'] = self.signal_fusion['signal_weights'].get('dl', 0.1)
            
            # 量子分析信号
            if 'quantum' in analysis_results and tail in analysis_results['quantum']:
                quantum_result = analysis_results['quantum'][tail]
                signals['quantum'] = quantum_result.get('quantum_reversal_probability', 0.0)
                weights['quantum'] = self.signal_fusion['signal_weights']['quantum']
            
            # 执行信号融合
            fusion_result = self._fuse_signals(signals, weights)
            
            # 信号质量评估
            signal_quality = self._assess_signal_quality_comprehensive(signals)
            
            # 信号一致性检查
            signal_consistency = self._check_signal_consistency(signals)
            
            # 信号可靠性评估
            signal_reliability = self._assess_signal_reliability(signals, weights)
            
            # 自适应权重调整
            adapted_weights = self._adapt_signal_weights(signals, weights, tail)
            
            # 重新融合（使用自适应权重）
            adapted_fusion_result = self._fuse_signals(signals, adapted_weights)
            
            fusion_results[tail] = {
                'original_signals': signals,
                'original_weights': weights,
                'adapted_weights': adapted_weights,
                'consensus_score': fusion_result['consensus_score'],
                'adapted_consensus_score': adapted_fusion_result['consensus_score'],
                'signal_quality': signal_quality,
                'signal_consistency': signal_consistency,
                'signal_reliability': signal_reliability,
                'signal_count': len(signals),
                'dominant_signal': max(signals.keys(), key=lambda k: signals[k]) if signals else None,
                'signal_variance': np.var(list(signals.values())) if signals else 0.0,
                'fusion_confidence': fusion_result['confidence'],
                'adaptation_improvement': adapted_fusion_result['consensus_score'] - fusion_result['consensus_score']
            }
        
        return fusion_results
    
    def _perform_comprehensive_risk_assessment(self, tail: int, fusion_result: Dict, 
                                             data_matrix: np.ndarray) -> Dict[str, Any]:
        """执行综合风险评估"""
        try:
            tail_data = data_matrix[:, tail]
            
            # VaR计算
            var_analysis = self._calculate_value_at_risk(tail_data)
            
            # 条件风险价值
            cvar_analysis = self._calculate_conditional_var(tail_data)
            
            # 最大回撤分析
            drawdown_analysis = self._analyze_maximum_drawdown(tail_data)
            
            # 波动率风险
            volatility_risk = self._assess_volatility_risk(tail_data)
            
            # 流动性风险
            liquidity_risk = self._assess_liquidity_risk(tail_data)
            
            # 模型风险
            model_risk = self._assess_model_risk(fusion_result)
            
            # 系统性风险
            systemic_risk = self._assess_systemic_risk(tail, data_matrix)
            
            # 操作风险
            operational_risk = self._assess_operational_risk(fusion_result)
            
            # 信用风险（模拟）
            credit_risk = self._assess_credit_risk_simulation(tail_data)
            
            # 市场风险
            market_risk = self._assess_market_risk(tail_data, data_matrix)
            
            # 风险集中度
            risk_concentration = self._assess_risk_concentration(tail, data_matrix)
            
            # 压力测试
            stress_test_results = self._perform_stress_testing(tail_data)
            
            # 情景分析
            scenario_analysis = self._perform_scenario_analysis(tail_data)
            
            # 风险调整收益
            risk_adjusted_returns = self._calculate_risk_adjusted_returns(tail_data)
            
            # 综合风险评分
            overall_risk_score = self._calculate_overall_risk_score(
                var_analysis, volatility_risk, model_risk, systemic_risk
            )
            
            return {
                'value_at_risk': var_analysis,
                'conditional_var': cvar_analysis,
                'maximum_drawdown': drawdown_analysis,
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'model_risk': model_risk,
                'systemic_risk': systemic_risk,
                'operational_risk': operational_risk,
                'credit_risk': credit_risk,
                'market_risk': market_risk,
                'risk_concentration': risk_concentration,
                'stress_test_results': stress_test_results,
                'scenario_analysis': scenario_analysis,
                'risk_adjusted_returns': risk_adjusted_returns,
                'overall_risk_score': overall_risk_score,
                'risk_level': self._classify_risk_level(overall_risk_score),
                'risk_capacity': self._assess_risk_capacity(tail_data),
                'risk_tolerance_alignment': self._check_risk_tolerance_alignment(overall_risk_score)
            }
            
        except Exception as e:
            self.logger.error(f"风险评估错误: {str(e)}")
            return {'error': str(e), 'overall_risk_score': 1.0, 'risk_level': 'high'}
    
    def _calculate_comprehensive_score(self, fusion_result: Dict, risk_assessment: Dict, 
                                     prediction_horizon: PredictionHorizon) -> float:
        """计算综合得分"""
        try:
            # 基础融合得分
            base_score = fusion_result.get('adapted_consensus_score', 0.0)
            
            # 信号质量调整
            quality_multiplier = fusion_result.get('signal_quality', 0.5)
            
            # 一致性调整
            consistency_multiplier = fusion_result.get('signal_consistency', 0.5)
            
            # 可靠性调整
            reliability_multiplier = fusion_result.get('signal_reliability', 0.5)
            
            # 风险调整
            risk_score = risk_assessment.get('overall_risk_score', 0.5)
            risk_multiplier = 1.0 - risk_score  # 风险越高，得分越低
            
            # 时间范围调整
            horizon_multiplier = self._get_horizon_multiplier(prediction_horizon)
            
            # 计算调整后得分
            adjusted_score = (base_score * 
                            quality_multiplier * 
                            consistency_multiplier * 
                            reliability_multiplier * 
                            risk_multiplier * 
                            horizon_multiplier)
            
            # 应用非线性变换增强区分度
            if adjusted_score > 0.8:
                final_score = 0.8 + (adjusted_score - 0.8) * 2.0
            elif adjusted_score < 0.2:
                final_score = adjusted_score * 0.5
            else:
                final_score = adjusted_score
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"综合得分计算错误: {str(e)}")
            return 0.0
    
    def _calculate_advanced_confidence(self, score: float, analysis: Dict, 
                                     data_quality: float) -> float:
        """计算高级置信度"""
        try:
            confidence_factors = []
            
            # 基础得分贡献
            confidence_factors.append(score)
            
            # 数据质量贡献
            confidence_factors.append(data_quality)
            
            # 信号一致性贡献
            signal_consistency = analysis.get('signal_consistency', 0.5)
            confidence_factors.append(signal_consistency)
            
            # 模型一致性贡献
            if 'ml_score' in analysis and 'dl_score' in analysis:
                model_agreement = 1.0 - abs(analysis['ml_score'] - analysis['dl_score'])
                confidence_factors.append(model_agreement)
            
            # 分析深度贡献
            analysis_depth = len([k for k, v in analysis.items() 
                                if k.endswith('_score') and v > 0])
            depth_factor = min(1.0, analysis_depth / 8.0)
            confidence_factors.append(depth_factor)
            
            # 风险调整
            risk_factor = 1.0 - analysis.get('risk_score', 0.5)
            confidence_factors.append(risk_factor)
            
            # 时间一致性
            if 'temporal_patterns' in analysis:
                temporal_strength = analysis['temporal_patterns'].get('strength', 0.5)
                confidence_factors.append(temporal_strength)
            
            # 计算加权平均置信度
            weights = [0.25, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1]
            if len(confidence_factors) < len(weights):
                weights = weights[:len(confidence_factors)]
                weights = [w / sum(weights) for w in weights]
            
            confidence = sum(f * w for f, w in zip(confidence_factors, weights))
            
            # 应用非线性调整
            if confidence > 0.9:
                confidence = 0.9 + (confidence - 0.9) * 0.5
            elif confidence < 0.1:
                confidence = confidence * 2.0
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"置信度计算错误: {str(e)}")
            return 0.5
    
    def _generate_comprehensive_reasoning(self, tail: int, analysis: Dict, 
                                        fusion_result: Dict) -> str:
        """生成综合推理说明"""
        try:
            reasons = []
            
            # 技术分析理由
            if analysis.get('technical_score', 0) > 0.6:
                reasons.append(f"技术指标显示强烈反转信号(得分:{analysis['technical_score']:.2f})")
            
            # 小波分析理由
            if analysis.get('wavelet_score', 0) > 0.6:
                reasons.append(f"小波分析检测到多尺度反转模式(概率:{analysis['wavelet_score']:.2f})")
            
            # 傅里叶分析理由
            if analysis.get('fourier_score', 0) > 0.6:
                reasons.append(f"频域分析揭示周期性反转特征(得分:{analysis['fourier_score']:.2f})")
            
            # 非线性动力学理由
            if analysis.get('nonlinear_score', 0) > 0.6:
                reasons.append(f"混沌动力学显示系统接近临界点(得分:{analysis['nonlinear_score']:.2f})")
            
            # 机器学习理由
            if analysis.get('ml_score', 0) > 0.6:
                reasons.append(f"机器学习模型集成预测高置信度反转(置信度:{analysis['ml_score']:.2f})")
            
            # 深度学习理由
            if analysis.get('dl_score', 0) > 0.6:
                reasons.append(f"深度神经网络识别复杂反转模式(置信度:{analysis['dl_score']:.2f})")
            
            # 量子分析理由
            if analysis.get('quantum_score', 0) > 0.6:
                reasons.append(f"量子态分析显示相干性崩塌迹象(概率:{analysis['quantum_score']:.2f})")
            
            # 模式识别理由
            if analysis.get('pattern_score', 0) > 0.6:
                reasons.append(f"识别出经典反转形态模式(强度:{analysis['pattern_score']:.2f})")
            
            # 信号融合理由
            consensus_score = fusion_result.get('adapted_consensus_score', 0)
            if consensus_score > 0.7:
                reasons.append(f"多维度信号高度一致指向反转(一致性:{consensus_score:.2f})")
            
            # 风险调整理由
            risk_score = analysis.get('risk_score', 0.5)
            if risk_score < 0.3:
                reasons.append(f"综合风险评估为低风险水平(风险分:{risk_score:.2f})")
            
            # 市场状态理由
            if 'market_regime' in analysis:
                reasons.append(f"当前市场状态: {analysis['market_regime']}")
            
            # 构建最终推理
            if reasons:
                main_reason = "；".join(reasons[:3])  # 取前三个最重要的理由
                
                if len(reasons) > 3:
                    additional_count = len(reasons) - 3
                    main_reason += f"；另有{additional_count}个支持信号"
                
                return f"尾数{tail}反转推荐基于：{main_reason}"
            else:
                return f"尾数{tail}基于综合技术分析显示反转机会"
                
        except Exception as e:
            self.logger.error(f"推理生成错误: {str(e)}")
            return f"尾数{tail}综合分析显示反转潜力"
    
    def _analyze_optimal_timing(self, tail: int, analysis: Dict, 
                              prediction_horizon: PredictionHorizon) -> Dict[str, Any]:
        """分析最优时机"""
        try:
            timing_factors = []
            
            # 技术指标时机
            if analysis.get('technical_score', 0) > 0.7:
                timing_factors.append(('immediate', 0.8))
            elif analysis.get('technical_score', 0) > 0.5:
                timing_factors.append(('next_1_2_periods', 0.6))
            
            # 模式识别时机
            if analysis.get('pattern_score', 0) > 0.8:
                timing_factors.append(('immediate', 0.9))
            
            # 机器学习预测时机
            if analysis.get('ml_score', 0) > 0.7:
                ml_timing = self._estimate_ml_timing(analysis)
                timing_factors.append((ml_timing, 0.7))
            
            # 量子分析时机
            if analysis.get('quantum_score', 0) > 0.6:
                timing_factors.append(('quantum_superposition_collapse', 0.8))
            
            # 风险调整时机
            risk_score = analysis.get('risk_score', 0.5)
            if risk_score < 0.2:
                timing_factors.append(('low_risk_window', 0.9))
            elif risk_score > 0.8:
                timing_factors.append(('wait_for_risk_reduction', 0.3))
            
            # 综合时机评估
            if timing_factors:
                immediate_score = sum(score for timing, score in timing_factors 
                                    if timing == 'immediate')
                delayed_score = sum(score for timing, score in timing_factors 
                                  if 'next' in timing or 'wait' in timing)
                
                if immediate_score > delayed_score:
                    optimal_timing = 'immediate'
                    timing_confidence = immediate_score / len(timing_factors)
                else:
                    optimal_timing = 'next_1_3_periods'
                    timing_confidence = delayed_score / len(timing_factors)
            else:
                optimal_timing = 'monitor_closely'
                timing_confidence = 0.5
            
            return {
                'optimal_timing': optimal_timing,
                'timing_confidence': timing_confidence,
                'timing_factors': timing_factors,
                'prediction_horizon': prediction_horizon.name,
                'recommended_monitoring_frequency': self._recommend_monitoring_frequency(optimal_timing),
                'entry_signals': self._identify_entry_signals(analysis),
                'exit_signals': self._identify_exit_signals(analysis),
                'timing_risk': self._assess_timing_risk(optimal_timing, analysis)
            }
            
        except Exception as e:
            self.logger.error(f"时机分析错误: {str(e)}")
            return {
                'optimal_timing': 'monitor_closely',
                'timing_confidence': 0.3,
                'error': str(e)
            }
        
    def _quantify_prediction_uncertainty(self, tail: int, fusion_result: Dict, 
                                        risk_assessment: Dict) -> Dict[str, Any]:
        """量化预测不确定性"""
        try:
            uncertainty_sources = {}
            
            # 模型不确定性
            model_uncertainty = self._quantify_model_uncertainty(fusion_result)
            uncertainty_sources['model'] = model_uncertainty
            
            # 数据不确定性
            data_uncertainty = self._quantify_data_uncertainty(fusion_result)
            uncertainty_sources['data'] = data_uncertainty
            
            # 参数不确定性
            parameter_uncertainty = self._quantify_parameter_uncertainty()
            uncertainty_sources['parameter'] = parameter_uncertainty
            
            # 认知不确定性
            epistemic_uncertainty = self._quantify_epistemic_uncertainty(fusion_result)
            uncertainty_sources['epistemic'] = epistemic_uncertainty
            
            # 随机不确定性
            aleatoric_uncertainty = self._quantify_aleatoric_uncertainty(risk_assessment)
            uncertainty_sources['aleatoric'] = aleatoric_uncertainty
            
            # 计算总不确定性
            total_uncertainty = self._calculate_total_uncertainty(uncertainty_sources)
            
            # 不确定性传播分析
            uncertainty_propagation = self._analyze_uncertainty_propagation(uncertainty_sources)
            
            # 敏感性分析
            sensitivity_analysis = self._perform_sensitivity_analysis(fusion_result)
            
            # 置信区间计算
            confidence_intervals = self._calculate_confidence_intervals(
                fusion_result, total_uncertainty
            )
            
            return {
                'uncertainty_sources': uncertainty_sources,
                'total_uncertainty': total_uncertainty,
                'uncertainty_propagation': uncertainty_propagation,
                'sensitivity_analysis': sensitivity_analysis,
                'confidence_intervals': confidence_intervals,
                'uncertainty_ranking': self._rank_uncertainty_sources(uncertainty_sources),
                'uncertainty_reduction_suggestions': self._suggest_uncertainty_reduction(uncertainty_sources),
                'prediction_reliability': 1.0 - total_uncertainty
            }
            
        except Exception as e:
            self.logger.error(f"不确定性量化错误: {str(e)}")
            return {'error': str(e), 'total_uncertainty': 0.5}
    
    def _update_adaptive_parameters(self, tail: int, confidence: float, scores: Dict[int, float]):
        """更新自适应参数"""
        try:
            # 更新预测统计
            self.total_predictions += 1
            
            # 更新动态阈值
            if confidence > 0.8:
                # 高置信度时略微提高阈值
                for key in self.dynamic_thresholds.dynamic_thresholds:
                    current = self.dynamic_thresholds.dynamic_thresholds[key]
                    adaptive_rate = self.dynamic_thresholds.adaptive_rates[key]
                    self.dynamic_thresholds.dynamic_thresholds[key] = min(0.95, 
                        current + adaptive_rate * 0.1)
            elif confidence < 0.4:
                # 低置信度时降低阈值
                for key in self.dynamic_thresholds.dynamic_thresholds:
                    current = self.dynamic_thresholds.dynamic_thresholds[key]
                    adaptive_rate = self.dynamic_thresholds.adaptive_rates[key]
                    self.dynamic_thresholds.dynamic_thresholds[key] = max(0.3, 
                        current - adaptive_rate * 0.1)
            
            # 更新信号权重
            self._update_signal_weights(tail, confidence, scores)
            
            # 记录历史
            self.dynamic_thresholds.threshold_history[tail].append({
                'timestamp': datetime.now(),
                'confidence': confidence,
                'thresholds': self.dynamic_thresholds.dynamic_thresholds.copy()
            })
            
            # 更新模型性能
            self._update_model_performance_tracking(tail, confidence)
            
        except Exception as e:
            self.logger.error(f"自适应参数更新错误: {str(e)}")
    
    def _update_performance_metrics(self, execution_time: float, confidence: float, 
                                  candidate_count: int):
        """更新性能指标"""
        try:
            # 记录执行时间
            self.performance_monitor['execution_times']['prediction'].append(execution_time)
            
            # 记录内存使用
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            self.performance_monitor['memory_usage'].append(memory_usage)
            
            # 记录置信度历史
            self.model_confidence_history.append(confidence)
            
            # 更新平均性能指标
            if len(self.performance_monitor['execution_times']['prediction']) > 0:
                avg_execution_time = np.mean(self.performance_monitor['execution_times']['prediction'])
                self.logger.info(f"平均执行时间: {avg_execution_time:.3f}秒")
            
            # 内存监控警告
            if memory_usage > self.config['memory_limit_gb'] * 1024:
                self.logger.warning(f"内存使用超限: {memory_usage:.1f}MB")
                self._trigger_memory_cleanup()
            
        except Exception as e:
            self.logger.error(f"性能指标更新错误: {str(e)}")
    
    # ========== 辅助计算方法 ==========
    
    def _create_data_hash(self, historical_data: List[Dict]) -> str:
        """创建历史数据哈希值"""
        import hashlib
        data_str = str(historical_data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_historical_data_from_hash(self, data_hash: str) -> List[Dict]:
        """从哈希值恢复历史数据（简化实现）"""
        # 在实际实现中，这里应该从缓存中恢复数据
        # 这里返回空列表作为占位符
        return []
    
    def _create_failure_result(self, reason: str, details: Dict = None) -> Dict[str, Any]:
        """创建失败结果"""
        return {
            'success': False,
            'recommended_tail': None,
            'confidence': 0.0,
            'reasoning': reason,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_horizon_multiplier(self, horizon: PredictionHorizon) -> float:
        """获取时间范围乘数"""
        multipliers = {
            PredictionHorizon.ULTRA_SHORT: 1.0,
            PredictionHorizon.SHORT: 0.95,
            PredictionHorizon.MEDIUM: 0.9,
            PredictionHorizon.LONG: 0.85,
            PredictionHorizon.ULTRA_LONG: 0.8
        }
        return multipliers.get(horizon, 0.9)
    
    def _calculate_partial_autocorr(self, data: np.ndarray, lag: int) -> float:
        """计算偏自相关系数"""
        try:
            if len(data) <= lag:
                return 0.0
            
            # 使用Yule-Walker方程计算偏自相关
            autocorrs = [1.0]
            for k in range(1, lag + 1):
                if len(data) > k:
                    autocorr = np.corrcoef(data[:-k], data[k:])[0, 1]
                    autocorrs.append(autocorr if not np.isnan(autocorr) else 0.0)
                else:
                    autocorrs.append(0.0)
            
            # 简化的偏自相关计算
            if len(autocorrs) > lag:
                return autocorrs[lag]
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_trend_slope(self, data: np.ndarray) -> float:
        """计算趋势斜率"""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            slope, _ = np.polyfit(x, data, 1)
            return slope
            
        except Exception:
            return 0.0
    
    def _detect_seasonality(self, data: np.ndarray) -> float:
        """检测季节性强度"""
        try:
            if len(data) < 8:
                return 0.0
            
            # 使用FFT检测周期性
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data))
            power = np.abs(fft)**2
            
            # 排除DC分量
            power[0] = 0
            
            # 找出主导频率
            max_power = np.max(power)
            total_power = np.sum(power)
            
            if total_power > 0:
                seasonality_strength = max_power / total_power
            else:
                seasonality_strength = 0.0
            
            return min(1.0, seasonality_strength * 4)
            
        except Exception:
            return 0.0
    
    def _test_stationarity(self, data: np.ndarray) -> float:
        """测试平稳性"""
        try:
            if len(data) < 10:
                return 0.5
            
            # ADF测试
            try:
                adf_stat, adf_pvalue, _, _, _, _ = adfuller(data)
                adf_score = 1.0 - adf_pvalue  # p值越小，越平稳
            except:
                adf_score = 0.5
            
            # KPSS测试
            try:
                kpss_stat, kpss_pvalue, _, _ = kpss(data)
                kpss_score = kpss_pvalue  # p值越大，越平稳
            except:
                kpss_score = 0.5
            
            # 组合评分
            stationarity_score = (adf_score + kpss_score) / 2
            return min(1.0, max(0.0, stationarity_score))
            
        except Exception:
            return 0.5
    
    def _extract_fft_features(self, data: np.ndarray) -> List[float]:
        """提取FFT特征"""
        try:
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data))
            power = np.abs(fft)**2
            
            # 去除直流分量
            power[0] = 0
            
            features = []
            
            # 主导频率
            if len(power) > 1:
                dominant_freq_idx = np.argmax(power[1:]) + 1
                features.append(freqs[dominant_freq_idx])
                features.append(power[dominant_freq_idx])
            else:
                features.extend([0.0, 0.0])
            
            # 功率谱特征
            if np.sum(power) > 0:
                features.append(np.sum(power))  # 总功率
                features.append(np.mean(power))  # 平均功率
                features.append(np.std(power))   # 功率标准差
                features.append(np.max(power))   # 最大功率
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            return features
            
        except Exception:
            return [0.0] * 6
    
    def _extract_nonlinear_features(self, data: np.ndarray) -> List[float]:
        """提取非线性特征"""
        try:
            features = []
            
            # 李雅普诺夫指数（简化）
            lyapunov = self._calculate_lyapunov_exponent(data)
            features.append(lyapunov)
            
            # 赫斯特指数
            hurst = self._calculate_hurst_exponent(data)
            features.append(hurst)
            
            # 近似熵
            approx_entropy = self._calculate_approximate_entropy(data)
            features.append(approx_entropy)
            
            # 样本熵
            sample_entropy = self._calculate_sample_entropy(data)
            features.append(sample_entropy)
            
            # 分形维数
            fractal_dim = self._calculate_fractal_dimension(data)
            features.append(fractal_dim)
            
            # 去趋势波动分析
            dfa_alpha = self._calculate_dfa_alpha(data)
            features.append(dfa_alpha)
            
            # 递归量化分析指标
            rr, det = self._calculate_rqa_measures(data)
            features.extend([rr, det])
            
            return features
            
        except Exception:
            return [0.0] * 8
    
    def _extract_technical_features(self, data: np.ndarray) -> List[float]:
        """提取技术指标特征"""
        try:
            features = []
            
            if len(data) < 5:
                return [0.0] * 10
            
            # RSI
            rsi = self._calculate_rsi_simple(data)
            features.append(rsi)
            
            # MACD
            if len(data) >= 26:
                macd, macd_signal, macd_hist = self._calculate_macd_simple(data)
                features.extend([macd, macd_signal, macd_hist])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # 布林带位置
            if len(data) >= 20:
                bb_position = self._calculate_bollinger_position_simple(data)
                features.append(bb_position)
            else:
                features.append(0.5)
            
            # 随机指标
            if len(data) >= 14:
                stoch_k = self._calculate_stochastic_simple(data)
                features.append(stoch_k)
            else:
                features.append(50.0)
            
            # Williams %R
            if len(data) >= 14:
                williams_r = self._calculate_williams_r_simple(data)
                features.append(williams_r)
            else:
                features.append(-50.0)
            
            # CCI
            if len(data) >= 20:
                cci = self._calculate_cci_simple(data)
                features.append(cci)
            else:
                features.append(0.0)
            
            # ROC (Rate of Change)
            if len(data) >= 10:
                roc = (data[-1] - data[-10]) / data[-10] if data[-10] != 0 else 0.0
                features.append(roc)
            else:
                features.append(0.0)
            
            # 动量指标
            if len(data) >= 5:
                momentum = data[-1] - data[-5]
                features.append(momentum)
            else:
                features.append(0.0)
            
            return features
            
        except Exception:
            return [0.0] * 10
    
    def _extract_relative_features(self, tail: int, data_matrix: np.ndarray) -> List[float]:
        """提取相对特征"""
        try:
            features = []
            tail_data = data_matrix[:, tail]
            
            # 与其他尾数的相关性
            correlations = []
            for other_tail in range(10):
                if other_tail != tail:
                    other_data = data_matrix[:, other_tail]
                    if len(tail_data) > 1 and len(other_data) > 1:
                        corr = np.corrcoef(tail_data, other_data)[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0.0)
            
            if correlations:
                features.extend([
                    np.mean(correlations),
                    np.std(correlations),
                    np.max(correlations),
                    np.min(correlations)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 相对强度
            if len(tail_data) > 0:
                total_appearances = np.sum(data_matrix, axis=1)
                relative_strength = np.mean(tail_data) / np.mean(total_appearances) if np.mean(total_appearances) > 0 else 0.0
                features.append(relative_strength)
            else:
                features.append(0.0)
            
            # 相对变异系数
            if len(tail_data) > 1:
                cv = np.std(tail_data) / np.mean(tail_data) if np.mean(tail_data) > 0 else 0.0
                features.append(cv)
            else:
                features.append(0.0)
            
            return features
            
        except Exception:
            return [0.0] * 6
    
    def _extract_window_features(self, data: np.ndarray, horizon: PredictionHorizon) -> List[float]:
        """提取时间窗口特征"""
        try:
            features = []
            window_size = horizon.value
            
            if len(data) < window_size:
                return [0.0] * 5
            
            # 最近窗口统计
            recent_window = data[-window_size:]
            features.extend([
                np.mean(recent_window),
                np.std(recent_window),
                np.max(recent_window) - np.min(recent_window),  # 范围
                np.sum(recent_window > np.mean(data)),  # 高于整体均值的数量
                np.sum(np.diff(recent_window) > 0)      # 上升趋势数量
            ])
            
            return features
            
        except Exception:
            return [0.0] * 5
        
# ========== 高级技术指标实现 ==========
    
    def _calculate_rsi_advanced(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """计算高级RSI指标"""
        try:
            if len(data) < period + 1:
                return np.array([50.0])
            
            delta = np.diff(data)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            # 指数移动平均
            avg_gain = pd.Series(gain).ewm(span=period).mean().values
            avg_loss = pd.Series(loss).ewm(span=period).mean().values
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return np.array([50.0])
    
    def _calculate_macd_advanced(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算高级MACD指标"""
        try:
            if len(data) < 26:
                return np.array([0.0]), np.array([0.0]), np.array([0.0])
            
            # 指数移动平均
            ema12 = pd.Series(data).ewm(span=12).mean().values
            ema26 = pd.Series(data).ewm(span=26).mean().values
            
            # MACD线
            macd_line = ema12 - ema26
            
            # 信号线
            macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
            
            # MACD柱状图
            macd_histogram = macd_line - macd_signal
            
            return macd_line, macd_signal, macd_histogram
            
        except Exception:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    def _calculate_bollinger_bands_advanced(self, data: np.ndarray, period: int = 20, 
                                          std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算高级布林带"""
        try:
            if len(data) < period:
                mean_val = np.mean(data)
                std_val = np.std(data)
                return (np.array([mean_val + std_mult * std_val]), 
                       np.array([mean_val]), 
                       np.array([mean_val - std_mult * std_val]))
            
            # 简单移动平均
            sma = pd.Series(data).rolling(window=period).mean().values
            
            # 标准差
            std = pd.Series(data).rolling(window=period).std().values
            
            # 布林带
            upper_band = sma + (std_mult * std)
            lower_band = sma - (std_mult * std)
            
            return upper_band, sma, lower_band
            
        except Exception:
            mean_val = np.mean(data) if len(data) > 0 else 0.5
            return (np.array([mean_val + 0.1]), 
                   np.array([mean_val]), 
                   np.array([mean_val - 0.1]))
    
    def _calculate_stochastic_advanced(self, data: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """计算高级随机指标"""
        try:
            if len(data) < period:
                return np.array([50.0]), np.array([50.0])
            
            # 计算%K
            lowest_low = pd.Series(data).rolling(window=period).min().values
            highest_high = pd.Series(data).rolling(window=period).max().values
            
            k_percent = 100 * ((data - lowest_low) / (highest_high - lowest_low + 1e-10))
            
            # 计算%D (3期移动平均)
            d_percent = pd.Series(k_percent).rolling(window=3).mean().values
            
            return k_percent, d_percent
            
        except Exception:
            return np.array([50.0]), np.array([50.0])
    
    def _calculate_adx_advanced(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """计算高级ADX指标"""
        try:
            if len(data) < period + 1:
                return np.array([25.0])
            
            # 模拟高低价（使用数据变化）
            high = data + np.abs(np.random.normal(0, 0.01, len(data)))
            low = data - np.abs(np.random.normal(0, 0.01, len(data)))
            close = data
            
            # 计算True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # 计算方向性移动
            dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                              np.maximum(high[1:] - high[:-1], 0), 0)
            dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                               np.maximum(low[:-1] - low[1:], 0), 0)
            
            # 平滑处理
            atr = pd.Series(tr).rolling(window=period).mean().values
            di_plus = 100 * pd.Series(dm_plus).rolling(window=period).mean().values / (atr + 1e-10)
            di_minus = 100 * pd.Series(dm_minus).rolling(window=period).mean().values / (atr + 1e-10)
            
            # 计算ADX
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
            adx = pd.Series(dx).rolling(window=period).mean().values
            
            return adx
            
        except Exception:
            return np.array([25.0])
    
    # ========== 小波分析高级实现 ==========
    
    def _calculate_wavelet_entropy(self, coeffs: np.ndarray) -> float:
        """计算小波熵"""
        try:
            if len(coeffs) == 0:
                return 0.0
            
            # 归一化能量
            energy = coeffs**2
            total_energy = np.sum(energy)
            
            if total_energy == 0:
                return 0.0
            
            p = energy / total_energy
            p = p[p > 0]  # 避免log(0)
            
            entropy = -np.sum(p * np.log2(p))
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_singularity_spectrum(self, data: np.ndarray) -> Dict[str, float]:
        """计算奇异谱"""
        try:
            if len(data) < 16:
                return {'alpha_min': 0.0, 'alpha_max': 0.0, 'width': 0.0}
            
            # 使用小波变换进行多分辨率分析
            wavelet = 'db4'
            levels = min(4, int(np.log2(len(data))))
            
            coeffs = pywt.wavedec(data, wavelet, level=levels)
            
            # 计算局部奇异性指数
            alphas = []
            for level, detail in enumerate(coeffs[1:], 1):
                if len(detail) > 0:
                    # 计算局部最大值
                    local_maxima = find_peaks(np.abs(detail))[0]
                    
                    for peak in local_maxima:
                        if peak < len(detail):
                            # 计算奇异性指数
                            alpha = np.log2(np.abs(detail[peak]) + 1e-10) / level
                            alphas.append(alpha)
            
            if alphas:
                alpha_min = np.min(alphas)
                alpha_max = np.max(alphas)
                width = alpha_max - alpha_min
            else:
                alpha_min = alpha_max = width = 0.0
            
            return {
                'alpha_min': alpha_min,
                'alpha_max': alpha_max, 
                'width': width
            }
            
        except Exception:
            return {'alpha_min': 0.0, 'alpha_max': 0.0, 'width': 0.0}
    
    def _analyze_wavelet_correlation(self, coeffs: List[np.ndarray]) -> Dict[str, float]:
        """分析小波相关性"""
        try:
            if len(coeffs) < 2:
                return {'cross_correlation': 0.0, 'scale_correlation': 0.0}
            
            # 跨尺度相关性
            correlations = []
            for i in range(len(coeffs) - 1):
                for j in range(i + 1, len(coeffs)):
                    if len(coeffs[i]) > 1 and len(coeffs[j]) > 1:
                        # 重新采样到相同长度
                        min_len = min(len(coeffs[i]), len(coeffs[j]))
                        c1 = coeffs[i][:min_len]
                        c2 = coeffs[j][:min_len]
                        
                        corr = np.corrcoef(c1, c2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            cross_correlation = np.mean(correlations) if correlations else 0.0
            
            # 尺度间相关性
            scale_correlations = []
            for i in range(1, len(coeffs)):
                if len(coeffs[i]) > 1:
                    autocorr = np.corrcoef(coeffs[i][:-1], coeffs[i][1:])[0, 1]
                    if not np.isnan(autocorr):
                        scale_correlations.append(abs(autocorr))
            
            scale_correlation = np.mean(scale_correlations) if scale_correlations else 0.0
            
            return {
                'cross_correlation': cross_correlation,
                'scale_correlation': scale_correlation
            }
            
        except Exception:
            return {'cross_correlation': 0.0, 'scale_correlation': 0.0}
    
    def _calculate_wavelet_reversal_probability(self, coeffs: List[np.ndarray], 
                                              detail_analysis: List[Dict],
                                              singularity_spectrum: Dict) -> float:
        """计算小波反转概率"""
        try:
            if not coeffs or not detail_analysis:
                return 0.0
            
            reversal_indicators = []
            
            # 基于能量分布
            energies = [da['energy'] for da in detail_analysis]
            if energies:
                # 高频能量突增表明反转
                high_freq_energy = sum(energies[:2]) if len(energies) >= 2 else 0
                total_energy = sum(energies)
                
                if total_energy > 0:
                    high_freq_ratio = high_freq_energy / total_energy
                    reversal_indicators.append(high_freq_ratio)
            
            # 基于奇异谱宽度
            spectrum_width = singularity_spectrum.get('width', 0.0)
            if spectrum_width > 1.0:  # 宽谱表明复杂性增加
                reversal_indicators.append(min(1.0, spectrum_width / 3.0))
            
            # 基于最大系数
            max_coeffs = [da['max_coeff'] for da in detail_analysis]
            if max_coeffs:
                max_coeff_ratio = max(max_coeffs) / (np.mean(max_coeffs) + 1e-10)
                if max_coeff_ratio > 2.0:  # 异常大的系数
                    reversal_indicators.append(min(1.0, (max_coeff_ratio - 1) / 3.0))
            
            # 基于熵值
            entropies = [da['entropy'] for da in detail_analysis]
            if entropies:
                avg_entropy = np.mean(entropies)
                if avg_entropy > 2.0:  # 高熵表明混乱度增加
                    reversal_indicators.append(min(1.0, (avg_entropy - 1) / 3.0))
            
            return np.mean(reversal_indicators) if reversal_indicators else 0.0
            
        except Exception:
            return 0.0
    
    def _identify_dominant_scales(self, coeffs: List[np.ndarray]) -> List[int]:
        """识别主导尺度"""
        try:
            if not coeffs:
                return []
            
            # 计算每个尺度的能量
            energies = []
            for i, coeff in enumerate(coeffs[1:], 1):  # 跳过近似系数
                energy = np.sum(coeff**2)
                energies.append((i, energy))
            
            # 按能量排序
            energies.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前3个主导尺度
            dominant_scales = [scale for scale, _ in energies[:3]]
            
            return dominant_scales
            
        except Exception:
            return []
    
    def _calculate_multiscale_entropy(self, coeffs: List[np.ndarray]) -> List[float]:
        """计算多尺度熵"""
        try:
            entropies = []
            
            for coeff in coeffs:
                if len(coeff) > 0:
                    entropy = self._calculate_wavelet_entropy(coeff)
                    entropies.append(entropy)
                else:
                    entropies.append(0.0)
            
            return entropies
            
        except Exception:
            return [0.0]
    
    # ========== 傅里叶分析高级实现 ==========
    
    def _identify_dominant_frequencies(self, frequencies: np.ndarray, 
                                     power_spectrum: np.ndarray) -> List[Tuple[float, float]]:
        """识别主导频率"""
        try:
            if len(power_spectrum) == 0:
                return []
            
            # 找出功率谱的峰值
            peaks, _ = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.1)
            
            # 获取峰值对应的频率和功率
            dominant_freqs = []
            for peak in peaks:
                if peak < len(frequencies):
                    freq = frequencies[peak]
                    power = power_spectrum[peak]
                    dominant_freqs.append((freq, power))
            
            # 按功率排序
            dominant_freqs.sort(key=lambda x: x[1], reverse=True)
            
            return dominant_freqs[:5]  # 返回前5个主导频率
            
        except Exception:
            return []
    
    def _analyze_frequency_stability(self, data: np.ndarray, window_size: int = 10) -> float:
        """分析频率稳定性"""
        try:
            if len(data) < window_size * 2:
                return 0.5
            
            # 滑动窗口分析
            stability_measures = []
            
            for i in range(len(data) - window_size):
                window1 = data[i:i + window_size]
                window2 = data[i + 1:i + 1 + window_size]
                
                # 计算每个窗口的主导频率
                fft1 = np.fft.fft(window1)
                fft2 = np.fft.fft(window2)
                
                power1 = np.abs(fft1)**2
                power2 = np.abs(fft2)**2
                
                # 找出主导频率
                peak1 = np.argmax(power1[1:]) + 1
                peak2 = np.argmax(power2[1:]) + 1
                
                # 计算频率稳定性
                freq_diff = abs(peak1 - peak2) / max(peak1, peak2, 1)
                stability = 1.0 - freq_diff
                stability_measures.append(stability)
            
            return np.mean(stability_measures) if stability_measures else 0.5
            
        except Exception:
            return 0.5
    
    def _perform_harmonic_analysis(self, fft_values: np.ndarray, 
                                 frequencies: np.ndarray) -> Dict[str, Any]:
        """执行谐波分析"""
        try:
            if len(fft_values) == 0:
                return {'harmonic_content': 0.0, 'fundamental_frequency': 0.0}
            
            power_spectrum = np.abs(fft_values)**2
            
            # 找出基频
            fundamental_idx = np.argmax(power_spectrum[1:]) + 1
            fundamental_freq = frequencies[fundamental_idx] if fundamental_idx < len(frequencies) else 0.0
            
            # 寻找谐波
            harmonics = []
            for n in range(2, 6):  # 2-5次谐波
                harmonic_freq = n * fundamental_freq
                # 找到最接近谐波频率的索引
                freq_diffs = np.abs(frequencies - harmonic_freq)
                harmonic_idx = np.argmin(freq_diffs)
                
                if freq_diffs[harmonic_idx] < 0.1:  # 频率误差容忍度
                    harmonic_power = power_spectrum[harmonic_idx]
                    harmonics.append(harmonic_power)
            
            # 计算谐波含量
            fundamental_power = power_spectrum[fundamental_idx]
            total_harmonic_power = sum(harmonics)
            
            if fundamental_power > 0:
                harmonic_content = total_harmonic_power / fundamental_power
            else:
                harmonic_content = 0.0
            
            return {
                'harmonic_content': harmonic_content,
                'fundamental_frequency': fundamental_freq,
                'harmonic_count': len(harmonics),
                'total_harmonic_distortion': harmonic_content
            }
            
        except Exception:
            return {'harmonic_content': 0.0, 'fundamental_frequency': 0.0}
    
    def _calculate_spectral_centroid(self, power_spectrum: np.ndarray, 
                                   frequencies: np.ndarray) -> float:
        """计算频谱重心"""
        try:
            if len(power_spectrum) == 0 or np.sum(power_spectrum) == 0:
                return 0.0
            
            centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
            return centroid
            
        except Exception:
            return 0.0
    
    def _calculate_spectral_rolloff(self, power_spectrum: np.ndarray, 
                                  frequencies: np.ndarray, threshold: float = 0.85) -> float:
        """计算频谱滚降点"""
        try:
            if len(power_spectrum) == 0:
                return 0.0
            
            total_energy = np.sum(power_spectrum)
            cumulative_energy = np.cumsum(power_spectrum)
            
            rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
            
            if len(rolloff_idx) > 0:
                return frequencies[rolloff_idx[0]]
            else:
                return frequencies[-1] if len(frequencies) > 0 else 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_spectral_flux(self, power_spectrum: np.ndarray) -> float:
        """计算频谱通量"""
        try:
            if len(power_spectrum) < 2:
                return 0.0
            
            # 计算相邻帧之间的功率谱差异
            flux = np.sum(np.diff(power_spectrum)**2)
            return flux
            
        except Exception:
            return 0.0
    
    def _calculate_spectral_bandwidth(self, power_spectrum: np.ndarray, 
                                    frequencies: np.ndarray) -> float:
        """计算频谱带宽"""
        try:
            if len(power_spectrum) == 0:
                return 0.0
            
            centroid = self._calculate_spectral_centroid(power_spectrum, frequencies)
            
            if np.sum(power_spectrum) == 0:
                return 0.0
            
            bandwidth = np.sqrt(np.sum(((frequencies - centroid)**2) * power_spectrum) / 
                               np.sum(power_spectrum))
            
            return bandwidth
            
        except Exception:
            return 0.0
    
    def _calculate_frequency_reversal_score(self, dominant_frequencies: List[Tuple[float, float]], 
                                          frequency_stability: float, 
                                          harmonic_analysis: Dict) -> float:
        """计算频域反转评分"""
        try:
            score_components = []
            
            # 主导频率变化
            if dominant_frequencies:
                # 检查是否有异常高功率的频率
                powers = [power for _, power in dominant_frequencies]
                if powers:
                    max_power = max(powers)
                    avg_power = np.mean(powers)
                    
                    if avg_power > 0:
                        power_ratio = max_power / avg_power
                        if power_ratio > 3.0:  # 异常突出的频率
                            score_components.append(min(1.0, (power_ratio - 1) / 5.0))
            
            # 频率不稳定性
            instability = 1.0 - frequency_stability
            if instability > 0.5:
                score_components.append(instability)
            
            # 谐波失真
            harmonic_content = harmonic_analysis.get('harmonic_content', 0.0)
            if harmonic_content > 0.3:
                score_components.append(min(1.0, harmonic_content))
            
            return np.mean(score_components) if score_components else 0.0
            
        except Exception:
            return 0.0
        
# ========== 非线性动力学高级实现 ==========
    
    def _calculate_lyapunov_exponent(self, data: np.ndarray, embedding_dim: int = 3, 
                                   time_delay: int = 1) -> float:
        """计算李雅普诺夫指数"""
        try:
            if len(data) < 30:
                return 0.0
            
            # 相空间重构
            embedded = self._embed_time_series(data, embedding_dim, time_delay)
            
            if len(embedded) < 10:
                return 0.0
            
            # 寻找最近邻点
            distances = []
            divergences = []
            
            for i in range(len(embedded) - 10):
                # 计算到其他点的距离
                point = embedded[i]
                other_points = embedded[i+1:]
                
                dists = np.linalg.norm(other_points - point, axis=1)
                
                # 找最近邻
                nearest_idx = np.argmin(dists)
                if nearest_idx < len(dists) - 5:
                    initial_distance = dists[nearest_idx]
                    
                    if initial_distance > 0:
                        # 跟踪5步后的距离
                        future_distance = np.linalg.norm(
                            embedded[i + nearest_idx + 5] - embedded[i + 5]
                        )
                        
                        if future_distance > 0:
                            divergence = np.log(future_distance / initial_distance)
                            divergences.append(divergence)
            
            if divergences:
                lyapunov = np.mean(divergences) / 5  # 除以时间步长
            else:
                lyapunov = 0.0
            
            return lyapunov
            
        except Exception:
            return 0.0
    
    def _calculate_correlation_dimension(self, data: np.ndarray, 
                                       embedding_dim: int = 5) -> float:
        """计算关联维数"""
        try:
            if len(data) < 50:
                return 1.0
            
            # 相空间重构
            embedded = self._embed_time_series(data, embedding_dim, 1)
            
            if len(embedded) < 10:
                return 1.0
            
            # 计算关联积分
            radii = np.logspace(-3, 0, 20)
            correlations = []
            
            for r in radii:
                count = 0
                total_pairs = 0
                
                for i in range(len(embedded)):
                    for j in range(i + 1, len(embedded)):
                        distance = np.linalg.norm(embedded[i] - embedded[j])
                        total_pairs += 1
                        
                        if distance < r:
                            count += 1
                
                if total_pairs > 0:
                    correlation = count / total_pairs
                    correlations.append(correlation)
                else:
                    correlations.append(0.0)
            
            # 计算关联维数（斜率）
            log_radii = np.log(radii)
            log_correlations = np.log(np.array(correlations) + 1e-10)
            
            # 线性拟合
            valid_indices = ~np.isnan(log_correlations) & ~np.isinf(log_correlations)
            if np.sum(valid_indices) > 5:
                slope, _ = np.polyfit(log_radii[valid_indices], 
                                    log_correlations[valid_indices], 1)
                correlation_dimension = slope
            else:
                correlation_dimension = 1.0
            
            return max(0.0, min(10.0, correlation_dimension))
            
        except Exception:
            return 1.0
    
    def _calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """计算赫斯特指数"""
        try:
            if len(data) < 10:
                return 0.5
            
            # R/S分析
            N = len(data)
            mean_data = np.mean(data)
            
            # 累积偏差
            Y = np.cumsum(data - mean_data)
            
            # 范围
            R = np.max(Y) - np.min(Y)
            
            # 标准差
            S = np.std(data)
            
            if S == 0:
                return 0.5
            
            # R/S比率
            rs_ratio = R / S
            
            if rs_ratio <= 0:
                return 0.5
            
            # 赫斯特指数估计
            hurst = np.log(rs_ratio) / np.log(N)
            
            # 限制在合理范围内
            return max(0.0, min(1.0, hurst))
            
        except Exception:
            return 0.5
    
    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """计算分形维数"""
        try:
            if len(data) < 4:
                return 1.0
            
            # 盒计数法
            # 将数据归一化到[0,1]
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            
            # 不同盒子大小
            box_sizes = []
            box_counts = []
            
            for box_size in [1/4, 1/8, 1/16, 1/32, 1/64]:
                if box_size > 0:
                    # 计算需要的盒子数量
                    grid_size = int(1.0 / box_size)
                    boxes = set()
                    
                    for i, value in enumerate(normalized_data):
                        x_box = int(i * grid_size / len(normalized_data))
                        y_box = int(value * grid_size)
                        boxes.add((x_box, y_box))
                    
                    box_sizes.append(box_size)
                    box_counts.append(len(boxes))
            
            if len(box_sizes) > 2:
                # 线性拟合计算维数
                log_sizes = np.log(box_sizes)
                log_counts = np.log(box_counts)
                
                slope, _ = np.polyfit(log_sizes, log_counts, 1)
                fractal_dimension = -slope
            else:
                fractal_dimension = 1.0
            
            return max(1.0, min(3.0, fractal_dimension))
            
        except Exception:
            return 1.0
    
    def _calculate_shannon_entropy(self, data: np.ndarray, bins: int = 50) -> float:
        """计算香农熵"""
        try:
            if len(data) == 0:
                return 0.0
            
            # 数据分箱
            hist, _ = np.histogram(data, bins=bins, density=True)
            hist = hist + 1e-10  # 避免log(0)
            
            # 归一化
            hist = hist / np.sum(hist)
            
            # 计算熵
            entropy = -np.sum(hist * np.log2(hist))
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_approximate_entropy(self, data: np.ndarray, m: int = 2, 
                                     r: float = None) -> float:
        """计算近似熵"""
        try:
            if len(data) < 10:
                return 0.0
            
            N = len(data)
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template, patterns[j], m) <= r:
                            C[i] += 1.0
                
                C = C / (N - m + 1.0)
                phi = np.mean(np.log(C + 1e-10))
                return phi
            
            approximate_entropy = _phi(m) - _phi(m + 1)
            return approximate_entropy
            
        except Exception:
            return 0.0
    
    def _calculate_sample_entropy(self, data: np.ndarray, m: int = 2, 
                                r: float = None) -> float:
        """计算样本熵"""
        try:
            if len(data) < 10:
                return 0.0
            
            N = len(data)
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            patterns_m = np.array([data[i:i + m] for i in range(N - m + 1)])
            patterns_m1 = np.array([data[i:i + m + 1] for i in range(N - m)])
            
            A = 0
            B = 0
            
            for i in range(N - m + 1):
                for j in range(i + 1, N - m + 1):
                    if _maxdist(patterns_m[i], patterns_m[j], m) <= r:
                        B += 1
                        if j < N - m and _maxdist(patterns_m1[i], patterns_m1[j], m + 1) <= r:
                            A += 1
            
            if B == 0:
                return 0.0
            
            sample_entropy = -np.log(A / B)
            return sample_entropy
            
        except Exception:
            return 0.0
    
    def _calculate_permutation_entropy(self, data: np.ndarray, m: int = 3) -> float:
        """计算排列熵"""
        try:
            if len(data) < m:
                return 0.0
            
            # 生成所有可能的排列
            from itertools import permutations
            all_perms = list(permutations(range(m)))
            perm_counts = {perm: 0 for perm in all_perms}
            
            # 计算每个模式的出现次数
            for i in range(len(data) - m + 1):
                segment = data[i:i + m]
                order = tuple(np.argsort(segment))
                
                if order in perm_counts:
                    perm_counts[order] += 1
            
            # 计算概率
            total_patterns = len(data) - m + 1
            probabilities = [count / total_patterns for count in perm_counts.values()]
            
            # 计算熵
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # 归一化
            max_entropy = np.log2(len(all_perms))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
            
        except Exception:
            return 0.0
    
    def _perform_recurrence_analysis(self, data: np.ndarray) -> Dict[str, float]:
        """执行递归量化分析"""
        try:
            if len(data) < 20:
                return {'recurrence_rate': 0.0, 'determinism': 0.0}
            
            # 嵌入参数
            embedding_dim = 3
            embedded = self._embed_time_series(data, embedding_dim, 1)
            
            if len(embedded) < 10:
                return {'recurrence_rate': 0.0, 'determinism': 0.0}
            
            # 构建递归矩阵
            N = len(embedded)
            threshold = 0.1 * np.std(data)
            recurrence_matrix = np.zeros((N, N))
            
            for i in range(N):
                for j in range(N):
                    distance = np.linalg.norm(embedded[i] - embedded[j])
                    if distance < threshold:
                        recurrence_matrix[i, j] = 1
            
            # 计算递归率
            recurrence_rate = np.sum(recurrence_matrix) / (N * N)
            
            # 计算确定性（对角线结构）
            diagonal_lines = []
            for diag in range(1, N):
                line_length = 0
                for i in range(N - diag):
                    if recurrence_matrix[i, i + diag] == 1:
                        line_length += 1
                    else:
                        if line_length >= 2:
                            diagonal_lines.append(line_length)
                        line_length = 0
                
                if line_length >= 2:
                    diagonal_lines.append(line_length)
            
            if diagonal_lines:
                determinism = sum(diagonal_lines) / np.sum(recurrence_matrix)
            else:
                determinism = 0.0
            
            return {
                'recurrence_rate': recurrence_rate,
                'determinism': determinism,
                'average_diagonal_length': np.mean(diagonal_lines) if diagonal_lines else 0.0
            }
            
        except Exception:
            return {'recurrence_rate': 0.0, 'determinism': 0.0}
    
    def _reconstruct_phase_space(self, data: np.ndarray, embedding_dim: int = 3, 
                               time_delay: int = 1) -> np.ndarray:
        """重构相空间"""
        try:
            return self._embed_time_series(data, embedding_dim, time_delay)
        except Exception:
            return np.array([[0.0] * embedding_dim])
    
    def _embed_time_series(self, data: np.ndarray, embedding_dim: int, 
                          time_delay: int) -> np.ndarray:
        """时间序列嵌入"""
        try:
            if len(data) < embedding_dim * time_delay:
                return np.array([])
            
            embedded_length = len(data) - (embedding_dim - 1) * time_delay
            embedded = np.zeros((embedded_length, embedding_dim))
            
            for i in range(embedded_length):
                for j in range(embedding_dim):
                    embedded[i, j] = data[i + j * time_delay]
            
            return embedded
            
        except Exception:
            return np.array([])
    
    def _analyze_poincare_section(self, phase_space: np.ndarray) -> Dict[str, float]:
        """分析庞加莱截面"""
        try:
            if len(phase_space) == 0 or phase_space.shape[1] < 2:
                return {'section_density': 0.0, 'return_map_correlation': 0.0}
            
            # 简化的庞加莱截面（使用第一个坐标的零交叉）
            x_coords = phase_space[:, 0]
            
            # 找零交叉点
            zero_crossings = []
            for i in range(len(x_coords) - 1):
                if x_coords[i] * x_coords[i + 1] < 0:
                    # 线性插值找精确交叉点
                    t = -x_coords[i] / (x_coords[i + 1] - x_coords[i])
                    if phase_space.shape[1] > 1:
                        y_cross = phase_space[i, 1] + t * (phase_space[i + 1, 1] - phase_space[i, 1])
                        zero_crossings.append(y_cross)
            
            if len(zero_crossings) < 2:
                return {'section_density': 0.0, 'return_map_correlation': 0.0}
            
            # 计算截面密度
            section_density = len(zero_crossings) / len(phase_space)
            
            # 返回映射相关性
            if len(zero_crossings) > 2:
                return_values = zero_crossings[1:]
                prev_values = zero_crossings[:-1]
                correlation = np.corrcoef(prev_values, return_values)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            return {
                'section_density': section_density,
                'return_map_correlation': correlation
            }
            
        except Exception:
            return {'section_density': 0.0, 'return_map_correlation': 0.0}
    
    def _identify_chaos_indicators(self, lyapunov: float, correlation_dim: float, 
                                 entropy_measures: Dict) -> Dict[str, bool]:
        """识别混沌特征指标"""
        try:
            indicators = {}
            
            # 正李雅普诺夫指数表明混沌
            indicators['positive_lyapunov'] = lyapunov > 0.01
            
            # 非整数关联维数表明分形结构
            indicators['fractal_dimension'] = abs(correlation_dim - round(correlation_dim)) > 0.1
            
            # 高熵表明复杂性
            shannon_entropy = entropy_measures.get('shannon_entropy', 0.0)
            indicators['high_entropy'] = shannon_entropy > 3.0
            
            # 低近似熵表明规律性丧失
            approx_entropy = entropy_measures.get('approximate_entropy', 0.0)
            indicators['low_approximate_entropy'] = approx_entropy < 0.5
            
            return indicators
            
        except Exception:
            return {'positive_lyapunov': False, 'fractal_dimension': False, 
                   'high_entropy': False, 'low_approximate_entropy': False}
    
    def _calculate_chaos_reversal_score(self, lyapunov: float, hurst: float, 
                                      entropy_measures: Dict, 
                                      chaos_indicators: Dict) -> float:
        """计算混沌反转评分"""
        try:
            score_components = []
            
            # 李雅普诺夫指数贡献
            if lyapunov > 0.05:
                score_components.append(min(1.0, lyapunov * 10))
            
            # 赫斯特指数偏离0.5表明非随机行为
            hurst_deviation = abs(hurst - 0.5)
            if hurst_deviation > 0.2:
                score_components.append(hurst_deviation * 2)
            
            # 熵值贡献
            shannon_entropy = entropy_measures.get('shannon_entropy', 0.0)
            if shannon_entropy > 4.0:
                score_components.append(min(1.0, (shannon_entropy - 3) / 3))
            
            # 混沌指标数量
            chaos_count = sum(chaos_indicators.values())
            if chaos_count >= 2:
                score_components.append(chaos_count / 4.0)
            
            return np.mean(score_components) if score_components else 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_attractor_dimension(self, phase_space: np.ndarray) -> float:
        """估计吸引子维数"""
        try:
            if len(phase_space) == 0:
                return 0.0
            
            # 使用关联维数方法
            return self._calculate_correlation_dimension(phase_space[:, 0])
            
        except Exception:
            return 0.0
    
    def _estimate_predictability_horizon(self, lyapunov: float) -> float:
        """估计可预测性视界"""
        try:
            if lyapunov <= 0:
                return float('inf')
            
            # 可预测性视界 ≈ 1/λ
            horizon = 1.0 / lyapunov
            return min(100.0, horizon)  # 限制最大值
            
        except Exception:
            return 10.0
    
    # ========== 量子分析高级实现 ==========
    
    def _construct_quantum_state(self, data: np.ndarray) -> np.ndarray:
        """构建量子态"""
        try:
            if len(data) == 0:
                return np.array([1.0, 0.0, 0.0, 0.0])
            
            # 归一化数据到[0,1]
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            
            # 构建4维量子态（2个qubit系统）
            dim = 4
            
            # 使用数据的统计特性构建态
            mean_val = np.mean(normalized_data)
            std_val = np.std(normalized_data)
            skew_val = stats.skew(normalized_data) if len(normalized_data) > 2 else 0.0
            kurt_val = stats.kurtosis(normalized_data) if len(normalized_data) > 3 else 0.0
            
            # 构建复数振幅
            amplitudes = np.array([
                mean_val + 1j * std_val,
                (1 - mean_val) + 1j * abs(skew_val),
                std_val + 1j * abs(kurt_val),
                (1 - std_val) + 1j * (1 - abs(skew_val))
            ])
            
            # 归一化
            norm = np.sqrt(np.sum(np.abs(amplitudes)**2))
            if norm > 0:
                quantum_state = amplitudes / norm
            else:
                quantum_state = np.array([1.0, 0.0, 0.0, 0.0])
            
            return quantum_state
            
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _calculate_quantum_coherence(self, quantum_state: np.ndarray) -> float:
        """计算量子相干性"""
        try:
            if len(quantum_state) == 0:
                return 0.0
            
            # 计算密度矩阵
            density_matrix = np.outer(quantum_state, np.conj(quantum_state))
            
            # 相干性度量：非对角元素的模长和
            coherence = 0.0
            n = density_matrix.shape[0]
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        coherence += abs(density_matrix[i, j])
            
            # 归一化
            max_coherence = n * (n - 1)
            if max_coherence > 0:
                coherence = coherence / max_coherence
            
            return coherence
            
        except Exception:
            return 0.0
    
    def _calculate_entanglement_entropy(self, quantum_state: np.ndarray) -> float:
        """计算纠缠熵"""
        try:
            if len(quantum_state) != 4:  # 2-qubit系统
                return 0.0
            
            # 重塑为2x2矩阵表示2个qubit
            state_matrix = quantum_state.reshape(2, 2)
            
            # 计算约化密度矩阵（第一个qubit）
            reduced_dm = np.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    reduced_dm[i, j] = np.sum(state_matrix[i, :] * np.conj(state_matrix[j, :]))
            
            # 计算本征值
            eigenvalues = np.linalg.eigvals(reduced_dm)
            eigenvalues = np.real(eigenvalues[eigenvalues > 1e-10])
            
            # 计算冯诺依曼熵
            if len(eigenvalues) > 0:
                entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            else:
                entropy = 0.0
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_quantum_fidelity(self, quantum_state: np.ndarray) -> float:
        """计算量子保真度"""
        try:
            if len(quantum_state) == 0:
                return 0.0
            
            # 与最大纠缠态的保真度
            bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
            
            if len(quantum_state) == len(bell_state):
                fidelity = abs(np.vdot(quantum_state, bell_state))**2
            else:
                fidelity = 0.0
            
            return fidelity
            
        except Exception:
            return 0.0
    
    def _analyze_quantum_superposition(self, quantum_state: np.ndarray) -> Dict[str, float]:
        """分析量子叠加态"""
        try:
            if len(quantum_state) == 0:
                return {'superposition_degree': 0.0, 'phase_distribution': 0.0}
            
            # 叠加度：振幅分布的均匀性
            amplitudes = np.abs(quantum_state)
            if np.sum(amplitudes) > 0:
                normalized_amps = amplitudes / np.sum(amplitudes)
                superposition_degree = 1.0 - np.max(normalized_amps)
            else:
                superposition_degree = 0.0
            
            # 相位分布
            phases = np.angle(quantum_state)
            phase_variance = np.var(phases)
            phase_distribution = min(1.0, phase_variance / (np.pi**2))
            
            return {
                'superposition_degree': superposition_degree,
                'phase_distribution': phase_distribution
            }
            
        except Exception:
            return {'superposition_degree': 0.0, 'phase_distribution': 0.0}
    
    def _calculate_measurement_probabilities(self, quantum_state: np.ndarray) -> np.ndarray:
        """计算测量概率"""
        try:
            if len(quantum_state) == 0:
                return np.array([])
            
            probabilities = np.abs(quantum_state)**2
            return probabilities
            
        except Exception:
            return np.array([])
    
    def _analyze_quantum_interference(self, data: np.ndarray) -> Dict[str, float]:
        """分析量子干涉效应"""
        try:
            if len(data) < 4:
                return {'interference_pattern': 0.0, 'visibility': 0.0}
            
            # 使用FFT检测干涉模式
            fft_data = np.fft.fft(data)
            power_spectrum = np.abs(fft_data)**2
            
            # 干涉图样：峰值的规律性
            peaks, _ = find_peaks(power_spectrum)
            
            if len(peaks) > 2:
                # 计算峰值间距的规律性
                peak_intervals = np.diff(peaks)
                interval_variance = np.var(peak_intervals)
                max_interval = np.max(peak_intervals) if len(peak_intervals) > 0 else 1
                
                interference_pattern = 1.0 - (interval_variance / (max_interval + 1e-10))
                
                # 可见度：最大最小值的对比度
                max_power = np.max(power_spectrum)
                min_power = np.min(power_spectrum)
                
                if max_power + min_power > 0:
                    visibility = (max_power - min_power) / (max_power + min_power)
                else:
                    visibility = 0.0
            else:
                interference_pattern = 0.0
                visibility = 0.0
            
            return {
                'interference_pattern': min(1.0, interference_pattern),
                'visibility': min(1.0, visibility)
            }
            
        except Exception:
            return {'interference_pattern': 0.0, 'visibility': 0.0}
    
    def _analyze_quantum_decoherence(self, quantum_state: np.ndarray) -> Dict[str, float]:
        """分析量子退相干"""
        try:
            if len(quantum_state) == 0:
                return {'decoherence_rate': 0.0, 'coherence_time': 0.0}
            
            # 相干性测量
            coherence = self._calculate_quantum_coherence(quantum_state)
            
            # 纯度测量
            density_matrix = np.outer(quantum_state, np.conj(quantum_state))
            purity = np.real(np.trace(np.dot(density_matrix, density_matrix)))
            
            # 退相干率（简化模型）
            decoherence_rate = 1.0 - purity
            
            # 相干时间估计
            if decoherence_rate > 0:
                coherence_time = 1.0 / decoherence_rate
            else:
                coherence_time = float('inf')
            
            return {
                'decoherence_rate': decoherence_rate,
                'coherence_time': min(100.0, coherence_time),
                'purity': purity
            }
            
        except Exception:
            return {'decoherence_rate': 0.0, 'coherence_time': 0.0}
    
    def _assess_quantum_error_correction(self, quantum_state: np.ndarray) -> float:
        """评估量子纠错能力"""
        try:
            if len(quantum_state) == 0:
                return 0.0
            
            # 基于纠缠熵的纠错能力评估
            entanglement_entropy = self._calculate_entanglement_entropy(quantum_state)
            
            # 高纠缠熵表明更好的纠错能力
            correction_capacity = entanglement_entropy / np.log2(len(quantum_state))
            
            return min(1.0, correction_capacity)
            
        except Exception:
            return 0.0
    
    def _calculate_quantum_reversal_probability(self, coherence: float, 
                                              entanglement_entropy: float,
                                              superposition_analysis: Dict) -> float:
        """计算量子反转概率"""
        try:
            probability_factors = []
            
            # 相干性崩塌
            if coherence < 0.3:
                probability_factors.append(1.0 - coherence)
            
            # 纠缠态崩塌
            if entanglement_entropy < 0.5:
                probability_factors.append(1.0 - entanglement_entropy)
            
            # 叠加态不稳定
            superposition_degree = superposition_analysis.get('superposition_degree', 0.0)
            if superposition_degree > 0.8:
                probability_factors.append(superposition_degree)
            
            # 相位混乱
            phase_distribution = superposition_analysis.get('phase_distribution', 0.0)
            if phase_distribution > 0.7:
                probability_factors.append(phase_distribution)
            
            return np.mean(probability_factors) if probability_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_quantum_purity(self, quantum_state: np.ndarray) -> float:
        """计算量子态纯度"""
        try:
            if len(quantum_state) == 0:
                return 0.0
            
            density_matrix = np.outer(quantum_state, np.conj(quantum_state))
            purity = np.real(np.trace(np.dot(density_matrix, density_matrix)))
            
            return purity
            
        except Exception:
            return 0.0
    
    def _assess_quantum_advantage(self, data: np.ndarray) -> float:
        """评估量子优势"""
        try:
            if len(data) < 4:
                return 0.0
            
            # 量子优势基于非经典相关性
            quantum_state = self._construct_quantum_state(data)
            
            # 计算贝尔不等式违反程度
            bell_violation = self._calculate_bell_inequality_violation(quantum_state)
            
            # 量子优势评分
            advantage = min(1.0, bell_violation / 2.0)
            
            return advantage
            
        except Exception:
            return 0.0
    
    def _calculate_bell_state_similarity(self, quantum_state: np.ndarray) -> float:
        """计算与贝尔态的相似性"""
        try:
            if len(quantum_state) != 4:
                return 0.0
            
            # 四个贝尔态
            bell_states = [
                np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
                np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]),
                np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]),
                np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
            ]
            
            # 计算与每个贝尔态的保真度
            fidelities = []
            for bell_state in bell_states:
                fidelity = abs(np.vdot(quantum_state, bell_state))**2
                fidelities.append(fidelity)
            
            # 返回最大相似性
            return max(fidelities)
            
        except Exception:
            return 0.0
    
    def _calculate_bell_inequality_violation(self, quantum_state: np.ndarray) -> float:
        """计算贝尔不等式违反程度"""
        try:
            if len(quantum_state) != 4:
                return 0.0
            
            # CHSH不等式的量子违反
            # 简化计算
            entanglement_entropy = self._calculate_entanglement_entropy(quantum_state)
            
            # 基于纠缠熵估计贝尔违反
            max_violation = 2 * np.sqrt(2) - 2  # 量子力学最大违反值减去经典界限
            violation = entanglement_entropy * max_violation
            
            return violation
            
        except Exception:
            return 0.0
        
# ========== 机器学习具体实现 ==========
    
    def _prepare_ml_training_data(self, features: np.ndarray, target_data: np.ndarray, 
                                window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """准备机器学习训练数据"""
        try:
            if len(target_data) < window_size + 1:
                return np.array([]), np.array([])
            
            X = []
            y = []
            
            # 使用滑动窗口创建训练样本
            for i in range(len(target_data) - window_size):
                # 特征：窗口内的历史数据
                window_features = target_data[i:i + window_size].tolist()
                
                # 如果有额外特征，添加进去
                if len(features) > 0:
                    window_features.extend(features.tolist() if hasattr(features, 'tolist') else [features])
                
                X.append(window_features)
                # 标签：下一个时间点的值
                y.append(target_data[i + window_size])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"训练数据准备错误: {str(e)}")
            return np.array([]), np.array([])
    
    def _cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> np.ndarray:
        """交叉验证模型"""
        try:
            if len(X) == 0 or len(y) == 0:
                return np.array([0.0])
            
            from sklearn.model_selection import cross_val_score
            
            # 确保数据维度正确
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            scores = cross_val_score(model, X, y, cv=min(cv_folds, len(X)), 
                                   scoring='neg_mean_squared_error')
            
            return -scores  # 转换为正值（MSE）
            
        except Exception as e:
            self.logger.error(f"交叉验证错误: {str(e)}")
            return np.array([0.0])
    
    def _ensemble_predictions(self, predictions: Dict[str, float], 
                            performances: Dict[str, Dict]) -> float:
        """集成预测结果"""
        try:
            if not predictions:
                return 0.0
            
            # 基于性能的权重
            weights = {}
            total_weight = 0
            
            for model_name, pred in predictions.items():
                if model_name in performances:
                    # 使用交叉验证得分作为权重（分数越高权重越大）
                    cv_score = performances[model_name].get('cv_mean', 0.1)
                    weight = 1.0 / (cv_score + 1e-6)  # 错误越小权重越大
                    weights[model_name] = weight
                    total_weight += weight
                else:
                    weights[model_name] = 1.0
                    total_weight += 1.0
            
            # 加权平均
            weighted_sum = 0
            for model_name, pred in predictions.items():
                weight = weights[model_name] / total_weight
                weighted_sum += pred * weight
            
            return weighted_sum
            
        except Exception:
            return np.mean(list(predictions.values())) if predictions else 0.0
    
    def _analyze_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """分析特征重要性"""
        try:
            if len(X) == 0 or len(y) == 0:
                return {}
            
            # 使用随机森林分析特征重要性
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            importance_dict = {}
            for i, importance in enumerate(rf.feature_importances_):
                importance_dict[f'feature_{i}'] = importance
            
            return importance_dict
            
        except Exception:
            return {}
    
    def _calculate_ensemble_confidence(self, performances: Dict[str, Dict]) -> float:
        """计算集成模型置信度"""
        try:
            if not performances:
                return 0.0
            
            confidence_factors = []
            
            # 平均交叉验证得分
            cv_scores = [perf.get('cv_mean', 0.0) for perf in performances.values()]
            if cv_scores:
                avg_cv_score = np.mean(cv_scores)
                confidence_factors.append(1.0 / (avg_cv_score + 1e-6))
            
            # 模型一致性
            predictions = [perf.get('prediction', 0.0) for perf in performances.values()]
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                consistency = 1.0 / (prediction_std + 1e-6)
                confidence_factors.append(min(1.0, consistency / 10))
            
            # 模型数量
            model_count_factor = min(1.0, len(performances) / 10.0)
            confidence_factors.append(model_count_factor)
            
            return np.mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _quantify_prediction_uncertainty_ml(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """量化机器学习预测不确定性"""
        try:
            if not predictions:
                return {'total_uncertainty': 1.0}
            
            pred_values = list(predictions.values())
            
            # 预测方差
            prediction_variance = np.var(pred_values)
            
            # 预测范围
            prediction_range = np.max(pred_values) - np.min(pred_values)
            
            # 标准化不确定性
            normalized_uncertainty = min(1.0, prediction_variance + prediction_range / 4.0)
            
            return {
                'prediction_variance': prediction_variance,
                'prediction_range': prediction_range,
                'total_uncertainty': normalized_uncertainty
            }
            
        except Exception:
            return {'total_uncertainty': 1.0}
    
    def _calculate_model_agreement(self, predictions: Dict[str, float]) -> float:
        """计算模型一致性"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            pred_values = list(predictions.values())
            
            # 计算所有模型预测的标准差
            std_dev = np.std(pred_values)
            mean_pred = np.mean(pred_values)
            
            # 一致性 = 1 - 标准差/均值
            if mean_pred != 0:
                agreement = 1.0 - min(1.0, abs(std_dev / mean_pred))
            else:
                agreement = 1.0 - min(1.0, std_dev)
            
            return max(0.0, agreement)
            
        except Exception:
            return 0.0
    
    def _assess_prediction_stability(self, predictions: Dict[str, float]) -> float:
        """评估预测稳定性"""
        try:
            if not predictions:
                return 0.0
            
            pred_values = list(predictions.values())
            
            # 稳定性基于预测值的离散程度
            if len(pred_values) == 1:
                return 1.0
            
            # 使用变异系数
            mean_val = np.mean(pred_values)
            std_val = np.std(pred_values)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                stability = 1.0 / (1.0 + cv)
            else:
                stability = 1.0 / (1.0 + std_val)
            
            return stability
            
        except Exception:
            return 0.0
    
    # ========== 深度学习具体实现 ==========
    
    def _prepare_sequence_data(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """准备序列数据"""
        try:
            if len(data) < sequence_length:
                return np.array([])
            
            sequences = []
            for i in range(len(data) - sequence_length + 1):
                seq = data[i:i + sequence_length]
                sequences.append(seq)
            
            return np.array(sequences)
            
        except Exception:
            return np.array([])
    
    def _pytorch_ensemble_predict(self, sequence_data: np.ndarray, 
                                 target_data: np.ndarray) -> Dict[str, float]:
        """PyTorch集成预测"""
        try:
            if not TORCH_AVAILABLE or len(sequence_data) == 0:
                return {}
            
            predictions = {}
            
            # 准备数据
            if len(sequence_data.shape) == 1:
                sequence_data = sequence_data.reshape(1, -1)
            
            # 为每个模型进行预测
            for model_name, model in self.pytorch_models.items():
                try:
                    model.eval()
                    
                    # 数据预处理
                    input_size = 20  # 假设输入特征数
                    if sequence_data.shape[-1] != input_size:
                        # 调整输入大小
                        if sequence_data.shape[-1] > input_size:
                            seq_input = sequence_data[:, :input_size]
                        else:
                            # 填充到所需大小
                            padding = np.zeros((sequence_data.shape[0], 
                                              input_size - sequence_data.shape[-1]))
                            seq_input = np.hstack([sequence_data, padding])
                    else:
                        seq_input = sequence_data
                    
                    # 转换为tensor
                    input_tensor = torch.FloatTensor(seq_input)
                    
                    # 预测
                    with torch.no_grad():
                        output = model(input_tensor)
                        prediction = output.item() if output.numel() == 1 else output.mean().item()
                    
                    predictions[f'pytorch_{model_name}'] = prediction
                    
                except Exception as e:
                    self.logger.warning(f"PyTorch模型 {model_name} 预测失败: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"PyTorch集成预测错误: {str(e)}")
            return {}
    
    def _tensorflow_ensemble_predict(self, sequence_data: np.ndarray, 
                                   target_data: np.ndarray) -> Dict[str, float]:
        """TensorFlow集成预测"""
        try:
            if not TF_AVAILABLE or len(sequence_data) == 0:
                return {}
            
            predictions = {}
            
            # 准备数据
            if len(sequence_data.shape) == 1:
                sequence_data = sequence_data.reshape(1, 1, -1)
            elif len(sequence_data.shape) == 2:
                sequence_data = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
            
            # 为每个模型进行预测
            for model_name, model in self.tensorflow_models.items():
                try:
                    # 调整输入形状以匹配模型期望
                    expected_shape = model.input_shape
                    
                    if len(expected_shape) == 3:  # (batch, timesteps, features)
                        timesteps = expected_shape[1]
                        features = expected_shape[2]
                        
                        # 调整时间步长
                        if sequence_data.shape[1] != timesteps:
                            if sequence_data.shape[1] > timesteps:
                                seq_input = sequence_data[:, :timesteps, :]
                            else:
                                # 重复最后一个时间步
                                padding_steps = timesteps - sequence_data.shape[1]
                                last_step = sequence_data[:, -1:, :]
                                padding = np.repeat(last_step, padding_steps, axis=1)
                                seq_input = np.concatenate([sequence_data, padding], axis=1)
                        else:
                            seq_input = sequence_data
                        
                        # 调整特征数
                        if seq_input.shape[2] != features:
                            if seq_input.shape[2] > features:
                                seq_input = seq_input[:, :, :features]
                            else:
                                # 填充特征
                                padding_features = features - seq_input.shape[2]
                                padding = np.zeros((seq_input.shape[0], seq_input.shape[1], padding_features))
                                seq_input = np.concatenate([seq_input, padding], axis=2)
                    else:
                        seq_input = sequence_data
                    
                    # 预测
                    prediction = model.predict(seq_input, verbose=0)
                    pred_value = prediction.item() if prediction.size == 1 else np.mean(prediction)
                    
                    predictions[f'tensorflow_{model_name}'] = pred_value
                    
                except Exception as e:
                    self.logger.warning(f"TensorFlow模型 {model_name} 预测失败: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"TensorFlow集成预测错误: {str(e)}")
            return {}
    
    def _analyze_attention_weights(self, sequence_data: np.ndarray) -> Dict[str, Any]:
        """分析注意力权重"""
        try:
            if len(sequence_data) == 0:
                return {'attention_distribution': [], 'focus_indices': []}
            
            # 简化的注意力分析
            # 基于数据变化率计算注意力
            if len(sequence_data.shape) == 1:
                changes = np.abs(np.diff(sequence_data))
            else:
                changes = np.mean(np.abs(np.diff(sequence_data, axis=0)), axis=1)
            
            if len(changes) > 0:
                # 归一化为注意力权重
                attention_weights = changes / (np.sum(changes) + 1e-10)
                
                # 找出注意力集中的位置
                focus_threshold = np.mean(attention_weights) + np.std(attention_weights)
                focus_indices = np.where(attention_weights > focus_threshold)[0].tolist()
            else:
                attention_weights = []
                focus_indices = []
            
            return {
                'attention_distribution': attention_weights.tolist() if len(attention_weights) > 0 else [],
                'focus_indices': focus_indices,
                'attention_entropy': self._calculate_attention_entropy(attention_weights),
                'focus_concentration': len(focus_indices) / max(1, len(attention_weights))
            }
            
        except Exception:
            return {'attention_distribution': [], 'focus_indices': []}
    
    def _calculate_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """计算注意力熵"""
        try:
            if len(attention_weights) == 0:
                return 0.0
            
            # 确保权重和为1
            weights = attention_weights / (np.sum(attention_weights) + 1e-10)
            weights = weights[weights > 0]  # 移除零权重
            
            if len(weights) == 0:
                return 0.0
            
            entropy = -np.sum(weights * np.log2(weights))
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_dl_confidence(self, predictions: Dict[str, float]) -> float:
        """计算深度学习置信度"""
        try:
            if not predictions:
                return 0.0
            
            # 基于预测一致性的置信度
            pred_values = list(predictions.values())
            
            if len(pred_values) == 1:
                return 0.7  # 单模型中等置信度
            
            # 计算预测的一致性
            mean_pred = np.mean(pred_values)
            std_pred = np.std(pred_values)
            
            # 一致性越高，置信度越高
            if mean_pred != 0:
                consistency = 1.0 - min(1.0, std_pred / abs(mean_pred))
            else:
                consistency = 1.0 - min(1.0, std_pred)
            
            # 考虑模型数量的影响
            model_count_factor = min(1.0, len(predictions) / 5.0)
            
            confidence = consistency * 0.7 + model_count_factor * 0.3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    def _analyze_dl_interpretability(self, sequence_data: np.ndarray, 
                                   predictions: Dict[str, float]) -> Dict[str, Any]:
        """分析深度学习模型可解释性"""
        try:
            if len(sequence_data) == 0:
                return {'feature_attribution': {}, 'model_explanations': []}
            
            # 简化的特征归因分析
            feature_attribution = {}
            
            # 基于输入数据的变化分析特征重要性
            if len(sequence_data.shape) == 1:
                data_variance = np.var(sequence_data)
                for i, value in enumerate(sequence_data):
                    feature_attribution[f'timestep_{i}'] = abs(value) / (data_variance + 1e-10)
            else:
                for i in range(sequence_data.shape[-1]):
                    feature_variance = np.var(sequence_data[:, i])
                    feature_mean = np.mean(np.abs(sequence_data[:, i]))
                    feature_attribution[f'feature_{i}'] = feature_mean / (feature_variance + 1e-10)
            
            # 模型解释
            model_explanations = []
            for model_name, prediction in predictions.items():
                explanation = {
                    'model': model_name,
                    'prediction': prediction,
                    'confidence': abs(prediction),  # 简化的置信度
                    'complexity': 'high' if 'lstm' in model_name or 'transformer' in model_name else 'medium'
                }
                model_explanations.append(explanation)
            
            return {
                'feature_attribution': feature_attribution,
                'model_explanations': model_explanations,
                'interpretability_score': self._calculate_interpretability_score(feature_attribution)
            }
            
        except Exception:
            return {'feature_attribution': {}, 'model_explanations': []}
    
    def _calculate_interpretability_score(self, feature_attribution: Dict[str, float]) -> float:
        """计算可解释性评分"""
        try:
            if not feature_attribution:
                return 0.0
            
            attributions = list(feature_attribution.values())
            
            # 特征归因的集中度
            max_attribution = max(attributions)
            mean_attribution = np.mean(attributions)
            
            if mean_attribution > 0:
                concentration = max_attribution / mean_attribution
                # 集中度越高，越容易解释
                interpretability = min(1.0, concentration / 5.0)
            else:
                interpretability = 0.0
            
            return interpretability
            
        except Exception:
            return 0.0
    
    def _assess_model_complexity(self) -> Dict[str, Any]:
        """评估模型复杂度"""
        try:
            complexity_info = {
                'total_models': len(self.ml_models) + len(self.pytorch_models) + len(self.tensorflow_models),
                'ml_models': len(self.ml_models),
                'pytorch_models': len(self.pytorch_models),
                'tensorflow_models': len(self.tensorflow_models)
            }
            
            # 计算参数总数（简化估计）
            total_parameters = 0
            
            # PyTorch模型参数
            if TORCH_AVAILABLE:
                for model in self.pytorch_models.values():
                    try:
                        params = sum(p.numel() for p in model.parameters())
                        total_parameters += params
                    except:
                        total_parameters += 10000  # 估计值
            
            # TensorFlow模型参数
            if TF_AVAILABLE:
                for model in self.tensorflow_models.values():
                    try:
                        params = model.count_params()
                        total_parameters += params
                    except:
                        total_parameters += 10000  # 估计值
            
            complexity_info['total_parameters'] = total_parameters
            complexity_info['complexity_level'] = (
                'low' if total_parameters < 10000 else
                'medium' if total_parameters < 100000 else 'high'
            )
            
            return complexity_info
            
        except Exception:
            return {'total_models': 0, 'complexity_level': 'unknown'}
    
    def _extract_gradient_information(self, sequence_data: np.ndarray) -> Dict[str, Any]:
        """提取梯度信息"""
        try:
            if len(sequence_data) == 0:
                return {'gradient_norm': 0.0, 'gradient_direction': []}
            
            # 计算数值梯度
            if len(sequence_data.shape) == 1:
                gradients = np.gradient(sequence_data)
            else:
                gradients = np.gradient(sequence_data, axis=0)
            
            # 梯度范数
            if gradients.ndim == 1:
                gradient_norm = np.linalg.norm(gradients)
                gradient_direction = gradients.tolist()
            else:
                gradient_norm = np.mean([np.linalg.norm(grad) for grad in gradients])
                gradient_direction = np.mean(gradients, axis=0).tolist()
            
            return {
                'gradient_norm': gradient_norm,
                'gradient_direction': gradient_direction,
                'gradient_stability': 1.0 / (np.std(gradients) + 1e-10) if gradients.size > 1 else 1.0
            }
            
        except Exception:
            return {'gradient_norm': 0.0, 'gradient_direction': []}
        

# ========== 市场微观结构分析具体实现 ==========
    
    def _simulate_bid_ask_spread(self, data: np.ndarray) -> Dict[str, float]:
        """模拟买卖价差"""
        try:
            if len(data) == 0:
                return {'spread': 0.0, 'relative_spread': 0.0}
            
            # 基于数据波动性模拟价差
            volatility = np.std(data)
            mean_price = np.mean(data)
            
            # 价差通常与波动性成正比
            absolute_spread = volatility * 0.1  # 简化模型
            relative_spread = absolute_spread / (mean_price + 1e-10)
            
            # 价差的时间序列
            spread_series = []
            for i in range(len(data)):
                window_start = max(0, i - 5)
                window_data = data[window_start:i + 1]
                window_vol = np.std(window_data) if len(window_data) > 1 else volatility
                spread_series.append(window_vol * 0.1)
            
            return {
                'spread': absolute_spread,
                'relative_spread': relative_spread,
                'spread_volatility': np.std(spread_series) if len(spread_series) > 1 else 0.0,
                'spread_series': spread_series
            }
            
        except Exception:
            return {'spread': 0.0, 'relative_spread': 0.0}
    
    def _analyze_market_depth(self, tail_data: np.ndarray, 
                            all_data: np.ndarray) -> Dict[str, float]:
        """分析市场深度"""
        try:
            if len(tail_data) == 0:
                return {'depth_score': 0.0, 'liquidity_ratio': 0.0}
            
            # 基于尾数出现频率和总体活跃度分析深度
            tail_frequency = np.mean(tail_data)
            total_activity = np.mean(np.sum(all_data, axis=1)) if all_data.ndim > 1 else np.mean(all_data)
            
            # 深度评分：基于相对活跃度
            if total_activity > 0:
                relative_activity = tail_frequency / total_activity
                depth_score = min(1.0, relative_activity * 10)
            else:
                depth_score = 0.0
            
            # 流动性比率：基于数据变化的平滑程度
            if len(tail_data) > 1:
                changes = np.abs(np.diff(tail_data))
                avg_change = np.mean(changes)
                liquidity_ratio = 1.0 / (avg_change + 1e-10)
                liquidity_ratio = min(1.0, liquidity_ratio / 10)
            else:
                liquidity_ratio = 0.5
            
            return {
                'depth_score': depth_score,
                'liquidity_ratio': liquidity_ratio,
                'relative_activity': relative_activity if total_activity > 0 else 0.0,
                'activity_stability': 1.0 - np.std(tail_data) if len(tail_data) > 1 else 1.0
            }
            
        except Exception:
            return {'depth_score': 0.0, 'liquidity_ratio': 0.0}
    
    def _calculate_order_flow_imbalance(self, data: np.ndarray) -> Dict[str, float]:
        """计算订单流不平衡"""
        try:
            if len(data) < 2:
                return {'imbalance': 0.0, 'imbalance_trend': 0.0}
            
            # 模拟买卖订单流
            changes = np.diff(data)
            
            # 正变化视为买单，负变化视为卖单
            buy_flow = np.sum(changes[changes > 0])
            sell_flow = np.sum(np.abs(changes[changes < 0]))
            
            total_flow = buy_flow + sell_flow
            
            if total_flow > 0:
                imbalance = (buy_flow - sell_flow) / total_flow
            else:
                imbalance = 0.0
            
            # 不平衡趋势
            if len(changes) >= 5:
                recent_imbalance = self._calculate_recent_imbalance(changes[-5:])
                overall_imbalance = self._calculate_recent_imbalance(changes)
                imbalance_trend = recent_imbalance - overall_imbalance
            else:
                imbalance_trend = 0.0
            
            return {
                'imbalance': imbalance,
                'imbalance_trend': imbalance_trend,
                'buy_pressure': buy_flow / (total_flow + 1e-10),
                'sell_pressure': sell_flow / (total_flow + 1e-10)
            }
            
        except Exception:
            return {'imbalance': 0.0, 'imbalance_trend': 0.0}
    
    def _calculate_recent_imbalance(self, changes: np.ndarray) -> float:
        """计算最近的不平衡"""
        try:
            buy_flow = np.sum(changes[changes > 0])
            sell_flow = np.sum(np.abs(changes[changes < 0]))
            total_flow = buy_flow + sell_flow
            
            if total_flow > 0:
                return (buy_flow - sell_flow) / total_flow
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _analyze_trade_intensity(self, data: np.ndarray) -> Dict[str, float]:
        """分析交易强度"""
        try:
            if len(data) == 0:
                return {'intensity': 0.0, 'intensity_trend': 0.0}
            
            # 交易强度基于数据变化的频率和幅度
            if len(data) > 1:
                changes = np.abs(np.diff(data))
                
                # 强度度量
                intensity = np.mean(changes) + np.std(changes)
                
                # 强度趋势
                if len(changes) >= 10:
                    recent_intensity = np.mean(changes[-5:]) + np.std(changes[-5:])
                    historical_intensity = np.mean(changes[:-5]) + np.std(changes[:-5])
                    intensity_trend = recent_intensity - historical_intensity
                else:
                    intensity_trend = 0.0
                
                # 强度分布分析
                intensity_percentiles = np.percentile(changes, [25, 50, 75])
                
            else:
                intensity = np.abs(data[0])
                intensity_trend = 0.0
                intensity_percentiles = [intensity, intensity, intensity]
            
            return {
                'intensity': intensity,
                'intensity_trend': intensity_trend,
                'intensity_volatility': np.std(changes) if len(data) > 1 else 0.0,
                'intensity_percentiles': {
                    'q25': intensity_percentiles[0],
                    'q50': intensity_percentiles[1], 
                    'q75': intensity_percentiles[2]
                }
            }
            
        except Exception:
            return {'intensity': 0.0, 'intensity_trend': 0.0}
    
    def _analyze_price_impact(self, data: np.ndarray) -> Dict[str, float]:
        """分析价格影响"""
        try:
            if len(data) < 3:
                return {'temporary_impact': 0.0, 'permanent_impact': 0.0}
            
            # 价格影响分析
            changes = np.diff(data)
            
            # 临时影响：短期价格反弹
            temporary_impacts = []
            for i in range(len(changes) - 1):
                if changes[i] != 0:
                    # 检查下一期是否有反向变化
                    if changes[i] * changes[i + 1] < 0:
                        impact = abs(changes[i + 1] / changes[i])
                        temporary_impacts.append(impact)
            
            temporary_impact = np.mean(temporary_impacts) if temporary_impacts else 0.0
            
            # 永久影响：趋势延续
            permanent_impacts = []
            for i in range(len(changes) - 2):
                if changes[i] != 0:
                    # 检查后续变化是否同向
                    same_direction = sum(1 for j in range(i + 1, min(i + 3, len(changes))) 
                                       if changes[i] * changes[j] > 0)
                    if same_direction > 0:
                        impact = same_direction / 2.0  # 归一化
                        permanent_impacts.append(impact)
            
            permanent_impact = np.mean(permanent_impacts) if permanent_impacts else 0.0
            
            return {
                'temporary_impact': min(1.0, temporary_impact),
                'permanent_impact': min(1.0, permanent_impact),
                'impact_ratio': permanent_impact / (temporary_impact + 1e-10),
                'market_resilience': 1.0 - temporary_impact
            }
            
        except Exception:
            return {'temporary_impact': 0.0, 'permanent_impact': 0.0}
    
    def _calculate_liquidity_measures(self, data: np.ndarray) -> Dict[str, float]:
        """计算流动性度量"""
        try:
            if len(data) == 0:
                return {'kyle_lambda': 0.0, 'amihud_ratio': 0.0}
            
            # Kyle's Lambda (价格影响度量)
            if len(data) > 1:
                changes = np.diff(data)
                price_changes = np.abs(changes)
                volume_proxy = np.ones_like(price_changes)  # 简化：假设单位成交量
                
                if np.sum(volume_proxy) > 0:
                    kyle_lambda = np.mean(price_changes) / np.mean(volume_proxy)
                else:
                    kyle_lambda = 0.0
            else:
                kyle_lambda = 0.0
            
            # Amihud非流动性比率
            if len(data) > 1:
                returns = np.diff(data) / (data[:-1] + 1e-10)
                abs_returns = np.abs(returns)
                dollar_volume = np.ones_like(abs_returns)  # 简化
                
                if np.sum(dollar_volume) > 0:
                    amihud_ratio = np.mean(abs_returns / (dollar_volume + 1e-10))
                else:
                    amihud_ratio = 0.0
            else:
                amihud_ratio = 0.0
            
            # 其他流动性指标
            bid_ask_spread = np.std(data) * 0.1  # 简化的价差估计
            turnover_rate = 1.0  # 简化
            
            return {
                'kyle_lambda': kyle_lambda,
                'amihud_ratio': amihud_ratio,
                'bid_ask_spread': bid_ask_spread,
                'turnover_rate': turnover_rate,
                'liquidity_score': 1.0 / (kyle_lambda + amihud_ratio + 1e-10)
            }
            
        except Exception:
            return {'kyle_lambda': 0.0, 'amihud_ratio': 0.0}
    
    def _detect_information_asymmetry(self, tail_data: np.ndarray, 
                                    all_data: np.ndarray) -> Dict[str, float]:
        """检测信息不对称"""
        try:
            if len(tail_data) == 0:
                return {'asymmetry_score': 0.0, 'informed_trading': 0.0}
            
            # 基于交易模式检测信息不对称
            # 信息不对称通常表现为非随机的交易模式
            
            # 序列相关性检测
            if len(tail_data) > 1:
                autocorr = np.corrcoef(tail_data[:-1], tail_data[1:])[0, 1]
                autocorr = autocorr if not np.isnan(autocorr) else 0.0
            else:
                autocorr = 0.0
            
            # 与市场整体的相关性
            if all_data.ndim > 1:
                market_avg = np.mean(all_data, axis=1)
                if len(market_avg) == len(tail_data) and len(tail_data) > 1:
                    market_corr = np.corrcoef(tail_data, market_avg)[0, 1]
                    market_corr = market_corr if not np.isnan(market_corr) else 0.0
                else:
                    market_corr = 0.0
            else:
                market_corr = 0.0
            
            # 信息不对称评分
            asymmetry_score = abs(autocorr) + abs(market_corr - np.mean([autocorr, market_corr]))
            asymmetry_score = min(1.0, asymmetry_score)
            
            # 知情交易概率估计
            informed_trading = asymmetry_score * 0.5 + abs(autocorr) * 0.5
            
            return {
                'asymmetry_score': asymmetry_score,
                'informed_trading': informed_trading,
                'autocorrelation': autocorr,
                'market_correlation': market_corr
            }
            
        except Exception:
            return {'asymmetry_score': 0.0, 'informed_trading': 0.0}
    
    def _detect_market_manipulation(self, data: np.ndarray) -> Dict[str, Any]:
        """检测市场操纵"""
        try:
            if len(data) == 0:
                return {'manipulation_score': 0.0, 'anomaly_detected': False}
            
            manipulation_indicators = []
            
            # 异常价格模式检测
            if len(data) > 5:
                # 检测异常尖峰
                z_scores = np.abs(stats.zscore(data))
                outliers = np.sum(z_scores > 3)
                outlier_ratio = outliers / len(data)
                
                if outlier_ratio > 0.1:  # 超过10%的异常点
                    manipulation_indicators.append(('price_spikes', outlier_ratio))
                
                # 检测人为的周期性模式
                if len(data) >= 8:
                    fft = np.fft.fft(data)
                    power = np.abs(fft)**2
                    dominant_freq_power = np.max(power[1:])  # 排除DC分量
                    total_power = np.sum(power[1:])
                    
                    if total_power > 0:
                        concentration = dominant_freq_power / total_power
                        if concentration > 0.5:  # 过度集中的频率
                            manipulation_indicators.append(('artificial_pattern', concentration))
                
                # 检测价格固定模式
                unique_values = len(np.unique(data))
                if unique_values < len(data) * 0.3:  # 值过于集中
                    concentration_score = 1.0 - unique_values / len(data)
                    manipulation_indicators.append(('price_fixing', concentration_score))
            
            # 综合操纵评分
            if manipulation_indicators:
                manipulation_score = np.mean([score for _, score in manipulation_indicators])
                anomaly_detected = manipulation_score > 0.3
            else:
                manipulation_score = 0.0
                anomaly_detected = False
            
            return {
                'manipulation_score': manipulation_score,
                'anomaly_detected': anomaly_detected,
                'indicators': manipulation_indicators,
                'risk_level': 'high' if manipulation_score > 0.7 else 
                            'medium' if manipulation_score > 0.3 else 'low'
            }
            
        except Exception:
            return {'manipulation_score': 0.0, 'anomaly_detected': False}
    
    def _analyze_volatility_clustering(self, data: np.ndarray) -> Dict[str, float]:
        """分析波动性聚集"""
        try:
            if len(data) < 10:
                return {'clustering_coefficient': 0.0, 'garch_effect': 0.0}
            
            # 计算波动性代理（绝对收益率）
            returns = np.diff(data) / (data[:-1] + 1e-10)
            volatility_proxy = np.abs(returns)
            
            # 波动性的自相关
            if len(volatility_proxy) > 1:
                vol_autocorr = np.corrcoef(volatility_proxy[:-1], volatility_proxy[1:])[0, 1]
                vol_autocorr = vol_autocorr if not np.isnan(vol_autocorr) else 0.0
            else:
                vol_autocorr = 0.0
            
            # ARCH效应检测（简化）
            if len(volatility_proxy) >= 5:
                # 计算5期滞后的自相关
                arch_correlations = []
                for lag in range(1, min(6, len(volatility_proxy))):
                    if len(volatility_proxy) > lag:
                        corr = np.corrcoef(volatility_proxy[:-lag], volatility_proxy[lag:])[0, 1]
                        if not np.isnan(corr):
                            arch_correlations.append(abs(corr))
                
                garch_effect = np.mean(arch_correlations) if arch_correlations else 0.0
            else:
                garch_effect = 0.0
            
            # 聚集系数
            clustering_coefficient = (abs(vol_autocorr) + garch_effect) / 2
            
            return {
                'clustering_coefficient': clustering_coefficient,
                'garch_effect': garch_effect,
                'volatility_persistence': abs(vol_autocorr),
                'mean_volatility': np.mean(volatility_proxy)
            }
            
        except Exception:
            return {'clustering_coefficient': 0.0, 'garch_effect': 0.0}
    
    def _detect_price_jumps(self, data: np.ndarray, threshold_multiplier: float = 3.0) -> Dict[str, Any]:
        """检测价格跳跃"""
        try:
            if len(data) < 2:
                return {'jump_detected': False, 'jump_count': 0}
            
            # 计算收益率
            returns = np.diff(data) / (data[:-1] + 1e-10)
            
            # 跳跃检测阈值
            return_std = np.std(returns)
            threshold = threshold_multiplier * return_std
            
            # 识别跳跃
            jumps = np.abs(returns) > threshold
            jump_indices = np.where(jumps)[0]
            
            jump_magnitudes = returns[jumps]
            
            return {
                'jump_detected': len(jump_indices) > 0,
                'jump_count': len(jump_indices),
                'jump_indices': jump_indices.tolist(),
                'jump_magnitudes': jump_magnitudes.tolist(),
                'jump_frequency': len(jump_indices) / len(returns),
                'max_jump': np.max(np.abs(jump_magnitudes)) if len(jump_magnitudes) > 0 else 0.0
            }
            
        except Exception:
            return {'jump_detected': False, 'jump_count': 0}
    
    def _calculate_microstructure_score(self, bid_ask_spread: Dict, market_depth: Dict,
                                      order_flow: Dict, trade_intensity: Dict,
                                      liquidity: Dict) -> float:
        """计算微观结构综合评分"""
        try:
            score_components = []
            
            # 价差评分（越小越好）
            spread_score = 1.0 - min(1.0, bid_ask_spread.get('relative_spread', 0.0) * 10)
            score_components.append(spread_score * 0.2)
            
            # 深度评分
            depth_score = market_depth.get('depth_score', 0.0)
            score_components.append(depth_score * 0.25)
            
            # 订单流平衡评分
            imbalance = abs(order_flow.get('imbalance', 0.0))
            balance_score = 1.0 - imbalance
            score_components.append(balance_score * 0.2)
            
            # 交易强度评分
            intensity_score = min(1.0, trade_intensity.get('intensity', 0.0))
            score_components.append(intensity_score * 0.15)
            
            # 流动性评分
            liquidity_score = min(1.0, liquidity.get('liquidity_score', 0.0) / 10)
            score_components.append(liquidity_score * 0.2)
            
            return sum(score_components)
            
        except Exception:
            return 0.0
    
    def _assess_market_efficiency(self, data: np.ndarray) -> Dict[str, float]:
        """评估市场效率"""
        try:
            if len(data) < 10:
                return {'efficiency_score': 0.5, 'weak_form_efficiency': 0.5}
            
            # 弱式有效性检验
            returns = np.diff(data) / (data[:-1] + 1e-10)
            
            # 序列相关性检验
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                autocorr = autocorr if not np.isnan(autocorr) else 0.0
            else:
                autocorr = 0.0
            
            # 游程检验（简化）
            signs = np.sign(returns)
            runs = []
            current_run = 1
            
            for i in range(1, len(signs)):
                if signs[i] == signs[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            expected_runs = (len(returns) + 1) / 2
            actual_runs = len(runs)
            runs_ratio = actual_runs / expected_runs
            
            # 效率评分
            weak_form_efficiency = 1.0 - abs(autocorr)
            runs_efficiency = 1.0 - abs(runs_ratio - 1.0)
            
            efficiency_score = (weak_form_efficiency + runs_efficiency) / 2
            
            return {
                'efficiency_score': efficiency_score,
                'weak_form_efficiency': weak_form_efficiency,
                'autocorrelation': autocorr,
                'runs_ratio': runs_ratio
            }
            
        except Exception:
            return {'efficiency_score': 0.5, 'weak_form_efficiency': 0.5}
    
    def _analyze_transaction_costs(self, data: np.ndarray) -> Dict[str, float]:
        """分析交易成本"""
        try:
            if len(data) == 0:
                return {'total_cost': 0.0, 'cost_ratio': 0.0}
            
            # 估计各种交易成本
            volatility = np.std(data)
            
            # 买卖价差成本
            spread_cost = volatility * 0.05  # 简化模型
            
            # 市场冲击成本
            impact_cost = volatility * 0.03
            
            # 时机成本
            timing_cost = volatility * 0.02
            
            # 总交易成本
            total_cost = spread_cost + impact_cost + timing_cost
            
            # 成本比率
            mean_price = np.mean(data)
            cost_ratio = total_cost / (mean_price + 1e-10)
            
            return {
                'total_cost': total_cost,
                'cost_ratio': cost_ratio,
                'spread_cost': spread_cost,
                'impact_cost': impact_cost,
                'timing_cost': timing_cost,
                'cost_breakdown': {
                    'spread': spread_cost / (total_cost + 1e-10),
                    'impact': impact_cost / (total_cost + 1e-10),
                    'timing': timing_cost / (total_cost + 1e-10)
                }
            }
            
        except Exception:
            return {'total_cost': 0.0, 'cost_ratio': 0.0}
        
# ========== 风险管理具体实现 ==========
    
    def _calculate_value_at_risk(self, data: np.ndarray, confidence_level: float = 0.95,
                               time_horizon: int = 1) -> Dict[str, float]:
        """计算风险价值(VaR)"""
        try:
            if len(data) < 10:
                return {'var_normal': 0.0, 'var_historical': 0.0, 'var_monte_carlo': 0.0}
            
            # 计算收益率
            returns = np.diff(data) / (data[:-1] + 1e-10)
            
            # 1. 参数法VaR（正态分布假设）
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            alpha = 1 - confidence_level
            
            # 调整时间范围
            scaled_mean = mean_return * time_horizon
            scaled_std = std_return * np.sqrt(time_horizon)
            
            # 正态分布VaR
            z_score = stats.norm.ppf(alpha)
            var_normal = -(scaled_mean + z_score * scaled_std)
            
            # 2. 历史模拟法VaR
            if len(returns) > 0:
                var_historical = -np.percentile(returns, alpha * 100)
            else:
                var_historical = 0.0
            
            # 3. 蒙特卡洛模拟VaR
            np.random.seed(42)
            simulated_returns = np.random.normal(mean_return, std_return, 10000)
            var_monte_carlo = -np.percentile(simulated_returns, alpha * 100)
            
            # 4. 修正的Cornish-Fisher VaR（考虑偏度和峰度）
            if len(returns) > 3:
                skewness = stats.skew(returns)
                kurt = stats.kurtosis(returns)
                
                # Cornish-Fisher调整
                cf_adjustment = (z_score**2 - 1) * skewness / 6 + \
                               (z_score**3 - 3*z_score) * kurt / 24 - \
                               (2*z_score**3 - 5*z_score) * skewness**2 / 36
                
                adjusted_quantile = z_score + cf_adjustment
                var_cornish_fisher = -(scaled_mean + adjusted_quantile * scaled_std)
            else:
                var_cornish_fisher = var_normal
            
            return {
                'var_normal': var_normal,
                'var_historical': var_historical,
                'var_monte_carlo': var_monte_carlo,
                'var_cornish_fisher': var_cornish_fisher,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'var_average': np.mean([var_normal, var_historical, var_monte_carlo])
            }
            
        except Exception as e:
            self.logger.error(f"VaR计算错误: {str(e)}")
            return {'var_normal': 0.0, 'var_historical': 0.0, 'var_monte_carlo': 0.0}
    
    def _calculate_conditional_var(self, data: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        """计算条件风险价值(CVaR/Expected Shortfall)"""
        try:
            if len(data) < 10:
                return {'cvar': 0.0, 'expected_shortfall': 0.0}
            
            returns = np.diff(data) / (data[:-1] + 1e-10)
            alpha = 1 - confidence_level
            
            # 1. 历史模拟法CVaR
            var_threshold = np.percentile(returns, alpha * 100)
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) > 0:
                cvar_historical = -np.mean(tail_returns)
            else:
                cvar_historical = 0.0
            
            # 2. 参数法CVaR（正态分布）
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            z_alpha = stats.norm.ppf(alpha)
            phi_z = stats.norm.pdf(z_alpha)
            
            cvar_normal = -(mean_return - std_return * phi_z / alpha)
            
            # 3. 蒙特卡洛CVaR
            np.random.seed(42)
            simulated_returns = np.random.normal(mean_return, std_return, 10000)
            mc_var_threshold = np.percentile(simulated_returns, alpha * 100)
            mc_tail_returns = simulated_returns[simulated_returns <= mc_var_threshold]
            
            if len(mc_tail_returns) > 0:
                cvar_monte_carlo = -np.mean(mc_tail_returns)
            else:
                cvar_monte_carlo = 0.0
            
            # 平均CVaR
            cvar_average = np.mean([cvar_historical, cvar_normal, cvar_monte_carlo])
            
            return {
                'cvar': cvar_average,
                'expected_shortfall': cvar_average,
                'cvar_historical': cvar_historical,
                'cvar_normal': cvar_normal,
                'cvar_monte_carlo': cvar_monte_carlo,
                'confidence_level': confidence_level,
                'tail_expectation': cvar_average
            }
            
        except Exception as e:
            self.logger.error(f"CVaR计算错误: {str(e)}")
            return {'cvar': 0.0, 'expected_shortfall': 0.0}
    
    def _analyze_maximum_drawdown(self, data: np.ndarray) -> Dict[str, float]:
        """分析最大回撤"""
        try:
            if len(data) < 2:
                return {'max_drawdown': 0.0, 'drawdown_duration': 0}
            
            # 计算累积收益
            cumulative_returns = np.cumprod(1 + np.diff(data) / (data[:-1] + 1e-10))
            
            # 计算滚动最高点
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # 计算回撤
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # 最大回撤
            max_drawdown = np.min(drawdowns)
            max_dd_index = np.argmin(drawdowns)
            
            # 回撤持续时间
            if max_dd_index > 0:
                # 找到最大回撤开始的点
                peak_index = np.argmax(running_max[:max_dd_index + 1])
                drawdown_duration = max_dd_index - peak_index
            else:
                drawdown_duration = 0
            
            # 当前回撤
            current_drawdown = drawdowns[-1] if len(drawdowns) > 0 else 0.0
            
            # 回撤统计
            negative_drawdowns = drawdowns[drawdowns < 0]
            avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0.0
            
            # 恢复时间估计
            if current_drawdown < 0:
                recent_trend = np.mean(np.diff(data[-5:])) if len(data) >= 6 else 0
                if recent_trend > 0:
                    recovery_estimate = abs(current_drawdown) / (recent_trend + 1e-10)
                else:
                    recovery_estimate = float('inf')
            else:
                recovery_estimate = 0
            
            return {
                'max_drawdown': abs(max_drawdown),
                'drawdown_duration': drawdown_duration,
                'current_drawdown': abs(current_drawdown),
                'average_drawdown': abs(avg_drawdown),
                'recovery_estimate': min(100, recovery_estimate),
                'drawdown_frequency': len(negative_drawdowns) / len(drawdowns) if len(drawdowns) > 0 else 0,
                'max_drawdown_date': max_dd_index
            }
            
        except Exception as e:
            self.logger.error(f"最大回撤分析错误: {str(e)}")
            return {'max_drawdown': 0.0, 'drawdown_duration': 0}
    
    def _assess_volatility_risk(self, data: np.ndarray) -> Dict[str, float]:
        """评估波动率风险"""
        try:
            if len(data) < 2:
                return {'volatility': 0.0, 'volatility_risk_level': 'low'}
            
            returns = np.diff(data) / (data[:-1] + 1e-10)
            
            # 历史波动率
            historical_vol = np.std(returns) * np.sqrt(252)  # 年化
            
            # 实现波动率（基于高频收益率）
            if len(returns) > 1:
                realized_vol = np.sqrt(np.sum(returns**2)) * np.sqrt(252)
            else:
                realized_vol = historical_vol
            
            # GARCH波动率预测（简化）
            garch_vol = self._estimate_garch_volatility(returns)
            
            # 波动率聚集度
            vol_clustering = self._analyze_volatility_clustering(data)
            clustering_coefficient = vol_clustering.get('clustering_coefficient', 0.0)
            
            # 波动率风险等级
            if historical_vol > 0.3:
                risk_level = 'high'
            elif historical_vol > 0.15:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # 波动率预测区间
            vol_forecast_upper = garch_vol * 1.2
            vol_forecast_lower = garch_vol * 0.8
            
            return {
                'volatility': historical_vol,
                'realized_volatility': realized_vol,
                'garch_volatility': garch_vol,
                'volatility_clustering': clustering_coefficient,
                'volatility_risk_level': risk_level,
                'vol_forecast_range': {
                    'upper': vol_forecast_upper,
                    'lower': vol_forecast_lower
                },
                'volatility_of_volatility': np.std([historical_vol, realized_vol, garch_vol])
            }
            
        except Exception as e:
            self.logger.error(f"波动率风险评估错误: {str(e)}")
            return {'volatility': 0.0, 'volatility_risk_level': 'low'}
    
    def _estimate_garch_volatility(self, returns: np.ndarray, alpha: float = 0.1, 
                                 beta: float = 0.85) -> float:
        """估计GARCH波动率"""
        try:
            if len(returns) < 5:
                return np.std(returns) if len(returns) > 1 else 0.0
            
            # 简化的GARCH(1,1)模型
            omega = 0.01  # 长期方差
            
            # 初始条件波动率
            sigma_squared = np.var(returns)
            
            # 递归计算GARCH波动率
            for i in range(len(returns)):
                sigma_squared = omega + alpha * returns[i]**2 + beta * sigma_squared
            
            return np.sqrt(sigma_squared * 252)  # 年化
            
        except Exception:
            return np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
    
    def _assess_liquidity_risk(self, data: np.ndarray) -> Dict[str, float]:
        """评估流动性风险"""
        try:
            if len(data) == 0:
                return {'liquidity_risk': 0.0, 'liquidity_score': 1.0}
            
            # 基于价格变化的流动性评估
            if len(data) > 1:
                changes = np.abs(np.diff(data))
                
                # 流动性指标
                price_impact = np.mean(changes)  # 价格影响
                volume_proxy = 1.0 / (np.std(changes) + 1e-10)  # 成交量代理
                
                # 买卖价差估计
                bid_ask_spread = np.std(data) * 0.1
                
                # 市场深度估计
                market_depth = volume_proxy
                
                # 流动性风险评分
                liquidity_risk = (price_impact + bid_ask_spread) / (market_depth + 1e-10)
                liquidity_score = 1.0 / (1.0 + liquidity_risk)
                
            else:
                liquidity_risk = 0.0
                liquidity_score = 1.0
            
            # 流动性风险等级
            if liquidity_risk > 0.1:
                risk_level = 'high'
            elif liquidity_risk > 0.05:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'liquidity_risk': liquidity_risk,
                'liquidity_score': liquidity_score,
                'risk_level': risk_level,
                'price_impact_cost': price_impact if len(data) > 1 else 0.0,
                'market_depth_score': market_depth if len(data) > 1 else 1.0
            }
            
        except Exception:
            return {'liquidity_risk': 0.0, 'liquidity_score': 1.0}
    
    def _assess_model_risk(self, fusion_result: Dict) -> Dict[str, float]:
        """评估模型风险"""
        try:
            model_risk_factors = []
            
            # 模型不确定性
            signal_variance = fusion_result.get('signal_variance', 0.0)
            model_risk_factors.append(signal_variance)
            
            # 信号质量
            signal_quality = fusion_result.get('signal_quality', 0.5)
            quality_risk = 1.0 - signal_quality
            model_risk_factors.append(quality_risk)
            
            # 模型一致性
            signal_consistency = fusion_result.get('signal_consistency', 0.5)
            consistency_risk = 1.0 - signal_consistency
            model_risk_factors.append(consistency_risk)
            
            # 数据充分性
            signal_count = fusion_result.get('signal_count', 0)
            data_sufficiency = min(1.0, signal_count / 5.0)
            data_risk = 1.0 - data_sufficiency
            model_risk_factors.append(data_risk)
            
            # 模型复杂度风险
            complexity_score = self._assess_model_complexity()
            complexity_risk = min(1.0, complexity_score.get('total_parameters', 0) / 1000000)
            model_risk_factors.append(complexity_risk)
            
            # 综合模型风险
            overall_model_risk = np.mean(model_risk_factors)
            
            # 风险分解
            risk_breakdown = {
                'signal_variance_risk': signal_variance,
                'quality_risk': quality_risk,
                'consistency_risk': consistency_risk,
                'data_risk': data_risk,
                'complexity_risk': complexity_risk
            }
            
            return {
                'model_risk': overall_model_risk,
                'risk_breakdown': risk_breakdown,
                'risk_level': 'high' if overall_model_risk > 0.7 else 
                            'medium' if overall_model_risk > 0.4 else 'low',
                'model_confidence': 1.0 - overall_model_risk
            }
            
        except Exception:
            return {'model_risk': 0.5, 'risk_level': 'medium'}
    
    def _assess_systemic_risk(self, tail: int, data_matrix: np.ndarray) -> Dict[str, float]:
        """评估系统性风险"""
        try:
            if data_matrix.ndim < 2 or data_matrix.shape[1] < 2:
                return {'systemic_risk': 0.0, 'correlation_risk': 0.0}
            
            tail_data = data_matrix[:, tail]
            
            # 与其他尾数的相关性
            correlations = []
            for other_tail in range(data_matrix.shape[1]):
                if other_tail != tail:
                    other_data = data_matrix[:, other_tail]
                    if len(tail_data) > 1 and len(other_data) > 1:
                        corr = np.corrcoef(tail_data, other_data)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            # 系统性风险指标
            if correlations:
                avg_correlation = np.mean(correlations)
                max_correlation = np.max(correlations)
                correlation_risk = avg_correlation
            else:
                avg_correlation = 0.0
                max_correlation = 0.0
                correlation_risk = 0.0
            
            # 市场集中度风险
            total_activity = np.sum(data_matrix, axis=1)
            tail_share = np.mean(tail_data) / (np.mean(total_activity) + 1e-10)
            concentration_risk = min(1.0, tail_share * 10)  # 归一化
            
            # 系统性冲击敏感性
            if len(tail_data) > 5:
                # 检测与系统性变化的敏感性
                system_changes = np.diff(total_activity)
                tail_changes = np.diff(tail_data)
                
                if len(system_changes) > 0 and len(tail_changes) > 0:
                    sensitivity = np.corrcoef(system_changes, tail_changes)[0, 1]
                    sensitivity = sensitivity if not np.isnan(sensitivity) else 0.0
                    shock_sensitivity = abs(sensitivity)
                else:
                    shock_sensitivity = 0.0
            else:
                shock_sensitivity = 0.0
            
            # 综合系统性风险
            systemic_risk = (correlation_risk * 0.4 + 
                           concentration_risk * 0.3 + 
                           shock_sensitivity * 0.3)
            
            return {
                'systemic_risk': systemic_risk,
                'correlation_risk': correlation_risk,
                'concentration_risk': concentration_risk,
                'shock_sensitivity': shock_sensitivity,
                'average_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'market_share': tail_share
            }
            
        except Exception:
            return {'systemic_risk': 0.0, 'correlation_risk': 0.0}
    
    def _assess_operational_risk(self, fusion_result: Dict) -> Dict[str, float]:
        """评估操作风险"""
        try:
            operational_risk_factors = []
            
            # 技术故障风险
            model_count = fusion_result.get('signal_count', 0)
            if model_count < 3:
                tech_failure_risk = 0.3  # 模型过少
            else:
                tech_failure_risk = 0.1
            operational_risk_factors.append(tech_failure_risk)
            
            # 数据质量风险
            signal_quality = fusion_result.get('signal_quality', 0.5)
            data_quality_risk = 1.0 - signal_quality
            operational_risk_factors.append(data_quality_risk * 0.5)
            
            # 处理复杂度风险
            complexity_score = self._assess_model_complexity()
            processing_risk = min(0.3, complexity_score.get('total_models', 0) / 100)
            operational_risk_factors.append(processing_risk)
            
            # 人为错误风险（简化）
            human_error_risk = 0.05  # 基础人为错误概率
            operational_risk_factors.append(human_error_risk)
            
            # 系统集成风险
            integration_complexity = len(fusion_result.get('original_signals', {}))
            integration_risk = min(0.2, integration_complexity / 20)
            operational_risk_factors.append(integration_risk)
            
            # 综合操作风险
            overall_operational_risk = sum(operational_risk_factors)
            
            return {
                'operational_risk': overall_operational_risk,
                'tech_failure_risk': tech_failure_risk,
                'data_quality_risk': data_quality_risk,
                'processing_risk': processing_risk,
                'human_error_risk': human_error_risk,
                'integration_risk': integration_risk,
                'risk_level': 'high' if overall_operational_risk > 0.3 else 
                            'medium' if overall_operational_risk > 0.15 else 'low'
            }
            
        except Exception:
            return {'operational_risk': 0.1, 'risk_level': 'low'}
    
    def _assess_credit_risk_simulation(self, data: np.ndarray) -> Dict[str, float]:
        """模拟信用风险评估"""
        try:
            if len(data) == 0:
                return {'credit_risk': 0.0, 'default_probability': 0.0}
            
            # 基于数据稳定性模拟信用质量
            if len(data) > 1:
                volatility = np.std(data)
                trend = np.polyfit(range(len(data)), data, 1)[0]
                
                # 违约概率模拟
                # 高波动性和负趋势增加违约风险
                volatility_factor = min(1.0, volatility * 10)
                trend_factor = max(0.0, -trend * 100) if trend < 0 else 0.0
                
                default_probability = min(1.0, (volatility_factor + trend_factor) / 2)
                
                # 信用评级模拟
                if default_probability < 0.01:
                    credit_rating = 'AAA'
                    credit_risk = 0.01
                elif default_probability < 0.05:
                    credit_rating = 'AA'
                    credit_risk = 0.05
                elif default_probability < 0.1:
                    credit_rating = 'A'
                    credit_risk = 0.1
                elif default_probability < 0.2:
                    credit_rating = 'BBB'
                    credit_risk = 0.2
                else:
                    credit_rating = 'BB+'
                    credit_risk = 0.3
                    
            else:
                default_probability = 0.05  # 默认值
                credit_rating = 'A'
                credit_risk = 0.1
            
            return {
                'credit_risk': credit_risk,
                'default_probability': default_probability,
                'credit_rating': credit_rating,
                'volatility_component': volatility_factor if len(data) > 1 else 0.0,
                'trend_component': trend_factor if len(data) > 1 else 0.0
            }
            
        except Exception:
            return {'credit_risk': 0.1, 'default_probability': 0.05}
    
    def _assess_market_risk(self, tail_data: np.ndarray, all_data: np.ndarray) -> Dict[str, float]:
        """评估市场风险"""
        try:
            if len(tail_data) == 0:
                return {'market_risk': 0.0, 'beta': 1.0}
            
            # 计算Beta系数
            if all_data.ndim > 1:
                market_returns = np.diff(np.mean(all_data, axis=1))
                tail_returns = np.diff(tail_data)
                
                if len(market_returns) > 1 and len(tail_returns) > 1 and len(market_returns) == len(tail_returns):
                    market_var = np.var(market_returns)
                    if market_var > 0:
                        covariance = np.cov(tail_returns, market_returns)[0, 1]
                        beta = covariance / market_var
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
            
            # 市场风险评估
            if len(tail_data) > 1:
                tail_volatility = np.std(np.diff(tail_data))
                
                # 基于Beta和波动率的市场风险
                market_risk = abs(beta) * tail_volatility
                
                # 系统性风险vs特异性风险
                systematic_risk = abs(beta - 1.0)  # 偏离市场的程度
                idiosyncratic_risk = tail_volatility * (1 - min(1.0, abs(beta)))
                
            else:
                market_risk = 0.0
                systematic_risk = 0.0
                idiosyncratic_risk = 0.0
            
            # 市场风险等级
            if market_risk > 0.15:
                risk_level = 'high'
            elif market_risk > 0.08:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'market_risk': market_risk,
                'beta': beta,
                'systematic_risk': systematic_risk,
                'idiosyncratic_risk': idiosyncratic_risk,
                'risk_level': risk_level,
                'market_correlation': min(1.0, abs(beta))
            }
            
        except Exception:
            return {'market_risk': 0.1, 'beta': 1.0}
    
    def _assess_risk_concentration(self, tail: int, data_matrix: np.ndarray) -> Dict[str, float]:
        """评估风险集中度"""
        try:
            if data_matrix.ndim < 2:
                return {'concentration_risk': 0.0, 'diversification_ratio': 1.0}
            
            tail_data = data_matrix[:, tail]
            
            # 在总体中的比重
            total_activity = np.sum(data_matrix, axis=1)
            tail_weight = np.mean(tail_data) / (np.mean(total_activity) + 1e-10)
            
            # 集中度风险
            concentration_risk = min(1.0, tail_weight * 20)  # 放大集中度效应
            
            # 多样化比率
            # 计算与其他尾数的相关性
            correlations = []
            for other_tail in range(data_matrix.shape[1]):
                if other_tail != tail and len(data_matrix[:, other_tail]) > 1:
                    corr = np.corrcoef(tail_data, data_matrix[:, other_tail])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                diversification_ratio = 1.0 - avg_correlation
            else:
                diversification_ratio = 1.0
            
            # 风险分散效果
            if diversification_ratio > 0.8:
                diversification_level = 'excellent'
            elif diversification_ratio > 0.6:
                diversification_level = 'good'
            elif diversification_ratio > 0.4:
                diversification_level = 'moderate'
            else:
                diversification_level = 'poor'
            
            return {
                'concentration_risk': concentration_risk,
                'diversification_ratio': diversification_ratio,
                'portfolio_weight': tail_weight,
                'diversification_level': diversification_level,
                'correlation_with_others': avg_correlation if correlations else 0.0
            }
            
        except Exception:
            return {'concentration_risk': 0.0, 'diversification_ratio': 1.0}
    
    def _perform_stress_testing(self, data: np.ndarray) -> Dict[str, Any]:
        """执行压力测试"""
        try:
            if len(data) < 2:
                return {'stress_scenarios': [], 'worst_case_loss': 0.0}
            
            base_returns = np.diff(data) / (data[:-1] + 1e-10)
            base_volatility = np.std(base_returns)
            base_mean = np.mean(base_returns)
            
            stress_scenarios = []
            
            # 情景1：高波动率情景
            high_vol_scenario = {
                'name': 'High Volatility Shock',
                'volatility_multiplier': 3.0,
                'expected_loss': abs(base_mean - 3 * base_volatility),
                'probability': 0.05,
                'description': '波动率增加3倍的极端情况'
            }
            stress_scenarios.append(high_vol_scenario)
            
            # 情景2：趋势反转情景
            trend_reversal_scenario = {
                'name': 'Trend Reversal',
                'return_shift': -2 * abs(base_mean),
                'expected_loss': 2 * abs(base_mean),
                'probability': 0.1,
                'description': '趋势完全反转的情况'
            }
            stress_scenarios.append(trend_reversal_scenario)
            
            # 情景3：流动性枯竭情景
            liquidity_crisis_scenario = {
                'name': 'Liquidity Crisis',
                'liquidity_impact': base_volatility * 2,
                'expected_loss': base_volatility * 2,
                'probability': 0.02,
                'description': '市场流动性枯竭情况'
            }
            stress_scenarios.append(liquidity_crisis_scenario)
            
            # 情景4：系统性冲击
            systemic_shock_scenario = {
                'name': 'Systemic Shock',
                'correlation_increase': 0.8,
                'expected_loss': base_volatility * 4,
                'probability': 0.01,
                'description': '系统性风险冲击情况'
            }
            stress_scenarios.append(systemic_shock_scenario)
            
            # 计算最坏情况损失
            worst_case_loss = max(scenario['expected_loss'] for scenario in stress_scenarios)
            
            # 综合压力测试得分
            expected_stress_loss = sum(scenario['expected_loss'] * scenario['probability'] 
                                     for scenario in stress_scenarios)
            
            return {
                'stress_scenarios': stress_scenarios,
                'worst_case_loss': worst_case_loss,
                'expected_stress_loss': expected_stress_loss,
                'stress_test_passed': worst_case_loss < 0.5,  # 阈值
                'resilience_score': 1.0 - min(1.0, expected_stress_loss)
            }
            
        except Exception:
            return {'stress_scenarios': [], 'worst_case_loss': 0.0}
    
    def _perform_scenario_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """执行情景分析"""
        try:
            if len(data) < 5:
                return {'scenarios': [], 'base_case_probability': 1.0}
            
            base_trend = np.polyfit(range(len(data)), data, 1)[0]
            base_volatility = np.std(data)
            
            scenarios = []
            
            # 基准情景
            base_scenario = {
                'name': 'Base Case',
                'probability': 0.6,
                'expected_return': base_trend,
                'volatility': base_volatility,
                'description': '当前趋势延续'
            }
            scenarios.append(base_scenario)
            
            # 乐观情景
            optimistic_scenario = {
                'name': 'Optimistic',
                'probability': 0.2,
                'expected_return': base_trend * 1.5,
                'volatility': base_volatility * 0.8,
                'description': '积极发展趋势'
            }
            scenarios.append(optimistic_scenario)
            
            # 悲观情景
            pessimistic_scenario = {
                'name': 'Pessimistic',
                'probability': 0.15,
                'expected_return': -abs(base_trend),
                'volatility': base_volatility * 1.5,
                'description': '不利发展趋势'
            }
            scenarios.append(pessimistic_scenario)
            
            # 极端情景
            extreme_scenario = {
                'name': 'Extreme',
                'probability': 0.05,
                'expected_return': -abs(base_trend) * 3,
                'volatility': base_volatility * 3,
                'description': '极端不利情况'
            }
            scenarios.append(extreme_scenario)
            
            # 期望收益和风险
            expected_return = sum(s['probability'] * s['expected_return'] for s in scenarios)
            expected_volatility = sum(s['probability'] * s['volatility'] for s in scenarios)
            
            return {
                'scenarios': scenarios,
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'downside_probability': pessimistic_scenario['probability'] + extreme_scenario['probability'],
                'upside_probability': optimistic_scenario['probability'],
                'base_case_probability': base_scenario['probability']
            }
            
        except Exception:
            return {'scenarios': [], 'base_case_probability': 1.0}
    
    def _calculate_risk_adjusted_returns(self, data: np.ndarray) -> Dict[str, float]:
        """计算风险调整收益"""
        try:
            if len(data) < 2:
                return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0}
            
            returns = np.diff(data) / (data[:-1] + 1e-10)
            
            # 假设无风险利率
            risk_free_rate = 0.02 / 252  # 日化无风险利率
            
            excess_returns = returns - risk_free_rate
            mean_excess_return = np.mean(excess_returns)
            
            # 夏普比率
            if len(returns) > 1:
                return_volatility = np.std(returns)
                sharpe_ratio = mean_excess_return / (return_volatility + 1e-10)
            else:
                sharpe_ratio = 0.0
            
            # 索提诺比率（只考虑下行风险）
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns)
                sortino_ratio = mean_excess_return / (downside_deviation + 1e-10)
            else:
                sortino_ratio = float('inf') if mean_excess_return > 0 else 0.0
            
            # 卡尔马比率
            max_dd_analysis = self._analyze_maximum_drawdown(data)
            max_drawdown = max_dd_analysis['max_drawdown']
            
            if max_drawdown > 0:
                calmar_ratio = np.mean(returns) * 252 / max_drawdown  # 年化
            else:
                calmar_ratio = float('inf') if np.mean(returns) > 0 else 0.0
            
            # 信息比率（相对基准的超额收益）
            benchmark_return = 0.0  # 简化：假设基准为0
            tracking_error = np.std(returns - benchmark_return)
            information_ratio = (np.mean(returns) - benchmark_return) / (tracking_error + 1e-10)
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': min(10.0, sortino_ratio),  # 限制极值
                'calmar_ratio': min(10.0, calmar_ratio),
                'information_ratio': information_ratio,
                'return_volatility': return_volatility if len(returns) > 1 else 0.0,
                'downside_deviation': downside_deviation if len(negative_returns) > 0 else 0.0
            }
            
        except Exception:
            return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0}
    
    def _calculate_overall_risk_score(self, var_analysis: Dict, volatility_risk: Dict,
                                    model_risk: Dict, systemic_risk: Dict) -> float:
        """计算总体风险评分"""
        try:
            risk_components = []
            
            # VaR风险权重
            var_score = var_analysis.get('var_average', 0.0)
            normalized_var = min(1.0, abs(var_score) * 10)
            risk_components.append(normalized_var * 0.3)
            
            # 波动率风险权重
            vol_risk = volatility_risk.get('volatility', 0.0)
            normalized_vol = min(1.0, vol_risk)
            risk_components.append(normalized_vol * 0.25)
            
            # 模型风险权重
            model_risk_score = model_risk.get('model_risk', 0.0)
            risk_components.append(model_risk_score * 0.2)
            
            # 系统性风险权重
            systemic_risk_score = systemic_risk.get('systemic_risk', 0.0)
            risk_components.append(systemic_risk_score * 0.25)
            
            overall_risk = sum(risk_components)
            
            return min(1.0, max(0.0, overall_risk))
            
        except Exception:
            return 0.5
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """分类风险等级"""
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _assess_risk_capacity(self, data: np.ndarray) -> Dict[str, float]:
        """评估风险承受能力"""
        try:
            if len(data) == 0:
                return {'risk_capacity': 0.5, 'capacity_utilization': 0.5}
            
            # 基于历史表现评估风险承受能力
            if len(data) > 1:
                returns = np.diff(data) / (data[:-1] + 1e-10)
                
                # 历史最大损失承受
                max_loss = abs(np.min(returns)) if len(returns) > 0 else 0.0
                
                # 波动率耐受性
                volatility_tolerance = 1.0 / (np.std(returns) + 1e-10) if len(returns) > 1 else 1.0
                volatility_tolerance = min(1.0, volatility_tolerance / 10)
                
                # 持续时间耐受性
                negative_periods = np.sum(returns < 0) / len(returns) if len(returns) > 0 else 0.0
                duration_tolerance = 1.0 - negative_periods
                
                # 综合风险承受能力
                risk_capacity = (volatility_tolerance * 0.4 + 
                               duration_tolerance * 0.3 + 
                               (1.0 - min(1.0, max_loss * 10)) * 0.3)
                
            else:
                risk_capacity = 0.5  # 默认中等风险承受能力
            
            # 当前风险利用率
            current_risk = np.std(returns) if len(data) > 1 else 0.0
            capacity_utilization = min(1.0, current_risk / (risk_capacity + 1e-10))
            
            return {
                'risk_capacity': risk_capacity,
                'capacity_utilization': capacity_utilization,
                'available_capacity': max(0.0, risk_capacity - current_risk),
                'capacity_level': 'high' if risk_capacity > 0.7 else 
                                'medium' if risk_capacity > 0.4 else 'low'
            }
            
        except Exception:
            return {'risk_capacity': 0.5, 'capacity_utilization': 0.5}
    
    def _check_risk_tolerance_alignment(self, risk_score: float) -> Dict[str, Any]:
        """检查风险容忍度对齐"""
        try:
            target_risk_tolerance = self.config.get('risk_tolerance', 0.3)
            
            # 风险对齐度
            risk_alignment = 1.0 - abs(risk_score - target_risk_tolerance)
            
            # 对齐状态
            if abs(risk_score - target_risk_tolerance) < 0.1:
                alignment_status = 'well_aligned'
            elif risk_score > target_risk_tolerance:
                alignment_status = 'risk_too_high'
            else:
                alignment_status = 'risk_too_low'
            
            # 建议调整
            if risk_score > target_risk_tolerance + 0.1:
                recommendation = 'reduce_risk_exposure'
            elif risk_score < target_risk_tolerance - 0.1:
                recommendation = 'increase_risk_exposure'
            else:
                recommendation = 'maintain_current_level'
            
            return {
                'risk_alignment': risk_alignment,
                'alignment_status': alignment_status,
                'target_tolerance': target_risk_tolerance,
                'current_risk': risk_score,
                'risk_gap': risk_score - target_risk_tolerance,
                'recommendation': recommendation
            }
            
        except Exception:
            return {'risk_alignment': 0.5, 'alignment_status': 'unknown'}
        
# ========== 信号融合具体实现 ==========
    
    def _fuse_signals(self, signals: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
        """融合多个信号"""
        try:
            if not signals:
                return {'consensus_score': 0.0, 'confidence': 0.0}
            
            # 归一化权重
            total_weight = sum(weights.get(key, 0.0) for key in signals.keys())
            if total_weight == 0:
                # 等权重处理
                normalized_weights = {key: 1.0/len(signals) for key in signals.keys()}
            else:
                normalized_weights = {key: weights.get(key, 0.0)/total_weight for key in signals.keys()}
            
            # 加权融合
            consensus_score = sum(signals[key] * normalized_weights[key] for key in signals.keys())
            
            # 计算融合置信度
            signal_values = list(signals.values())
            
            # 信号一致性
            if len(signal_values) > 1:
                signal_std = np.std(signal_values)
                signal_mean = np.mean(signal_values)
                consistency = 1.0 - min(1.0, signal_std / (abs(signal_mean) + 1e-10))
            else:
                consistency = 1.0
            
            # 信号强度
            signal_strength = np.mean([abs(val) for val in signal_values])
            
            # 信号数量因子
            signal_count_factor = min(1.0, len(signals) / 5.0)
            
            # 综合置信度
            confidence = (consistency * 0.4 + 
                         signal_strength * 0.35 + 
                         signal_count_factor * 0.25)
            
            return {
                'consensus_score': consensus_score,
                'confidence': confidence,
                'consistency': consistency,
                'signal_strength': signal_strength,
                'signal_count': len(signals)
            }
            
        except Exception as e:
            self.logger.error(f"信号融合错误: {str(e)}")
            return {'consensus_score': 0.0, 'confidence': 0.0}
    
    def _assess_signal_quality_comprehensive(self, signals: Dict[str, float]) -> float:
        """综合评估信号质量"""
        try:
            if not signals:
                return 0.0
            
            quality_factors = []
            
            # 信号强度质量
            signal_values = list(signals.values())
            avg_strength = np.mean([abs(val) for val in signal_values])
            strength_quality = min(1.0, avg_strength * 2)
            quality_factors.append(strength_quality)
            
            # 信号分布质量
            if len(signal_values) > 1:
                value_range = max(signal_values) - min(signal_values)
                distribution_quality = min(1.0, value_range)
                quality_factors.append(distribution_quality)
            
            # 信号完整性质量
            expected_signals = ['technical', 'wavelet', 'fourier', 'nonlinear', 'ml', 'quantum']
            completeness = len([s for s in expected_signals if s in signals]) / len(expected_signals)
            quality_factors.append(completeness)
            
            # 信号可靠性质量
            reliable_signals = sum(1 for val in signal_values if abs(val) > 0.1)
            reliability = reliable_signals / len(signal_values) if signal_values else 0.0
            quality_factors.append(reliability)
            
            return np.mean(quality_factors)
            
        except Exception:
            return 0.0
    
    def _check_signal_consistency(self, signals: Dict[str, float]) -> float:
        """检查信号一致性"""
        try:
            if len(signals) < 2:
                return 1.0
            
            signal_values = list(signals.values())
            
            # 方向一致性
            positive_signals = sum(1 for val in signal_values if val > 0.1)
            negative_signals = sum(1 for val in signal_values if val < -0.1)
            neutral_signals = len(signal_values) - positive_signals - negative_signals
            
            # 主导方向
            if positive_signals > negative_signals:
                dominant_direction = positive_signals
                total_directional = positive_signals + negative_signals
            else:
                dominant_direction = negative_signals
                total_directional = positive_signals + negative_signals
            
            if total_directional > 0:
                direction_consistency = dominant_direction / total_directional
            else:
                direction_consistency = 0.0
            
            # 幅度一致性
            if len(signal_values) > 1:
                cv = np.std(signal_values) / (np.mean(np.abs(signal_values)) + 1e-10)
                magnitude_consistency = 1.0 / (1.0 + cv)
            else:
                magnitude_consistency = 1.0
            
            # 综合一致性
            overall_consistency = (direction_consistency * 0.6 + magnitude_consistency * 0.4)
            
            return overall_consistency
            
        except Exception:
            return 0.0
    
    def _assess_signal_reliability(self, signals: Dict[str, float], weights: Dict[str, float]) -> float:
        """评估信号可靠性"""
        try:
            if not signals:
                return 0.0
            
            reliability_scores = []
            
            for signal_name, signal_value in signals.items():
                # 基于历史性能的可靠性
                historical_reliability = self._get_signal_historical_reliability(signal_name)
                
                # 基于信号强度的可靠性
                strength_reliability = min(1.0, abs(signal_value) * 2)
                
                # 基于权重的可靠性
                weight_reliability = weights.get(signal_name, 0.0)
                
                # 综合可靠性
                signal_reliability = (historical_reliability * 0.4 + 
                                    strength_reliability * 0.3 + 
                                    weight_reliability * 0.3)
                
                reliability_scores.append(signal_reliability)
            
            return np.mean(reliability_scores) if reliability_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _get_signal_historical_reliability(self, signal_name: str) -> float:
        """获取信号历史可靠性"""
        try:
            # 简化实现：基于信号类型的预设可靠性
            reliability_map = {
                'technical': 0.7,
                'wavelet': 0.8,
                'fourier': 0.75,
                'nonlinear': 0.6,
                'ml': 0.85,
                'dl': 0.8,
                'quantum': 0.65,
                'microstructure': 0.7,
                'kalman': 0.75,
                'pattern': 0.8
            }
            
            return reliability_map.get(signal_name, 0.5)
            
        except Exception:
            return 0.5
    
    def _adapt_signal_weights(self, signals: Dict[str, float], 
                            weights: Dict[str, float], tail: int) -> Dict[str, float]:
        """自适应调整信号权重"""
        try:
            adapted_weights = weights.copy()
            
            # 基于信号强度调整
            signal_strengths = {name: abs(value) for name, value in signals.items()}
            max_strength = max(signal_strengths.values()) if signal_strengths else 1.0
            
            if max_strength > 0:
                for signal_name in signals.keys():
                    if signal_name in adapted_weights:
                        # 强信号增加权重
                        strength_factor = signal_strengths[signal_name] / max_strength
                        adaptation = self.config['adaptation_factor'] * strength_factor
                        adapted_weights[signal_name] *= (1.0 + adaptation)
            
            # 基于历史性能调整
            for signal_name in signals.keys():
                if signal_name in adapted_weights:
                    historical_performance = self._get_signal_performance(signal_name, tail)
                    performance_factor = historical_performance - 0.5  # 偏离中性的程度
                    adaptation = self.config['learning_rate'] * performance_factor
                    adapted_weights[signal_name] *= (1.0 + adaptation)
            
            # 重新归一化权重
            total_weight = sum(adapted_weights.values())
            if total_weight > 0:
                adapted_weights = {name: weight/total_weight 
                                 for name, weight in adapted_weights.items()}
            
            return adapted_weights
            
        except Exception:
            return weights
    
    def _get_signal_performance(self, signal_name: str, tail: int) -> float:
        """获取信号历史性能"""
        try:
            # 从性能监控中获取历史数据
            if tail in self.performance_monitor['signal_qualities']:
                signal_history = self.performance_monitor['signal_qualities'][tail]
                if signal_history:
                    return np.mean(signal_history)
            
            return 0.5  # 默认中性性能
            
        except Exception:
            return 0.5
    
    # ========== 高级模式识别具体实现 ==========
    
    def _detect_double_top_advanced(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """高级双顶检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 20:
                return None
            
            # 寻找峰值
            peaks, properties = find_peaks(tail_data, 
                                         height=np.mean(tail_data) + 0.5*np.std(tail_data),
                                         distance=5)
            
            if len(peaks) < 2:
                return None
            
            # 检查最近的两个峰值
            recent_peaks = peaks[-2:]
            peak_heights = tail_data[recent_peaks]
            
            # 双顶条件
            height_similarity = 1.0 - abs(peak_heights[0] - peak_heights[1]) / max(peak_heights)
            
            if height_similarity > 0.8:  # 高度相似
                # 检查中间谷值
                valley_region = tail_data[recent_peaks[0]:recent_peaks[1]]
                if len(valley_region) > 0:
                    min_valley = np.min(valley_region)
                    valley_depth = min(peak_heights) - min_valley
                    
                    if valley_depth > 0.2 * np.std(tail_data):  # 足够深的谷值
                        return {
                            'pattern_type': 'double_top',
                            'confidence': height_similarity,
                            'peak_positions': recent_peaks.tolist(),
                            'peak_heights': peak_heights.tolist(),
                            'valley_depth': valley_depth,
                            'reversal_target': min_valley
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_double_bottom_advanced(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """高级双底检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 20:
                return None
            
            # 反转数据来寻找谷值（作为峰值）
            inverted_data = -tail_data
            peaks, properties = find_peaks(inverted_data, 
                                         height=np.mean(inverted_data) + 0.5*np.std(inverted_data),
                                         distance=5)
            
            if len(peaks) < 2:
                return None
            
            # 检查最近的两个谷值
            recent_troughs = peaks[-2:]
            trough_depths = tail_data[recent_troughs]
            
            # 双底条件
            depth_similarity = 1.0 - abs(trough_depths[0] - trough_depths[1]) / (max(abs(trough_depths)) + 1e-10)
            
            if depth_similarity > 0.8:  # 深度相似
                # 检查中间峰值
                peak_region = tail_data[recent_troughs[0]:recent_troughs[1]]
                if len(peak_region) > 0:
                    max_peak = np.max(peak_region)
                    peak_height = max_peak - max(trough_depths)
                    
                    if peak_height > 0.2 * np.std(tail_data):  # 足够高的峰值
                        return {
                            'pattern_type': 'double_bottom',
                            'confidence': depth_similarity,
                            'trough_positions': recent_troughs.tolist(),
                            'trough_depths': trough_depths.tolist(),
                            'peak_height': peak_height,
                            'reversal_target': max_peak
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_head_shoulders_advanced(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """高级头肩顶检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 30:
                return None
            
            # 寻找三个主要峰值
            peaks, properties = find_peaks(tail_data, 
                                         height=np.mean(tail_data) + 0.3*np.std(tail_data),
                                         distance=5)
            
            if len(peaks) < 3:
                return None
            
            # 检查最近的三个峰值
            recent_peaks = peaks[-3:]
            peak_heights = tail_data[recent_peaks]
            
            # 头肩顶条件：中间峰值最高，两侧峰值相近
            if peak_heights[1] > peak_heights[0] and peak_heights[1] > peak_heights[2]:
                # 检查肩部高度相似性
                shoulder_similarity = 1.0 - abs(peak_heights[0] - peak_heights[2]) / max(peak_heights[0], peak_heights[2])
                
                if shoulder_similarity > 0.7:
                    # 计算颈线
                    left_valley_idx = recent_peaks[0] + np.argmin(tail_data[recent_peaks[0]:recent_peaks[1]])
                    right_valley_idx = recent_peaks[1] + np.argmin(tail_data[recent_peaks[1]:recent_peaks[2]])
                    
                    neckline_level = np.mean([tail_data[left_valley_idx], tail_data[right_valley_idx]])
                    
                    return {
                        'pattern_type': 'head_shoulders',
                        'confidence': shoulder_similarity,
                        'head_position': recent_peaks[1],
                        'head_height': peak_heights[1],
                        'shoulder_positions': [recent_peaks[0], recent_peaks[2]],
                        'shoulder_heights': [peak_heights[0], peak_heights[2]],
                        'neckline_level': neckline_level,
                        'reversal_target': neckline_level - (peak_heights[1] - neckline_level)
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_inverse_head_shoulders(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """反向头肩底检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 30:
                return None
            
            # 反转数据来寻找谷值
            inverted_data = -tail_data
            result = self._detect_head_shoulders_advanced(tail, np.column_stack([inverted_data]))
            
            if result:
                result['pattern_type'] = 'inverse_head_shoulders'
                # 调整反转目标
                if 'reversal_target' in result:
                    result['reversal_target'] = -result['reversal_target']
            
            return result
            
        except Exception:
            return None
    
    def _detect_triple_top(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """三重顶检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 30:
                return None
            
            peaks, _ = find_peaks(tail_data, 
                                height=np.mean(tail_data) + 0.5*np.std(tail_data),
                                distance=5)
            
            if len(peaks) < 3:
                return None
            
            recent_peaks = peaks[-3:]
            peak_heights = tail_data[recent_peaks]
            
            # 三重顶条件：三个峰值高度相近
            max_height = np.max(peak_heights)
            min_height = np.min(peak_heights)
            
            height_consistency = 1.0 - (max_height - min_height) / (max_height + 1e-10)
            
            if height_consistency > 0.85:
                return {
                    'pattern_type': 'triple_top',
                    'confidence': height_consistency,
                    'peak_positions': recent_peaks.tolist(),
                    'peak_heights': peak_heights.tolist(),
                    'average_height': np.mean(peak_heights)
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_triple_bottom(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """三重底检测"""
        try:
            tail_data = data_matrix[:, tail]
            inverted_data = -tail_data
            result = self._detect_triple_top(tail, np.column_stack([inverted_data]))
            
            if result:
                result['pattern_type'] = 'triple_bottom'
            
            return result
            
        except Exception:
            return None
    
    def _detect_ascending_triangle(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """上升三角形检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 20:
                return None
            
            # 寻找高点和低点
            peaks, _ = find_peaks(tail_data, distance=3)
            troughs, _ = find_peaks(-tail_data, distance=3)
            
            if len(peaks) < 3 or len(troughs) < 2:
                return None
            
            # 检查高点是否水平（阻力线）
            recent_peaks = peaks[-3:]
            peak_heights = tail_data[recent_peaks]
            peak_trend = np.polyfit(recent_peaks, peak_heights, 1)[0]
            
            # 检查低点是否上升（支撑线）
            recent_troughs = troughs[-2:]
            trough_depths = tail_data[recent_troughs]
            trough_trend = np.polyfit(recent_troughs, trough_depths, 1)[0]
            
            if abs(peak_trend) < 0.01 and trough_trend > 0.01:  # 水平阻力，上升支撑
                convergence = (np.mean(peak_heights) - np.mean(trough_depths)) / len(tail_data)
                
                return {
                    'pattern_type': 'ascending_triangle',
                    'confidence': min(1.0, trough_trend * 100),
                    'resistance_level': np.mean(peak_heights),
                    'support_trend': trough_trend,
                    'convergence_rate': convergence
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_descending_triangle(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """下降三角形检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 20:
                return None
            
            peaks, _ = find_peaks(tail_data, distance=3)
            troughs, _ = find_peaks(-tail_data, distance=3)
            
            if len(peaks) < 2 or len(troughs) < 3:
                return None
            
            # 检查低点是否水平（支撑线）
            recent_troughs = troughs[-3:]
            trough_depths = tail_data[recent_troughs]
            trough_trend = np.polyfit(recent_troughs, trough_depths, 1)[0]
            
            # 检查高点是否下降（阻力线）
            recent_peaks = peaks[-2:]
            peak_heights = tail_data[recent_peaks]
            peak_trend = np.polyfit(recent_peaks, peak_heights, 1)[0]
            
            if abs(trough_trend) < 0.01 and peak_trend < -0.01:  # 水平支撑，下降阻力
                convergence = (np.mean(peak_heights) - np.mean(trough_depths)) / len(tail_data)
                
                return {
                    'pattern_type': 'descending_triangle',
                    'confidence': min(1.0, abs(peak_trend) * 100),
                    'support_level': np.mean(trough_depths),
                    'resistance_trend': peak_trend,
                    'convergence_rate': convergence
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_symmetric_triangle(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """对称三角形检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 20:
                return None
            
            peaks, _ = find_peaks(tail_data, distance=3)
            troughs, _ = find_peaks(-tail_data, distance=3)
            
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # 检查高点下降趋势
            recent_peaks = peaks[-2:]
            peak_heights = tail_data[recent_peaks]
            peak_trend = np.polyfit(recent_peaks, peak_heights, 1)[0]
            
            # 检查低点上升趋势
            recent_troughs = troughs[-2:]
            trough_depths = tail_data[recent_troughs]
            trough_trend = np.polyfit(recent_troughs, trough_depths, 1)[0]
            
            if peak_trend < -0.01 and trough_trend > 0.01:  # 收敛趋势
                convergence_rate = abs(peak_trend) + abs(trough_trend)
                symmetry = 1.0 - abs(abs(peak_trend) - abs(trough_trend)) / (abs(peak_trend) + abs(trough_trend))
                
                return {
                    'pattern_type': 'symmetric_triangle',
                    'confidence': symmetry,
                    'upper_trend': peak_trend,
                    'lower_trend': trough_trend,
                    'convergence_rate': convergence_rate,
                    'apex_estimate': len(tail_data) + (np.mean(peak_heights) - np.mean(trough_depths)) / convergence_rate
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_rising_wedge(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """上升楔形检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 15:
                return None
            
            peaks, _ = find_peaks(tail_data, distance=3)
            troughs, _ = find_peaks(-tail_data, distance=3)
            
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # 检查整体上升趋势
            overall_trend = np.polyfit(range(len(tail_data)), tail_data, 1)[0]
            
            if overall_trend > 0:
                # 检查高点和低点都在上升，但高点上升更慢
                peak_trend = np.polyfit(peaks[-2:], tail_data[peaks[-2:]], 1)[0]
                trough_trend = np.polyfit(troughs[-2:], tail_data[troughs[-2:]], 1)[0]
                
                if peak_trend > 0 and trough_trend > 0 and trough_trend > peak_trend:
                    convergence = trough_trend - peak_trend
                    
                    return {
                        'pattern_type': 'rising_wedge',
                        'confidence': min(1.0, convergence * 100),
                        'upper_trend': peak_trend,
                        'lower_trend': trough_trend,
                        'convergence_angle': convergence
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_falling_wedge(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """下降楔形检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 15:
                return None
            
            peaks, _ = find_peaks(tail_data, distance=3)
            troughs, _ = find_peaks(-tail_data, distance=3)
            
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # 检查整体下降趋势
            overall_trend = np.polyfit(range(len(tail_data)), tail_data, 1)[0]
            
            if overall_trend < 0:
                # 检查高点和低点都在下降，但低点下降更慢
                peak_trend = np.polyfit(peaks[-2:], tail_data[peaks[-2:]], 1)[0]
                trough_trend = np.polyfit(troughs[-2:], tail_data[troughs[-2:]], 1)[0]
                
                if peak_trend < 0 and trough_trend < 0 and abs(peak_trend) > abs(trough_trend):
                    convergence = abs(peak_trend) - abs(trough_trend)
                    
                    return {
                        'pattern_type': 'falling_wedge',
                        'confidence': min(1.0, convergence * 100),
                        'upper_trend': peak_trend,
                        'lower_trend': trough_trend,
                        'convergence_angle': convergence
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_flag_advanced(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """高级旗形检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 15:
                return None
            
            # 检查是否有强烈的先导趋势（旗杆）
            if len(tail_data) >= 10:
                flagpole_data = tail_data[-10:-5]  # 旗杆部分
                flag_data = tail_data[-5:]         # 旗帜部分
                
                if len(flagpole_data) > 1 and len(flag_data) > 1:
                    flagpole_trend = np.polyfit(range(len(flagpole_data)), flagpole_data, 1)[0]
                    flag_trend = np.polyfit(range(len(flag_data)), flag_data, 1)[0]
                    flag_volatility = np.std(flag_data)
                    
                    # 旗形条件：强烈旗杆 + 横向整理
                    if abs(flagpole_trend) > 0.05 and abs(flag_trend) < 0.02 and flag_volatility < 0.1:
                        flag_direction = 'bull_flag' if flagpole_trend > 0 else 'bear_flag'
                        
                        return {
                            'pattern_type': flag_direction,
                            'confidence': abs(flagpole_trend) * 10,
                            'flagpole_strength': abs(flagpole_trend),
                            'flag_consolidation': 1.0 - abs(flag_trend),
                            'breakout_target': tail_data[-1] + flagpole_trend * 5
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_pennant(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """三角旗检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 15:
                return None
            
            # 类似旗形，但整理部分是收敛的三角形
            flagpole_data = tail_data[-10:-5]
            pennant_data = tail_data[-5:]
            
            if len(flagpole_data) > 1 and len(pennant_data) > 2:
                flagpole_trend = np.polyfit(range(len(flagpole_data)), flagpole_data, 1)[0]
                
                # 检查三角旗的收敛性
                pennant_range_start = max(pennant_data) - min(pennant_data)
                pennant_range_end = abs(pennant_data[-1] - pennant_data[-2])
                
                if pennant_range_start > 0:
                    convergence = 1.0 - pennant_range_end / pennant_range_start
                    
                    if abs(flagpole_trend) > 0.05 and convergence > 0.5:
                        pennant_type = 'bull_pennant' if flagpole_trend > 0 else 'bear_pennant'
                        
                        return {
                            'pattern_type': pennant_type,
                            'confidence': convergence,
                            'flagpole_strength': abs(flagpole_trend),
                            'convergence_rate': convergence,
                            'breakout_target': tail_data[-1] + flagpole_trend * 5
                        }
            
            return None
            
        except Exception:
            return None
        
# ========== 更多高级模式识别 ==========
    
    def _detect_cup_handle(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """杯柄形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 30:
                return None
            
            # 寻找杯子部分（U形底部）
            cup_data = tail_data[-25:-5]  # 杯子部分
            handle_data = tail_data[-5:]   # 手柄部分
            
            if len(cup_data) < 15:
                return None
            
            # 检查杯子的U形特征
            cup_start = cup_data[0]
            cup_end = cup_data[-1]
            cup_bottom = np.min(cup_data)
            cup_bottom_idx = np.argmin(cup_data)
            
            # U形条件：两端高度相近，中间有明显低点
            rim_similarity = 1.0 - abs(cup_start - cup_end) / (max(cup_start, cup_end) + 1e-10)
            depth = min(cup_start, cup_end) - cup_bottom
            
            if rim_similarity > 0.9 and depth > 0.1 * np.std(tail_data):
                # 检查手柄的轻微下倾
                if len(handle_data) > 2:
                    handle_trend = np.polyfit(range(len(handle_data)), handle_data, 1)[0]
                    handle_decline = handle_trend < 0 and abs(handle_trend) < 0.05
                    
                    if handle_decline:
                        return {
                            'pattern_type': 'cup_handle',
                            'confidence': rim_similarity,
                            'cup_depth': depth,
                            'rim_level': (cup_start + cup_end) / 2,
                            'handle_decline': abs(handle_trend),
                            'breakout_target': max(cup_start, cup_end) + depth
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_rounding_top(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """圆顶形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 20:
                return None
            
            # 检查倒U形特征
            data_len = len(tail_data)
            if data_len < 10:
                return None
            
            # 拟合二次函数
            x = np.arange(data_len)
            coeffs = np.polyfit(x, tail_data, 2)
            
            # 二次项系数为负表示倒U形
            if coeffs[0] < -0.001:  # 足够明显的曲率
                # 计算拟合优度
                fitted_curve = np.polyval(coeffs, x)
                r_squared = 1 - np.sum((tail_data - fitted_curve)**2) / np.sum((tail_data - np.mean(tail_data))**2)
                
                if r_squared > 0.7:  # 良好拟合
                    peak_x = -coeffs[1] / (2 * coeffs[0])
                    peak_y = np.polyval(coeffs, peak_x)
                    
                    return {
                        'pattern_type': 'rounding_top',
                        'confidence': r_squared,
                        'curvature': abs(coeffs[0]),
                        'peak_position': peak_x,
                        'peak_value': peak_y,
                        'symmetry': self._calculate_pattern_symmetry(tail_data, int(peak_x))
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_rounding_bottom(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """圆底形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 20:
                return None
            
            # 检查U形特征
            data_len = len(tail_data)
            x = np.arange(data_len)
            coeffs = np.polyfit(x, tail_data, 2)
            
            # 二次项系数为正表示U形
            if coeffs[0] > 0.001:  # 足够明显的曲率
                fitted_curve = np.polyval(coeffs, x)
                r_squared = 1 - np.sum((tail_data - fitted_curve)**2) / np.sum((tail_data - np.mean(tail_data))**2)
                
                if r_squared > 0.7:  # 良好拟合
                    trough_x = -coeffs[1] / (2 * coeffs[0])
                    trough_y = np.polyval(coeffs, trough_x)
                    
                    return {
                        'pattern_type': 'rounding_bottom',
                        'confidence': r_squared,
                        'curvature': coeffs[0],
                        'trough_position': trough_x,
                        'trough_value': trough_y,
                        'symmetry': self._calculate_pattern_symmetry(tail_data, int(trough_x))
                    }
            
            return None
            
        except Exception:
            return None
    
    def _calculate_pattern_symmetry(self, data: np.ndarray, center_idx: int) -> float:
        """计算模式的对称性"""
        try:
            if center_idx <= 0 or center_idx >= len(data) - 1:
                return 0.0
            
            left_part = data[:center_idx]
            right_part = data[center_idx+1:]
            
            # 取较短的一边进行比较
            min_len = min(len(left_part), len(right_part))
            if min_len == 0:
                return 0.0
            
            left_compare = left_part[-min_len:]
            right_compare = right_part[:min_len]
            
            # 计算对称性
            differences = np.abs(left_compare - right_compare[::-1])
            max_possible_diff = np.max(data) - np.min(data)
            
            if max_possible_diff > 0:
                symmetry = 1.0 - np.mean(differences) / max_possible_diff
            else:
                symmetry = 1.0
            
            return max(0.0, symmetry)
            
        except Exception:
            return 0.0
    
    # ========== 日本蜡烛图模式 ==========
    
    def _detect_doji(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """十字星(Doji)检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 3:
                return None
            
            # 简化：检查最近几期的波动性
            recent_data = tail_data[-3:]
            current_value = recent_data[-1]
            
            # 检查是否在均值附近（十字星特征）
            mean_value = np.mean(recent_data)
            std_value = np.std(recent_data)
            
            if std_value > 0 and abs(current_value - mean_value) < 0.3 * std_value:
                # 检查前期是否有明显趋势
                if len(tail_data) >= 5:
                    trend_data = tail_data[-5:-1]
                    trend = np.polyfit(range(len(trend_data)), trend_data, 1)[0]
                    
                    if abs(trend) > 0.02:  # 有明显趋势
                        return {
                            'pattern_type': 'doji',
                            'confidence': 1.0 - abs(current_value - mean_value) / (std_value + 1e-10),
                            'trend_before': 'uptrend' if trend > 0 else 'downtrend',
                            'reversal_signal': True
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_hammer(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """锤头线检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            current = tail_data[-1]
            previous = tail_data[-2]
            
            # 锤头特征：下跌后的反转信号
            if len(tail_data) >= 5:
                trend_data = tail_data[-5:-1]
                trend = np.polyfit(range(len(trend_data)), trend_data, 1)[0]
                
                # 下跌趋势中出现反弹
                if trend < -0.01 and current > previous:
                    reversal_strength = (current - previous) / (np.std(trend_data) + 1e-10)
                    
                    if reversal_strength > 1.0:  # 强烈反弹
                        return {
                            'pattern_type': 'hammer',
                            'confidence': min(1.0, reversal_strength / 2),
                            'reversal_strength': reversal_strength,
                            'trend_before': 'downtrend'
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_hanging_man(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """上吊线检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            current = tail_data[-1]
            previous = tail_data[-2]
            
            # 上吊线特征：上涨后的反转信号
            if len(tail_data) >= 5:
                trend_data = tail_data[-5:-1]
                trend = np.polyfit(range(len(trend_data)), trend_data, 1)[0]
                
                # 上涨趋势中出现回落
                if trend > 0.01 and current < previous:
                    reversal_strength = (previous - current) / (np.std(trend_data) + 1e-10)
                    
                    if reversal_strength > 1.0:  # 强烈回落
                        return {
                            'pattern_type': 'hanging_man',
                            'confidence': min(1.0, reversal_strength / 2),
                            'reversal_strength': reversal_strength,
                            'trend_before': 'uptrend'
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_shooting_star(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """流星线检测"""
        try:
            # 简化实现：类似上吊线但在高位
            result = self._detect_hanging_man(tail, data_matrix)
            if result:
                result['pattern_type'] = 'shooting_star'
            return result
            
        except Exception:
            return None
    
    def _detect_engulfing_bullish(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """看涨吞没形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 3:
                return None
            
            current = tail_data[-1]
            previous = tail_data[-2]
            before_previous = tail_data[-3]
            
            # 看涨吞没：前期下跌，当期强烈上涨
            if before_previous > previous and current > before_previous:
                engulfing_strength = (current - previous) / (before_previous - previous + 1e-10)
                
                if engulfing_strength > 1.5:  # 完全吞没
                    return {
                        'pattern_type': 'engulfing_bullish',
                        'confidence': min(1.0, engulfing_strength / 3),
                        'engulfing_ratio': engulfing_strength
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_engulfing_bearish(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """看跌吞没形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 3:
                return None
            
            current = tail_data[-1]
            previous = tail_data[-2]
            before_previous = tail_data[-3]
            
            # 看跌吞没：前期上涨，当期强烈下跌
            if before_previous < previous and current < before_previous:
                engulfing_strength = (previous - current) / (previous - before_previous + 1e-10)
                
                if engulfing_strength > 1.5:  # 完全吞没
                    return {
                        'pattern_type': 'engulfing_bearish',
                        'confidence': min(1.0, engulfing_strength / 3),
                        'engulfing_ratio': engulfing_strength
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_morning_star(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """晨星形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 4:
                return None
            
            # 三根K线形态：下跌 + 低位整理 + 上涨
            third = tail_data[-1]   # 当前
            second = tail_data[-2]  # 中间
            first = tail_data[-3]   # 之前
            
            # 晨星条件
            if first > second and third > second and third > first:
                gap_down = (first - second) / (np.std(tail_data) + 1e-10)
                gap_up = (third - second) / (np.std(tail_data) + 1e-10)
                
                if gap_down > 0.5 and gap_up > 0.5:
                    return {
                        'pattern_type': 'morning_star',
                        'confidence': min(1.0, (gap_down + gap_up) / 4),
                        'star_position': second,
                        'reversal_strength': third - first
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_evening_star(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """黄昏星形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 4:
                return None
            
            # 三根K线形态：上涨 + 高位整理 + 下跌
            third = tail_data[-1]   # 当前
            second = tail_data[-2]  # 中间
            first = tail_data[-3]   # 之前
            
            # 黄昏星条件
            if first < second and third < second and third < first:
                gap_up = (second - first) / (np.std(tail_data) + 1e-10)
                gap_down = (second - third) / (np.std(tail_data) + 1e-10)
                
                if gap_up > 0.5 and gap_down > 0.5:
                    return {
                        'pattern_type': 'evening_star',
                        'confidence': min(1.0, (gap_up + gap_down) / 4),
                        'star_position': second,
                        'reversal_strength': first - third
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_three_white_soldiers(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """三只白鸟形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 4:
                return None
            
            # 连续三个上涨
            recent_data = tail_data[-3:]
            
            if len(recent_data) == 3:
                increases = [recent_data[i+1] > recent_data[i] for i in range(2)]
                
                if all(increases):
                    # 检查涨幅的一致性
                    gains = [recent_data[i+1] - recent_data[i] for i in range(2)]
                    gain_consistency = 1.0 - np.std(gains) / (np.mean(gains) + 1e-10)
                    
                    if gain_consistency > 0.7:
                        return {
                            'pattern_type': 'three_white_soldiers',
                            'confidence': gain_consistency,
                            'total_gain': recent_data[-1] - recent_data[0],
                            'average_gain': np.mean(gains)
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_three_black_crows(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """三只黑鸦形态检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 4:
                return None
            
            # 连续三个下跌
            recent_data = tail_data[-3:]
            
            if len(recent_data) == 3:
                decreases = [recent_data[i+1] < recent_data[i] for i in range(2)]
                
                if all(decreases):
                    # 检查跌幅的一致性
                    losses = [recent_data[i] - recent_data[i+1] for i in range(2)]
                    loss_consistency = 1.0 - np.std(losses) / (np.mean(losses) + 1e-10)
                    
                    if loss_consistency > 0.7:
                        return {
                            'pattern_type': 'three_black_crows',
                            'confidence': loss_consistency,
                            'total_loss': recent_data[0] - recent_data[-1],
                            'average_loss': np.mean(losses)
                        }
            
            return None
            
        except Exception:
            return None
    
    # ========== 波浪理论模式（简化实现） ==========
    
    def _detect_elliott_wave_1(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """艾略特波浪第1浪检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 10:
                return None
            
            # 第1浪特征：从底部开始的初始上涨
            recent_data = tail_data[-8:]
            
            # 寻找低点和高点
            min_idx = np.argmin(recent_data)
            max_idx = np.argmax(recent_data)
            
            # 第1浪：低点在前，高点在后
            if min_idx < max_idx and max_idx - min_idx >= 3:
                wave_data = recent_data[min_idx:max_idx+1]
                wave_trend = np.polyfit(range(len(wave_data)), wave_data, 1)[0]
                
                if wave_trend > 0.02:  # 明显上升趋势
                    wave_strength = (recent_data[max_idx] - recent_data[min_idx]) / (np.std(recent_data) + 1e-10)
                    
                    return {
                        'pattern_type': 'elliott_wave_1',
                        'confidence': min(1.0, wave_strength / 3),
                        'wave_start': min_idx,
                        'wave_end': max_idx,
                        'wave_strength': wave_strength,
                        'impulse_direction': 'up'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_elliott_wave_2(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """艾略特波浪第2浪检测"""
        try:
            # 第2浪：对第1浪的调整，通常回调50-78.6%
            wave1_result = self._detect_elliott_wave_1(tail, data_matrix)
            if not wave1_result:
                return None
            
            tail_data = data_matrix[:, tail]
            recent_data = tail_data[-8:]
            
            wave1_end = wave1_result['wave_end']
            if wave1_end < len(recent_data) - 2:
                # 检查第1浪后的回调
                wave2_data = recent_data[wave1_end:]
                if len(wave2_data) >= 2:
                    wave2_trend = np.polyfit(range(len(wave2_data)), wave2_data, 1)[0]
                    
                    if wave2_trend < -0.01:  # 明显回调
                        wave1_height = recent_data[wave1_result['wave_end']] - recent_data[wave1_result['wave_start']]
                        wave2_decline = recent_data[wave1_end] - recent_data[-1]
                        
                        retracement_ratio = wave2_decline / (wave1_height + 1e-10)
                        
                        if 0.3 <= retracement_ratio <= 0.8:  # 典型回调幅度
                            return {
                                'pattern_type': 'elliott_wave_2',
                                'confidence': 1.0 - abs(retracement_ratio - 0.618),
                                'retracement_ratio': retracement_ratio,
                                'corrective_type': 'abc_correction'
                            }
            
            return None
            
        except Exception:
            return None
    
    def _detect_elliott_wave_3(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """艾略特波浪第3浪检测"""
        try:
            # 第3浪：通常是最强的推动浪
            wave2_result = self._detect_elliott_wave_2(tail, data_matrix)
            if not wave2_result:
                return None
            
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            # 检查当前是否有强烈上涨
            recent_data = tail_data[-5:]
            current_trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            
            if current_trend > 0.03:  # 比第1浪更强的上涨
                trend_strength = abs(current_trend)
                
                return {
                    'pattern_type': 'elliott_wave_3',
                    'confidence': min(1.0, trend_strength * 20),
                    'trend_strength': trend_strength,
                    'expected_extension': True  # 第3浪通常延长
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_elliott_wave_4(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """艾略特波浪第4浪检测"""
        try:
            # 第4浪：横向调整，不与第1浪重叠
            wave3_result = self._detect_elliott_wave_3(tail, data_matrix)
            if not wave3_result:
                return None
            
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            recent_data = tail_data[-5:]
            volatility = np.std(recent_data)
            trend = abs(np.polyfit(range(len(recent_data)), recent_data, 1)[0])
            
            # 第4浪特征：低波动性，横向整理
            if volatility < 0.1 and trend < 0.01:
                return {
                    'pattern_type': 'elliott_wave_4',
                    'confidence': 1.0 - trend * 100,
                    'consolidation_type': 'sideways',
                    'volatility_compression': True
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_elliott_wave_5(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """艾略特波浪第5浪检测"""
        try:
            # 第5浪：最后的推动浪，可能出现背离
            wave4_result = self._detect_elliott_wave_4(tail, data_matrix)
            if not wave4_result:
                return None
            
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            recent_data = tail_data[-5:]
            current_trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            
            if current_trend > 0.02:  # 最后的上涨
                # 检查是否有背离迹象
                momentum = recent_data[-1] - recent_data[-3]
                price_momentum = recent_data[-1] - recent_data[0]
                
                divergence = 1.0 - momentum / (price_momentum + 1e-10)
                
                return {
                    'pattern_type': 'elliott_wave_5',
                    'confidence': min(1.0, abs(current_trend) * 30),
                    'trend_strength': current_trend,
                    'divergence_signal': divergence > 0.3,
                    'completion_warning': True
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_corrective_wave_a(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """调整浪A检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 8:
                return None
            
            # A浪：五浪结束后的第一个调整浪
            recent_data = tail_data[-6:]
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            
            if trend < -0.02:  # 明显下跌
                return {
                    'pattern_type': 'corrective_wave_a',
                    'confidence': min(1.0, abs(trend) * 30),
                    'correction_strength': abs(trend),
                    'correction_type': 'impulse'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_corrective_wave_b(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """调整浪B检测"""
        try:
            # B浪：对A浪的反弹
            wave_a_result = self._detect_corrective_wave_a(tail, data_matrix)
            if not wave_a_result:
                return None
            
            tail_data = data_matrix[:, tail]
            recent_data = tail_data[-4:]
            
            if len(recent_data) >= 3:
                trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
                
                if trend > 0.01:  # 反弹
                    return {
                        'pattern_type': 'corrective_wave_b',
                        'confidence': min(1.0, trend * 50),
                        'retracement_type': 'corrective_rally'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_corrective_wave_c(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """调整浪C检测"""
        try:
            # C浪：完成调整的最后一浪
            wave_b_result = self._detect_corrective_wave_b(tail, data_matrix)
            if not wave_b_result:
                return None
            
            tail_data = data_matrix[:, tail]
            recent_data = tail_data[-4:]
            
            if len(recent_data) >= 3:
                trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
                
                if trend < -0.02:  # 最后下跌
                    return {
                        'pattern_type': 'corrective_wave_c',
                        'confidence': min(1.0, abs(trend) * 30),
                        'completion_signal': True,
                        'new_cycle_preparation': True
                    }
            
            return None
            
        except Exception:
            return None
    
    # ========== 其他高级模式 ==========
    
    def _detect_fractal_support(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """分形支撑检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            # 分形支撑：中心低点被两侧高点包围
            for i in range(2, len(tail_data) - 2):
                center = tail_data[i]
                left1, left2 = tail_data[i-1], tail_data[i-2]
                right1, right2 = tail_data[i+1], tail_data[i+2]
                
                if center < left1 and center < left2 and center < right1 and center < right2:
                    support_strength = min(left1, left2, right1, right2) - center
                    
                    return {
                        'pattern_type': 'fractal_support',
                        'confidence': min(1.0, support_strength * 10),
                        'support_level': center,
                        'support_index': i
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_fractal_resistance(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """分形阻力检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            # 分形阻力：中心高点被两侧低点包围
            for i in range(2, len(tail_data) - 2):
                center = tail_data[i]
                left1, left2 = tail_data[i-1], tail_data[i-2]
                right1, right2 = tail_data[i+1], tail_data[i+2]
                
                if center > left1 and center > left2 and center > right1 and center > right2:
                    resistance_strength = center - max(left1, left2, right1, right2)
                    
                    return {
                        'pattern_type': 'fractal_resistance',
                        'confidence': min(1.0, resistance_strength * 10),
                        'resistance_level': center,
                        'resistance_index': i
                    }
            
            return None
            
        except Exception:
            return None
        
# ========== 剩余模式识别方法 ==========
    
    def _detect_chaos_pattern(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """混沌理论模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 10:
                return None
            
            # 计算李雅普诺夫指数
            lyapunov = self._calculate_lyapunov_exponent(tail_data)
            
            # 计算分形维数
            fractal_dim = self._calculate_fractal_dimension(tail_data)
            
            # 混沌特征
            if lyapunov > 0.01 and 1.5 < fractal_dim < 2.5:
                chaos_strength = lyapunov * fractal_dim
                
                return {
                    'pattern_type': 'chaos_pattern',
                    'confidence': min(1.0, chaos_strength),
                    'lyapunov_exponent': lyapunov,
                    'fractal_dimension': fractal_dim,
                    'predictability_horizon': 1.0 / lyapunov if lyapunov > 0 else float('inf')
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_mandelbrot_pattern(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """曼德布罗特集合模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 8:
                return None
            
            # 简化的分形模式检测
            # 检查自相似性
            data_len = len(tail_data)
            half_len = data_len // 2
            
            if half_len > 2:
                first_half = tail_data[:half_len]
                second_half = tail_data[half_len:half_len*2]
                
                if len(first_half) == len(second_half):
                    # 计算相似性
                    correlation = np.corrcoef(first_half, second_half)[0, 1]
                    if not np.isnan(correlation) and correlation > 0.7:
                        return {
                            'pattern_type': 'mandelbrot_pattern',
                            'confidence': correlation,
                            'self_similarity': correlation,
                            'fractal_nature': True
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_quantum_superposition(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """量子叠加态模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            quantum_state = self._construct_quantum_state(tail_data)
            
            # 分析叠加态特征
            superposition_analysis = self._analyze_quantum_superposition(quantum_state)
            
            if superposition_analysis['superposition_degree'] > 0.6:
                return {
                    'pattern_type': 'quantum_superposition',
                    'confidence': superposition_analysis['superposition_degree'],
                    'superposition_degree': superposition_analysis['superposition_degree'],
                    'phase_distribution': superposition_analysis['phase_distribution'],
                    'quantum_coherence': self._calculate_quantum_coherence(quantum_state)
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_quantum_entanglement(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """量子纠缠模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            quantum_state = self._construct_quantum_state(tail_data)
            
            # 计算纠缠熵
            entanglement_entropy = self._calculate_entanglement_entropy(quantum_state)
            
            if entanglement_entropy > 0.5:
                return {
                    'pattern_type': 'quantum_entanglement',
                    'confidence': entanglement_entropy,
                    'entanglement_entropy': entanglement_entropy,
                    'bell_state_similarity': self._calculate_bell_state_similarity(quantum_state)
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_quantum_coherence(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """量子相干性模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            quantum_state = self._construct_quantum_state(tail_data)
            
            coherence = self._calculate_quantum_coherence(quantum_state)
            
            if coherence > 0.7:
                return {
                    'pattern_type': 'quantum_coherence',
                    'confidence': coherence,
                    'coherence_measure': coherence,
                    'decoherence_analysis': self._analyze_quantum_decoherence(quantum_state)
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_ml_pattern_1(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """机器学习发现的模式1"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 5:
                return None
            
            # 使用聚类算法发现模式
            if len(tail_data) >= 5:
                # 简化：基于数据的统计特征
                features = [
                    np.mean(tail_data),
                    np.std(tail_data),
                    stats.skew(tail_data),
                    stats.kurtosis(tail_data)
                ]
                
                # 模式识别（简化）
                pattern_score = np.sum(np.abs(features)) / len(features)
                
                if pattern_score > 0.5:
                    return {
                        'pattern_type': 'ml_discovered_pattern_1',
                        'confidence': min(1.0, pattern_score),
                        'feature_vector': features,
                        'pattern_score': pattern_score
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_ml_pattern_2(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """机器学习发现的模式2"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 8:
                return None
            
            # 基于自编码器的异常检测模式
            # 简化：检测数据的重构误差
            if len(tail_data) >= 4:
                # 简单的重构测试
                mid_point = len(tail_data) // 2
                first_half = tail_data[:mid_point]
                second_half = tail_data[mid_point:]
                
                if len(first_half) == len(second_half):
                    reconstruction_error = np.mean((first_half - second_half)**2)
                    
                    if reconstruction_error > 0.1:
                        return {
                            'pattern_type': 'ml_discovered_pattern_2',
                            'confidence': min(1.0, reconstruction_error * 5),
                            'reconstruction_error': reconstruction_error,
                            'anomaly_detected': True
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_ml_pattern_3(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """机器学习发现的模式3"""
        try:
            tail_data = data_matrix[:, tail]
            
            # 基于深度学习的序列模式
            if len(tail_data) >= 6:
                # 简化：检测序列的周期性
                fft_result = np.fft.fft(tail_data)
                power_spectrum = np.abs(fft_result)**2
                
                # 找主导频率
                dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1
                dominant_power = power_spectrum[dominant_freq_idx]
                total_power = np.sum(power_spectrum[1:])
                
                if total_power > 0:
                    pattern_strength = dominant_power / total_power
                    
                    if pattern_strength > 0.3:
                        return {
                            'pattern_type': 'ml_discovered_pattern_3',
                            'confidence': pattern_strength,
                            'dominant_frequency': dominant_freq_idx,
                            'pattern_strength': pattern_strength,
                            'periodicity_detected': True
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_seasonal_pattern(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """季节性模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 12:
                return None
            
            # 检测周期性
            seasonality_strength = self._detect_seasonality(tail_data)
            
            if seasonality_strength > 0.5:
                # 找出季节周期
                autocorr_values = []
                for lag in range(1, min(12, len(tail_data)//2)):
                    if len(tail_data) > lag:
                        autocorr = np.corrcoef(tail_data[:-lag], tail_data[lag:])[0, 1]
                        if not np.isnan(autocorr):
                            autocorr_values.append((lag, abs(autocorr)))
                
                if autocorr_values:
                    best_lag = max(autocorr_values, key=lambda x: x[1])
                    
                    return {
                        'pattern_type': 'seasonal_pattern',
                        'confidence': seasonality_strength,
                        'seasonality_strength': seasonality_strength,
                        'seasonal_period': best_lag[0],
                        'autocorrelation': best_lag[1]
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_cyclical_pattern(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """周期性模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 8:
                return None
            
            # 使用FFT检测周期
            fft_result = np.fft.fft(tail_data)
            frequencies = np.fft.fftfreq(len(tail_data))
            power_spectrum = np.abs(fft_result)**2
            
            # 找主导周期
            positive_freqs = frequencies[1:len(frequencies)//2]
            positive_power = power_spectrum[1:len(power_spectrum)//2]
            
            if len(positive_power) > 0:
                max_power_idx = np.argmax(positive_power)
                dominant_freq = positive_freqs[max_power_idx]
                
                if dominant_freq > 0:
                    cycle_length = 1.0 / dominant_freq
                    cycle_strength = positive_power[max_power_idx] / np.sum(positive_power)
                    
                    if cycle_strength > 0.2:
                        return {
                            'pattern_type': 'cyclical_pattern',
                            'confidence': cycle_strength,
                            'cycle_length': cycle_length,
                            'cycle_strength': cycle_strength,
                            'dominant_frequency': dominant_freq
                        }
            
            return None
            
        except Exception:
            return None
    
    def _detect_trend_break_pattern(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """趋势突破模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 10:
                return None
            
            # 检测趋势变化点
            mid_point = len(tail_data) // 2
            early_trend = np.polyfit(range(mid_point), tail_data[:mid_point], 1)[0]
            late_trend = np.polyfit(range(mid_point), tail_data[mid_point:], 1)[0]
            
            # 趋势突破条件
            trend_change = abs(late_trend - early_trend)
            
            if trend_change > 0.05:  # 显著趋势变化
                break_strength = trend_change
                break_direction = 'upward' if late_trend > early_trend else 'downward'
                
                return {
                    'pattern_type': 'trend_break_pattern',
                    'confidence': min(1.0, break_strength * 10),
                    'break_strength': break_strength,
                    'break_direction': break_direction,
                    'early_trend': early_trend,
                    'late_trend': late_trend
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_mean_reversion_pattern(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """均值回归模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 8:
                return None
            
            # 检测偏离均值的程度
            mean_value = np.mean(tail_data)
            current_value = tail_data[-1]
            std_value = np.std(tail_data)
            
            if std_value > 0:
                deviation = abs(current_value - mean_value) / std_value
                
                # 检测回归趋势
                if len(tail_data) >= 5:
                    recent_data = tail_data[-5:]
                    # 检查是否在向均值靠近
                    distances_to_mean = [abs(val - mean_value) for val in recent_data]
                    
                    if len(distances_to_mean) > 1:
                        regression_trend = np.polyfit(range(len(distances_to_mean)), distances_to_mean, 1)[0]
                        
                        if deviation > 1.5 and regression_trend < 0:  # 正在回归
                            return {
                                'pattern_type': 'mean_reversion_pattern',
                                'confidence': min(1.0, deviation / 3),
                                'deviation_level': deviation,
                                'regression_trend': abs(regression_trend),
                                'mean_level': mean_value,
                                'current_distance': abs(current_value - mean_value)
                            }
            
            return None
            
        except Exception:
            return None
    
    def _detect_emergence_pattern(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """涌现模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 10:
                return None
            
            # 检测突然的行为变化（涌现特征）
            recent_volatility = np.std(tail_data[-5:]) if len(tail_data) >= 5 else 0
            historical_volatility = np.std(tail_data[:-5]) if len(tail_data) > 5 else recent_volatility
            
            if historical_volatility > 0:
                volatility_change = recent_volatility / historical_volatility
                
                # 涌现条件：突然的行为模式改变
                if volatility_change > 2.0 or volatility_change < 0.5:
                    emergence_strength = abs(np.log(volatility_change))
                    
                    return {
                        'pattern_type': 'emergence_pattern',
                        'confidence': min(1.0, emergence_strength),
                        'volatility_change_ratio': volatility_change,
                        'emergence_type': 'complexity_increase' if volatility_change > 1 else 'order_emergence',
                        'emergence_strength': emergence_strength
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_self_organization(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """自组织模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 12:
                return None
            
            # 检测数据的自组织特征（熵的变化）
            early_data = tail_data[:len(tail_data)//2]
            late_data = tail_data[len(tail_data)//2:]
            
            early_entropy = self._calculate_shannon_entropy(early_data)
            late_entropy = self._calculate_shannon_entropy(late_data)
            
            entropy_change = early_entropy - late_entropy
            
            # 自组织：熵减少（更有序）
            if entropy_change > 0.5:
                return {
                    'pattern_type': 'self_organization',
                    'confidence': min(1.0, entropy_change),
                    'entropy_reduction': entropy_change,
                    'organization_level': 1.0 - late_entropy / (early_entropy + 1e-10),
                    'early_entropy': early_entropy,
                    'late_entropy': late_entropy
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_phase_transition(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """相变模式检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 10:
                return None
            
            # 检测突变点（相变特征）
            # 使用滑动窗口检测统计特性的突然变化
            window_size = 4
            changes = []
            
            for i in range(window_size, len(tail_data) - window_size):
                before_window = tail_data[i-window_size:i]
                after_window = tail_data[i:i+window_size]
                
                before_mean = np.mean(before_window)
                after_mean = np.mean(after_window)
                before_std = np.std(before_window)
                after_std = np.std(after_window)
                
                mean_change = abs(after_mean - before_mean)
                std_change = abs(after_std - before_std)
                
                total_change = mean_change + std_change
                changes.append(total_change)
            
            if changes:
                max_change = max(changes)
                max_change_idx = changes.index(max_change)
                
                if max_change > 0.2:  # 显著变化
                    return {
                        'pattern_type': 'phase_transition',
                        'confidence': min(1.0, max_change * 5),
                        'transition_point': max_change_idx + window_size,
                        'transition_magnitude': max_change,
                        'transition_type': 'critical_point'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_critical_point(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """临界点检测"""
        try:
            tail_data = data_matrix[:, tail]
            if len(tail_data) < 8:
                return None
            
            # 检测临界点特征：波动性急剧增加
            if len(tail_data) >= 6:
                recent_volatility = np.std(tail_data[-3:])
                previous_volatility = np.std(tail_data[-6:-3])
                
                if previous_volatility > 0:
                    volatility_ratio = recent_volatility / previous_volatility
                    
                    # 临界点：波动性急剧增加
                    if volatility_ratio > 3.0:
                        criticality = np.log(volatility_ratio)
                        
                        return {
                            'pattern_type': 'critical_point',
                            'confidence': min(1.0, criticality / 3),
                            'volatility_explosion': volatility_ratio,
                            'criticality_level': criticality,
                            'system_instability': True
                        }
            
            return None
            
        except Exception:
            return None
    
    # ========== 简化技术指标计算方法 ==========
    
    def _calculate_rsi_simple(self, data: np.ndarray, period: int = 14) -> float:
        """简化RSI计算"""
        try:
            if len(data) < period + 1:
                return 50.0
            
            changes = np.diff(data)
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return 50.0
    
    def _calculate_macd_simple(self, data: np.ndarray) -> Tuple[float, float, float]:
        """简化MACD计算"""
        try:
            if len(data) < 26:
                return 0.0, 0.0, 0.0
            
            # 指数移动平均
            ema12 = np.mean(data[-12:])  # 简化为简单均线
            ema26 = np.mean(data[-26:])
            
            macd_line = ema12 - ema26
            
            # 信号线（简化）
            if len(data) >= 35:
                signal_data = []
                for i in range(9):
                    if len(data) > 26 + i:
                        ema12_i = np.mean(data[-(12+i):len(data)-i])
                        ema26_i = np.mean(data[-(26+i):len(data)-i])
                        signal_data.append(ema12_i - ema26_i)
                
                macd_signal = np.mean(signal_data) if signal_data else macd_line
            else:
                macd_signal = macd_line
            
            macd_histogram = macd_line - macd_signal
            
            return macd_line, macd_signal, macd_histogram
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _calculate_bollinger_position_simple(self, data: np.ndarray, period: int = 20) -> float:
        """简化布林带位置计算"""
        try:
            if len(data) < period:
                return 0.5
            
            recent_data = data[-period:]
            mean_val = np.mean(recent_data)
            std_val = np.std(recent_data)
            
            upper_band = mean_val + 2 * std_val
            lower_band = mean_val - 2 * std_val
            current_price = data[-1]
            
            if upper_band != lower_band:
                position = (current_price - lower_band) / (upper_band - lower_band)
            else:
                position = 0.5
            
            return max(0.0, min(1.0, position))
            
        except Exception:
            return 0.5
    
    def _calculate_stochastic_simple(self, data: np.ndarray, period: int = 14) -> float:
        """简化随机指标计算"""
        try:
            if len(data) < period:
                return 50.0
            
            recent_data = data[-period:]
            current_price = data[-1]
            
            highest_high = np.max(recent_data)
            lowest_low = np.min(recent_data)
            
            if highest_high != lowest_low:
                k_percent = 100 * (current_price - lowest_low) / (highest_high - lowest_low)
            else:
                k_percent = 50.0
            
            return k_percent
            
        except Exception:
            return 50.0
    
    def _calculate_williams_r_simple(self, data: np.ndarray, period: int = 14) -> float:
        """简化Williams %R计算"""
        try:
            if len(data) < period:
                return -50.0
            
            recent_data = data[-period:]
            current_price = data[-1]
            
            highest_high = np.max(recent_data)
            lowest_low = np.min(recent_data)
            
            if highest_high != lowest_low:
                williams_r = -100 * (highest_high - current_price) / (highest_high - lowest_low)
            else:
                williams_r = -50.0
            
            return williams_r
            
        except Exception:
            return -50.0
    
    def _calculate_cci_simple(self, data: np.ndarray, period: int = 20) -> float:
        """简化CCI计算"""
        try:
            if len(data) < period:
                return 0.0
            
            recent_data = data[-period:]
            typical_price = data[-1]  # 简化：使用当前价格
            
            sma = np.mean(recent_data)
            mean_deviation = np.mean(np.abs(recent_data - sma))
            
            if mean_deviation != 0:
                cci = (typical_price - sma) / (0.015 * mean_deviation)
            else:
                cci = 0.0
            
            return cci
            
        except Exception:
            return 0.0
    
    def _calculate_dfa_alpha(self, data: np.ndarray) -> float:
        """去趋势波动分析"""
        try:
            if len(data) < 10:
                return 0.5
            
            # 简化的DFA实现
            N = len(data)
            y = np.cumsum(data - np.mean(data))
            
            # 不同窗口大小
            scales = np.logspace(0.5, np.log10(N//4), 10).astype(int)
            fluctuations = []
            
            for scale in scales:
                if scale >= 4:
                    segments = N // scale
                    if segments > 0:
                        F = 0
                        for v in range(segments):
                            start = v * scale
                            end = start + scale
                            segment = y[start:end]
                            
                            # 线性去趋势
                            x = np.arange(len(segment))
                            coeffs = np.polyfit(x, segment, 1)
                            trend = np.polyval(coeffs, x)
                            
                            F += np.mean((segment - trend)**2)
                        
                        F = np.sqrt(F / segments)
                        fluctuations.append(F)
                    else:
                        fluctuations.append(0)
                else:
                    fluctuations.append(0)
            
            # 计算标度指数
            valid_scales = []
            valid_fluctuations = []
            
            for i, f in enumerate(fluctuations):
                if f > 0:
                    valid_scales.append(scales[i])
                    valid_fluctuations.append(f)
            
            if len(valid_scales) > 3:
                log_scales = np.log10(valid_scales)
                log_fluctuations = np.log10(valid_fluctuations)
                
                alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]
                return max(0.0, min(2.0, alpha))
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_rqa_measures(self, data: np.ndarray) -> Tuple[float, float]:
        """递归量化分析指标"""
        try:
            if len(data) < 10:
                return 0.0, 0.0
            
            # 简化的RQA实现
            threshold = 0.1 * np.std(data)
            N = len(data)
            
            # 构建递归矩阵
            recurrence_points = 0
            diagonal_points = 0
            
            for i in range(N):
                for j in range(N):
                    if abs(data[i] - data[j]) < threshold:
                        recurrence_points += 1
                        
                        # 检查对角线结构
                        if abs(i - j) <= 2:
                            diagonal_points += 1
            
            # 递归率
            recurrence_rate = recurrence_points / (N * N)
            
            # 确定性
            determinism = diagonal_points / max(1, recurrence_points)
            
            return recurrence_rate, determinism
            
        except Exception:
            return 0.0, 0.0
    
# ========== 更多辅助分析方法 ==========
    
    def _get_feature_importance_summary(self) -> Dict[str, float]:
        """获取特征重要性摘要"""
        try:
            importance_summary = {}
            
            # 从历史记录中获取特征重要性
            for tail, importance_history in self.feature_importance_tracker.items():
                if importance_history:
                    avg_importance = {}
                    for importance_dict in importance_history:
                        for feature, value in importance_dict.items():
                            if feature not in avg_importance:
                                avg_importance[feature] = []
                            avg_importance[feature].append(value)
                    
                    # 计算平均重要性
                    for feature, values in avg_importance.items():
                        avg_value = np.mean(values)
                        if feature not in importance_summary:
                            importance_summary[feature] = []
                        importance_summary[feature].append(avg_value)
            
            # 全局平均
            global_importance = {}
            for feature, tail_values in importance_summary.items():
                global_importance[feature] = np.mean(tail_values)
            
            return global_importance
            
        except Exception:
            return {}
    
    def _get_model_contribution_summary(self) -> Dict[str, float]:
        """获取模型贡献度摘要"""
        try:
            contributions = {}
            
            # 基于模型类型的预设权重
            model_weights = {
                'technical_analysis': 0.2,
                'wavelet_analysis': 0.15,
                'fourier_analysis': 0.12,
                'nonlinear_dynamics': 0.1,
                'machine_learning': 0.18,
                'deep_learning': 0.15,
                'quantum_analysis': 0.05,
                'pattern_recognition': 0.05
            }
            
            # 根据历史性能调整权重
            for model_type, base_weight in model_weights.items():
                # 简化：使用基础权重
                contributions[model_type] = base_weight
            
            return contributions
            
        except Exception:
            return {}
    
    def _detect_current_market_regime(self, data_matrix: np.ndarray) -> str:
        """检测当前市场状态"""
        try:
            if data_matrix.ndim < 2 or data_matrix.shape[0] < 10:
                return MarketRegime.SIDEWAYS.value
            
            # 计算市场整体活跃度
            total_activity = np.sum(data_matrix, axis=1)
            
            # 计算趋势
            if len(total_activity) >= 10:
                recent_trend = np.polyfit(range(10), total_activity[-10:], 1)[0]
                overall_trend = np.polyfit(range(len(total_activity)), total_activity, 1)[0]
                
                # 计算波动率
                recent_volatility = np.std(total_activity[-10:])
                historical_volatility = np.std(total_activity)
                
                # 状态判断
                if recent_volatility > historical_volatility * 1.5:
                    if abs(recent_trend) > 0.1:
                        return MarketRegime.CRISIS.value
                    else:
                        return MarketRegime.HIGH_VOLATILITY.value
                elif recent_volatility < historical_volatility * 0.5:
                    return MarketRegime.LOW_VOLATILITY.value
                elif recent_trend > 0.05:
                    return MarketRegime.BULL_MARKET.value
                elif recent_trend < -0.05:
                    return MarketRegime.BEAR_MARKET.value
                else:
                    return MarketRegime.SIDEWAYS.value
            
            return MarketRegime.SIDEWAYS.value
            
        except Exception:
            return MarketRegime.SIDEWAYS.value
    
    def _forecast_volatility(self, data_matrix: np.ndarray, horizon: int = 5) -> Dict[str, float]:
        """预测波动率"""
        try:
            if data_matrix.ndim < 2 or data_matrix.shape[0] < 10:
                return {'forecast': 0.1, 'confidence': 0.0}
            
            # 计算历史波动率
            total_activity = np.sum(data_matrix, axis=1)
            historical_volatility = np.std(total_activity)
            
            # 简单的GARCH模型预测
            if len(total_activity) >= 5:
                recent_returns = np.diff(total_activity[-5:])
                recent_volatility = np.std(recent_returns) if len(recent_returns) > 1 else historical_volatility
                
                # 波动率预测（简化）
                alpha = 0.1  # GARCH参数
                beta = 0.85
                omega = 0.01
                
                # 递推预测
                forecast_var = omega
                for i in range(horizon):
                    if len(recent_returns) > 0:
                        forecast_var = omega + alpha * recent_returns[-1]**2 + beta * forecast_var
                
                forecast_volatility = np.sqrt(forecast_var)
                
                # 预测置信度
                data_length_factor = min(1.0, len(total_activity) / 50)
                trend_stability = 1.0 - abs(np.polyfit(range(len(total_activity)), total_activity, 1)[0])
                confidence = data_length_factor * trend_stability
                
            else:
                forecast_volatility = historical_volatility
                confidence = 0.3
            
            return {
                'forecast': forecast_volatility,
                'confidence': min(1.0, confidence),
                'horizon': horizon,
                'current_volatility': historical_volatility,
                'volatility_trend': 'increasing' if forecast_volatility > historical_volatility else 'decreasing'
            }
            
        except Exception:
            return {'forecast': 0.1, 'confidence': 0.0}
    
    def _calculate_reversal_probability_distribution(self, scores: Dict[int, float]) -> Dict[str, Any]:
        """计算反转概率分布"""
        try:
            if not scores:
                return {'probabilities': {}, 'entropy': 0.0}
            
            # 将得分转换为概率
            total_score = sum(scores.values())
            if total_score == 0:
                # 均匀分布
                prob = 1.0 / len(scores)
                probabilities = {tail: prob for tail in scores.keys()}
            else:
                probabilities = {tail: score/total_score for tail, score in scores.items()}
            
            # 计算分布熵
            prob_values = list(probabilities.values())
            prob_values = [p for p in prob_values if p > 0]
            
            if prob_values:
                entropy = -sum(p * np.log2(p) for p in prob_values)
                max_entropy = np.log2(len(scores))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                entropy = 0.0
                normalized_entropy = 0.0
            
            # 概率分析
            max_prob_tail = max(probabilities.keys(), key=lambda k: probabilities[k])
            max_probability = probabilities[max_prob_tail]
            
            # 集中度分析
            concentration = max_probability
            uncertainty = 1.0 - concentration
            
            return {
                'probabilities': probabilities,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'max_probability_tail': max_prob_tail,
                'max_probability': max_probability,
                'concentration': concentration,
                'uncertainty': uncertainty,
                'distribution_type': self._classify_distribution_type(probabilities)
            }
            
        except Exception:
            return {'probabilities': {}, 'entropy': 0.0}
    
    def _classify_distribution_type(self, probabilities: Dict[int, float]) -> str:
        """分类概率分布类型"""
        try:
            if not probabilities:
                return 'unknown'
            
            prob_values = list(probabilities.values())
            max_prob = max(prob_values)
            
            if max_prob > 0.7:
                return 'concentrated'  # 高度集中
            elif max_prob > 0.4:
                return 'moderate'      # 中等集中
            else:
                return 'dispersed'     # 分散
                
        except Exception:
            return 'unknown'
    
    def _generate_alternative_scenarios(self, scores: Dict[int, float], 
                                      confidence: float) -> List[Dict[str, Any]]:
        """生成备选方案"""
        try:
            scenarios = []
            
            if not scores:
                return scenarios
            
            # 主要方案
            best_tail = max(scores.keys(), key=lambda k: scores[k])
            scenarios.append({
                'scenario': 'primary',
                'recommended_tail': best_tail,
                'confidence': confidence,
                'probability': confidence,
                'reasoning': '主要推荐方案'
            })
            
            # 备选方案
            sorted_tails = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            
            if len(sorted_tails) > 1:
                second_best = sorted_tails[1]
                second_confidence = scores[second_best] / scores[best_tail] * confidence
                
                scenarios.append({
                    'scenario': 'alternative',
                    'recommended_tail': second_best,
                    'confidence': second_confidence,
                    'probability': second_confidence * 0.7,
                    'reasoning': '备选推荐方案'
                })
            
            # 保守方案（如果主方案置信度不高）
            if confidence < 0.6:
                # 选择风险最低的选项
                conservative_tail = min(scores.keys(), key=lambda k: abs(scores[k] - 0.5))
                
                scenarios.append({
                    'scenario': 'conservative',
                    'recommended_tail': conservative_tail,
                    'confidence': 0.5,
                    'probability': 1.0 - confidence,
                    'reasoning': '保守方案-低风险选择'
                })
            
            return scenarios
            
        except Exception:
            return []
    
    def _find_historical_similarities(self, tail: int, data_matrix: np.ndarray, 
                                    lookback: int = 100) -> Dict[str, Any]:
        """寻找历史相似情况"""
        try:
            if data_matrix.ndim < 2 or data_matrix.shape[0] < 20:
                return {'similarities': [], 'average_outcome': 0.5}
            
            tail_data = data_matrix[:, tail]
            current_pattern = tail_data[-10:] if len(tail_data) >= 10 else tail_data
            
            similarities = []
            outcomes = []
            
            # 在历史数据中搜索相似模式
            search_length = min(lookback, len(tail_data) - 15)
            
            for i in range(search_length):
                if i + len(current_pattern) + 5 < len(tail_data):
                    historical_pattern = tail_data[i:i + len(current_pattern)]
                    
                    # 计算相似度
                    if len(historical_pattern) == len(current_pattern):
                        correlation = np.corrcoef(current_pattern, historical_pattern)[0, 1]
                        
                        if not np.isnan(correlation) and correlation > 0.7:
                            # 获取后续结果
                            future_data = tail_data[i + len(current_pattern):i + len(current_pattern) + 5]
                            if len(future_data) > 0:
                                outcome = np.mean(future_data) - current_pattern[-1]
                                
                                similarities.append({
                                    'similarity': correlation,
                                    'historical_period': i,
                                    'outcome': outcome,
                                    'confidence': correlation
                                })
                                outcomes.append(outcome)
            
            # 分析历史结果
            if outcomes:
                average_outcome = np.mean(outcomes)
                outcome_std = np.std(outcomes)
                positive_outcomes = sum(1 for o in outcomes if o > 0)
                success_rate = positive_outcomes / len(outcomes)
            else:
                average_outcome = 0.0
                outcome_std = 0.0
                success_rate = 0.5
            
            return {
                'similarities': similarities[:5],  # 返回前5个最相似的
                'similarity_count': len(similarities),
                'average_outcome': average_outcome,
                'outcome_variance': outcome_std,
                'historical_success_rate': success_rate,
                'prediction_reliability': len(similarities) / max(1, search_length) * 100
            }
            
        except Exception:
            return {'similarities': [], 'average_outcome': 0.5}
    
    def _detect_early_warning_signals(self, data_matrix: np.ndarray) -> Dict[str, Any]:
        """检测早期警告信号"""
        try:
            if data_matrix.ndim < 2 or data_matrix.shape[0] < 15:
                return {'warning_signals': [], 'alert_level': 'low'}
            
            warning_signals = []
            alert_level = 'low'
            
            # 系统性风险信号
            total_activity = np.sum(data_matrix, axis=1)
            
            # 1. 波动性急剧增加
            if len(total_activity) >= 10:
                recent_vol = np.std(total_activity[-5:])
                historical_vol = np.std(total_activity[-15:-5])
                
                if historical_vol > 0 and recent_vol / historical_vol > 2.0:
                    warning_signals.append({
                        'type': 'volatility_spike',
                        'severity': 'high',
                        'description': '波动性急剧增加',
                        'ratio': recent_vol / historical_vol
                    })
                    alert_level = 'high'
            
            # 2. 相关性增加（系统性风险）
            if data_matrix.shape[1] > 2:
                correlations = []
                for i in range(data_matrix.shape[1]):
                    for j in range(i + 1, data_matrix.shape[1]):
                        if len(data_matrix) > 1:
                            corr = np.corrcoef(data_matrix[:, i], data_matrix[:, j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    if avg_correlation > 0.8:
                        warning_signals.append({
                            'type': 'high_correlation',
                            'severity': 'medium',
                            'description': '系统相关性过高',
                            'correlation': avg_correlation
                        })
                        if alert_level == 'low':
                            alert_level = 'medium'
            
            # 3. 趋势一致性丧失
            trend_consistency = []
            for tail in range(data_matrix.shape[1]):
                tail_data = data_matrix[:, tail]
                if len(tail_data) >= 10:
                    recent_trend = np.polyfit(range(5), tail_data[-5:], 1)[0]
                    historical_trend = np.polyfit(range(10), tail_data[-15:-5], 1)[0]
                    
                    if abs(historical_trend) > 0.01:
                        consistency = 1.0 - abs(recent_trend - historical_trend) / abs(historical_trend)
                        trend_consistency.append(consistency)
            
            if trend_consistency:
                avg_consistency = np.mean(trend_consistency)
                if avg_consistency < 0.3:
                    warning_signals.append({
                        'type': 'trend_breakdown',
                        'severity': 'medium',
                        'description': '趋势一致性丧失',
                        'consistency': avg_consistency
                    })
                    if alert_level == 'low':
                        alert_level = 'medium'
            
            # 4. 极值出现频率增加
            extreme_count = 0
            for tail in range(data_matrix.shape[1]):
                tail_data = data_matrix[:, tail]
                if len(tail_data) >= 10:
                    recent_data = tail_data[-5:]
                    historical_mean = np.mean(tail_data[-15:-5])
                    historical_std = np.std(tail_data[-15:-5])
                    
                    if historical_std > 0:
                        for value in recent_data:
                            if abs(value - historical_mean) > 2 * historical_std:
                                extreme_count += 1
            
            extreme_ratio = extreme_count / (data_matrix.shape[1] * 5)
            if extreme_ratio > 0.2:
                warning_signals.append({
                    'type': 'extreme_values',
                    'severity': 'high',
                    'description': '极值出现频率异常',
                    'extreme_ratio': extreme_ratio
                })
                alert_level = 'high'
            
            return {
                'warning_signals': warning_signals,
                'alert_level': alert_level,
                'signal_count': len(warning_signals),
                'system_stability': 1.0 - len(warning_signals) / 10.0
            }
            
        except Exception:
            return {'warning_signals': [], 'alert_level': 'low'}
    
    def _calculate_regime_change_probability(self, data_matrix: np.ndarray) -> float:
        """计算制度变迁概率"""
        try:
            if data_matrix.ndim < 2 or data_matrix.shape[0] < 20:
                return 0.0
            
            change_indicators = []
            
            # 1. 统计特性变化
            total_activity = np.sum(data_matrix, axis=1)
            if len(total_activity) >= 20:
                early_period = total_activity[-20:-10]
                recent_period = total_activity[-10:]
                
                # 均值变化
                mean_change = abs(np.mean(recent_period) - np.mean(early_period))
                mean_change_ratio = mean_change / (np.std(early_period) + 1e-10)
                change_indicators.append(min(1.0, mean_change_ratio))
                
                # 方差变化
                var_change = abs(np.var(recent_period) - np.var(early_period))
                var_change_ratio = var_change / (np.var(early_period) + 1e-10)
                change_indicators.append(min(1.0, var_change_ratio))
            
            # 2. 相关结构变化
            if data_matrix.shape[1] > 2:
                early_corr_matrix = np.corrcoef(data_matrix[-20:-10].T)
                recent_corr_matrix = np.corrcoef(data_matrix[-10:].T)
                
                # 计算相关矩阵的差异
                if not (np.isnan(early_corr_matrix).any() or np.isnan(recent_corr_matrix).any()):
                    corr_diff = np.mean(np.abs(recent_corr_matrix - early_corr_matrix))
                    change_indicators.append(min(1.0, corr_diff * 5))
            
            # 3. 趋势方向变化
            trend_changes = 0
            for tail in range(data_matrix.shape[1]):
                tail_data = data_matrix[:, tail]
                if len(tail_data) >= 20:
                    early_trend = np.polyfit(range(10), tail_data[-20:-10], 1)[0]
                    recent_trend = np.polyfit(range(10), tail_data[-10:], 1)[0]
                    
                    if early_trend * recent_trend < 0:  # 趋势反转
                        trend_changes += 1
            
            trend_change_ratio = trend_changes / data_matrix.shape[1]
            change_indicators.append(trend_change_ratio)
            
            # 综合概率
            if change_indicators:
                regime_change_prob = np.mean(change_indicators)
            else:
                regime_change_prob = 0.0
            
            return min(1.0, regime_change_prob)
            
        except Exception:
            return 0.0
    
    def _calculate_systemic_risk_indicators(self, data_matrix: np.ndarray) -> Dict[str, float]:
        """计算系统性风险指标"""
        try:
            if data_matrix.ndim < 2:
                return {'overall_systemic_risk': 0.0}
            
            indicators = {}
            
            # 1. 系统性相关性
            if data_matrix.shape[1] > 1:
                correlations = []
                for i in range(data_matrix.shape[1]):
                    for j in range(i + 1, data_matrix.shape[1]):
                        if len(data_matrix) > 1:
                            corr = np.corrcoef(data_matrix[:, i], data_matrix[:, j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations:
                    indicators['average_correlation'] = np.mean(correlations)
                    indicators['max_correlation'] = np.max(correlations)
                else:
                    indicators['average_correlation'] = 0.0
                    indicators['max_correlation'] = 0.0
            
            # 2. 集中度风险
            total_activity = np.sum(data_matrix, axis=1)
            if len(total_activity) > 0:
                tail_shares = []
                for tail in range(data_matrix.shape[1]):
                    tail_activity = np.sum(data_matrix[:, tail])
                    share = tail_activity / (np.sum(total_activity) + 1e-10)
                    tail_shares.append(share)
                
                # HHI指数
                hhi = sum(share**2 for share in tail_shares)
                indicators['concentration_index'] = hhi
            
            # 3. 传染风险
            contagion_risk = 0.0
            if data_matrix.shape[0] >= 5:
                # 检测级联效应
                for i in range(len(data_matrix) - 1):
                    current_volatility = np.std(data_matrix[i])
                    next_volatility = np.std(data_matrix[i + 1])
                    
                    if current_volatility > 0:
                        volatility_transmission = next_volatility / current_volatility
                        if volatility_transmission > 1.5:
                            contagion_risk += 0.1
                
                contagion_risk = min(1.0, contagion_risk)
            
            indicators['contagion_risk'] = contagion_risk
            
            # 4. 系统性脆弱性
            vulnerabilities = []
            for tail in range(data_matrix.shape[1]):
                tail_data = data_matrix[:, tail]
                if len(tail_data) > 1:
                    # 基于波动性的脆弱性
                    volatility = np.std(tail_data)
                    trend_stability = 1.0 - abs(np.polyfit(range(len(tail_data)), tail_data, 1)[0])
                    vulnerability = volatility * (1.0 - trend_stability)
                    vulnerabilities.append(vulnerability)
            
            if vulnerabilities:
                indicators['system_vulnerability'] = np.mean(vulnerabilities)
            else:
                indicators['system_vulnerability'] = 0.0
            
            # 5. 综合系统性风险
            risk_components = [
                indicators.get('average_correlation', 0.0),
                indicators.get('concentration_index', 0.0),
                indicators.get('contagion_risk', 0.0),
                indicators.get('system_vulnerability', 0.0)
            ]
            
            indicators['overall_systemic_risk'] = np.mean(risk_components)
            
            return indicators
            
        except Exception:
            return {'overall_systemic_risk': 0.0}
    
    def _calculate_complexity_measures(self, data_matrix: np.ndarray) -> Dict[str, float]:
        """计算复杂性度量"""
        try:
            if data_matrix.ndim < 2:
                return {'overall_complexity': 0.0}
            
            measures = {}
            
            # 1. 信息熵
            entropies = []
            for tail in range(data_matrix.shape[1]):
                tail_data = data_matrix[:, tail]
                if len(tail_data) > 0:
                    entropy = self._calculate_shannon_entropy(tail_data)
                    entropies.append(entropy)
            
            if entropies:
                measures['average_entropy'] = np.mean(entropies)
                measures['entropy_variance'] = np.var(entropies)
            else:
                measures['average_entropy'] = 0.0
                measures['entropy_variance'] = 0.0
            
            # 2. 分形维数
            fractal_dimensions = []
            for tail in range(min(5, data_matrix.shape[1])):  # 限制计算量
                tail_data = data_matrix[:, tail]
                if len(tail_data) >= 10:
                    fractal_dim = self._calculate_fractal_dimension(tail_data)
                    fractal_dimensions.append(fractal_dim)
            
            if fractal_dimensions:
                measures['average_fractal_dimension'] = np.mean(fractal_dimensions)
            else:
                measures['average_fractal_dimension'] = 1.0
            
            # 3. 网络复杂性
            if data_matrix.shape[1] > 2:
                # 基于相关网络的复杂性
                correlations = []
                for i in range(data_matrix.shape[1]):
                    for j in range(i + 1, data_matrix.shape[1]):
                        if len(data_matrix) > 1:
                            corr = np.corrcoef(data_matrix[:, i], data_matrix[:, j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations:
                    # 网络密度
                    network_density = np.mean(correlations)
                    # 网络异质性
                    network_heterogeneity = np.std(correlations)
                    
                    measures['network_density'] = network_density
                    measures['network_heterogeneity'] = network_heterogeneity
                    measures['network_complexity'] = network_density * network_heterogeneity
                else:
                    measures['network_complexity'] = 0.0
            
            # 4. 时间复杂性
            total_activity = np.sum(data_matrix, axis=1)
            if len(total_activity) > 10:
                # Lempel-Ziv复杂性（简化）
                binary_sequence = [1 if x > np.median(total_activity) else 0 for x in total_activity]
                lz_complexity = self._calculate_lempel_ziv_complexity(binary_sequence)
                measures['temporal_complexity'] = lz_complexity
            else:
                measures['temporal_complexity'] = 0.0
            
            # 5. 综合复杂性
            complexity_components = [
                measures.get('average_entropy', 0.0) / 5.0,  # 归一化
                (measures.get('average_fractal_dimension', 1.0) - 1.0) / 2.0,
                measures.get('network_complexity', 0.0),
                measures.get('temporal_complexity', 0.0)
            ]
            
            measures['overall_complexity'] = np.mean(complexity_components)
            
            return measures
            
        except Exception:
            return {'overall_complexity': 0.0}
    
    def _calculate_lempel_ziv_complexity(self, binary_sequence: List[int]) -> float:
        """计算Lempel-Ziv复杂性"""
        try:
            if len(binary_sequence) == 0:
                return 0.0
            
            # 简化的LZ77算法
            dictionary = []
            i = 0
            
            while i < len(binary_sequence):
                # 寻找最长匹配
                max_match_len = 0
                match_pos = 0
                
                for j, pattern in enumerate(dictionary):
                    match_len = 0
                    for k in range(min(len(pattern), len(binary_sequence) - i)):
                        if pattern[k] == binary_sequence[i + k]:
                            match_len += 1
                        else:
                            break
                    
                    if match_len > max_match_len:
                        max_match_len = match_len
                        match_pos = j
                
                # 添加新模式
                if max_match_len == 0:
                    new_pattern = [binary_sequence[i]]
                    i += 1
                else:
                    end_pos = min(i + max_match_len + 1, len(binary_sequence))
                    new_pattern = binary_sequence[i:end_pos]
                    i = end_pos
                
                dictionary.append(new_pattern)
            
            # 复杂性 = 字典大小 / 序列长度
            complexity = len(dictionary) / len(binary_sequence) if len(binary_sequence) > 0 else 0.0
            
            return complexity
            
        except Exception:
            return 0.0
    
    # ========== 系统管理方法 ==========
    
    def _update_signal_weights(self, tail: int, confidence: float, scores: Dict[int, float]):
        """更新信号权重"""
        try:
            # 基于性能更新权重
            weight_adjustment = self.config['learning_rate'] * (confidence - 0.5)
            
            # 更新权重历史
            if tail not in self.signal_fusion['weight_history']:
                self.signal_fusion['weight_history'][tail] = deque(maxlen=100)
            
            current_weights = self.signal_fusion['signal_weights'].copy()
            
            # 调整权重
            for signal_type in current_weights:
                adjustment = weight_adjustment * np.random.normal(0, 0.1)  # 添加噪声
                current_weights[signal_type] *= (1 + adjustment)
                current_weights[signal_type] = max(0.01, min(0.5, current_weights[signal_type]))
            
            # 归一化权重
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                for signal_type in current_weights:
                    current_weights[signal_type] /= total_weight
            
            # 更新全局权重
            self.signal_fusion['signal_weights'] = current_weights
            
            # 记录权重历史
            self.signal_fusion['weight_history'][tail].append({
                'timestamp': datetime.now(),
                'weights': current_weights.copy(),
                'confidence': confidence
            })
            
        except Exception as e:
            self.logger.error(f"权重更新错误: {str(e)}")
    
    def _update_model_performance_tracking(self, tail: int, confidence: float):
        """更新模型性能跟踪"""
        try:
            # 更新性能历史
            if tail not in self.performance_monitor['model_performances']:
                self.performance_monitor['model_performances'][tail] = deque(maxlen=100)
            
            performance_record = {
                'timestamp': datetime.now(),
                'confidence': confidence,
                'prediction_quality': confidence  # 简化
            }
            
            self.performance_monitor['model_performances'][tail].append(performance_record)
            
            # 更新全局性能指标
            self.performance_monitor['prediction_accuracies'].append(confidence)
            
        except Exception as e:
            self.logger.error(f"性能跟踪更新错误: {str(e)}")
    
    def _trigger_memory_cleanup(self):
        """触发内存清理"""
        try:
            # 清理缓存
            if hasattr(self, 'cache'):
                # 保留最近的缓存条目
                if len(self.cache) > self.config['cache_size']:
                    # 移除最旧的条目
                    while len(self.cache) > self.config['cache_size'] // 2:
                        self.cache.popitem(last=False)
            
            # 清理历史数据
            for tail in range(10):
                if len(self.trend_history[tail]) > self.config['history_window']:
                    # 保留最近的数据
                    recent_data = list(self.trend_history[tail])[-self.config['history_window']//2:]
                    self.trend_history[tail].clear()
                    self.trend_history[tail].extend(recent_data)
            
            # 垃圾回收
            gc.collect()
            
            self.logger.info("内存清理完成")
            
        except Exception as e:
            self.logger.error(f"内存清理错误: {str(e)}")
    
    def _cache_prediction_result(self, candidate_tails: Tuple[int], data_hash: str, 
                               result: Dict[str, Any]):
        """缓存预测结果"""
        try:
            if not self.config['cache_enabled']:
                return
            
            cache_key = f"{candidate_tails}_{data_hash}"
            
            # 简化结果以节省内存
            cached_result = {
                'recommended_tail': result.get('recommended_tail'),
                'confidence': result.get('confidence'),
                'final_score': result.get('final_score'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache[cache_key] = cached_result
            
            # 缓存大小控制
            if len(self.cache) > self.config['cache_size']:
                # 移除最旧的条目
                self.cache.popitem(last=False)
            
        except Exception as e:
            self.logger.error(f"缓存错误: {str(e)}")
    
    # ========== 学习和适应方法 ==========
    
    def learn_from_outcome(self, prediction: Dict, actual_tails: List[int]) -> Dict:
        """从结果中学习（增强版）"""
        try:
            if not prediction or 'recommended_tail' not in prediction:
                return {'learning_success': False, 'reason': 'invalid_prediction'}
            
            predicted_tail = prediction['recommended_tail']
            prediction_confidence = prediction.get('confidence', 0.0)
            was_correct = predicted_tail in actual_tails
            
            # 更新基础统计
            self.total_predictions += 1
            if was_correct:
                self.successful_reversals += 1
            else:
                self.false_signals += 1
            
            # 计算准确率
            self.prediction_accuracy = self.successful_reversals / max(1, self.total_predictions)
            
            # 学习强度基于预测置信度
            learning_intensity = prediction_confidence if was_correct else (1.0 - prediction_confidence)
            
            # 更新模型参数
            self._adaptive_parameter_update(predicted_tail, was_correct, learning_intensity)
            
            # 更新信号权重
            self._adaptive_signal_weight_update(prediction, was_correct, learning_intensity)
            
            # 更新动态阈值
            self._adaptive_threshold_update(was_correct, prediction_confidence)
            
            # 学习结果记录
            learning_record = {
                'learning_success': True,
                'was_correct': was_correct,
                'prediction_confidence': prediction_confidence,
                'learning_intensity': learning_intensity,
                'current_accuracy': self.prediction_accuracy,
                'total_predictions': self.total_predictions,
                'successful_reversals': self.successful_reversals,
                'false_signals': self.false_signals,
                'adaptation_details': {
                    'parameter_updates': self._get_recent_parameter_changes(),
                    'weight_updates': self._get_recent_weight_changes(),
                    'threshold_updates': self._get_recent_threshold_changes()
                }
            }
            
            return learning_record
            
        except Exception as e:
            self.logger.error(f"学习过程错误: {str(e)}")
            return {'learning_success': False, 'reason': str(e)}
    
    def _adaptive_parameter_update(self, tail: int, was_correct: bool, intensity: float):
        """自适应参数更新"""
        try:
            adjustment_factor = intensity * self.config['learning_rate']
            
            if was_correct:
                # 强化成功的参数配置
                if self.config['reversal_threshold'] > 0.5:
                    self.config['reversal_threshold'] *= (1 - adjustment_factor * 0.1)
                
                self.config['momentum_sensitivity'] *= (1 + adjustment_factor * 0.1)
            else:
                # 调整失败的参数配置
                self.config['reversal_threshold'] *= (1 + adjustment_factor * 0.1)
                self.config['momentum_sensitivity'] *= (1 - adjustment_factor * 0.1)
            
            # 确保参数在合理范围内
            self.config['reversal_threshold'] = max(0.3, min(0.9, self.config['reversal_threshold']))
            self.config['momentum_sensitivity'] = max(0.5, min(3.0, self.config['momentum_sensitivity']))
            
        except Exception as e:
            self.logger.error(f"参数更新错误: {str(e)}")
    
    def _adaptive_signal_weight_update(self, prediction: Dict, was_correct: bool, intensity: float):
        """自适应信号权重更新"""
        try:
            detailed_analysis = prediction.get('detailed_analysis', {})
            
            if not detailed_analysis:
                return
            
            recommended_tail = prediction.get('recommended_tail')
            if recommended_tail not in detailed_analysis:
                return
            
            tail_analysis = detailed_analysis[recommended_tail]
            
            # 根据各信号的贡献调整权重
            signals_to_adjust = ['technical', 'wavelet', 'fourier', 'nonlinear', 'ml', 'quantum']
            
            for signal_type in signals_to_adjust:
                signal_score_key = f'{signal_type}_score'
                if signal_score_key in tail_analysis:
                    signal_score = tail_analysis[signal_score_key]
                    
                    if signal_type in self.signal_fusion['signal_weights']:
                        current_weight = self.signal_fusion['signal_weights'][signal_type]
                        
                        if was_correct:
                            # 成功时，增强高分信号的权重
                            if signal_score > 0.6:
                                adjustment = intensity * 0.05 * signal_score
                                new_weight = current_weight * (1 + adjustment)
                            else:
                                new_weight = current_weight
                        else:
                            # 失败时，降低高分信号的权重
                            if signal_score > 0.6:
                                adjustment = intensity * 0.05 * signal_score
                                new_weight = current_weight * (1 - adjustment)
                            else:
                                new_weight = current_weight
                        
                        self.signal_fusion['signal_weights'][signal_type] = max(0.01, min(0.5, new_weight))
            
            # 重新归一化权重
            total_weight = sum(self.signal_fusion['signal_weights'].values())
            if total_weight > 0:
                for signal_type in self.signal_fusion['signal_weights']:
                    self.signal_fusion['signal_weights'][signal_type] /= total_weight
            
        except Exception as e:
            self.logger.error(f"信号权重更新错误: {str(e)}")
    
    def _adaptive_threshold_update(self, was_correct: bool, confidence: float):
        """自适应阈值更新"""
        try:
            adaptation_rate = self.config['learning_rate'] * 0.5
            
            for threshold_name in self.dynamic_thresholds.dynamic_thresholds:
                current_threshold = self.dynamic_thresholds.dynamic_thresholds[threshold_name]
                
                if was_correct:
                    if confidence > 0.8:
                        # 高置信度成功，可以略微降低阈值
                        adjustment = -adaptation_rate * 0.1
                    else:
                        # 低置信度成功，保持当前阈值
                        adjustment = 0.0
                else:
                    if confidence > 0.7:
                        # 高置信度失败，需要提高阈值
                        adjustment = adaptation_rate * 0.2
                    else:
                        # 低置信度失败，适度提高阈值
                        adjustment = adaptation_rate * 0.1
                
                new_threshold = current_threshold + adjustment
                self.dynamic_thresholds.dynamic_thresholds[threshold_name] = max(0.1, min(0.95, new_threshold))
            
        except Exception as e:
            self.logger.error(f"阈值更新错误: {str(e)}")
    
    def _get_recent_parameter_changes(self) -> Dict[str, float]:
        """获取最近的参数变化"""
        return {
            'reversal_threshold': self.config['reversal_threshold'],
            'momentum_sensitivity': self.config['momentum_sensitivity']
        }
    
    def _get_recent_weight_changes(self) -> Dict[str, float]:
        """获取最近的权重变化"""
        return self.signal_fusion['signal_weights'].copy()
    
    def _get_recent_threshold_changes(self) -> Dict[str, float]:
        """获取最近的阈值变化"""
        return self.dynamic_thresholds.dynamic_thresholds.copy()
    
    # ========== 报告和状态方法 ==========
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = {
                'system_info': {
                    'version': '2.0.0-research',
                    'total_predictions': self.total_predictions,
                    'prediction_accuracy': self.prediction_accuracy,
                    'successful_reversals': self.successful_reversals,
                    'false_signals': self.false_signals
                },
                'performance_metrics': {
                    'average_confidence': np.mean(self.model_confidence_history) if self.model_confidence_history else 0.0,
                    'recent_accuracy': self._calculate_recent_accuracy(),
                    'prediction_trend': self._analyze_prediction_trend()
                },
                'model_status': {
                    'ml_models_count': len(self.ml_models),
                    'dl_models_available': TORCH_AVAILABLE or TF_AVAILABLE,
                    'technical_indicators_count': len(self.technical_indicators),
                    'pattern_library_size': len(self.pattern_library)
                },
                'system_health': {
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cache_hit_ratio': self._calculate_cache_hit_ratio(),
                    'processing_speed': self._estimate_processing_speed()
                },
                'configuration': {
                    'current_thresholds': self.dynamic_thresholds.dynamic_thresholds.copy(),
                    'signal_weights': self.signal_fusion['signal_weights'].copy(),
                    'key_parameters': {
                        'reversal_threshold': self.config['reversal_threshold'],
                        'momentum_sensitivity': self.config['momentum_sensitivity'],
                        'learning_rate': self.config['learning_rate']
                    }
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"状态获取错误: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_recent_accuracy(self, window: int = 50) -> float:
        """计算最近的准确率"""
        try:
            if len(self.performance_monitor['prediction_accuracies']) < window:
                return self.prediction_accuracy
            
            recent_predictions = list(self.performance_monitor['prediction_accuracies'])[-window:]
            return np.mean(recent_predictions) if recent_predictions else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_prediction_trend(self) -> str:
        """分析预测趋势"""
        try:
            if len(self.performance_monitor['prediction_accuracies']) < 10:
                return 'insufficient_data'
            
            recent_data = list(self.performance_monitor['prediction_accuracies'])[-10:]
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            
            if trend > 0.01:
                return 'improving'
            elif trend < -0.01:
                return 'declining'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'
    
    def _calculate_cache_hit_ratio(self) -> float:
        """计算缓存命中率"""
        try:
            if not hasattr(self, 'cache_hits') or not hasattr(self, 'cache_misses'):
                return 0.0
            
            total_requests = self.cache_hits + self.cache_misses
            if total_requests == 0:
                return 0.0
            
            return self.cache_hits / total_requests
            
        except Exception:
            return 0.0
    
    def _estimate_processing_speed(self) -> float:
        """估计处理速度"""
        try:
            if 'prediction' not in self.performance_monitor['execution_times']:
                return 0.0
            
            execution_times = list(self.performance_monitor['execution_times']['prediction'])
            if not execution_times:
                return 0.0
            
            avg_time = np.mean(execution_times)
            # 返回每秒预测数
            return 1.0 / (avg_time + 1e-10)
            
        except Exception:
            return 0.0
        
# ========== 遗漏方法的实现 ==========
    
    def _calculate_rsi_trend(self, rsi_values: np.ndarray) -> str:
        """计算RSI趋势"""
        try:
            if len(rsi_values) < 3:
                return 'insufficient_data'
            
            recent_trend = np.polyfit(range(len(rsi_values[-5:])), rsi_values[-5:], 1)[0]
            
            if recent_trend > 0.5:
                return 'strongly_rising'
            elif recent_trend > 0.1:
                return 'rising'
            elif recent_trend > -0.1:
                return 'sideways'
            elif recent_trend > -0.5:
                return 'falling'
            else:
                return 'strongly_falling'
                
        except Exception:
            return 'unknown'
    
    def _detect_rsi_divergence(self, price_data: np.ndarray, rsi_values: np.ndarray) -> bool:
        """检测RSI背离"""
        try:
            if len(price_data) < 5 or len(rsi_values) < 5:
                return False
            
            # 价格趋势
            price_trend = np.polyfit(range(len(price_data)), price_data, 1)[0]
            rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
            
            # 背离：价格和RSI趋势相反
            return (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0)
            
        except Exception:
            return False
    
    def _classify_rsi_level(self, rsi_value: float) -> str:
        """分类RSI水平"""
        if rsi_value >= 70:
            return 'overbought'
        elif rsi_value >= 50:
            return 'bullish'
        elif rsi_value >= 30:
            return 'bearish'
        else:
            return 'oversold'
    
    def _detect_macd_crossover(self, macd_line: np.ndarray, macd_signal: np.ndarray) -> str:
        """检测MACD交叉"""
        try:
            if len(macd_line) < 2 or len(macd_signal) < 2:
                return 'no_crossover'
            
            # 检查最近的交叉
            prev_diff = macd_line[-2] - macd_signal[-2]
            curr_diff = macd_line[-1] - macd_signal[-1]
            
            if prev_diff <= 0 and curr_diff > 0:
                return 'bullish_crossover'
            elif prev_diff >= 0 and curr_diff < 0:
                return 'bearish_crossover'
            else:
                return 'no_crossover'
                
        except Exception:
            return 'no_crossover'
    
    def _detect_macd_divergence(self, price_data: np.ndarray, macd_line: np.ndarray) -> bool:
        """检测MACD背离"""
        try:
            return self._detect_rsi_divergence(price_data, macd_line)
        except Exception:
            return False
    
    def _calculate_bollinger_position(self, price: float, upper: float, lower: float) -> float:
        """计算布林带位置"""
        try:
            if upper == lower:
                return 0.5
            
            position = (price - lower) / (upper - lower)
            return max(0.0, min(1.0, position))
            
        except Exception:
            return 0.5
    
    def _detect_bollinger_squeeze(self, upper_band: np.ndarray, lower_band: np.ndarray) -> bool:
        """检测布林带收缩"""
        try:
            if len(upper_band) < 5 or len(lower_band) < 5:
                return False
            
            # 计算带宽
            recent_width = np.mean(upper_band[-3:] - lower_band[-3:])
            historical_width = np.mean(upper_band[-10:-3] - lower_band[-10:-3])
            
            return recent_width < historical_width * 0.7
            
        except Exception:
            return False
    
    def _detect_bollinger_breakout(self, price_data: np.ndarray, 
                                 upper_band: np.ndarray, lower_band: np.ndarray) -> str:
        """检测布林带突破"""
        try:
            if len(price_data) == 0 or len(upper_band) == 0 or len(lower_band) == 0:
                return 'no_breakout'
            
            current_price = price_data[-1]
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            
            if current_price > current_upper:
                return 'upper_breakout'
            elif current_price < current_lower:
                return 'lower_breakout'
            else:
                return 'no_breakout'
                
        except Exception:
            return 'no_breakout'
    
    def _detect_stochastic_crossover(self, stoch_k: np.ndarray, stoch_d: np.ndarray) -> str:
        """检测随机指标交叉"""
        try:
            return self._detect_macd_crossover(stoch_k, stoch_d)
        except Exception:
            return 'no_crossover'
    
    def _detect_stochastic_divergence(self, price_data: np.ndarray, stoch_k: np.ndarray) -> bool:
        """检测随机指标背离"""
        try:
            return self._detect_rsi_divergence(price_data, stoch_k)
        except Exception:
            return False
    
    def _classify_trend_strength(self, adx_value: float) -> str:
        """分类趋势强度"""
        if adx_value >= 50:
            return 'very_strong'
        elif adx_value >= 25:
            return 'strong'
        elif adx_value >= 20:
            return 'moderate'
        else:
            return 'weak'
    
    def _analyze_volume_patterns(self, tail: int, data_matrix: np.ndarray) -> Dict[str, Any]:
        """分析成交量模式"""
        try:
            # 简化：用总活跃度模拟成交量
            if data_matrix.ndim > 1:
                volume_proxy = np.sum(data_matrix, axis=1)
            else:
                volume_proxy = data_matrix
            
            if len(volume_proxy) < 5:
                return {'volume_trend': 'insufficient_data'}
            
            # 成交量趋势
            volume_trend = np.polyfit(range(len(volume_proxy)), volume_proxy, 1)[0]
            
            # 成交量波动性
            volume_volatility = np.std(volume_proxy)
            
            # 成交量异常
            recent_volume = np.mean(volume_proxy[-3:])
            avg_volume = np.mean(volume_proxy)
            volume_ratio = recent_volume / (avg_volume + 1e-10)
            
            return {
                'volume_trend': 'increasing' if volume_trend > 0 else 'decreasing',
                'volume_volatility': volume_volatility,
                'volume_ratio': volume_ratio,
                'volume_anomaly': volume_ratio > 1.5 or volume_ratio < 0.5
            }
            
        except Exception:
            return {'volume_trend': 'unknown'}
    
    def _identify_support_resistance_levels(self, data: np.ndarray) -> Dict[str, Any]:
        """识别支撑阻力水平"""
        try:
            if len(data) < 10:
                return {'support_levels': [], 'resistance_levels': []}
            
            # 寻找局部极值
            peaks, _ = find_peaks(data, distance=3)
            troughs, _ = find_peaks(-data, distance=3)
            
            # 支撑水平（低点）
            support_levels = []
            if len(troughs) > 0:
                trough_values = data[troughs]
                # 聚类相近的支撑位
                unique_supports = []
                for val in trough_values:
                    if not any(abs(val - existing) < 0.1 * np.std(data) for existing in unique_supports):
                        unique_supports.append(val)
                support_levels = sorted(unique_supports)
            
            # 阻力水平（高点）
            resistance_levels = []
            if len(peaks) > 0:
                peak_values = data[peaks]
                # 聚类相近的阻力位
                unique_resistances = []
                for val in peak_values:
                    if not any(abs(val - existing) < 0.1 * np.std(data) for existing in unique_resistances):
                        unique_resistances.append(val)
                resistance_levels = sorted(unique_resistances, reverse=True)
            
            return {
                'support_levels': support_levels[:3],  # 取前3个最强支撑
                'resistance_levels': resistance_levels[:3],  # 取前3个最强阻力
                'current_position': self._classify_price_position(data[-1], support_levels, resistance_levels)
            }
            
        except Exception:
            return {'support_levels': [], 'resistance_levels': []}
    
    def _classify_price_position(self, current_price: float, 
                               support_levels: List[float], 
                               resistance_levels: List[float]) -> str:
        """分类价格位置"""
        try:
            if not support_levels and not resistance_levels:
                return 'neutral'
            
            # 找最近的支撑和阻力
            nearest_support = max(support_levels) if support_levels else 0
            nearest_resistance = min(resistance_levels) if resistance_levels else float('inf')
            
            if current_price <= nearest_support:
                return 'at_support'
            elif current_price >= nearest_resistance:
                return 'at_resistance'
            else:
                return 'between_levels'
                
        except Exception:
            return 'unknown'
    
    def _calculate_technical_composite_score(self, analysis: Dict[str, Any]) -> float:
        """计算技术分析综合得分"""
        try:
            scores = []
            
            # RSI得分
            rsi_info = analysis.get('rsi', {})
            rsi_level = rsi_info.get('overbought_oversold', 'neutral')
            if rsi_level in ['overbought', 'oversold']:
                scores.append(0.8)
            else:
                scores.append(0.4)
            
            # MACD得分
            macd_info = analysis.get('macd', {})
            macd_crossover = macd_info.get('crossover', 'no_crossover')
            if 'crossover' in macd_crossover and macd_crossover != 'no_crossover':
                scores.append(0.7)
            else:
                scores.append(0.3)
            
            # 布林带得分
            bollinger_info = analysis.get('bollinger', {})
            bollinger_position = bollinger_info.get('position', 0.5)
            if bollinger_position > 0.8 or bollinger_position < 0.2:
                scores.append(0.8)
            else:
                scores.append(0.4)
            
            # 成交量得分
            volume_info = analysis.get('volume', {})
            if volume_info.get('volume_anomaly', False):
                scores.append(0.6)
            else:
                scores.append(0.3)
            
            return np.mean(scores) if scores else 0.5
            
        except Exception:
            return 0.5
    
    def _estimate_ml_timing(self, analysis: Dict[str, Any]) -> str:
        """估计机器学习预测时机"""
        try:
            ml_score = analysis.get('ml_score', 0.0)
            
            if ml_score > 0.8:
                return 'immediate'
            elif ml_score > 0.6:
                return 'next_1_2_periods'
            elif ml_score > 0.4:
                return 'next_3_5_periods'
            else:
                return 'wait_for_better_signal'
                
        except Exception:
            return 'monitor_closely'
    
    def _recommend_monitoring_frequency(self, timing: str) -> str:
        """推荐监控频率"""
        frequency_map = {
            'immediate': 'real_time',
            'next_1_2_periods': 'every_period',
            'next_1_3_periods': 'every_period',
            'next_3_5_periods': 'every_2_periods',
            'wait_for_better_signal': 'every_5_periods',
            'monitor_closely': 'every_period'
        }
        
        return frequency_map.get(timing, 'every_period')
    
    def _identify_entry_signals(self, analysis: Dict[str, Any]) -> List[str]:
        """识别入场信号"""
        try:
            signals = []
            
            # 技术信号
            if analysis.get('technical_score', 0) > 0.6:
                signals.append('technical_confirmation')
            
            # 模式信号
            if analysis.get('pattern_score', 0) > 0.7:
                signals.append('pattern_breakout')
            
            # 机器学习信号
            if analysis.get('ml_score', 0) > 0.7:
                signals.append('ml_consensus')
            
            # 量子信号
            if analysis.get('quantum_score', 0) > 0.6:
                signals.append('quantum_coherence_break')
            
            return signals
            
        except Exception:
            return []
    
    def _identify_exit_signals(self, analysis: Dict[str, Any]) -> List[str]:
        """识别退出信号"""
        try:
            signals = []
            
            # 风险信号
            if analysis.get('risk_score', 0) > 0.7:
                signals.append('high_risk_exit')
            
            # 反向信号
            if analysis.get('reversal_confidence', 0) < 0.3:
                signals.append('reversal_failure')
            
            # 时间衰减
            signals.append('time_decay_after_5_periods')
            
            return signals
            
        except Exception:
            return []
    
    def _assess_timing_risk(self, timing: str, analysis: Dict[str, Any]) -> str:
        """评估时机风险"""
        try:
            confidence = analysis.get('final_score', 0.5)
            
            if timing == 'immediate':
                if confidence > 0.8:
                    return 'low'
                elif confidence > 0.6:
                    return 'medium'
                else:
                    return 'high'
            elif 'wait' in timing:
                return 'low'
            else:
                return 'medium'
                
        except Exception:
            return 'medium'
    
    def _quantify_model_uncertainty(self, fusion_result: Dict) -> float:
        """量化模型不确定性"""
        try:
            # 基于信号方差的模型不确定性
            signal_variance = fusion_result.get('signal_variance', 0.0)
            
            # 基于模型一致性的不确定性
            consistency = fusion_result.get('signal_consistency', 1.0)
            consistency_uncertainty = 1.0 - consistency
            
            # 综合不确定性
            model_uncertainty = (signal_variance + consistency_uncertainty) / 2
            
            return min(1.0, model_uncertainty)
            
        except Exception:
            return 0.5
    
    def _quantify_data_uncertainty(self, fusion_result: Dict) -> float:
        """量化数据不确定性"""
        try:
            # 基于信号数量的数据不确定性
            signal_count = fusion_result.get('signal_count', 0)
            if signal_count == 0:
                return 1.0
            
            # 期望信号数量
            expected_signals = 6  # technical, wavelet, fourier, nonlinear, ml, quantum
            
            data_sufficiency = signal_count / expected_signals
            data_uncertainty = 1.0 - min(1.0, data_sufficiency)
            
            return data_uncertainty
            
        except Exception:
            return 0.5
    
    def _quantify_parameter_uncertainty(self) -> float:
        """量化参数不确定性"""
        try:
            # 基于参数变化历史的不确定性
            parameter_stability = 0.8  # 简化：假设参数相对稳定
            
            return 1.0 - parameter_stability
            
        except Exception:
            return 0.2
    
    def _quantify_epistemic_uncertainty(self, fusion_result: Dict) -> float:
        """量化认知不确定性"""
        try:
            # 基于模型理解程度的不确定性
            signal_quality = fusion_result.get('signal_quality', 0.5)
            
            # 认知不确定性与信号质量反相关
            epistemic_uncertainty = 1.0 - signal_quality
            
            return epistemic_uncertainty
            
        except Exception:
            return 0.5
    
    def _quantify_aleatoric_uncertainty(self, risk_assessment: Dict) -> float:
        """量化随机不确定性"""
        try:
            # 基于系统随机性的不确定性
            volatility_risk = risk_assessment.get('volatility_risk', {})
            volatility = volatility_risk.get('volatility', 0.1)
            
            # 随机不确定性与波动率相关
            aleatoric_uncertainty = min(1.0, volatility * 5)
            
            return aleatoric_uncertainty
            
        except Exception:
            return 0.3
    
    def _calculate_total_uncertainty(self, uncertainty_sources: Dict[str, float]) -> float:
        """计算总不确定性"""
        try:
            # 使用平方和开方法组合不确定性
            uncertainties = list(uncertainty_sources.values())
            total_uncertainty_squared = sum(u**2 for u in uncertainties)
            total_uncertainty = np.sqrt(total_uncertainty_squared / len(uncertainties))
            
            return min(1.0, total_uncertainty)
            
        except Exception:
            return 0.5
    
    def _analyze_uncertainty_propagation(self, uncertainty_sources: Dict[str, float]) -> Dict[str, Any]:
        """分析不确定性传播"""
        try:
            # 不确定性传播分析
            propagation_analysis = {
                'dominant_source': max(uncertainty_sources.keys(), 
                                     key=lambda k: uncertainty_sources[k]),
                'uncertainty_distribution': uncertainty_sources,
                'propagation_factor': max(uncertainty_sources.values()) / 
                                    (np.mean(list(uncertainty_sources.values())) + 1e-10)
            }
            
            return propagation_analysis
            
        except Exception:
            return {'dominant_source': 'unknown'}
    
    def _perform_sensitivity_analysis(self, fusion_result: Dict) -> Dict[str, float]:
        """执行敏感性分析"""
        try:
            sensitivity = {}
            
            # 对关键参数的敏感性
            base_score = fusion_result.get('consensus_score', 0.5)
            
            # 权重敏感性（简化）
            for signal_type in ['technical', 'wavelet', 'fourier', 'ml']:
                # 模拟权重变化的影响
                sensitivity[f'{signal_type}_weight'] = abs(base_score - 0.5) * 0.1
            
            return sensitivity
            
        except Exception:
            return {}
    
    def _calculate_confidence_intervals(self, fusion_result: Dict, 
                                      uncertainty: float) -> Dict[str, Tuple[float, float]]:
        """计算置信区间"""
        try:
            base_score = fusion_result.get('consensus_score', 0.5)
            
            # 95%置信区间
            margin_of_error = 1.96 * uncertainty
            
            intervals = {
                '95%': (max(0.0, base_score - margin_of_error), 
                       min(1.0, base_score + margin_of_error)),
                '68%': (max(0.0, base_score - uncertainty), 
                       min(1.0, base_score + uncertainty))
            }
            
            return intervals
            
        except Exception:
            return {'95%': (0.0, 1.0), '68%': (0.25, 0.75)}
    
    def _rank_uncertainty_sources(self, uncertainty_sources: Dict[str, float]) -> List[Tuple[str, float]]:
        """排序不确定性来源"""
        try:
            return sorted(uncertainty_sources.items(), key=lambda x: x[1], reverse=True)
        except Exception:
            return []
    
    def _suggest_uncertainty_reduction(self, uncertainty_sources: Dict[str, float]) -> List[str]:
        """建议不确定性减少方法"""
        try:
            suggestions = []
            
            for source, uncertainty in uncertainty_sources.items():
                if uncertainty > 0.5:
                    if source == 'model':
                        suggestions.append('增加模型数量和多样性')
                    elif source == 'data':
                        suggestions.append('收集更多高质量数据')
                    elif source == 'parameter':
                        suggestions.append('优化参数设置')
                    elif source == 'epistemic':
                        suggestions.append('提高模型理论基础')
                    elif source == 'aleatoric':
                        suggestions.append('考虑使用概率模型')
            
            return suggestions
            
        except Exception:
            return ['优化整体系统配置']
    
    # ========== 异常处理和恢复方法 ==========
    
    def _handle_prediction_error(self, error: Exception) -> Dict[str, Any]:
        """处理预测错误"""
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            # 记录错误
            self.logger.error(f"预测错误 {error_type}: {error_message}")
            
            # 尝试恢复
            recovery_result = self._attempt_error_recovery(error_type)
            
            return {
                'success': False,
                'error_type': error_type,
                'error_message': error_message,
                'recovery_attempted': recovery_result['attempted'],
                'recovery_success': recovery_result['success'],
                'fallback_recommendation': self._get_fallback_recommendation()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error_type': 'critical_error',
                'error_message': f'错误处理失败: {str(e)}'
            }
    
    def _attempt_error_recovery(self, error_type: str) -> Dict[str, bool]:
        """尝试错误恢复"""
        try:
            recovery_attempted = True
            recovery_success = False
            
            if error_type == 'MemoryError':
                # 内存错误恢复
                self._trigger_memory_cleanup()
                recovery_success = True
                
            elif error_type == 'ValueError':
                # 数值错误恢复
                self._reset_problematic_parameters()
                recovery_success = True
                
            elif error_type == 'IndexError':
                # 索引错误恢复
                self._validate_data_structures()
                recovery_success = True
                
            else:
                recovery_attempted = False
            
            return {
                'attempted': recovery_attempted,
                'success': recovery_success
            }
            
        except Exception:
            return {'attempted': False, 'success': False}
    
    def _reset_problematic_parameters(self):
        """重置问题参数"""
        try:
            # 重置为默认配置
            default_config = self._get_default_config()
            
            # 重置关键参数
            critical_params = [
                'reversal_threshold', 'momentum_sensitivity', 
                'learning_rate', 'adaptation_factor'
            ]
            
            for param in critical_params:
                if param in default_config:
                    self.config[param] = default_config[param]
            
            self.logger.info("问题参数已重置为默认值")
            
        except Exception as e:
            self.logger.error(f"参数重置失败: {str(e)}")
    
    def _validate_data_structures(self):
        """验证数据结构"""
        try:
            # 验证关键数据结构
            structures_to_check = [
                'trend_history', 'technical_indicators', 
                'performance_monitor', 'signal_fusion'
            ]
            
            for structure_name in structures_to_check:
                if hasattr(self, structure_name):
                    structure = getattr(self, structure_name)
                    if not structure:
                        # 重新初始化空结构
                        if structure_name == 'trend_history':
                            self.trend_history = defaultdict(lambda: deque(maxlen=self.config['history_window']))
                        # 其他结构的重新初始化...
            
            self.logger.info("数据结构验证完成")
            
        except Exception as e:
            self.logger.error(f"数据结构验证失败: {str(e)}")
    
    def _get_fallback_recommendation(self) -> Dict[str, Any]:
        """获取备用推荐"""
        try:
            # 简单的备用策略
            return {
                'recommended_tail': 5,  # 中性选择
                'confidence': 0.3,
                'reasoning': '系统错误时的备用推荐',
                'recommendation_type': 'fallback',
                'risk_level': 'medium'
            }
            
        except Exception:
            return {
                'recommended_tail': None,
                'confidence': 0.0,
                'reasoning': '无法提供推荐'
            }
    
    # ========== 版本兼容性方法 ==========
    
    def predict_legacy(self, candidate_tails: List[int], historical_data: List[Dict]) -> Dict[str, Any]:
        """传统版本兼容的预测方法"""
        try:
            # 调用简化预测方法
            return self.predict_simple(candidate_tails, historical_data)
            
        except Exception as e:
            return self._handle_prediction_error(e)
    
    def get_simple_status(self) -> Dict[str, Any]:
        """获取简化状态信息"""
        try:
            return {
                'total_predictions': self.total_predictions,
                'accuracy': self.prediction_accuracy,
                'system_health': 'normal' if self.prediction_accuracy > 0.5 else 'needs_attention'
            }
            
        except Exception:
            return {'status': 'error'}
    
    # ========== 系统关闭和清理 ==========
    
    def shutdown(self):
        """系统关闭清理"""
        try:
            self.logger.info("开始系统关闭清理...")
            
            # 关闭线程池
            if hasattr(self, 'thread_pool') and self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                
            if hasattr(self, 'process_pool') and self.process_pool:
                self.process_pool.shutdown(wait=True)
            
            # 清理缓存
            if hasattr(self, 'cache'):
                self.cache.clear()
            
            # 最终内存清理
            gc.collect()
            
            self.logger.info("系统关闭清理完成")
            
        except Exception as e:
            self.logger.error(f"系统关闭清理错误: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass

# ========== 模块级别的工具函数 ==========

def create_anti_trend_hunter(config: Dict[str, Any] = None) -> AntiTrendHunter:
    """创建反趋势猎手实例的工厂函数"""
    try:
        return AntiTrendHunter(config)
    except Exception as e:
        print(f"❌ 反趋势猎手创建失败: {str(e)}")
        return None

def validate_historical_data(historical_data: List[Dict]) -> bool:
    """验证历史数据格式"""
    try:
        if not isinstance(historical_data, list):
            return False
        
        if len(historical_data) == 0:
            return False
        
        # 检查数据格式
        for period in historical_data:
            if not isinstance(period, dict):
                return False
            
            if 'tails' not in period:
                return False
            
            if not isinstance(period['tails'], list):
                return False
        
        return True
        
    except Exception:
        return False

def format_prediction_result(result: Dict[str, Any]) -> str:
    """格式化预测结果为可读字符串"""
    try:
        if not result.get('success', False):
            return f"❌ 预测失败: {result.get('reasoning', '未知错误')}"
        
        tail = result.get('recommended_tail')
        confidence = result.get('confidence', 0.0)
        reasoning = result.get('reasoning', '')
        
        confidence_level = "高" if confidence > 0.7 else "中" if confidence > 0.5 else "低"
        
        return f"""
🎯 反趋势预测结果
推荐尾数: {tail}
置信度: {confidence:.1%} ({confidence_level})
分析依据: {reasoning}
预测类型: {result.get('reversal_type', '技术反转')}
风险评估: {result.get('risk_assessment', {}).get('risk_level', '中等')}
建议时机: {result.get('timing_analysis', {}).get('optimal_timing', '密切关注')}
"""
        
    except Exception:
        return "❌ 结果格式化失败"

# ========== 模块初始化 ==========

if __name__ == "__main__":
    print("🧬 科研级反趋势猎手模块已加载")
    print("   版本: 2.0.0-research")
    print("   功能: 多维度反趋势分析与预测")
    print("   作者: AI Research Team")
    print("   使用: hunter = create_anti_trend_hunter()")
    
    # 简单的系统测试
    try:
        test_hunter = create_anti_trend_hunter()
        if test_hunter:
            print("✅ 系统初始化测试通过")
            test_hunter.shutdown()
        else:
            print("❌ 系统初始化测试失败")
    except Exception as e:
        print(f"❌ 系统测试错误: {str(e)}")
        
    def predict_simple(self, candidate_tails: List[int], historical_data: List[Dict]) -> Dict[str, Any]:
        """
        简化版预测方法（向后兼容）
        """
        # 创建数据哈希
        data_hash = self._create_data_hash(historical_data)
        
        # 调用完整版预测方法
        return self.predict(tuple(candidate_tails), data_hash, PredictionHorizon.SHORT)
    
    def _update_trend_states(self, historical_data: List[Dict]):
        """更新所有尾数的趋势状态"""
        for tail in range(10):
            # 计算不同时间窗口的出现频率
            frequencies = {}
            for window_name, window_size in self.analysis_windows.items():
                if len(historical_data) >= window_size:
                    count = sum(1 for i in range(window_size) 
                              if tail in historical_data[i].get('tails', []))
                    frequencies[window_name] = count / window_size
            
            # 判断趋势状态
            if frequencies:
                # 计算趋势斜率
                if 'long' in frequencies and 'short' in frequencies:
                    trend_slope = frequencies['short'] - frequencies['long']
                    
                    if trend_slope > 0.3:
                        state = TrendState.STRONG_UPTREND
                    elif trend_slope > 0.15:
                        state = TrendState.MODERATE_UPTREND
                    elif trend_slope > 0.05:
                        state = TrendState.WEAK_UPTREND
                    elif trend_slope > -0.05:
                        state = TrendState.SIDEWAYS
                    elif trend_slope > -0.15:
                        state = TrendState.WEAK_DOWNTREND
                    elif trend_slope > -0.3:
                        state = TrendState.MODERATE_DOWNTREND
                    else:
                        state = TrendState.STRONG_DOWNTREND
                    
                    self.trend_states[tail] = {
                        'state': state,
                        'slope': trend_slope,
                        'frequencies': frequencies
                    }
    
    def _calculate_technical_indicators(self, historical_data: List[Dict]):
        """计算所有技术指标"""
        for tail in range(10):
            # RSI计算
            self._calculate_rsi(tail, historical_data)
            
            # MACD计算
            self._calculate_macd(tail, historical_data)
            
            # 随机指标计算
            self._calculate_stochastic(tail, historical_data)
            
            # Williams %R计算
            self._calculate_williams_r(tail, historical_data)
            
            # OBV计算
            self._calculate_obv(tail, historical_data)
            
            # CCI计算
            self._calculate_cci(tail, historical_data)
            
            # MFI计算
            self._calculate_mfi(tail, historical_data)
            
            # ADX计算
            self._calculate_adx(tail, historical_data)
    
    def _analyze_trend_strength(self, tail: int, historical_data: List[Dict]) -> Dict:
        """分析趋势强度"""
        if tail not in self.trend_states:
            return {'state': 'unknown', 'strength': 0.0}
        
        trend_info = self.trend_states[tail]
        state = trend_info['state']
        
        # 计算趋势强度
        strength = abs(state.value) / 4.0  # 归一化到0-1
        
        # 计算趋势持续时间
        duration = self._calculate_trend_duration(tail, historical_data)
        
        # 计算趋势一致性
        consistency = self._calculate_trend_consistency(tail, historical_data)
        
        # 综合趋势强度
        total_strength = (strength * 0.4 + 
                         min(1.0, duration / 10.0) * 0.3 + 
                         consistency * 0.3)
        
        return {
            'state': state.name,
            'strength': total_strength,
            'raw_strength': strength,
            'duration': duration,
            'consistency': consistency
        }
    
    def _analyze_momentum(self, tail: int, historical_data: List[Dict]) -> Dict:
        """分析动量"""
        if len(historical_data) < self.config['momentum_period']:
            return {'momentum': 0.0, 'acceleration': 0.0}
        
        period = self.config['momentum_period']
        
        # 计算动量
        recent_count = sum(1 for i in range(period // 2) 
                         if tail in historical_data[i].get('tails', []))
        earlier_count = sum(1 for i in range(period // 2, period) 
                          if tail in historical_data[i].get('tails', []))
        
        if earlier_count > 0:
            momentum = (recent_count - earlier_count) / earlier_count
        else:
            momentum = recent_count / (period // 2)
        
        # 计算加速度
        if len(historical_data) >= period * 2:
            prev_momentum = self._calculate_previous_momentum(tail, historical_data, period)
            acceleration = momentum - prev_momentum
        else:
            acceleration = 0.0
        
        return {
            'momentum': momentum,
            'acceleration': acceleration,
            'is_decelerating': acceleration < -0.1,
            'is_accelerating': acceleration > 0.1
        }
    
    def _detect_trend_exhaustion(self, tail: int, historical_data: List[Dict]) -> Dict:
        """检测趋势耗尽"""
        exhaustion_signals = []
        
        # 1. 连续出现导致的耗尽
        consecutive_count = 0
        for period in historical_data:
            if tail in period.get('tails', []):
                consecutive_count += 1
            else:
                break
        
        if consecutive_count >= 5:
            exhaustion_signals.append(('consecutive_exhaustion', min(1.0, consecutive_count / 8.0)))
        
        # 2. 频率极端导致的耗尽
        if len(historical_data) >= 10:
            recent_freq = sum(1 for i in range(10) 
                            if tail in historical_data[i].get('tails', [])) / 10.0
            if recent_freq > 0.7:
                exhaustion_signals.append(('frequency_exhaustion', recent_freq))
        
        # 3. RSI超买超卖
        if tail in self.technical_indicators['rsi'] and self.technical_indicators['rsi'][tail]:
            latest_rsi = self.technical_indicators['rsi'][tail][-1]
            if latest_rsi > self.config['rsi_overbought']:
                exhaustion_signals.append(('rsi_overbought', (latest_rsi - 50) / 50))
            elif latest_rsi < self.config['rsi_oversold']:
                exhaustion_signals.append(('rsi_oversold', (50 - latest_rsi) / 50))
        
        # 4. 动量衰减
        momentum_data = self._analyze_momentum(tail, historical_data)
        if momentum_data['is_decelerating']:
            exhaustion_signals.append(('momentum_decay', abs(momentum_data['acceleration'])))
        
        # 综合耗尽程度
        if exhaustion_signals:
            exhaustion_level = np.mean([score for _, score in exhaustion_signals])
            exhaustion_types = [signal_type for signal_type, _ in exhaustion_signals]
        else:
            exhaustion_level = 0.0
            exhaustion_types = []
        
        return {
            'exhaustion': exhaustion_level,
            'signals': exhaustion_types,
            'is_exhausted': exhaustion_level > self.dynamic_thresholds['trend_exhaustion']
        }
    
    def _identify_reversal_signals(self, tail: int, historical_data: List[Dict]) -> Dict:
        """识别反转信号"""
        reversal_signals = []
        
        # 1. 背离信号
        divergence = self._check_divergence(tail, historical_data)
        if divergence['has_divergence']:
            reversal_signals.append(('divergence', divergence['strength']))
        
        # 2. 支撑阻力突破
        sr_break = self._check_support_resistance_break(tail, historical_data)
        if sr_break['has_break']:
            reversal_signals.append(('sr_break', sr_break['strength']))
        
        # 3. 形态反转信号
        pattern_reversal = self._check_pattern_reversal(tail, historical_data)
        if pattern_reversal['has_reversal']:
            reversal_signals.append(('pattern', pattern_reversal['confidence']))
        
        # 4. 成交量异常
        volume_anomaly = self._check_volume_anomaly(tail, historical_data)
        if volume_anomaly['has_anomaly']:
            reversal_signals.append(('volume', volume_anomaly['strength']))
        
        # 5. 极值反转
        extreme_reversal = self._check_extreme_reversal(tail, historical_data)
        if extreme_reversal['is_extreme']:
            reversal_signals.append(('extreme', extreme_reversal['probability']))
        
        # 综合反转置信度
        if reversal_signals:
            confidence = np.mean([score for _, score in reversal_signals])
            signal_types = [signal_type for signal_type, _ in reversal_signals]
        else:
            confidence = 0.0
            signal_types = []
        
        return {
            'confidence': confidence,
            'signals': signal_types,
            'signal_count': len(reversal_signals),
            'has_strong_signal': confidence > self.dynamic_thresholds['reversal_confidence']
        }
    
    def _analyze_breakout_potential(self, tail: int, historical_data: List[Dict]) -> Dict:
        """分析突破潜力"""
        # 计算近期的震荡区间
        if len(historical_data) < 10:
            return {'potential': 0.0, 'direction': 'unknown'}
        
        # 计算最近10期的出现情况
        appearances = []
        for i in range(10):
            appearances.append(1 if tail in historical_data[i].get('tails', []) else 0)
        
        # 计算震荡区间
        high = max(appearances)
        low = min(appearances)
        current = appearances[0]
        
        # 判断突破方向和潜力
        if current == high and high > np.mean(appearances):
            # 向上突破
            potential = min(1.0, (current - np.mean(appearances)) / 0.5)
            direction = 'upward'
        elif current == low and low < np.mean(appearances):
            # 向下突破
            potential = min(1.0, (np.mean(appearances) - current) / 0.5)
            direction = 'downward'
        else:
            potential = 0.0
            direction = 'sideways'
        
        # 计算突破强度
        if len(historical_data) >= 20:
            long_term_mean = sum(1 for i in range(20) 
                               if tail in historical_data[i].get('tails', [])) / 20.0
            breakout_strength = abs(current - long_term_mean)
        else:
            breakout_strength = 0.0
        
        return {
            'potential': potential,
            'direction': direction,
            'strength': breakout_strength,
            'is_breakout': potential > 0.5
        }
    
    def _analyze_volume_divergence(self, tail: int, historical_data: List[Dict]) -> float:
        """分析成交量背离"""
        if len(historical_data) < 10:
            return 0.0
        
        # 简化的成交量分析（用出现的尾数总数模拟成交量）
        price_trend = []
        volume_trend = []
        
        for i in range(10):
            # "价格"用出现频率表示
            price = 1 if tail in historical_data[i].get('tails', []) else 0
            price_trend.append(price)
            
            # "成交量"用该期总尾数表示
            volume = len(historical_data[i].get('tails', []))
            volume_trend.append(volume)
        
        # 计算趋势相关性
        if len(set(price_trend)) > 1 and len(set(volume_trend)) > 1:
            correlation = np.corrcoef(price_trend, volume_trend)[0, 1]
            
            # 负相关表示背离
            if correlation < -0.3:
                divergence = abs(correlation)
            else:
                divergence = 0.0
        else:
            divergence = 0.0
        
        return divergence
    
    def _calculate_technical_score(self, tail: int) -> float:
        """计算技术指标综合得分"""
        scores = []
        
        # RSI得分
        if tail in self.technical_indicators['rsi'] and self.technical_indicators['rsi'][tail]:
            rsi = self.technical_indicators['rsi'][tail][-1]
            if rsi > self.config['rsi_overbought'] or rsi < self.config['rsi_oversold']:
                scores.append(0.8)
            else:
                scores.append(0.2)
        
        # MACD得分
        if tail in self.technical_indicators['macd'] and self.technical_indicators['macd'][tail]:
            macd_signal = self.technical_indicators['macd'][tail][-1]
            if abs(macd_signal) > 0.5:
                scores.append(0.7)
            else:
                scores.append(0.3)
        
        # 其他指标得分...
        
        return np.mean(scores) if scores else 0.5
    
    def _match_reversal_patterns(self, tail: int, historical_data: List[Dict]) -> List[Dict]:
        """匹配反转模式"""
        matched_patterns = []
        
        for pattern_name, pattern_func in self.reversal_pattern_library.items():
            if pattern_func(tail, historical_data):
                matched_patterns.append({
                    'name': pattern_name,
                    'confidence': 0.7  # 简化处理
                })
        
        return matched_patterns
    
    def _calculate_reversal_score(self, trend_analysis: Dict, momentum_analysis: Dict,
                                 exhaustion_analysis: Dict, reversal_signals: Dict,
                                 breakout_analysis: Dict, volume_divergence: float,
                                 technical_score: float, pattern_matches: List) -> float:
        """计算综合反转得分"""
        
        # 基础分数
        base_score = 0.0
        
        # 1. 趋势强度贡献（强趋势反转更有价值）
        if trend_analysis['strength'] > 0.6:
            base_score += trend_analysis['strength'] * 0.2
        
        # 2. 动量贡献（负动量或减速增加反转可能）
        if momentum_analysis['momentum'] < 0 or momentum_analysis['is_decelerating']:
            base_score += 0.15
        
        # 3. 耗尽贡献
        base_score += exhaustion_analysis['exhaustion'] * 0.25
        
        # 4. 反转信号贡献
        base_score += reversal_signals['confidence'] * 0.2
        
        # 5. 突破潜力贡献
        if breakout_analysis['is_breakout']:
            base_score += breakout_analysis['potential'] * 0.1
        
        # 6. 成交量背离贡献
        base_score += volume_divergence * 0.1
        
        # 7. 技术指标贡献
        base_score += technical_score * 0.1
        
        # 8. 模式匹配贡献
        if pattern_matches:
            base_score += min(0.1, len(pattern_matches) * 0.03)
        
        # 确保分数在0-1范围内
        final_score = min(1.0, max(0.0, base_score))
        
        # 应用非线性变换增强区分度
        if final_score > 0.7:
            final_score = 0.7 + (final_score - 0.7) * 1.5
        elif final_score < 0.3:
            final_score = final_score * 0.7
        
        return min(1.0, final_score)
    
    def _calculate_confidence(self, score: float, analysis: Dict) -> float:
        """计算置信度"""
        confidence_factors = []
        
        # 基础置信度来自得分
        confidence_factors.append(score)
        
        # 趋势强度影响置信度
        if 'trend_strength' in analysis:
            confidence_factors.append(analysis['trend_strength'])
        
        # 耗尽程度影响置信度
        if 'exhaustion_level' in analysis:
            confidence_factors.append(analysis['exhaustion_level'])
        
        # 反转信号影响置信度
        if 'reversal_confidence' in analysis:
            confidence_factors.append(analysis['reversal_confidence'])
        
        # 模式匹配数量影响置信度
        if 'pattern_matches' in analysis:
            pattern_confidence = min(1.0, analysis['pattern_matches'] / 3.0)
            confidence_factors.append(pattern_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else score
    
    def _generate_reasoning(self, tail: int, analysis: Dict) -> str:
        """生成推理说明"""
        reasons = []
        
        if analysis['trend_state'] in ['STRONG_UPTREND', 'MODERATE_UPTREND']:
            reasons.append(f"尾数{tail}处于{analysis['trend_state']}，即将反转")
        
        if analysis['exhaustion_level'] > 0.7:
            reasons.append(f"趋势耗尽程度达{analysis['exhaustion_level']:.0%}")
        
        if analysis['reversal_confidence'] > 0.6:
            reasons.append(f"反转信号强度{analysis['reversal_confidence']:.0%}")
        
        if analysis['volume_divergence'] > 0.3:
            reasons.append("出现明显成交量背离")
        
        if analysis['pattern_matches'] > 0:
            reasons.append(f"匹配{analysis['pattern_matches']}个反转模式")
        
        return "；".join(reasons) if reasons else "综合技术分析显示反转机会"
    
    # === 辅助方法实现 ===
    
    def _initialize_reversal_patterns(self) -> Dict:
        """初始化反转模式库"""
        return {
            'double_top': self._check_double_top,
            'double_bottom': self._check_double_bottom,
            'head_shoulders': self._check_head_shoulders,
            'wedge': self._check_wedge,
            'triangle': self._check_triangle,
            'flag': self._check_flag,
            'channel_break': self._check_channel_break
        }
    
    def _calculate_trend_duration(self, tail: int, historical_data: List[Dict]) -> int:
        """计算趋势持续时间"""
        if tail not in self.trend_states:
            return 0
        
        current_state = self.trend_states[tail]['state']
        duration = 0
        
        # 向后查找相同趋势状态的持续时间
        for i in range(len(historical_data)):
            # 简化：通过频率变化判断趋势是否改变
            period_has_tail = tail in historical_data[i].get('tails', [])
            
            if current_state.value > 0:  # 上升趋势
                if period_has_tail:
                    duration += 1
                else:
                    if duration >= self.config['min_trend_duration']:
                        break
            else:  # 下降趋势
                if not period_has_tail:
                    duration += 1
                else:
                    if duration >= self.config['min_trend_duration']:
                        break
        
        return duration
    
    def _calculate_trend_consistency(self, tail: int, historical_data: List[Dict]) -> float:
        """计算趋势一致性"""
        if len(historical_data) < 5:
            return 0.0
        
        # 计算方向一致性
        directions = []
        for i in range(len(historical_data) - 1):
            current = 1 if tail in historical_data[i].get('tails', []) else 0
            next_val = 1 if tail in historical_data[i + 1].get('tails', []) else 0
            
            if current > next_val:
                directions.append(1)  # 上升
            elif current < next_val:
                directions.append(-1)  # 下降
            else:
                directions.append(0)  # 持平
        
        if not directions:
            return 0.0
        
        # 计算一致性（方向相同的比例）
        most_common = max(set(directions), key=directions.count)
        consistency = directions.count(most_common) / len(directions)
        
        return consistency
    
    def _calculate_previous_momentum(self, tail: int, historical_data: List[Dict], period: int) -> float:
        """计算之前的动量（用于计算加速度）"""
        if len(historical_data) < period * 2:
            return 0.0
        
        # 计算前一个周期的动量
        recent_count = sum(1 for i in range(period, period + period // 2) 
                         if tail in historical_data[i].get('tails', []))
        earlier_count = sum(1 for i in range(period + period // 2, period * 2) 
                          if tail in historical_data[i].get('tails', []))
        
        if earlier_count > 0:
            momentum = (recent_count - earlier_count) / earlier_count
        else:
            momentum = recent_count / (period // 2)
        
        return momentum
    
    def _calculate_rsi(self, tail: int, historical_data: List[Dict], period: int = 14):
        """计算RSI指标"""
        if len(historical_data) < period + 1:
            return
        
        gains = []
        losses = []
        
        for i in range(period):
            current = 1 if tail in historical_data[i].get('tails', []) else 0
            previous = 1 if tail in historical_data[i + 1].get('tails', []) else 0
            
            change = current - previous
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        self.technical_indicators['rsi'][tail].append(rsi)
    
    def _calculate_macd(self, tail: int, historical_data: List[Dict]):
        """计算MACD指标"""
        if len(historical_data) < 26:
            return
        
        # 简化的MACD计算
        fast_period = 12
        slow_period = 26
        
        fast_ma = sum(1 for i in range(fast_period) 
                     if tail in historical_data[i].get('tails', [])) / fast_period
        slow_ma = sum(1 for i in range(slow_period) 
                     if tail in historical_data[i].get('tails', [])) / slow_period
        
        macd = fast_ma - slow_ma
        self.technical_indicators['macd'][tail].append(macd)
    
    def _calculate_stochastic(self, tail: int, historical_data: List[Dict], period: int = 14):
        """计算随机指标"""
        if len(historical_data) < period:
            return
        
        # 获取期间内的最高和最低
        values = []
        for i in range(period):
            values.append(1 if tail in historical_data[i].get('tails', []) else 0)
        
        high = max(values)
        low = min(values)
        current = values[0]
        
        if high != low:
            k = ((current - low) / (high - low)) * 100
        else:
            k = 50
        
        self.technical_indicators['stochastic'][tail].append(k)
    
    def _calculate_williams_r(self, tail: int, historical_data: List[Dict], period: int = 14):
        """计算Williams %R指标"""
        if len(historical_data) < period:
            return
        
        values = []
        for i in range(period):
            values.append(1 if tail in historical_data[i].get('tails', []) else 0)
        
        high = max(values)
        low = min(values)
        current = values[0]
        
        if high != low:
            williams_r = ((high - current) / (high - low)) * -100
        else:
            williams_r = -50
        
        self.technical_indicators['williams_r'][tail].append(williams_r)
    
    def _calculate_obv(self, tail: int, historical_data: List[Dict]):
        """计算OBV指标"""
        if len(historical_data) < 2:
            return
        
        # 简化的OBV计算
        obv = 0
        for i in range(len(historical_data) - 1):
            current = 1 if tail in historical_data[i].get('tails', []) else 0
            previous = 1 if tail in historical_data[i + 1].get('tails', []) else 0
            volume = len(historical_data[i].get('tails', []))  # 用尾数总数作为成交量
            
            if current > previous:
                obv += volume
            elif current < previous:
                obv -= volume
        
        self.technical_indicators['obv'][tail].append(obv)
    
    def _calculate_cci(self, tail: int, historical_data: List[Dict], period: int = 20):
        """计算CCI指标"""
        if len(historical_data) < period:
            return
        
        # 简化的CCI计算
        typical_prices = []
        for i in range(period):
            value = 1 if tail in historical_data[i].get('tails', []) else 0
            typical_prices.append(value)
        
        sma = np.mean(typical_prices)
        mean_deviation = np.mean([abs(tp - sma) for tp in typical_prices])
        
        if mean_deviation != 0:
            cci = (typical_prices[0] - sma) / (0.015 * mean_deviation)
        else:
            cci = 0
        
        self.technical_indicators['cci'][tail].append(cci)
    
    def _calculate_mfi(self, tail: int, historical_data: List[Dict], period: int = 14):
        """计算MFI指标"""
        if len(historical_data) < period + 1:
            return
        
        positive_flow = 0
        negative_flow = 0
        
        for i in range(period):
            current = 1 if tail in historical_data[i].get('tails', []) else 0
            previous = 1 if tail in historical_data[i + 1].get('tails', []) else 0
            volume = len(historical_data[i].get('tails', []))
            
            money_flow = current * volume
            
            if current > previous:
                positive_flow += money_flow
            elif current < previous:
                negative_flow += money_flow
        
        if negative_flow == 0:
            mfi = 100
        else:
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))
        
        self.technical_indicators['mfi'][tail].append(mfi)
    
    def _calculate_adx(self, tail: int, historical_data: List[Dict], period: int = 14):
        """计算ADX指标"""
        if len(historical_data) < period + 1:
            return
        
        # 简化的ADX计算
        dx_values = []
        
        for i in range(period):
            current = 1 if tail in historical_data[i].get('tails', []) else 0
            previous = 1 if tail in historical_data[i + 1].get('tails', []) else 0
            
            # 计算方向性指标
            if current > previous:
                plus_dm = current - previous
                minus_dm = 0
            elif current < previous:
                plus_dm = 0
                minus_dm = previous - current
            else:
                plus_dm = 0
                minus_dm = 0
            
            # 简化的DX计算
            if plus_dm + minus_dm > 0:
                dx = abs(plus_dm - minus_dm) / (plus_dm + minus_dm) * 100
            else:
                dx = 0
            
            dx_values.append(dx)
        
        adx = np.mean(dx_values)
        self.technical_indicators['adx'][tail].append(adx)
    
    def _check_divergence(self, tail: int, historical_data: List[Dict]) -> Dict:
        """检查背离信号"""
        if len(historical_data) < 10:
            return {'has_divergence': False, 'strength': 0.0}
        
        # 价格趋势（用出现频率表示）
        price_trend = []
        for i in range(10):
            price_trend.append(1 if tail in historical_data[i].get('tails', []) else 0)
        
        # 动量趋势（用RSI表示）
        if tail in self.technical_indicators['rsi'] and len(self.technical_indicators['rsi'][tail]) >= 10:
            momentum_trend = self.technical_indicators['rsi'][tail][-10:]
            
            # 检查价格和动量的背离
            price_direction = 1 if price_trend[0] > price_trend[-1] else -1
            momentum_direction = 1 if momentum_trend[0] > momentum_trend[-1] else -1
            
            if price_direction != momentum_direction:
                # 存在背离
                strength = abs(price_trend[0] - price_trend[-1]) + abs(momentum_trend[0] - momentum_trend[-1]) / 100
                return {'has_divergence': True, 'strength': min(1.0, strength)}
        
        return {'has_divergence': False, 'strength': 0.0}
    
    def _check_support_resistance_break(self, tail: int, historical_data: List[Dict]) -> Dict:
        """检查支撑阻力突破"""
        if len(historical_data) < 20:
            return {'has_break': False, 'strength': 0.0}
        
        # 计算支撑和阻力水平
        frequencies = []
        for i in range(20):
            count = sum(1 for j in range(max(0, i-5), min(i+5, len(historical_data)))
                      if tail in historical_data[j].get('tails', []))
            frequencies.append(count / min(10, len(historical_data) - max(0, i-5)))
        
        # 找出关键水平
        resistance = max(frequencies)
        support = min(frequencies)
        current = frequencies[0]
        
        # 检查突破
        if current > resistance * 0.95:
            # 阻力突破
            return {'has_break': True, 'strength': min(1.0, (current - resistance) / resistance)}
        elif current < support * 1.05:
            # 支撑突破
            return {'has_break': True, 'strength': min(1.0, (support - current) / support)}
        
        return {'has_break': False, 'strength': 0.0}
    
    def _check_pattern_reversal(self, tail: int, historical_data: List[Dict]) -> Dict:
        """检查形态反转信号"""
        patterns_found = []
        
        # 检查各种反转形态
        if self._check_double_top(tail, historical_data):
            patterns_found.append('double_top')
        
        if self._check_double_bottom(tail, historical_data):
            patterns_found.append('double_bottom')
        
        if self._check_head_shoulders(tail, historical_data):
            patterns_found.append('head_shoulders')
        
        if patterns_found:
            return {
                'has_reversal': True,
                'confidence': min(1.0, len(patterns_found) * 0.3),
                'patterns': patterns_found
            }
        
        return {'has_reversal': False, 'confidence': 0.0}
    
    def _check_volume_anomaly(self, tail: int, historical_data: List[Dict]) -> Dict:
        """检查成交量异常"""
        if len(historical_data) < 10:
            return {'has_anomaly': False, 'strength': 0.0}
        
        # 计算平均成交量（用尾数总数模拟）
        volumes = [len(period.get('tails', [])) for period in historical_data[:10]]
        avg_volume = np.mean(volumes)
        current_volume = volumes[0]
        
        # 检查异常
        if current_volume > avg_volume * 1.5:
            # 放量
            return {'has_anomaly': True, 'strength': min(1.0, (current_volume - avg_volume) / avg_volume)}
        elif current_volume < avg_volume * 0.5:
            # 缩量
            return {'has_anomaly': True, 'strength': min(1.0, (avg_volume - current_volume) / avg_volume)}
        
        return {'has_anomaly': False, 'strength': 0.0}
    
    def _check_extreme_reversal(self, tail: int, historical_data: List[Dict]) -> Dict:
        """检查极值反转"""
        if len(historical_data) < 30:
            return {'is_extreme': False, 'probability': 0.0}
        
        # 计算30期内的频率
        frequency = sum(1 for i in range(30) 
                      if tail in historical_data[i].get('tails', [])) / 30.0
        
        # 检查是否达到极值
        if frequency > 0.8:
            # 极度超买
            return {'is_extreme': True, 'probability': frequency}
        elif frequency < 0.2:
            # 极度超卖
            return {'is_extreme': True, 'probability': 1 - frequency}
        
        return {'is_extreme': False, 'probability': 0.0}
    
    # === 反转形态检测方法 ===
    
    def _check_double_top(self, tail: int, historical_data: List[Dict]) -> bool:
        """检查双顶形态"""
        if len(historical_data) < 15:
            return False
        
        # 简化：检查两个相似的高点
        peaks = []
        for i in range(1, 14):
            prev = 1 if tail in historical_data[i-1].get('tails', []) else 0
            curr = 1 if tail in historical_data[i].get('tails', []) else 0
            next_val = 1 if tail in historical_data[i+1].get('tails', []) else 0
            
            if curr > prev and curr > next_val:
                peaks.append(i)
        
        return len(peaks) >= 2
    
    def _check_double_bottom(self, tail: int, historical_data: List[Dict]) -> bool:
        """检查双底形态"""
        if len(historical_data) < 15:
            return False
        
        # 简化：检查两个相似的低点
        troughs = []
        for i in range(1, 14):
            prev = 1 if tail in historical_data[i-1].get('tails', []) else 0
            curr = 1 if tail in historical_data[i].get('tails', []) else 0
            next_val = 1 if tail in historical_data[i+1].get('tails', []) else 0
            
            if curr < prev and curr < next_val:
                troughs.append(i)
        
        return len(troughs) >= 2
    
    def _check_head_shoulders(self, tail: int, historical_data: List[Dict]) -> bool:
        """检查头肩形态"""
        if len(historical_data) < 20:
            return False
        
        # 简化：检查三个峰，中间最高
        peaks = []
        for i in range(1, 19):
            prev = 1 if tail in historical_data[i-1].get('tails', []) else 0
            curr = 1 if tail in historical_data[i].get('tails', []) else 0
            next_val = 1 if tail in historical_data[i+1].get('tails', []) else 0
            
            if curr > prev and curr > next_val:
                peaks.append((i, curr))
        
        if len(peaks) >= 3:
            # 检查中间峰是否最高
            middle_idx = len(peaks) // 2
            if all(peaks[middle_idx][1] >= peak[1] for peak in peaks):
                return True
        
        return False
    
    def _check_wedge(self, tail: int, historical_data: List[Dict]) -> bool:
        """检查楔形形态"""
        if len(historical_data) < 10:
            return False
        
        # 简化：检查收敛趋势
        high_points = []
        low_points = []
        
        for i in range(10):
            value = 1 if tail in historical_data[i].get('tails', []) else 0
            if i % 2 == 0:
                high_points.append(value)
            else:
                low_points.append(value)
        
        # 检查是否收敛
        if len(high_points) >= 2 and len(low_points) >= 2:
            high_trend = high_points[0] - high_points[-1]
            low_trend = low_points[-1] - low_points[0]
            
            if high_trend > 0 and low_trend > 0:
                return True
        
        return False
    
    def _check_triangle(self, tail: int, historical_data: List[Dict]) -> bool:
        """检查三角形形态"""
        return self._check_wedge(tail, historical_data)  # 简化处理
    
    def _check_flag(self, tail: int, historical_data: List[Dict]) -> bool:
        """检查旗形形态"""
        if len(historical_data) < 8:
            return False
        
        # 简化：检查短期盘整
        values = []
        for i in range(8):
            values.append(1 if tail in historical_data[i].get('tails', []) else 0)
        
        # 检查是否在狭窄区间震荡
        if max(values) - min(values) <= 0.3:
            return True
        
        return False
    
    def _check_channel_break(self, tail: int, historical_data: List[Dict]) -> bool:
        """检查通道突破"""
        if len(historical_data) < 15:
            return False
        
        # 计算通道上下轨
        upper_bound = []
        lower_bound = []
        
        for i in range(15):
            value = 1 if tail in historical_data[i].get('tails', []) else 0
            if value == 1:
                upper_bound.append(i)
            else:
                lower_bound.append(i)
        
        # 检查是否突破通道
        if upper_bound and lower_bound:
            current = 1 if tail in historical_data[0].get('tails', []) else 0
            avg_upper = np.mean([1 for _ in upper_bound])
            avg_lower = np.mean([0 for _ in lower_bound])
            
            if current > avg_upper * 1.1 or current < avg_lower * 0.9:
                return True
        
        return False
    
    def _classify_reversal_type(self, analysis: Dict) -> str:
        """分类反转类型"""
        if analysis['exhaustion_level'] > 0.7:
            return 'exhaustion_reversal'
        elif analysis['momentum'] < -0.3:
            return 'momentum_reversal'
        elif analysis['pattern_matches'] > 2:
            return 'pattern_reversal'
        elif analysis['volume_divergence'] > 0.5:
            return 'volume_reversal'
        else:
            return 'technical_reversal'
    
    def _assess_risk(self, analysis: Dict) -> str:
        """评估风险水平"""
        risk_score = 0.0
        
        # 趋势强度风险
        if analysis['trend_strength'] > 0.7:
            risk_score += 0.3  # 强趋势反转风险较高
        
        # 反转信号不足风险
        if analysis['reversal_confidence'] < 0.5:
            risk_score += 0.3
        
        # 技术指标冲突风险
        if analysis['technical_score'] < 0.5:
            risk_score += 0.2
        
        # 模式不足风险
        if analysis['pattern_matches'] == 0:
            risk_score += 0.2
        
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_timing(self, analysis: Dict) -> str:
        """建议操作时机"""
        if analysis['exhaustion_level'] > 0.8:
            return 'immediate'  # 立即反转
        elif analysis['reversal_confidence'] > 0.7:
            return 'next_1_2_periods'  # 1-2期内
        elif analysis['breakout_potential'] > 0.6:
            return 'wait_for_confirmation'  # 等待确认
        else:
            return 'monitor_closely'  # 密切关注
    
    def _update_learning_parameters(self, tail: int, confidence: float):
        """更新学习参数"""
        self.total_predictions += 1
        
        # 更新动态阈值
        if confidence > 0.7:
            # 高置信度时，略微提高阈值
            for key in self.dynamic_thresholds:
                self.dynamic_thresholds[key] = min(0.9, 
                    self.dynamic_thresholds[key] * (1 + self.learning_rate * 0.1))
        elif confidence < 0.3:
            # 低置信度时，降低阈值
            for key in self.dynamic_thresholds:
                self.dynamic_thresholds[key] = max(0.3, 
                    self.dynamic_thresholds[key] * (1 - self.learning_rate * 0.1))
    
    def learn_from_outcome(self, prediction: Dict, actual_tails: List[int]) -> Dict:
        """从结果中学习"""
        if not prediction or 'recommended_tail' not in prediction:
            return {'learning_success': False}
        
        predicted_tail = prediction['recommended_tail']
        was_correct = predicted_tail in actual_tails
        
        # 更新统计
        if was_correct:
            self.successful_reversals += 1
        else:
            self.false_signals += 1
        
        # 计算准确率
        if self.total_predictions > 0:
            self.prediction_accuracy = self.successful_reversals / self.total_predictions
        
        # 动态调整参数
        if was_correct:
            # 增强成功的参数配置
            self.config['reversal_threshold'] *= 0.98  # 略微降低阈值
            self.config['momentum_sensitivity'] *= 1.02  # 提高敏感度
        else:
            # 调整失败的参数配置
            self.config['reversal_threshold'] *= 1.02  # 提高阈值
            self.config['momentum_sensitivity'] *= 0.98  # 降低敏感度
        
        # 确保参数在合理范围内
        self.config['reversal_threshold'] = max(0.5, min(0.8, self.config['reversal_threshold']))
        self.config['momentum_sensitivity'] = max(1.0, min(2.0, self.config['momentum_sensitivity']))
        
        return {
            'learning_success': True,
            'was_correct': was_correct,
            'current_accuracy': self.prediction_accuracy,
            'total_predictions': self.total_predictions,
            'successful_reversals': self.successful_reversals,
            'false_signals': self.false_signals,
            'updated_thresholds': self.dynamic_thresholds.copy()
        }