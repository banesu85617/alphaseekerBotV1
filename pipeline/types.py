"""
数据类型定义模块

定义多策略信号处理管道中使用的所有数据类型和结构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

# 枚举定义
class SignalDirection(Enum):
    """信号方向枚举"""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"

class ConfidenceLevel(Enum):
    """置信度等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class StrategyType(Enum):
    """策略类型"""
    TECHNICAL_INDICATOR = "technical_indicator"
    ML_PREDICTION = "ml_prediction"
    RISK_MODEL = "risk_model"
    BACKTEST_REFERENCE = "backtest_reference"

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# 数据结构定义
@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    ohlcv: Optional[pd.DataFrame] = None
    order_book: Optional[Dict] = None
    data_freshness: float = 0.0  # 数据新鲜度(秒)

@dataclass
class TechnicalIndicators:
    """技术指标数据"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    adx: Optional[float] = None
    atr: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

@dataclass
class RiskMetrics:
    """风险指标数据"""
    garch_volatility: Optional[float] = None
    var_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

@dataclass
class MLFeatures:
    """机器学习特征数据"""
    spread: Optional[float] = None
    order_imbalance: Optional[float] = None
    depth_imbalance: Optional[float] = None
    wap_1: Optional[float] = None
    wap_5: Optional[float] = None
    volatility_60s: Optional[float] = None
    mid_price: Optional[float] = None
    volume_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class MLPrediction:
    """机器学习预测结果"""
    label: int  # 1: 买入, 0: 持有, -1: 卖出
    probability_scores: Dict[int, float] = field(default_factory=dict)
    confidence: float = 0.0
    model_version: str = ""
    prediction_time: datetime = field(default_factory=datetime.now)

@dataclass
class LLMAssessment:
    """LLM评估结果"""
    trade_direction: SignalDirection
    optimal_entry: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_assessment: RiskLevel
    analysis_summary: str = ""
    reasoning: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """回测结果"""
    score: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_count: int
    backtest_period: str
    strategy_name: str = ""
    additional_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class StrategySignal:
    """策略信号"""
    strategy_type: StrategyType
    direction: SignalDirection
    confidence: float
    score: float
    timestamp: datetime
    symbol: str
    market_data: MarketData
    technical_indicators: Optional[TechnicalIndicators] = None
    risk_metrics: Optional[RiskMetrics] = None
    ml_prediction: Optional[MLPrediction] = None
    llm_assessment: Optional[LLMAssessment] = None
    backtest_result: Optional[BacktestResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusionResult:
    """策略融合结果"""
    symbol: str
    final_direction: SignalDirection
    final_score: float
    combined_confidence: float
    risk_reward_ratio: float
    component_scores: Dict[StrategyType, float] = field(default_factory=dict)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    decision_reason: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """管道配置"""
    # 基础配置
    max_concurrent_tasks: int = 16
    timeout_seconds: int = 10
    batch_size: int = 100
    
    # 第一层LightGBM配置
    ml_probability_threshold: float = 0.65
    ml_confidence_threshold: float = 0.6
    
    # 第二层LLM配置
    llm_confidence_threshold: float = 0.65
    llm_timeout: float = 3.0
    llm_provider: str = "local"  # local, lm_studio, ollama
    
    # 策略融合配置
    strategy_weights: Dict[StrategyType, float] = field(default_factory=lambda: {
        StrategyType.TECHNICAL_INDICATOR: 0.4,
        StrategyType.ML_PREDICTION: 0.2,
        StrategyType.RISK_MODEL: 0.2,
        StrategyType.BACKTEST_REFERENCE: 0.2
    })
    
    # 风险控制配置
    min_risk_reward_ratio: float = 1.0
    max_position_size: float = 0.1
    max_leverage: float = 10.0
    
    # 性能配置
    cache_ttl: Dict[str, int] = field(default_factory=lambda: {
        "indicators": 300,  # 5分钟
        "ml_predictions": 60,  # 1分钟
        "llm_assessments": 600,  # 10分钟
        "backtest_results": 604800  # 7天
    })
    
    # 过滤条件
    min_backtest_score: float = 0.60
    require_sma_alignment: bool = True
    min_adx_threshold: float = 20.0
    max_symbols_per_scan: int = 100
    top_n_results: int = 10

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_processing_time: float
    throughput: float  # 每秒处理符号数
    accuracy_metrics: Dict[str, float]
    latency_breakdown: Dict[str, float]
    cache_hit_rates: Dict[str, float]
    error_rates: Dict[str, float]
    strategy_contribution: Dict[StrategyType, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScanRequest:
    """扫描请求"""
    symbols: List[str]
    timeframe: str = "1m"
    max_symbols: Optional[int] = None
    top_n: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScanResult:
    """扫描结果"""
    request_id: str
    results: List[FusionResult]
    processing_time: float
    total_symbols: int
    filtered_symbols: int
    timestamp: datetime = field(default_factory=datetime.now)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# 异常定义
class PipelineError(Exception):
    """管道基础异常"""
    pass

class ConfigurationError(PipelineError):
    """配置错误"""
    pass

class DataInsufficientError(PipelineError):
    """数据不足错误"""
    pass

class ModelUnavailableError(PipelineError):
    """模型不可用错误"""
    pass

class ProcessingTimeoutError(PipelineError):
    """处理超时错误"""
    pass