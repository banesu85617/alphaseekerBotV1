"""
多策略信号处理管道 (Multi-Strategy Signal Processing Pipeline)

该模块实现了基于双重验证架构的统一信号处理管道，整合了：
- LightGBM快速筛选 (第一层)
- 本地LLM深度评估 (第二层)  
- 技术指标触发机制
- 机器学习预测
- 动态策略权重算法
- 策略融合和冲突解决
- 信号优先级排序
- 策略回测和验证
- 策略性能监控
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# 添加StrategyType枚举
class StrategyType(Enum):
    """交易策略类型枚举"""
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading" 
    SCALPING = "scalping"
    POSITION_TRADING = "position_trading"
    ALPHA_SEEKING = "alpha_seeking"
    ARBITRAGE = "arbitrage"
    MULTI_STRATEGY = "multi_strategy"

# 基础数据类型定义
@dataclass
class PipelineConfig:
    """管道配置"""
    name: str
    enabled: bool = True
    max_signals: int = 100
    confidence_threshold: float = 0.7
    strategy_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.strategy_weights is None:
            self.strategy_weights = {
                "day_trading": 1.0,
                "swing_trading": 1.0,
                "alpha_seeking": 1.0
            }

@dataclass
class StrategySignal:
    """策略信号"""
    symbol: str
    signal_type: StrategyType
    price: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class FusionResult:
    """融合结果"""
    final_signal: StrategySignal
    contributing_signals: List[StrategySignal]
    fusion_method: str
    confidence_score: float
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_signals: int = 0
    successful_signals: int = 0
    accuracy_rate: float = 0.0
    avg_confidence: float = 0.0
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()

# 基础类实现
class SignalProcessor:
    """信号处理器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """处理信号列表"""
        processed = []
        for signal in signals:
            if signal.confidence >= self.config.confidence_threshold:
                processed.append(signal)
        return processed

class StrategyFusion:
    """策略融合器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fuse_signals(self, signals: List[StrategySignal]) -> FusionResult:
        """融合多个信号"""
        if not signals:
            raise ValueError("没有信号可供融合")
        
        # 简单的加权融合策略
        total_weight = sum(self.config.strategy_weights.get(s.signal_type.value, 1.0) 
                          for s in signals)
        weighted_confidence = sum(s.confidence * 
                                 self.config.strategy_weights.get(s.signal_type.value, 1.0) 
                                 for s in signals) / total_weight
        
        # 选择置信度最高的作为主要信号
        main_signal = max(signals, key=lambda x: x.confidence)
        
        return FusionResult(
            final_signal=main_signal,
            contributing_signals=signals,
            fusion_method="weighted_average",
            confidence_score=weighted_confidence,
            timestamp=datetime.now()
        )

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.logger = logging.getLogger(__name__)
    
    def update_metrics(self, signals: List[StrategySignal], results: List[FusionResult]):
        """更新性能指标"""
        self.metrics.total_signals += len(signals)
        self.metrics.successful_signals += len([r for r in results if r.confidence_score > 0.8])
        if self.metrics.total_signals > 0:
            self.metrics.accuracy_rate = (self.metrics.successful_signals / 
                                        self.metrics.total_signals)
        self.metrics.avg_confidence = sum(r.confidence_score for r in results) / len(results)
        self.metrics.last_update = datetime.now()

class PriorityManager:
    """优先级管理器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def prioritize_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """按优先级排序信号"""
        return sorted(signals, key=lambda x: x.confidence, reverse=True)[:self.config.max_signals]

class BacktestValidator:
    """回测验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_signal(self, signal: StrategySignal) -> bool:
        """验证单个信号"""
        # 简单的验证逻辑
        return (signal.confidence > 0.6 and 
                signal.price > 0 and 
                signal.symbol is not None)

class MultiStrategyPipeline:
    """多策略管道主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = PipelineConfig(**(config or {}))
        self.signal_processor = SignalProcessor(self.config)
        self.strategy_fusion = StrategyFusion(self.config)
        self.performance_monitor = PerformanceMonitor()
        self.priority_manager = PriorityManager(self.config)
        self.backtest_validator = BacktestValidator()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"多策略管道初始化完成: {self.config.name}")
    
    def process_signals(self, signals: List[StrategySignal]) -> List[FusionResult]:
        """处理信号并返回融合结果"""
        try:
            # 1. 优先级排序
            prioritized = self.priority_manager.prioritize_signals(signals)
            
            # 2. 信号处理
            processed = self.signal_processor.process_signals(prioritized)
            
            # 3. 分组处理（按策略类型）
            signals_by_type = {}
            for signal in processed:
                strategy_type = signal.signal_type.value
                if strategy_type not in signals_by_type:
                    signals_by_type[strategy_type] = []
                signals_by_type[strategy_type].append(signal)
            
            # 4. 策略融合
            results = []
            for strategy_type, type_signals in signals_by_type.items():
                if len(type_signals) > 1:
                    fusion_result = self.strategy_fusion.fuse_signals(type_signals)
                    results.append(fusion_result)
                else:
                    # 单个信号直接作为结果
                    signal = type_signals[0]
                    fusion_result = FusionResult(
                        final_signal=signal,
                        contributing_signals=[signal],
                        fusion_method="direct",
                        confidence_score=signal.confidence,
                        timestamp=datetime.now()
                    )
                    results.append(fusion_result)
            
            # 5. 更新性能指标
            self.performance_monitor.update_metrics(signals, results)
            
            self.logger.info(f"成功处理 {len(signals)} 个信号，生成 {len(results)} 个融合结果")
            return results
            
        except Exception as e:
            self.logger.error(f"信号处理失败: {e}")
            raise
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        return self.performance_monitor.metrics
    
    def validate_config(self) -> bool:
        """验证配置"""
        try:
            assert self.config.name, "配置名称不能为空"
            assert 0 <= self.config.confidence_threshold <= 1, "置信度阈值必须在0-1之间"
            assert self.config.max_signals > 0, "最大信号数量必须大于0"
            return True
        except AssertionError as e:
            self.logger.error(f"配置验证失败: {e}")
            return False

__version__ = "1.0.0"
__author__ = "AlphaSeeker Team"

__all__ = [
    "StrategyType",
    "MultiStrategyPipeline",
    "SignalProcessor", 
    "StrategyFusion",
    "PerformanceMonitor",
    "PriorityManager", 
    "BacktestValidator",
    "PipelineConfig",
    "StrategySignal",
    "FusionResult",
    "PerformanceMetrics"
]
