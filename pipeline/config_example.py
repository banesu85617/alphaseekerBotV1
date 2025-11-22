"""
配置示例文件

展示如何使用多策略信号处理管道的各种配置选项。
"""

from .types import PipelineConfig, StrategyType
from datetime import timedelta

def get_default_config() -> PipelineConfig:
    """获取默认配置"""
    return PipelineConfig(
        max_concurrent_tasks=16,
        timeout_seconds=10,
        batch_size=100,
        ml_probability_threshold=0.65,
        ml_confidence_threshold=0.6,
        llm_confidence_threshold=0.65,
        llm_timeout=3.0,
        llm_provider="local",
        strategy_weights={
            StrategyType.TECHNICAL_INDICATOR: 0.4,
            StrategyType.ML_PREDICTION: 0.2,
            StrategyType.RISK_MODEL: 0.2,
            StrategyType.BACKTEST_REFERENCE: 0.2
        },
        min_risk_reward_ratio=1.0,
        max_position_size=0.1,
        max_leverage=10.0,
        cache_ttl={
            "indicators": 300,
            "ml_predictions": 60,
            "llm_assessments": 600,
            "backtest_results": 604800
        },
        min_backtest_score=0.60,
        require_sma_alignment=True,
        min_adx_threshold=20.0,
        max_symbols_per_scan=100,
        top_n_results=10
    )

def get_high_performance_config() -> PipelineConfig:
    """高性能配置 - 针对高吞吐量优化"""
    config = get_default_config()
    
    # 提升并发能力
    config.max_concurrent_tasks = 32
    config.batch_size = 200
    config.timeout_seconds = 8  # 更严格的超时
    
    # 降低置信度阈值，提高召回率
    config.ml_probability_threshold = 0.55
    config.ml_confidence_threshold = 0.5
    config.llm_confidence_threshold = 0.6
    
    # 提升缓存效率
    config.cache_ttl = {
        "indicators": 180,      # 3分钟
        "ml_predictions": 30,   # 30秒
        "llm_assessments": 300, # 5分钟
        "backtest_results": 86400  # 1天
    }
    
    # 扩展处理能力
    config.max_symbols_per_scan = 200
    config.top_n_results = 20
    
    return config

def get_high_accuracy_config() -> PipelineConfig:
    """高精度配置 - 针对高准确率优化"""
    config = get_default_config()
    
    # 降低并发，确保质量
    config.max_concurrent_tasks = 8
    config.batch_size = 50
    config.timeout_seconds = 15  # 更宽松的超时
    
    # 提高置信度阈值，降低误报率
    config.ml_probability_threshold = 0.75
    config.ml_confidence_threshold = 0.7
    config.llm_confidence_threshold = 0.75
    
    # 调整策略权重，突出ML预测
    config.strategy_weights = {
        StrategyType.TECHNICAL_INDICATOR: 0.25,
        StrategyType.ML_PREDICTION: 0.4,
        StrategyType.RISK_MODEL: 0.25,
        StrategyType.BACKTEST_REFERENCE: 0.1
    }
    
    # 更严格的过滤条件
    config.min_risk_reward_ratio = 1.5
    config.min_backtest_score = 0.75
    config.min_adx_threshold = 25.0
    
    return config

def get_conservative_config() -> PipelineConfig:
    """保守配置 - 针对低风险偏好"""
    config = get_default_config()
    
    # 减少并发和批量大小
    config.max_concurrent_tasks = 4
    config.batch_size = 25
    config.timeout_seconds = 20
    
    # 提高所有阈值
    config.ml_probability_threshold = 0.8
    config.ml_confidence_threshold = 0.75
    config.llm_confidence_threshold = 0.8
    
    # 风险优先的权重分配
    config.strategy_weights = {
        StrategyType.TECHNICAL_INDICATOR: 0.3,
        StrategyType.ML_PREDICTION: 0.15,
        StrategyType.RISK_MODEL: 0.45,  # 风险模型权重最高
        StrategyType.BACKTEST_REFERENCE: 0.1
    }
    
    # 严格的过滤条件
    config.min_risk_reward_ratio = 2.0
    config.max_position_size = 0.05  # 更小的仓位
    config.max_leverage = 5.0       # 更低的杠杆
    config.min_backtest_score = 0.8
    config.min_adx_threshold = 30.0
    
    # 减少处理数量
    config.max_symbols_per_scan = 30
    config.top_n_results = 5
    
    return config

def get_aggressive_config() -> PipelineConfig:
    """激进配置 - 针对高收益追求"""
    config = get_default_config()
    
    # 最大化并发和批量
    config.max_concurrent_tasks = 64
    config.batch_size = 500
    config.timeout_seconds = 6
    
    # 降低阈值，提高灵敏度
    config.ml_probability_threshold = 0.45
    config.ml_confidence_threshold = 0.4
    config.llm_confidence_threshold = 0.5
    
    # 技术分析和ML优先的权重
    config.strategy_weights = {
        StrategyType.TECHNICAL_INDICATOR: 0.5,
        StrategyType.ML_PREDICTION: 0.3,
        StrategyType.RISK_MODEL: 0.1,
        StrategyType.BACKTEST_REFERENCE: 0.1
    }
    
    # 宽松的过滤条件
    config.min_risk_reward_ratio = 0.8
    config.max_position_size = 0.2   # 更大的仓位
    config.max_leverage = 20.0      # 更高的杠杆
    config.min_backtest_score = 0.4
    config.min_adx_threshold = 15.0
    
    # 大量处理
    config.max_symbols_per_scan = 500
    config.top_n_results = 50
    
    return config

def get_custom_config(**kwargs) -> PipelineConfig:
    """获取自定义配置"""
    config = get_default_config()
    
    # 应用自定义参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"未知配置参数: {key}")
    
    return config

# 预定义配置常量
PRESET_CONFIGS = {
    "default": get_default_config,
    "high_performance": get_high_performance_config,
    "high_accuracy": get_high_accuracy_config,
    "conservative": get_conservative_config,
    "aggressive": get_aggressive_config
}

def get_preset_config(name: str) -> PipelineConfig:
    """获取预设配置"""
    if name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"未知预设配置: {name}. 可用配置: {available}")
    
    return PRESET_CONFIGS[name]()

def validate_config(config: PipelineConfig) -> bool:
    """验证配置有效性"""
    try:
        # 检查基础参数
        if config.max_concurrent_tasks <= 0:
            return False
        
        if config.timeout_seconds <= 0:
            return False
        
        if config.batch_size <= 0:
            return False
        
        # 检查阈值范围
        if not (0 <= config.ml_probability_threshold <= 1):
            return False
        
        if not (0 <= config.ml_confidence_threshold <= 1):
            return False
        
        if not (0 <= config.llm_confidence_threshold <= 1):
            return False
        
        if not (0 <= config.min_risk_reward_ratio <= 10):
            return False
        
        # 检查策略权重
        if not config.strategy_weights:
            return False
        
        weight_sum = sum(config.strategy_weights.values())
        if abs(weight_sum - 1.0) > 0.01:  # 允许小误差
            return False
        
        # 检查数值范围
        if not (0 < config.max_position_size <= 1):
            return False
        
        if config.max_leverage <= 0:
            return False
        
        if config.min_backtest_score < 0 or config.min_backtest_score > 1:
            return False
        
        if config.min_adx_threshold < 0:
            return False
        
        if config.max_symbols_per_scan <= 0:
            return False
        
        if config.top_n_results <= 0:
            return False
        
        # 检查缓存TTL
        for cache_type, ttl in config.cache_ttl.items():
            if ttl <= 0:
                return False
        
        return True
        
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False

def print_config_summary(config: PipelineConfig):
    """打印配置摘要"""
    print("\n=== 管道配置摘要 ===")
    print(f"并发任务数: {config.max_concurrent_tasks}")
    print(f"超时时间: {config.timeout_seconds}秒")
    print(f"批处理大小: {config.batch_size}")
    print(f"ML概率阈值: {config.ml_probability_threshold}")
    print(f"ML置信度阈值: {config.ml_confidence_threshold}")
    print(f"LLM置信度阈值: {config.llm_confidence_threshold}")
    print(f"最小风险回报比: {config.min_risk_reward_ratio}")
    print(f"最大仓位大小: {config.max_position_size}")
    print(f"最大杠杆: {config.max_leverage}")
    print(f"最小回测分数: {config.min_backtest_score}")
    print(f"最小ADX阈值: {config.min_adx_threshold}")
    print(f"最大扫描符号数: {config.max_symbols_per_scan}")
    print(f"返回结果数: {config.top_n_results}")
    
    print("\n策略权重:")
    for strategy_type, weight in config.strategy_weights.items():
        print(f"  {strategy_type.value}: {weight:.1%}")
    
    print("\n缓存TTL:")
    for cache_type, ttl in config.cache_ttl.items():
        print(f"  {cache_type}: {ttl}秒")
    
    print("=" * 50)