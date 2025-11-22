"""
双重验证机制核心模块

实现了LightGBM快速筛选 + 本地LLM深度验证的两层验证流程。
提供高性能、可配置、可监控的信号验证解决方案。
"""

from .coordinator import (
    SignalValidationCoordinator,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    ValidationPriority,
    Layer1Result,
    Layer2Result
)

from .lgbm_filter import LightGBMFilter, LightGBMConfig
from .llm_evaluator import LLMEvaluator, LLMConfig, LLMProvider
from .fusion_algorithm import ValidationFusion, FusionConfig, FusionStrategy
from .config import ValidationConfig, load_config
from .monitoring import ValidationMonitor, PerformanceMetrics, AggregatedMetrics
from .utils import (
    TimeoutManager, RetryManager, AsyncCache, RateLimiter,
    CircuitBreaker, ValidationUtils, BatchProcessor,
    TimeoutConfig, RetryConfig
)

__version__ = "1.0.0"
__author__ = "AlphaSeeker Team"

__all__ = [
    # 核心协调器
    "SignalValidationCoordinator",
    "ValidationRequest", 
    "ValidationResult",
    "ValidationStatus",
    "ValidationPriority",
    "Layer1Result",
    "Layer2Result",
    
    # LightGBM筛选器
    "LightGBMFilter",
    "LightGBMConfig",
    
    # LLM评估器
    "LLMEvaluator", 
    "LLMConfig",
    "LLMProvider",
    
    # 融合算法
    "ValidationFusion",
    "FusionConfig", 
    "FusionStrategy",
    
    # 配置管理
    "ValidationConfig",
    "load_config",
    
    # 性能监控
    "ValidationMonitor",
    "PerformanceMetrics",
    "AggregatedMetrics",
    
    # 工具类
    "TimeoutManager",
    "RetryManager", 
    "AsyncCache",
    "RateLimiter",
    "CircuitBreaker",
    "ValidationUtils",
    "BatchProcessor",
    "TimeoutConfig",
    "RetryConfig"
]

# 模块描述
__doc__ = """
双重验证机制核心模块

这个模块实现了AlphaSeeker系统的双重验证机制，包括：

1. SignalValidationCoordinator: 核心协调器，管理整个验证流程
2. LightGBMFilter: 第一层快速筛选器，基于机器学习模型
3. LLMEvaluator: 第二层深度评估器，使用本地LLM进行参数建议
4. ValidationFusion: 结果融合算法，综合评分和决策
5. ValidationMonitor: 性能监控器，实时跟踪验证性能
6. ValidationConfig: 配置管理，支持多种环境配置

主要特性：
- 高性能异步处理
- 配置化验证流程
- 实时性能监控
- 智能重试和错误处理
- 支持批量验证
- 完善的日志和统计

使用方法：
    import asyncio
    from validation import SignalValidationCoordinator, ValidationConfig, ValidationRequest
    
    async def main():
        # 创建配置
        config = ValidationConfig.create_development_config()
        
        # 创建协调器
        async with SignalValidationCoordinator(config) as coordinator:
            # 创建验证请求
            request = ValidationRequest(
                symbol="BTCUSDT",
                timeframe="1h", 
                current_price=45000.0,
                features={'feature1': 0.5},
                indicators={'rsi': 45.0},
                risk_context={'volatility': 0.02}
            )
            
            # 执行验证
            result = await coordinator.validate_signal(request)
            
            print(f"验证结果: {result.status.value}")
            print(f"综合评分: {result.combined_score:.3f}")
    
    asyncio.run(main())
"""