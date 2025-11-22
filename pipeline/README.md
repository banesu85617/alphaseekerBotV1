# 多策略信号处理管道 (Multi-Strategy Signal Processing Pipeline)

基于AlphaSeeker双重验证架构的统一信号处理管道，整合了技术指标、机器学习预测、风险建模和回测验证的多策略融合系统。

## 核心特性

### 🏗️ 架构特点
- **双重验证架构**: LightGBM快速筛选 + 本地LLM深度评估
- **多策略融合**: 技术指标 + ML预测 + 风险模型 + 回测参考
- **动态权重调整**: 基于历史性能的智能权重优化
- **冲突解决机制**: 自动检测和解决策略冲突
- **实时优先级排序**: 基于数据新鲜度、置信度、流动性等因素

### ⚡ 性能特点
- **高吞吐量**: 支持批量处理数百个交易对
- **低延迟**: 10秒内完成端到端信号处理
- **并发优化**: 智能任务调度和资源分配
- **缓存机制**: 多层缓存提升处理效率

### 📊 监控功能
- **实时性能监控**: 延迟、吞吐量、准确率等关键指标
- **智能告警系统**: 自动检测性能异常
- **策略贡献度分析**: 各策略效果的量化评估
- **回测验证**: 历史数据验证和参数优化

## 安装和设置

### 环境要求
- Python 3.8+
- 依赖包: pandas, numpy, asyncio, logging

### 基本设置
```python
from pipeline import MultiStrategyPipeline
from pipeline.types import PipelineConfig, StrategyType

# 创建配置
config = PipelineConfig(
    max_concurrent_tasks=16,
    timeout_seconds=10,
    ml_probability_threshold=0.65,
    llm_confidence_threshold=0.65,
    strategy_weights={
        StrategyType.TECHNICAL_INDICATOR: 0.4,
        StrategyType.ML_PREDICTION: 0.2,
        StrategyType.RISK_MODEL: 0.2,
        StrategyType.BACKTEST_REFERENCE: 0.2
    }
)

# 初始化管道
pipeline = MultiStrategyPipeline(config)
```

## 使用示例

### 1. 基本使用 - 单个符号处理

```python
import asyncio
from datetime import datetime
from pipeline.types import MarketData, TechnicalIndicators, MLPrediction

async def basic_example():
    # 启动管道
    await pipeline.start()
    
    try:
        # 准备数据
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            price=45000.0,
            volume=1000000.0,
            data_freshness=5.0
        )
        
        technical_indicators = TechnicalIndicators(
            rsi=65.5,
            macd=120.5,
            adx=28.3,
            sma_50=44500.0,
            sma_200=42000.0
        )
        
        ml_prediction = MLPrediction(
            label=1,  # 买入
            probability_scores={-1: 0.15, 0: 0.25, 1: 0.60},
            confidence=0.72,
            model_version="v1.2.3"
        )
        
        # 处理信号
        result = await pipeline.process_single_symbol(
            symbol="BTCUSDT",
            market_data=market_data,
            technical_indicators=technical_indicators,
            ml_prediction=ml_prediction
        )
        
        print(f"最终方向: {result.final_direction.value}")
        print(f"综合评分: {result.final_score:.3f}")
        print(f"置信度: {result.combined_confidence:.3f}")
        
    finally:
        await pipeline.stop()

# 运行示例
asyncio.run(basic_example())
```

### 2. 批量扫描 - 多个符号处理

```python
from pipeline.types import ScanRequest

async def batch_scan_example():
    await pipeline.start()
    
    try:
        # 准备批量数据
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
        symbol_data_map = {
            symbol: {
                "price": 40000 + hash(symbol) % 10000,
                "volume": 1000000,
                "data_freshness": 10,
                "technical_indicators": {
                    "rsi": 50 + hash(symbol) % 40,
                    "macd": 100 + hash(symbol) % 200,
                    "adx": 20 + hash(symbol) % 20
                },
                "ml_prediction": {
                    "label": hash(symbol) % 3 - 1,
                    "probability_scores": {-1: 0.2, 0: 0.3, 1: 0.5},
                    "confidence": 0.6,
                    "model_version": "v1.2.3"
                }
            }
            for symbol in symbols
        }
        
        # 创建扫描请求
        scan_request = ScanRequest(
            symbols=symbols,
            max_symbols=4,
            top_n=3,
            filters={
                "min_confidence": 0.6,
                "min_score": 0.3,
                "allowed_directions": ["long", "short"]
            }
        )
        
        # 执行批量扫描
        scan_result = await pipeline.batch_scan(scan_request, symbol_data_map)
        
        print(f"处理时间: {scan_result.processing_time:.2f}秒")
        print(f"有效结果: {len(scan_result.results)}")
        
        for i, result in enumerate(scan_result.results, 1):
            print(f"{i}. {result.symbol}: {result.final_direction.value} "
                  f"(评分: {result.final_score:.3f})")
        
    finally:
        await pipeline.stop()

asyncio.run(batch_scan_example())
```

### 3. 性能监控

```python
async def performance_monitoring_example():
    await pipeline.start()
    
    try:
        # 处理多个符号产生性能数据
        # ... 处理逻辑 ...
        
        # 获取性能指标
        metrics = await pipeline.get_performance_metrics()
        print(f"吞吐量: {metrics.throughput:.2f} 信号/秒")
        print(f"准确率: {metrics.accuracy_metrics.get('overall_accuracy', 0):.2%}")
        
        # 获取详细报告
        report = await pipeline.get_performance_report()
        print(f"系统健康: {report['summary']['system_health']}")
        print(f"活跃告警: {report['summary']['total_alerts']}")
        
        # 检查告警
        alerts = await pipeline.check_alerts()
        if alerts:
            print("检测到告警:")
            for alert in alerts:
                print(f"  - {alert['message']}")
        
    finally:
        await pipeline.stop()
```

### 4. 回测验证

```python
import pandas as pd
from pipeline.types import StrategySignal, StrategyType

async def backtest_validation_example():
    await pipeline.start()
    
    try:
        # 准备历史数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'open': 40000 + pd.Series(range(len(dates))) % 2000,
            'high': 41000 + pd.Series(range(len(dates))) % 2000,
            'low': 39000 + pd.Series(range(len(dates))) % 2000,
            'close': 40000 + pd.Series(range(len(dates))) % 2000,
            'volume': 1000000 + pd.Series(range(len(dates))) % 500000
        })
        
        # 创建策略信号
        signal = StrategySignal(
            strategy_type=StrategyType.TECHNICAL_INDICATOR,
            direction=SignalDirection.LONG,
            confidence=0.8,
            score=0.75,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            market_data=MarketData(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                price=45000.0,
                volume=1000000.0
            ),
            technical_indicators=TechnicalIndicators(
                rsi=35.0,  # 超卖
                macd=150.0,
                adx=25.0
            )
        )
        
        # 执行回测验证
        backtest_result = await pipeline.validate_signal_backtest(
            signal=signal,
            historical_data=historical_data,
            validation_period=timedelta(days=30)
        )
        
        print(f"回测评分: {backtest_result.score:.3f}")
        print(f"收益率: {backtest_result.total_return:.2%}")
        print(f"胜率: {backtest_result.win_rate:.2%}")
        print(f"夏普比率: {backtest_result.sharpe_ratio:.3f}")
        
    finally:
        await pipeline.stop()
```

## 配置选项

### 预设配置

系统提供了多种预设配置，适应不同使用场景：

```python
from pipeline.config_example import get_preset_config

# 默认配置
config = get_preset_config("default")

# 高性能配置（高吞吐量）
config = get_preset_config("high_performance")

# 高精度配置（低误报率）
config = get_preset_config("high_accuracy")

# 保守配置（低风险偏好）
config = get_preset_config("conservative")

# 激进配置（高收益追求）
config = get_preset_config("aggressive")
```

### 自定义配置

```python
from pipeline.types import PipelineConfig, StrategyType

config = PipelineConfig(
    # 基础性能
    max_concurrent_tasks=32,
    timeout_seconds=8,
    batch_size=200,
    
    # 模型阈值
    ml_probability_threshold=0.65,
    ml_confidence_threshold=0.6,
    llm_confidence_threshold=0.65,
    
    # 策略权重
    strategy_weights={
        StrategyType.TECHNICAL_INDICATOR: 0.35,
        StrategyType.ML_PREDICTION: 0.3,
        StrategyType.RISK_MODEL: 0.25,
        StrategyType.BACKTEST_REFERENCE: 0.1
    },
    
    # 风险控制
    min_risk_reward_ratio=1.0,
    max_position_size=0.1,
    max_leverage=10.0,
    
    # 处理限制
    max_symbols_per_scan=100,
    top_n_results=10,
    
    # 缓存设置
    cache_ttl={
        "indicators": 300,      # 5分钟
        "ml_predictions": 60,   # 1分钟
        "llm_assessments": 600, # 10分钟
        "backtest_results": 604800  # 7天
    }
)
```

## 数据结构

### 核心数据类型

#### MarketData (市场数据)
```python
market_data = MarketData(
    symbol="BTCUSDT",
    timestamp=datetime.now(),
    price=45000.0,
    volume=1000000.0,
    ohlcv=None,           # 可选：OHLCV DataFrame
    order_book=None,      # 可选：订单簿数据
    data_freshness=5.0    # 数据新鲜度（秒）
)
```

#### TechnicalIndicators (技术指标)
```python
indicators = TechnicalIndicators(
    rsi=65.5,
    macd=120.5,
    macd_signal=110.2,
    bollinger_upper=43200.0,
    bollinger_middle=42500.0,
    bollinger_lower=41800.0,
    adx=28.3,
    atr=850.0,
    sma_20=42100.0,
    sma_50=41900.0,
    sma_200=40500.0,
    ema_12=42650.0,
    ema_26=42380.0
)
```

#### MLPrediction (机器学习预测)
```python
ml_prediction = MLPrediction(
    label=1,  # -1: 卖出, 0: 持有, 1: 买入
    probability_scores={-1: 0.15, 0: 0.25, 1: 0.60},
    confidence=0.72,
    model_version="lightgbm_v2.1.0",
    prediction_time=datetime.now()
)
```

#### FusionResult (融合结果)
```python
result = FusionResult(
    symbol="BTCUSDT",
    final_direction=SignalDirection.LONG,
    final_score=0.825,
    combined_confidence=0.78,
    risk_reward_ratio=1.5,
    component_scores={
        StrategyType.TECHNICAL_INDICATOR: 0.35,
        StrategyType.ML_PREDICTION: 0.25,
        StrategyType.RISK_MODEL: 0.15,
        StrategyType.BACKTEST_REFERENCE: 0.10
    },
    confidence_breakdown={
        "technical_indicator": 0.82,
        "ml_prediction": 0.75,
        "risk_model": 0.68,
        "backtest_reference": 0.71
    },
    decision_reason=[
        "技术指标显示RSI超卖信号",
        "ML模型预测上涨概率60%",
        "风险回报比满足要求"
    ]
)
```

## 性能优化

### 1. 吞吐量优化
- 增加 `max_concurrent_tasks` 和 `batch_size`
- 降低置信度阈值提高召回率
- 优化缓存策略减少重复计算

### 2. 准确率优化
- 提高模型阈值减少误报
- 增加风险模型权重
- 强化数据质量检查

### 3. 延迟优化
- 使用高性能配置
- 减少批量处理大小
- 优化并发控制

### 4. 资源优化
- 监控内存使用
- 调整缓存大小
- 优化任务调度

## 监控和告警

### 关键指标
- **处理时延**: 各阶段延迟统计 (P95/P99)
- **吞吐量**: 信号处理速度 (信号/秒)
- **准确率**: 策略预测准确性
- **缓存命中率**: 缓存效率
- **错误率**: 各类错误统计

### 告警条件
- 总处理时间超过阈值
- 吞吐量低于预期
- 错误率过高
- 缓存命中率过低
- 准确率下降

### 告警级别
- **高**: 影响核心功能，需要立即处理
- **中**: 影响性能，建议尽快处理
- **低**: 轻微影响，可延后处理

## 最佳实践

### 1. 配置选择
- **高吞吐量场景**: 使用 `high_performance` 配置
- **高精度场景**: 使用 `high_accuracy` 配置
- **低风险场景**: 使用 `conservative` 配置
- **高收益场景**: 使用 `aggressive` 配置

### 2. 数据质量
- 确保OHLCV数据完整性
- 检查时间戳对齐
- 验证技术指标计算准确性
- 监控数据新鲜度

### 3. 性能调优
- 定期监控性能指标
- 根据实际场景调整权重
- 优化缓存策略
- 平衡准确率和吞吐量

### 4. 风险管理
- 设置合理的风险回报比
- 监控最大回撤
- 控制仓位大小
- 实施多层过滤

## 故障排除

### 常见问题

#### 1. 处理速度慢
- 检查 `max_concurrent_tasks` 设置
- 确认数据源响应速度
- 验证网络连接稳定性
- 监控系统资源使用

#### 2. 准确率低
- 检查模型训练数据质量
- 验证特征工程过程
- 确认数据时间对齐
- 调整融合权重

#### 3. 内存使用高
- 减少 `batch_size`
- 清理过期缓存
- 优化数据结构
- 监控内存泄漏

#### 4. 错误率高
- 检查数据源质量
- 验证输入数据格式
- 确认网络连接
- 监控外部服务状态

### 调试技巧
1. 启用详细日志记录
2. 分步验证数据流程
3. 监控各组件性能
4. 使用性能分析工具
5. 逐步优化参数

## 扩展和定制

### 添加新策略类型
1. 扩展 `StrategyType` 枚举
2. 在 `SignalProcessor` 中添加处理逻辑
3. 更新权重计算
4. 调整融合算法

### 集成外部服务
1. 实现数据接口适配器
2. 配置服务连接参数
3. 添加错误处理机制
4. 实施超时控制

### 自定义融合算法
1. 继承 `StrategyFusion` 类
2. 重写融合逻辑
3. 添加新的权重计算方法
4. 更新配置参数

## 版本历史

### v1.0.0
- 初始版本发布
- 基础管道功能
- 多策略融合
- 性能监控
- 回测验证

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 仓库
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱: [联系邮箱]
- GitHub Issues: [GitHub链接]
- 文档: [文档链接]

---

*此管道基于 AlphaSeeker 双重验证架构设计，旨在提供高效、准确、可扩展的多策略信号处理解决方案。*