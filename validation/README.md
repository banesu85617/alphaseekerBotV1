# 双重验证机制核心协调器

这是AlphaSeeker系统的核心验证模块，实现了"LightGBM快速筛选 + 本地LLM深度评估"的双重验证机制。

## 功能特性

### 🧠 双重验证架构
- **第一层**: LightGBM机器学习模型进行毫秒级快速筛选
- **第二层**: 本地LLM进行深度评估和参数建议
- **结果融合**: 智能评分算法综合两层验证结果

### ⚡ 高性能设计
- 异步处理架构，支持并发验证
- 智能缓存机制，提升响应速度
- 可配置的超时和重试策略
- 批量验证支持，提高处理效率

### 📊 全面监控
- 实时性能指标收集
- 详细的统计分析和趋势分析
- 健康状态监控和预警
- 可导出的性能报告

### 🔧 灵活配置
- 支持多种环境配置（开发/测试/生产）
- 可调整的验证参数和阈值
- 支持不同的融合策略
- 模块化的组件设计

## 核心组件

### 1. SignalValidationCoordinator
核心协调器，负责管理整个验证流程：
- 协调两层验证的执行
- 处理异步任务和并发控制
- 管理验证队列和优先级
- 集成监控和统计功能

### 2. LightGBMFilter
第一层验证器：
- 基于LightGBM模型的快速分类
- 支持多种特征预处理
- 可配置的概率阈值和门控条件
- 支持批量预测

### 3. LLMEvaluator
第二层验证器：
- 支持多种本地LLM提供商（Ollama、LM Studio、AnythingLLM）
- 结构化输出和结果验证
- 智能参数建议和风险评估
- 可配置的超时和重试机制

### 4. ValidationFusion
结果融合算法：
- 多种融合策略（等权重、自适应、性能基、置信度加权）
- 动态权重调整
- 风险回报比计算
- 综合评分算法

### 5. ValidationMonitor
性能监控器：
- 实时性能指标收集
- 聚合统计和分析
- 健康状态检查
- 预警和告警机制

## 快速开始

### 安装依赖
```bash
pip install aiohttp httpx numpy pandas lightgbm pyyaml
```

### 基础使用示例
```python
import asyncio
from validation import (
    SignalValidationCoordinator, 
    ValidationConfig, 
    ValidationRequest,
    ValidationPriority
)

async def main():
    # 1. 创建配置
    config = ValidationConfig.create_development_config()
    
    # 2. 创建协调器
    async with SignalValidationCoordinator(config) as coordinator:
        # 3. 创建验证请求
        request = ValidationRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=45000.0,
            features={
                'mid_price': 45000.0,
                'spread': 2.5,
                'volatility_60s': 0.025
            },
            indicators={
                'rsi': 45.2,
                'macd': -125.3,
                'adx': 28.5
            },
            risk_context={
                'volatility': 0.035,
                'var_95': 0.025
            },
            priority=ValidationPriority.MEDIUM
        )
        
        # 4. 执行验证
        result = await coordinator.validate_signal(request)
        
        # 5. 处理结果
        print(f"验证状态: {result.status.value}")
        print(f"综合评分: {result.combined_score:.3f}")
        
        if result.layer1_result:
            print(f"第一层: 标签={result.layer1_result.label}, 概率={result.layer1_result.probability:.3f}")
        
        if result.layer2_result:
            print(f"第二层: 方向={result.layer2_result.direction}, 置信度={result.layer2_result.confidence:.3f}")

asyncio.run(main())
```

### 批量验证示例
```python
async def batch_example():
    config = ValidationConfig.create_development_config()
    
    async with SignalValidationCoordinator(config) as coordinator:
        # 创建多个验证请求
        requests = []
        for symbol in ["BTCUSDT", "ETHUSDT", "ADAUSDT"]:
            request = ValidationRequest(
                symbol=symbol,
                timeframe="1h",
                current_price=current_prices[symbol],
                features=sample_features,
                indicators=sample_indicators,
                risk_context=sample_risk_context
            )
            requests.append(request)
        
        # 批量验证
        results = await coordinator.batch_validate(requests)
        
        # 处理结果
        for result in results:
            print(f"{result.symbol}: {result.status.value}, 评分={result.combined_score:.3f}")
```

### 性能监控示例
```python
async def monitoring_example():
    config = ValidationConfig.create_development_config()
    
    async with SignalValidationCoordinator(config) as coordinator:
        # 执行多个验证请求
        # ... (执行验证代码)
        
        # 获取性能摘要
        perf_summary = await coordinator.monitor.get_performance_summary()
        print(f"性能摘要: {perf_summary}")
        
        # 获取实时统计
        real_time = await coordinator.monitor.get_real_time_stats()
        print(f"实时统计: {real_time}")
        
        # 检查健康状态
        health = await coordinator.monitor.check_health_status()
        print(f"健康状态: {health}")

asyncio.run(monitoring_example())
```

## 配置管理

### 环境配置
```python
# 开发环境
dev_config = ValidationConfig.create_development_config()

# 生产环境  
prod_config = ValidationConfig.create_production_config()

# 测试环境
test_config = ValidationConfig.create_test_config()
```

### 自定义配置
```python
from validation import ValidationConfig, LightGBMConfig, LLMConfig, FusionConfig

# 创建自定义配置
config = ValidationConfig(
    max_concurrent_tasks=32,
    lgbm_config=LightGBMConfig(
        probability_threshold=0.7,
        batch_size=100
    ),
    llm_config=LLMConfig(
        provider=LLMProvider.OLLAMA,
        base_url="http://localhost:11434",
        model_name="llama2:13b"
    ),
    fusion_config=FusionConfig(
        strategy=FusionStrategy.ADAPTIVE_WEIGHT,
        risk_reward_threshold=1.2
    )
)
```

### 配置文件
```yaml
# config.yaml
lgbm_config:
  model_path: "models/lightgbm_model.txt"
  probability_threshold: 0.65
  confidence_threshold: 0.6

llm_config:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model_name: "llama2"
  
fusion_config:
  strategy: "equal_weight"
  layer1_weight: 0.3
  layer2_weight: 0.4

timeout_config:
  layer1_timeout: 2.0
  layer2_timeout: 5.0
```

```python
# 从配置文件加载
config = ValidationConfig.from_yaml("config.yaml")
```

## API参考

### SignalValidationCoordinator
主要的验证协调器类。

#### 主要方法
- `validate_signal(request)`: 执行单个信号验证
- `batch_validate(requests)`: 批量验证信号
- `get_performance_stats()`: 获取性能统计
- `shutdown()`: 关闭协调器

### ValidationRequest
验证请求数据类。

#### 主要属性
- `symbol`: 交易对符号
- `timeframe`: 时间周期
- `current_price`: 当前价格
- `features`: 特征数据
- `indicators`: 技术指标
- `risk_context`: 风险上下文
- `priority`: 优先级

### ValidationResult
验证结果数据类。

#### 主要属性
- `status`: 验证状态
- `layer1_result`: 第一层结果
- `layer2_result`: 第二层结果
- `combined_score`: 综合评分
- `risk_reward_ratio`: 风险回报比
- `total_processing_time`: 总处理时间

## 监控指标

### 性能指标
- 平均处理时间
- P50/P95/P99延迟
- 成功率/错误率
- 超时率

### 业务指标
- 验证通过率
- 综合评分分布
- 符号处理统计
- 状态分布

### 健康指标
- 服务可用性
- 组件健康状态
- 预警和告警

## 部署建议

### 开发环境
- 使用轻量级配置
- 禁用缓存
- 开启调试日志
- 较小的并发数

### 生产环境
- 高并发配置
- 启用缓存
- 完善监控
- 熔断和重试
- 资源限制

### 监控建议
- 设置性能阈值
- 配置告警规则
- 定期导出报告
- 跟踪趋势变化

## 故障排除

### 常见问题
1. **LLM连接失败**: 检查本地LLM服务状态
2. **LightGBM模型加载失败**: 验证模型文件路径
3. **超时错误**: 调整超时配置或检查系统资源
4. **内存不足**: 减少并发数或增加缓存TTL

### 日志分析
- 查看验证流程日志
- 监控性能指标
- 检查错误统计
- 分析失败原因

## 开发指南

### 添加新的LLM提供商
1. 继承`LLMEvaluator`类
2. 实现`provider_handler`方法
3. 添加到`LLMProvider`枚举
4. 更新配置验证

### 自定义融合策略
1. 继承`ValidationFusion`类
2. 实现融合算法
3. 添加到`FusionStrategy`枚举
4. 更新权重计算逻辑

### 扩展监控指标
1. 在`ValidationMonitor`中添加新指标
2. 更新性能摘要方法
3. 添加相应的告警阈值
4. 更新文档说明

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request。请确保：
1. 代码符合PEP8规范
2. 添加必要的测试用例
3. 更新相关文档
4. 通过所有CI检查

## 更新日志

### v1.0.0
- 初始版本发布
- 实现双重验证机制
- 支持多种LLM提供商
- 完整的监控和配置系统