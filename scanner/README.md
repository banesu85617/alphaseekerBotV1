# AlphaSeeker 市场扫描和深度分析系统

一个高效、模块化的市场扫描和深度分析系统，支持数百交易对的并行处理。

## 🚀 主要特性

### 核心功能
- **并行市场扫描**: 支持数百交易对的并行处理，高效的异步架构
- **智能交易对筛选**: 多维度筛选算法，动态优先级排序
- **深度分析触发**: 基于评分的智能深度分析触发机制
- **实时市场监控**: 实时监控市场状况和系统性能
- **可配置策略**: 灵活的策略配置系统，支持自定义

### 技术架构
- **多级缓存系统**: 内存缓存 + Redis分布式缓存
- **性能监控**: 全面的性能指标监控和分析
- **警报系统**: 灵活的警报规则和多种通知方式
- **配置管理**: 完整的配置管理和预设方案

## 📋 目录结构

```
code/scanner/
├── __init__.py              # 主包初始化
├── demo.py                  # 演示脚本
├── core/                    # 核心模块
│   ├── __init__.py
│   └── market_scanner.py    # 主扫描器类
├── cache/                   # 缓存系统
│   ├── __init__.py
│   ├── memory_cache.py      # 内存缓存
│   └── redis_cache.py       # Redis缓存
├── strategies/              # 策略系统
│   ├── __init__.py
│   └── scan_strategies.py   # 扫描策略
├── monitoring/              # 监控系统
│   ├── __init__.py
│   ├── performance_monitor.py  # 性能监控
│   └── alert_manager.py     # 警报管理
├── utils/                   # 工具类
│   ├── __init__.py
│   ├── data_processor.py    # 数据处理
│   └── metrics_calculator.py # 指标计算
└── config/                  # 配置管理
    ├── __init__.py
    └── scanner_config.py    # 配置管理
```

## 🛠 安装和设置

### 环境要求
- Python 3.8+
- Redis (可选，用于分布式缓存)
- 所需依赖包:
```bash
pip install asyncio pandas numpy redis psutil aiohttp aiosmtplib
```

### 快速开始

```python
import asyncio
from scanner import create_scanner, PresetConfigs

async def main():
    # 使用预设配置创建扫描器
    config = PresetConfigs.balanced_config()
    scanner = create_scanner(config)
    
    # 执行市场扫描
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    report = await scanner.scan_markets(symbols)
    
    print(f"扫描完成: {report.total_symbols} 个交易对")
    print(f"最佳机会: {report.top_opportunities[0].symbol}")

asyncio.run(main())
```

## 📖 使用指南

### 1. 基本扫描

```python
from scanner import MarketScanner, ScanConfig

# 创建扫描配置
config = ScanConfig(
    max_tickers=100,
    batch_size=20,
    enable_deep_analysis=True,
    deep_analysis_threshold=0.8
)

# 创建扫描器
scanner = MarketScanner(config)

# 执行扫描
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
report = await scanner.scan_markets(symbols)
```

### 2. 缓存系统

```python
from scanner import MemoryCache, RedisCache

# 内存缓存
memory_cache = MemoryCache(default_ttl=300)
await memory_cache.set("key", "value")
result = memory_cache.get("key")

# Redis缓存 (需要Redis服务器)
redis_cache = RedisCache()
await redis_cache.set("key", "value")
result = await redis_cache.get("key")
```

### 3. 策略系统

```python
from scanner.strategies import StrategyFactory

# 创建优先级策略
volume_strategy = StrategyFactory.create_priority_strategy("volume")
filtered_data = volume_strategy.apply(market_data)

# 创建过滤策略
strict_filter = StrategyFactory.create_filter_strategy("strict")
filtered_data = strict_filter.apply(market_data)
```

### 4. 性能监控

```python
from scanner.monitoring import PerformanceMonitor

# 创建监控器
monitor = PerformanceMonitor()
await monitor.start_monitoring(interval=10.0)

# 获取统计信息
stats = monitor.get_statistics()
print(f"系统状态: {stats}")

# 停止监控
await monitor.stop_monitoring()
```

### 5. 警报系统

```python
from scanner.monitoring import AlertManager, create_opportunity_alert

# 创建警报管理器
alert_manager = AlertManager()

# 发送机会警报
alert = create_opportunity_alert("BTCUSDT", 0.92, "Strong signals")
await alert_manager.send_alert(alert)

# 获取警报统计
stats = alert_manager.get_statistics()
```

### 6. 配置管理

```python
from scanner.config import ConfigManager, PresetConfigs

# 加载配置
config_manager = ConfigManager("config.json")
config = config_manager.load_config()

# 使用预设配置
hf_config = PresetConfigs.high_frequency_config()
quality_config = PresetConfigs.quality_focused_config()

# 保存配置
config_manager.save_config(config, "my_config.json")
```

## ⚙️ 预设配置

### 高频扫描配置 (High Frequency)
- 适用于快速扫描和高频交易
- 更快的处理速度，但牺牲部分准确性
- 关闭深度分析以提高速度
- 严格的过滤条件

```python
config = PresetConfigs.high_frequency_config()
```

### 质量优先配置 (Quality Focused)
- 适用于寻找高质量机会
- 启用深度分析
- 严格的质量控制
- 更长的处理时间

```python
config = PresetConfigs.quality_focused_config()
```

### 平衡配置 (Balanced)
- 平衡速度和质量的默认配置
- 适中的参数设置
- 标准扫描流程

```python
config = PresetConfigs.balanced_config()
```

## 📊 性能指标

### 系统性能
- **吞吐量**: 支持100+ 交易对/秒
- **延迟**: 平均扫描延迟 < 10秒
- **内存使用**: < 1GB (标准配置)
- **CPU使用**: < 80% (标准配置)

### 评分算法
- **技术评分**: RSI、MACD、布林带等15+指标
- **情绪评分**: 基于价格变化和成交量趋势
- **流动性评分**: 成交量、买卖价差、订单簿深度
- **动量评分**: 动量、ROC、Williams %R等
- **风险评分**: 波动率、价格稳定性等

## 🔧 自定义配置

### 扫描配置
```python
scanner_config = {
    'max_tickers': 100,        # 最大交易对数量
    'batch_size': 20,          # 批处理大小
    'max_workers': 10,         # 最大工作线程
    'timeout': 30.0,           # 超时时间(秒)
    'enable_deep_analysis': True,     # 启用深度分析
    'deep_analysis_threshold': 0.7,   # 深度分析阈值
    'cache_ttl': 60            # 缓存TTL(秒)
}
```

### 策略配置
```python
strategy_config = {
    'priority_weights': {      # 优先级权重
        'volume': 0.3,
        'volatility': 0.2,
        'trend': 0.2,
        'liquidity': 0.15,
        'quality': 0.15
    },
    'filter_strategy': 'balanced',   # 过滤策略
    'priority_strategy': 'volume'    # 优先级策略
}
```

### 警报配置
```python
alert_config = {
    'enable_alerts': True,           # 启用警报
    'high_opportunity_threshold': 0.9,  # 高机会阈值
    'performance_alert_threshold': 30.0, # 性能警报阈值
    'error_rate_threshold': 0.1,     # 错误率阈值
    'alert_cooldown': 300            # 警报冷却时间(秒)
}
```

## 📈 监控和警报

### 性能监控
- 扫描持续时间和吞吐量
- 内存和CPU使用情况
- 缓存命中率
- 错误率和延迟

### 警报类型
- **机会警报**: 发现高价值交易机会
- **性能警报**: 系统性能异常
- **错误警报**: 系统错误和故障
- **资源警报**: 资源使用超标

### 警报处理方式
- **日志记录**: 写入系统日志
- **Webhook**: 发送到外部API
- **邮件通知**: 发送邮件给指定用户
- **控制台输出**: 实时控制台显示

## 🚀 演示程序

运行完整的演示程序来了解系统功能：

```bash
cd code/scanner
python demo.py
```

演示程序包含：
1. 基本扫描功能演示
2. 高级策略系统演示
3. 缓存系统演示
4. 性能监控演示
5. 警报系统演示
6. 深度分析演示
7. 性能优化演示
8. 配置管理演示
9. 实时监控演示

## 🔌 扩展接口

### 深度分析接口
```python
async def custom_deep_analysis(symbol: str, metadata: dict) -> dict:
    \"\"\"自定义深度分析函数\"\"\"
    # 实现您的深度分析逻辑
    return {
        'pattern': 'custom_pattern',
        'confidence': 0.85,
        'action': 'buy'
    }

# 设置回调
scanner.callbacks['deep_analysis_callback'] = custom_deep_analysis
```

### 自定义策略
```python
from scanner.strategies import BaseStrategy

class CustomStrategy(BaseStrategy):
    def apply(self, data):
        # 实现您的自定义策略
        return processed_data

# 注册策略
custom_strategy = CustomStrategy(config)
```

### 自定义警报规则
```python
from scanner.monitoring.alert_manager import AlertRule

custom_rule = AlertRule(
    id="custom_rule",
    name="Custom Rule",
    condition=lambda data: data.get('custom_metric', 0) > threshold
)

alert_manager.add_rule(custom_rule)
```

## 📝 日志和调试

### 日志配置
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scanner.log'),
        logging.StreamHandler()
    ]
)
```

### 调试模式
```python
config.debug = True
config.log_level = "DEBUG"
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

如果您遇到问题或有疑问，请：

1. 查看 [文档](https://github.com/your-repo/scanner/wiki)
2. 搜索 [Issues](https://github.com/your-repo/scanner/issues)
3. 创建新的 Issue
4. 联系维护团队

## 🎯 路线图

- [ ] 支持更多交易所API
- [ ] 增加机器学习预测模型
- [ ] 添加图形化配置界面
- [ ] 支持集群部署
- [ ] 增加更多的技术指标
- [ ] 实现自适应参数调整

## 🏆 致谢

感谢所有为这个项目做出贡献的开发者！

---

**AlphaSeeker Scanner** - 让市场扫描更智能、更高效！