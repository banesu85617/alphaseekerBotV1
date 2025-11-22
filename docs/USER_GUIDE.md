# AlphaSeeker 使用指南

## 概述

本指南提供AlphaSeeker集成系统的完整使用说明，包括基本操作、高级功能、API接口和最佳实践。

## 快速开始

### 1. 启动系统

```bash
# 克隆项目
git clone <repository-url>
cd alphaseeker

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，设置必要参数

# 启动系统
python main_integration.py
```

系统将在 `http://localhost:8000` 启动，API文档可在 `http://localhost:8000/docs` 查看。

### 2. 验证安装

```bash
# 健康检查
curl http://localhost:8000/health

# 获取系统状态
curl http://localhost:8000/api/v1/system/status

# 获取组件信息
curl http://localhost:8000/api/v1/components
```

## API接口使用

### 1. 交易信号分析

#### 单个信号分析

```bash
curl -X POST "http://localhost:8000/api/v1/signal/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "market_data": {
      "price": 45000.0,
      "volume": 1000000.0,
      "timestamp": 1640995200
    },
    "indicators": {
      "rsi": 65.5,
      "macd": 120.5,
      "adx": 28.3,
      "sma_50": 44500.0,
      "sma_200": 42000.0
    },
    "features": {
      "mid_price": 45000.0,
      "spread": 2.5,
      "volatility_60s": 0.025,
      "bid_ask_ratio": 1.2
    }
  }'
```

#### Python客户端示例

```python
import asyncio
import aiohttp

async def analyze_signal(symbol, market_data, indicators, features):
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8000/api/v1/signal/analyze"
        payload = {
            "symbol": symbol,
            "market_data": market_data,
            "indicators": indicators,
            "features": features
        }
        
        async with session.post(url, json=payload) as response:
            return await response.json()

# 使用示例
market_data = {
    "price": 45000.0,
    "volume": 1000000.0,
    "timestamp": 1640995200
}

indicators = {
    "rsi": 65.5,
    "macd": 120.5,
    "adx": 28.3,
    "sma_50": 44500.0,
    "sma_200": 42000.0
}

features = {
    "mid_price": 45000.0,
    "spread": 2.5,
    "volatility_60s": 0.025
}

result = asyncio.run(analyze_signal("BTCUSDT", market_data, indicators, features))
print(f"信号方向: {result['signal_direction']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 2. 市场扫描

#### 批量市场扫描

```bash
curl -X POST "http://localhost:8000/api/v1/scan/market" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"],
    "max_results": 10
  }'
```

#### Python批量扫描示例

```python
import asyncio
import aiohttp
from typing import List, Dict

async def scan_market_batch(symbols: List[str], max_results: int = 10):
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8000/api/v1/scan/market"
        payload = {
            "symbols": symbols,
            "max_results": max_results
        }
        
        async with session.post(url, json=payload) as response:
            return await response.json()

# 使用示例
symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
result = asyncio.run(scan_market_batch(symbols, 5))

print(f"扫描结果: {len(result['results'])} 个机会")
print(f"平均置信度: {result['summary']['avg_confidence']:.3f}")

for opportunity in result['results']:
    print(f"{opportunity['symbol']}: {opportunity['signal_direction']} "
          f"(置信度: {opportunity['confidence']:.3f})")
```

### 3. 系统监控

#### 获取性能指标

```bash
curl http://localhost:8000/api/v1/performance
```

响应示例：
```json
{
  "performance": {
    "total_requests": 150,
    "successful_requests": 145,
    "failed_requests": 5,
    "success_rate": 96.67,
    "avg_processing_time": 0.245,
    "uptime": 3600.0
  },
  "system_info": {
    "uptime": 3600.0,
    "version": "1.0.0",
    "config": {
      "max_concurrent_tasks": 32,
      "batch_size": 100,
      "enable_cache": true
    }
  }
}
```

## 高级使用

### 1. 配置文件自定义

#### 修改主配置

编辑 `config/main_config.yaml`:

```yaml
# 提高并发处理能力
performance:
  max_concurrent_tasks: 64
  batch_size: 200

# 调整ML模型阈值
components:
  pipeline:
    ml_probability_threshold: 0.7      # 提高阈值
    llm_confidence_threshold: 0.7      # 提高阈值

# 配置LLM提供商
  validation:
    llm_config:
      provider: "lm_studio"            # 切换到LM Studio
      base_url: "http://localhost:1234"
      model_name: "local-llama-13b"
```

#### 环境变量配置

在 `.env` 文件中设置:

```bash
# 提高性能
ALPHASEEKER_MAX_CONCURRENT_TASKS=64
ALPHASEEKER_BATCH_SIZE=200

# 调整阈值
ALPHASEEKER_ML_PROBABILITY_THRESHOLD=0.7
ALPHASEEKER_LLM_CONFIDENCE_THRESHOLD=0.7

# 配置LLM
ALPHASEEKER_LLM_PROVIDER=lm_studio
ALPHASEEKER_LLM_BASE_URL=http://localhost:1234
ALPHASEEKER_LLM_MODEL_NAME=local-llama-13b

# 启用详细日志
ALPHASEEKER_LOG_LEVEL=DEBUG
```

### 2. 实时监控

#### 监控系统状态

```python
import asyncio
import aiohttp
import json
from datetime import datetime

async def monitor_system():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                # 获取系统状态
                async with session.get('http://localhost:8000/health') as response:
                    status = await response.json()
                    
                    # 获取性能指标
                    async with session.get('http://localhost:8000/api/v1/performance') as perf_response:
                        perf = await perf_response.json()
                        
                        # 打印状态
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] 状态: {status['status']}")
                        print(f"  请求总数: {perf['performance']['total_requests']}")
                        print(f"  成功率: {perf['performance']['success_rate']:.2f}%")
                        print(f"  平均处理时间: {perf['performance']['avg_processing_time']:.3f}s")
                        print(f"  系统运行时间: {perf['performance']['uptime']:.0f}s")
                        
                        # 检查告警
                        if perf['performance']['success_rate'] < 95:
                            print("⚠️  警告: 成功率低于95%")
                        
                        if perf['performance']['avg_processing_time'] > 5.0:
                            print("⚠️  警告: 平均处理时间过长")
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
        except Exception as e:
            print(f"监控错误: {e}")
            await asyncio.sleep(10)

# 启动监控
asyncio.run(monitor_system())
```

### 3. 批量处理

#### 大批量市场扫描

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

async def process_large_batch(symbols: list, batch_size: int = 50):
    """大批量市场扫描"""
    
    # 分批处理
    results = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}: {len(batch)} 个交易对")
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "symbols": batch,
                "max_results": 20
            }
            
            async with session.post('http://localhost:8000/api/v1/scan/market', json=payload) as response:
                batch_result = await response.json()
                results.extend(batch_result['results'])
        
        # 短暂延迟避免过载
        await asyncio.sleep(2)
    
    return results

# 使用示例
all_symbols = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
    "UNIUSDT", "AAVEUSDT", "COMPUSDT", "SUSHIUSDT", "YFIUSDT",
    # ... 更多交易对
]

results = asyncio.run(process_large_batch(all_symbols, batch_size=10))
print(f"总计找到 {len(results)} 个交易机会")

# 筛选高置信度机会
high_confidence = [r for r in results if r['confidence'] >= 0.8]
print(f"高置信度机会: {len(high_confidence)} 个")

for opportunity in sorted(high_confidence, key=lambda x: x['confidence'], reverse=True)[:10]:
    print(f"{opportunity['symbol']}: {opportunity['signal_direction']} "
          f"(置信度: {opportunity['confidence']:.3f})")
```

### 4. 自定义策略

#### 实现自定义ML模型

```python
from ml_engine import AlphaSeekerMLEngine
from pipeline.types import MarketData, MLPrediction
import pandas as pd

class CustomAlphaSeekerML(AlphaSeekerMLEngine):
    """自定义AlphaSeeker ML引擎"""
    
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.custom_features = [
            'volume_spike',
            'price_acceleration',
            'order_flow_imbalance'
        ]
    
    def create_custom_features(self, market_data: dict) -> dict:
        """创建自定义特征"""
        features = market_data.copy()
        
        # 成交量爆发
        if 'volume' in features and 'avg_volume' in features:
            features['volume_spike'] = features['volume'] / features['avg_volume']
        
        # 价格加速度
        if 'price_velocity' in features:
            features['price_acceleration'] = features['price_velocity'] - features['prev_velocity']
        
        # 订单流不平衡
        if 'bid_volume' in features and 'ask_volume' in features:
            total_volume = features['bid_volume'] + features['ask_volume']
            if total_volume > 0:
                features['order_flow_imbalance'] = (
                    features['bid_volume'] - features['ask_volume']
                ) / total_volume
        
        return features
    
    def predict(self, market_data: dict, position: str = 'FLAT'):
        """自定义预测逻辑"""
        # 添加自定义特征
        enhanced_data = self.create_custom_features(market_data)
        
        # 基础预测
        base_prediction = super().predict(enhanced_data, position)
        
        # 应用自定义逻辑
        if enhanced_data.get('volume_spike', 1) > 2.0:
            base_prediction['confidence'] *= 1.1  # 提高成交量爆发时的置信度
        
        if enhanced_data.get('order_flow_imbalance', 0) > 0.5:
            base_prediction['signal_label'] = 1  # 强烈买方不平衡时做多
        
        return base_prediction

# 使用自定义引擎
custom_ml = CustomAlphaSeekerML(config, logger)
result = custom_ml.predict(market_data)
```

### 5. 风险管理

#### 自定义风险管理

```python
from pipeline.risk.manager import RiskManager
from pipeline.types import MarketData

class AdvancedRiskManager(RiskManager):
    """高级风险管理"""
    
    def __init__(self, config):
        super().__init__(config)
        self.max_drawdown_threshold = 0.15  # 15%最大回撤
        self.var_confidence = 0.05  # 95% VaR
        self.correlation_limit = 0.7  # 相关性限制
    
    def calculate_position_size(self, signal: dict, account_balance: float, 
                              market_data: MarketData, portfolio_risk: dict) -> dict:
        """计算仓位大小"""
        
        # 基础仓位计算
        base_position = super().calculate_position_size(
            signal, account_balance, market_data, portfolio_risk
        )
        
        # VaR调整
        var_adjustment = self._calculate_var_adjustment(market_data)
        
        # 相关性调整
        correlation_adjustment = self._calculate_correlation_adjustment(
            market_data['symbol'], portfolio_risk
        )
        
        # 回撤检查
        drawdown_check = self._check_drawdown_limit(portfolio_risk)
        
        # 综合调整
        total_adjustment = var_adjustment * correlation_adjustment * drawdown_check
        
        adjusted_position = {
            **base_position,
            'size': base_position['size'] * total_adjustment,
            'adjustments': {
                'var': var_adjustment,
                'correlation': correlation_adjustment,
                'drawdown': drawdown_check
            }
        }
        
        return adjusted_position
    
    def _calculate_var_adjustment(self, market_data: MarketData) -> float:
        """计算VaR调整"""
        volatility = market_data.get('volatility', 0.02)
        var_multiplier = volatility / 0.02  # 基准波动率2%
        return max(0.1, 1 / var_multiplier)  # 波动率越高，仓位越小
    
    def _calculate_correlation_adjustment(self, symbol: str, portfolio_risk: dict) -> float:
        """计算相关性调整"""
        # 获取与现有持仓的相关性
        correlations = portfolio_risk.get('correlations', {})
        max_correlation = max(correlations.values()) if correlations else 0
        
        if max_correlation > self.correlation_limit:
            return 0.5  # 高相关性时降低仓位
        
        return 1.0
    
    def _check_drawdown_limit(self, portfolio_risk: dict) -> float:
        """检查回撤限制"""
        current_drawdown = portfolio_risk.get('current_drawdown', 0)
        
        if current_drawdown > self.max_drawdown_threshold:
            return 0.1  # 接近最大回撤时大幅降低仓位
        
        if current_drawdown > self.max_drawdown_threshold * 0.8:
            return 0.5
        
        return 1.0

# 使用高级风险管理
advanced_risk = AdvancedRiskManager(risk_config)
position = advanced_risk.calculate_position_size(signal, balance, market_data, portfolio_risk)
```

## 最佳实践

### 1. 性能优化

#### 批量处理建议

```python
# 1. 合理的批大小
# 避免过大的批次导致内存压力
OPTIMAL_BATCH_SIZE = 50

# 2. 并发控制
# 限制同时请求数量
MAX_CONCURRENT_REQUESTS = 10

# 3. 缓存策略
# 启用特征缓存减少重复计算
ENABLE_FEATURE_CACHE = True
CACHE_TTL = 300  # 5分钟

# 4. 监控性能
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            if processing_time > 5.0:
                logger.warning(f"慢查询: {func.__name__} 耗时 {processing_time:.2f}s")
            
            return result
        except Exception as e:
            logger.error(f"执行失败: {func.__name__} - {e}")
            raise
    return wrapper
```

### 2. 错误处理

```python
import asyncio
import aiohttp
from typing import Optional

async def robust_api_call(url: str, data: dict, retries: int = 3) -> Optional[dict]:
    """健壮的API调用"""
    
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status}: {await response.text()}")
        
        except asyncio.TimeoutError:
            logger.warning(f"超时，第{attempt + 1}次重试")
        except aiohttp.ClientError as e:
            logger.warning(f"客户端错误: {e}")
        except Exception as e:
            logger.error(f"未知错误: {e}")
        
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt)  # 指数退避
    
    return None

# 使用示例
result = await robust_api_call(
    "http://localhost:8000/api/v1/signal/analyze",
    signal_data
)

if result:
    print("分析成功:", result['signal_direction'])
else:
    print("分析失败，请检查系统状态")
```

### 3. 数据质量检查

```python
def validate_market_data(data: dict) -> bool:
    """验证市场数据质量"""
    
    required_fields = ['price', 'volume', 'timestamp']
    
    # 检查必需字段
    for field in required_fields:
        if field not in data:
            logger.error(f"缺少必需字段: {field}")
            return False
    
    # 检查数据有效性
    if data['price'] <= 0:
        logger.error("价格必须大于0")
        return False
    
    if data['volume'] <= 0:
        logger.error("成交量必须大于0")
        return False
    
    if data['timestamp'] > time.time() + 3600:  # 未来时间不超过1小时
        logger.warning("时间戳可能在未来")
    
    # 检查异常值
    if abs(data['price']) > 1000000:  # 超过100万的价格
        logger.warning("价格可能异常")
    
    return True

def validate_indicators(indicators: dict) -> bool:
    """验证技术指标"""
    
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        if not (0 <= rsi <= 100):
            logger.error(f"RSI值异常: {rsi}")
            return False
    
    if 'macd' in indicators:
        macd = indicators['macd']
        if abs(macd) > 10000:  # 异常大的MACD值
            logger.warning(f"MACD值可能异常: {macd}")
    
    return True

# 数据预处理示例
def preprocess_data(raw_data: dict) -> dict:
    """预处理数据"""
    
    processed = raw_data.copy()
    
    # 价格标准化
    if 'price' in processed:
        processed['price'] = float(processed['price'])
    
    # 时间戳标准化
    if 'timestamp' in processed:
        processed['timestamp'] = float(processed['timestamp'])
    
    # 填充缺失值
    if 'volume' not in processed or processed['volume'] <= 0:
        processed['volume'] = 1000000  # 默认成交量
    
    return processed
```

### 4. 日志记录

```python
import logging
import json
from datetime import datetime

def setup_structured_logging():
    """设置结构化日志"""
    
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # 添加异常信息
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            # 添加自定义字段
            if hasattr(record, 'custom_fields'):
                log_entry.update(record.custom_fields)
            
            return json.dumps(log_entry)
    
    # 创建格式化器
    formatter = StructuredFormatter()
    
    # 文件处理器
    file_handler = logging.FileHandler('logs/alphaseeker_structured.log')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 使用结构化日志
logger = setup_structured_logging()

def log_signal_analysis(symbol: str, result: dict, processing_time: float):
    """记录信号分析日志"""
    
    # 添加自定义字段
    extra = {
        'custom_fields': {
            'event_type': 'signal_analysis',
            'symbol': symbol,
            'signal_direction': result['signal_direction'],
            'confidence': result['confidence'],
            'processing_time': processing_time
        }
    }
    
    logger.info(f"信号分析完成: {symbol}", extra=extra)
```

## 故障排除

### 1. 常见问题

#### 系统启动失败

```bash
# 检查端口占用
lsof -i :8000

# 检查依赖
pip list | grep -E "(fastapi|uvicorn|lightgbm)"

# 检查配置
python -c "from main_integration import CONFIG; print(CONFIG)"

# 调试模式启动
export ALPHASEEKER_DEBUG=true
python main_integration.py
```

#### API调用失败

```python
import aiohttp
import asyncio

async def test_api_health():
    """测试API健康状态"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as response:
                if response.status == 200:
                    print("✅ API服务正常")
                    health_data = await response.json()
                    print(f"系统状态: {health_data['status']}")
                    print(f"组件数量: {len(health_data['components'])}")
                else:
                    print(f"❌ API服务异常: {response.status}")
                    print(f"响应: {await response.text()}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")

asyncio.run(test_api_health())
```

#### 性能问题

```python
import psutil
import asyncio

async def monitor_performance():
    """监控性能指标"""
    
    while True:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用
        memory = psutil.virtual_memory()
        
        # 磁盘I/O
        disk_io = psutil.disk_io_counters()
        
        print(f"CPU: {cpu_percent:.1f}% | "
              f"内存: {memory.percent:.1f}% | "
              f"磁盘I/O: {disk_io.read_bytes if disk_io else 0} bytes")
        
        if cpu_percent > 80:
            print("⚠️  CPU使用率过高")
        
        if memory.percent > 85:
            print("⚠️  内存使用率过高")
        
        await asyncio.sleep(10)

asyncio.run(monitor_performance())
```

### 2. 调试技巧

#### 启用详细日志

```python
import logging

# 启用DEBUG级别日志
logging.getLogger('alphaseeker').setLevel(logging.DEBUG)
logging.getLogger('ml_engine').setLevel(logging.DEBUG)
logging.getLogger('pipeline').setLevel(logging.DEBUG)
logging.getLogger('validation').setLevel(logging.DEBUG)

# 单独启用某个组件的日志
logger = logging.getLogger('pipeline.strategy_fusion')
logger.setLevel(logging.DEBUG)
```

#### 性能分析

```python
import cProfile
import pstats
import io
from functools import wraps

def profile_performance(func):
    """性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            pr.disable()
            
            # 输出性能统计
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            with open('performance_profile.txt', 'w') as f:
                f.write(s.getvalue())
            
            print("性能分析已保存到 performance_profile.txt")
    
    return wrapper

# 使用示例
@profile_performance
async def analyze_signal_heavy(symbol: str):
    # 复杂的信号分析逻辑
    pass
```

## 总结

本指南涵盖了AlphaSeeker系统的基本使用、高级功能、最佳实践和故障排除。通过合理使用这些功能和最佳实践，您可以：

1. **快速集成** - 使用REST API快速集成到现有系统
2. **优化性能** - 通过批量处理和并发控制提高效率
3. **确保质量** - 通过数据验证和错误处理保证结果可靠性
4. **便于维护** - 通过日志记录和监控及时发现和解决问题

更多详细信息请参考API文档和技术文档。

---

**注意**: 在生产环境中使用前，请务必进行充分测试和验证。