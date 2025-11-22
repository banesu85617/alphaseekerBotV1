# AlphaSeeker-Bot机器学习引擎

基于AlphaSeeker-Bot项目分析文档提取并整合的独立机器学习模块，实现高性能加密资产交易信号生成和风险管理。

## 核心特性

### 🚀 高性能推理
- **目标延迟**: 0.5秒内完成模型推理
- **批量处理**: 支持批量预测提升吞吐
- **缓存优化**: 特征缓存和预测缓存机制
- **轻量化模型**: 针对推理速度优化的模型参数

### 🧠 智能特征工程
- **60+微结构特征**: 价差、订单不平衡、深度不平衡、WAP、波动率等
- **自动特征选择**: 基于因子评估的智能特征筛选
- **多时间框架**: 支持1分钟、5分钟、15分钟等时间框架
- **实时特征生成**: 支持实时数据流的特征计算

### 📊 因子评估系统
- **AAA-E分级**: 基于IC、分位数收益价差、Sharpe、Sortino、Calmar和p值的综合评分
- **IC分析**: 信息系数计算和滚动IC监控
- **显著性检验**: 统计显著性检验和Bonferroni校正
- **风险调整收益**: 多维度风险调整收益指标

### ⚖️ 风险管理体系
- **动态止损**: 基于波动率的动态止损机制
- **固定止盈止损**: 0.4%/0.4%固定风控参数
- **仓位管理**: 基于信号强度和风险预算的仓位计算
- **风险监控**: 实时风险指标监控和预警

### 🔧 模块化设计
- **核心模型**: LightGBM多分类模型
- **特征工程**: 可扩展的特征工程框架
- **训练流水线**: 端到端模型训练和验证
- **推理引擎**: 高性能推理和信号生成

## 快速开始

### 1. 安装依赖

```bash
pip install lightgbm scikit-learn pandas numpy scipy joblib
```

### 2. 基础使用

```python
from ml_engine import create_ml_engine

# 创建ML引擎实例
ml_engine = create_ml_engine(log_level="INFO")

# 健康检查
health = ml_engine.health_check()
print(f"引擎状态: {health['overall_status']}")
```

### 3. 模型训练

```python
# 准备训练数据 (CSV文件包含市场数据)
train_results = ml_engine.train_model(
    data="market_data.csv",
    price_col="close"
)

if train_results["success"]:
    print(f"模型训练完成: {train_results['model_path']}")
```

### 4. 模型推理

```python
# 加载模型
ml_engine.load_model("models/trading_model.joblib")

# 准备市场数据
market_data = {
    "bid_price": 50000,
    "ask_price": 50001,
    "bid_volume": 10,
    "ask_volume": 8,
    "close": 50000.5,
    "volume": 100,
    "timestamp": time.time()
}

# 预测交易信号
signal = ml_engine.predict(market_data)
print(f"交易信号: {signal['signal_label']}")
print(f"置信度: {signal['confidence']:.3f}")
print(f"推理延迟: {signal['latency_ms']:.2f}ms")
```

### 5. 风险管理

```python
# 执行风险管理
risk_result = ml_engine.manage_risk(
    market_data=market_data,
    signal=signal,
    account_balance=10000
)

print(f"风控动作: {risk_result['action']}")
print(f"风险等级: {risk_result.get('risk_level', 'N/A')}")
```

## API 参考

### AlphaSeekerMLEngine

主要的机器学习引擎类，提供所有ML功能的统一接口。

#### 主要方法

##### `train_model(data, price_col='close')`
训练机器学习模型。

**参数:**
- `data` (str|pd.DataFrame): 训练数据文件路径或DataFrame
- `price_col` (str): 价格列名，默认为'close'

**返回:**
- 训练结果字典，包含成功标志、模型路径和评估结果

##### `load_model(model_path)`
加载预训练的模型。

**参数:**
- `model_path` (str): 模型文件路径

**返回:**
- 加载成功标志

##### `predict(market_data, position='FLAT')`
预测交易信号。

**参数:**
- `market_data` (dict): 市场数据字典
- `position` (str): 当前仓位状态 ('FLAT', 'LONG', 'SHORT')

**返回:**
- 预测结果字典，包含信号、置信度、概率分布和延迟信息

##### `manage_risk(market_data, signal, account_balance)`
执行风险管理。

**参数:**
- `market_data` (dict): 市场数据
- `signal` (dict): 信号信息
- `account_balance` (float): 账户余额

**返回:**
- 风险管理结果，包含风控动作和风险状态

##### `get_performance_stats()`
获取性能统计信息。

**返回:**
- 性能统计字典，包含推理性能、模型信息和风险指标

##### `evaluate_factors(data)`
评估alpha因子。

**参数:**
- `data` (str|pd.DataFrame): 评估数据

**返回:**
- 因子评估结果，包含分级、报告和顶级因子列表

##### `health_check()`
系统健康检查。

**返回:**
- 健康状态字典，包含各组件状态

### 数据格式要求

#### 训练数据格式
```csv
timestamp,bid_price,ask_price,bid_volume,ask_volume,close,volume
2025-01-01 00:00:00,50000,50001,10,8,50000.5,100
2025-01-01 00:00:01,50001,50002,12,9,50001.5,120
...
```

#### 推理数据格式
```python
{
    "bid_price": 50000,      # 买一价格
    "ask_price": 50001,      # 卖一价格
    "bid_volume": 10,        # 买一量
    "ask_volume": 8,         # 卖一量
    "close": 50000.5,        # 最新价格
    "volume": 100,           # 成交量
    "timestamp": time.time() # 时间戳
}
```

## 配置选项

### 模型配置 (MODEL_CONFIG)
```python
MODEL_CONFIG = {
    "objective": "multiclass",     # 多分类任务
    "num_class": 3,               # 三分类: 买入/持有/卖出
    "random_state": 42,           # 随机种子
    "class_weight": "balanced",   # 类别平衡
    "num_leaves": 31,             # 叶子节点数
    "learning_rate": 0.1,         # 学习率
    "feature_fraction": 0.8,      # 特征采样比例
}
```

### 风控配置 (RISK_CONFIG)
```python
RISK_CONFIG = {
    "TAKE_PROFIT_PCT": 0.004,      # 止盈4%
    "STOP_LOSS_PCT": 0.004,        # 止损4%
    "TRANSACTION_COST_PCT": 0.0005, # 交易成本0.05%
    "MAX_POSITION_SIZE": 1.0,      # 最大仓位
    "MAX_DAILY_LOSS": 0.02,        # 最大日亏损2%
    "VOLATILITY_BASED_SL": True,   # 基于波动的动态止损
}
```

### 推理配置 (INFERENCE_CONFIG)
```python
INFERENCE_CONFIG = {
    "TARGET_LATENCY_MS": 500,     # 目标延迟0.5秒
    "ENABLE_CACHING": True,       # 启用缓存
    "BATCH_SIZE": 32,             # 批量大小
    "PRECISION": "float32",       # 推理精度
}
```

## 性能优化

### 推理速度优化
1. **特征缓存**: 启用特征缓存避免重复计算
2. **轻量化模型**: 使用较少的树和叶子节点
3. **批量推理**: 使用批量处理提升吞吐
4. **并行计算**: 充分利用多核CPU

### 内存优化
1. **数据流处理**: 避免一次性加载大量数据
2. **定期清理**: 清理过期的历史记录和缓存
3. **数据类型优化**: 使用合适的数据类型

## 监控和诊断

### 性能监控
```python
# 获取性能统计
stats = ml_engine.get_performance_stats()
print(f"平均延迟: {stats['inference_performance']['latency_stats']['mean_ms']:.2f}ms")
print(f"准确率: {stats['model_info']['accuracy']:.4f}")
```

### 健康检查
```python
# 系统健康检查
health = ml_engine.health_check()
print(f"总体状态: {health['overall_status']}")
for component, status in health['components'].items():
    print(f"{component}: {status['status']}")
```

### 模型评估
```python
# 因子评估
factor_results = ml_engine.evaluate_factors("market_data.csv")
print(f"评估因子数量: {len(factor_results['factor_results'])}")
print(f"顶级因子: {factor_results['top_factors'][:5]}")
```

## 扩展开发

### 自定义特征工程
```python
from ml_engine.features.feature_engineer import MicrostructureFeatureEngineer

# 创建自定义特征工程器
feature_engineer = MicrostructureFeatureEngineer()

# 添加自定义特征
def create_custom_features(df):
    df = df.copy()
    df['my_custom_feature'] = df['close'] * df['volume']
    return df

# 集成到训练流程
train_results = ml_engine.train_model("data.csv")
```

### 自定义风险策略
```python
from ml_engine.risk.manager import RiskManager

# 创建自定义风险管理器
risk_manager = RiskManager({
    "custom_stop_loss": 0.003,  # 3%自定义止损
    "dynamic_position_sizing": True
})

# 集成到引擎
ml_engine.risk_manager = risk_manager
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件格式完整

2. **推理延迟过高**
   - 启用特征缓存
   - 减少特征数量
   - 使用批量预测

3. **数据质量警告**
   - 检查数据源质量
   - 清理异常值和缺失值
   - 验证数据格式

4. **内存使用过高**
   - 清理历史记录
   - 调整缓存大小
   - 使用数据流处理

### 日志分析
```python
# 查看详细日志
import logging
logging.getLogger('alpha_seeker_ml').setLevel(logging.DEBUG)
```

## 版本信息

- **版本**: 1.0.0
- **基于**: AlphaSeeker-Bot项目分析
- **Python要求**: 3.8+
- **主要依赖**: LightGBM, scikit-learn, pandas, numpy

## 许可证

本项目基于原始AlphaSeeker-Bot项目的开源许可证。

## 支持

如有问题或建议，请提交issue或联系开发团队。