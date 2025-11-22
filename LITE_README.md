# AlphaSeeker Lite - 轻量级版本说明

## 🎯 概述

**AlphaSeeker Lite** 是专为**受限环境**设计的轻量级版本，不依赖外部包，完全使用Python标准库实现。

## 🔧 解决的问题

您的环境存在以下限制：
- ❌ **权限限制**: 无法使用sudo
- ❌ **包管理限制**: pip安装被阻止
- ❌ **网络限制**: 可能无法访问外部API

## ✅ AlphaSeeker Lite 的特性

### 🌟 核心功能
- ✅ **模拟数据生成**: 生成真实的市场数据和交易信号
- ✅ **双重验证机制**: LightGBM + LLM模拟评估
- ✅ **实时信号处理**: 基于技术指标的智能信号生成
- ✅ **Web界面**: 完整的HTML界面展示结果
- ✅ **API接口**: RESTful API供程序调用

### 📊 技术特性
- ✅ **零依赖**: 仅使用Python标准库
- ✅ **轻量级**: 启动快速，内存占用低
- ✅ **多线程**: 支持并发处理
- ✅ **模块化设计**: 易于扩展和维护

## 🚀 快速启动

### 方法1: 运行环境检查并启动
```bash
cd /workspace/code
python3 lite_check.py
```

### 方法2: 直接启动
```bash
cd /workspace/code
python3 alphaseeker_lite.py
```

## 🌐 访问地址

启动成功后，您可以访问：

| 功能 | 地址 | 说明 |
|------|------|------|
| 主页 | http://localhost:8000 | 系统概览和导航 |
| 健康检查 | http://localhost:8000/health | 系统状态JSON |
| 市场扫描 | http://localhost:8000/scan | 实时信号扫描 |
| 性能统计 | http://localhost:8000/performance | 系统性能数据 |
| 分析示例 | http://localhost:8000/analyze/BTCUSDT | BTCUSDT详细分析 |

## 📈 模拟功能说明

### 1. 市场数据模拟
- **交易对**: 10个主流加密货币
- **价格波动**: 模拟真实市场波动(±5%)
- **技术指标**: RSI, MACD, 布林带, 移动平均线等

### 2. 信号生成逻辑
- **LightGBM模拟**: 概率预测 + 置信度评估
- **LLM评估**: 情绪分析 + 推理说明
- **双重验证**: 结合两种方法的优势

### 3. 信号类型
- 🟢 **BUY**: 买入信号 (概率 > 0.7, RSI < 70)
- 🔴 **SELL**: 卖出信号 (概率 < 0.3, RSI > 70)  
- 🟡 **HOLD**: 持有信号 (中性情况)

## 🔍 示例输出

### 市场扫描结果
```
交易对    | 信号  | 价格    | 置信度 | 24h变化 | 推理
BTCUSDT   | BUY   | $45678.90| 78.5% | +2.3%  | 技术指标显示上升趋势
ETHUSDT   | HOLD  | $3045.67 | 45.2% | -0.8%  | 成交量放大确认趋势
ADAUSDT   | SELL  | $0.5234  | 72.1% | +1.2%  | RSI超买信号
```

### 双重验证结果
```json
{
  "symbol": "BTCUSDT",
  "dual_validation": {
    "lightgbm": {
      "prediction": "BUY",
      "probability": 0.78,
      "confidence": 0.85,
      "passed": true
    },
    "llm": {
      "sentiment": "positive",
      "reasoning": "技术指标显示上升趋势",
      "confidence": 0.82
    },
    "fusion": {
      "final_signal": "BUY",
      "confidence": 0.83,
      "recommendation": "BUY BTCUSDT (置信度: 83.0%)"
    }
  }
}
```

## 🛠️ 技术架构

```
AlphaSeeker Lite
├── MockDataGenerator     # 模拟数据生成器
├── AlphaSeekerLite       # 主系统逻辑
├── HTTPRequestHandler    # HTTP请求处理
├── ThreadingHTTPServer   # 多线程服务器
└── 核心组件
    ├── TradingSignal     # 交易信号结构
    ├── MarketData        # 市场数据结构
    └── 性能监控
```

## 📋 系统要求

- **Python**: 3.6+
- **操作系统**: 任意（Linux/Windows/macOS）
- **内存**: 50MB+
- **磁盘**: 10MB
- **网络**: 本地访问（无外网要求）

## 🔧 开发和扩展

### 添加新的技术指标
在 `MockDataGenerator.generate_market_data()` 方法中添加：
```python
indicators = {
    'new_indicator': random.uniform(0, 100),
    # ... 其他指标
}
```

### 修改信号生成逻辑
在 `MockDataGenerator.generate_trading_signal()` 方法中调整条件判断。

### 扩展API接口
在 `HTTPRequestHandler.do_GET()` 方法中添加新的路由。

## ⚠️ 注意事项

1. **这是模拟版本**: 不进行真实交易
2. **数据为模拟**: 所有价格和指标都是模拟数据
3. **仅用于测试**: 适合系统测试和功能演示
4. **扩展性强**: 可以轻松扩展为真实版本

## 🚀 下一步

如果您需要真实版本：
1. 解决环境权限问题
2. 安装必要的依赖包
3. 配置交易所API密钥
4. 启动完整版本：`python3 main_integration.py`

## 📞 技术支持

如果遇到问题：
1. 检查Python版本: `python3 --version`
2. 运行环境检查: `python3 lite_check.py check`
3. 查看日志输出获取详细信息

---

**AlphaSeeker Lite** 让您在任何环境下都能体验AI驱动的交易信号系统！