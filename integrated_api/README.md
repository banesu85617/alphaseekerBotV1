# AlphaSeeker-API 重构版本

🚀 **AI驱动的加密货币技术分析与市场扫描引擎** - 现已集成本地LLM支持

## ✨ 特性

### 🎯 核心功能
- **15+种技术指标计算**: RSI、MACD、布林带、ADX、CCI、OBV等
- **智能信号评估**: 基于技术指标的趋势判断
- **风险建模**: GARCH波动率预测、VaR计算
- **简化回测**: RSI策略历史回测
- **市场扫描**: 批量扫描和排名功能

### 🤖 本地LLM集成
- **统一接口**: 支持LM Studio、Ollama、AnythingLLM
- **OpenAI兼容**: 保持与原API的兼容性
- **流式响应**: 支持实时生成
- **自动重试**: 智能重试机制

### ⚡ 性能优化
- **异步处理**: 全异步架构
- **并发控制**: 可配置并发任务数
- **缓存机制**: 智能数据缓存
- **批处理**: 批量数据处理

### 🔧 配置管理
- **环境变量**: 灵活配置
- **类型安全**: Pydantic配置验证
- **热重载**: 开发环境支持

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env 文件配置您的LLM服务
```

### 3. 启动服务

```bash
python main.py
```

## 🔧 LLM配置

### LM Studio配置

1. 下载并启动LM Studio
2. 下载LLM模型（如llama-3-8b-instruct）
3. 启动本地服务器（默认端口1234）
4. 配置环境变量：

```bash
LLM_PROVIDER=lm_studio
LLM_BASE_URL=http://localhost:1234
LLM_MODEL_NAME=llama-3-8b-instruct
```

### Ollama配置

1. 安装Ollama: https://ollama.ai/
2. 下载模型：
   ```bash
   ollama pull llama3:8b
   ```
3. 启动服务（默认端口11434）
4. 配置环境变量：
   ```bash
   LLM_PROVIDER=ollama
   LLM_MODEL_NAME=llama3:8b
   ```

### AnythingLLM配置

1. 下载AnythingLLM
2. 启动本地实例
3. 配置环境变量：
   ```bash
   LLM_PROVIDER=anything_llm
   LLM_BASE_URL=http://localhost:3001
   LLM_MODEL_NAME=your-configured-model
   ```

## 📊 API接口

### 获取交易对列表
```http
GET /api/crypto/tickers
```

### 分析单个交易对
```http
POST /api/crypto/analyze
Content-Type: application/json

{
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "lookback": 1000,
    "accountBalance": 1000.0,
    "maxLeverage": 10.0
}
```

### 市场扫描
```http
POST /api/crypto/scan
Content-Type: application/json

{
    "timeframe": "1m",
    "max_tickers": 50,
    "top_n": 10,
    "min_gpt_confidence": 0.65,
    "min_backtest_score": 0.60,
    "max_concurrent_tasks": 16
}
```

### 系统状态
```http
GET /health
GET /api/system/status
GET /api/llm/health
```

## 🎛️ 配置选项

### 应用配置
- `ENVIRONMENT`: 运行环境 (development/production)
- `DEBUG`: 调试模式
- `API_HOST`: API主机
- `API_PORT`: API端口
- `API_RELOAD`: 热重载

### LLM配置
- `LLM_PROVIDER`: LLM提供商 (lm_studio/ollama/anything_llm/openai)
- `LLM_BASE_URL`: LLM服务器地址
- `LLM_MODEL_NAME`: 模型名称
- `LLM_MAX_TOKENS`: 最大令牌数
- `LLM_TEMPERATURE`: 温度参数

### 性能配置
- `MAX_CONCURRENT_TASKS`: 最大并发任务数
- `BATCH_PROCESSING`: 启用批处理
- `BATCH_SIZE`: 批处理大小

## 🏗️ 项目结构

```
integrated_api/
├── config/          # 配置管理
│   ├── settings.py
│   └── llm_config.py
├── core/           # 核心模块
│   ├── llm_interface.py
│   ├── data_fetcher.py
│   ├── indicators.py
│   ├── models.py
│   └── exceptions.py
├── services/       # 服务层
│   ├── llm_service.py
│   ├── analysis_service.py
│   └── scanner_service.py
├── utils/          # 工具模块
│   ├── validation.py
│   └── performance.py
├── main.py         # 主应用
└── requirements.txt
```

## 🔄 与原版对比

| 功能 | 原版 | 重构版 |
|------|------|--------|
| 技术指标 | ✅ 15+种 | ✅ 保持所有 |
| LLM服务 | OpenAI GPT-4o | 本地LLM + OpenAI兼容 |
| 架构 | 单文件 | 模块化 |
| 性能 | 基础 | 优化 |
| 配置 | 硬编码 | 环境变量 |
| 错误处理 | 基础 | 完善 |
| 文档 | 基础 | 完整 |

## 🚨 注意事项

1. **LLM服务**: 确保您的LLM服务正在运行
2. **网络连接**: 需要互联网连接获取市场数据
3. **内存使用**: 大批量扫描可能消耗较多内存
4. **速率限制**: CCXT有内置速率限制

## 📝 开发指南

### 添加新的LLM提供商

1. 在 `core/llm_interface.py` 中继承 `BaseLLMClient`
2. 实现必要的接口方法
3. 在 `LLMInterface._init_client` 中添加实例化逻辑

### 添加新的技术指标

1. 在 `core/indicators.py` 中添加计算方法
2. 在 `apply_all_indicators` 中调用
3. 在 `get_latest_indicators` 中添加字段

### 自定义过滤条件

1. 在 `services/scanner_service.py` 中修改 `_passes_filters` 方法
2. 在模型定义中添加相应字段

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。

## ⚠️ 免责声明

本软件仅供教育和研究目的，不构成投资建议。使用前请充分了解相关风险。