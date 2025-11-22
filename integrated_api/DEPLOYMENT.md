# AlphaSeeker-API 部署指南

## 🚀 快速部署

### 前置要求
- Python 3.9+
- 网络连接（用于获取市场数据）
- 本地LLM服务（LM Studio/Ollama/AnythingLLM）

### 一键部署

```bash
# 1. 进入项目目录
cd code/integrated_api

# 2. 运行启动脚本
chmod +x start.sh
./start.sh

# 或者手动执行
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 🔧 配置指南

### 1. LLM服务配置

#### LM Studio (推荐)
```bash
# 下载并安装LM Studio
# 下载模型（如llama-3-8b-instruct）
# 启动本地服务器（默认端口1234）

# 配置.env
LLM_PROVIDER=lm_studio
LLM_BASE_URL=http://localhost:1234
LLM_MODEL_NAME=llama-3-8b-instruct
```

#### Ollama
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull llama3:8b

# 启动服务
ollama serve

# 配置.env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL_NAME=llama3:8b
```

#### AnythingLLM
```bash
# 下载AnythingLLM
# 启动服务（默认端口3001）

# 配置.env
LLM_PROVIDER=anything_llm
LLM_BASE_URL=http://localhost:3001
LLM_MODEL_NAME=your-configured-model
```

### 2. 性能调优

```bash
# 根据机器配置调整并发数
MAX_CONCURRENT_TASKS=8  # 4核CPU推荐
# MAX_CONCURRENT_TASKS=16  # 8核CPU推荐

# 启用批处理
BATCH_PROCESSING=true
BATCH_SIZE=10

# 内存优化
# 如有需要可调整缓存大小
```

### 3. 生产环境配置

```bash
# 关闭调试
DEBUG=false
ENVIRONMENT=production

# 生产端口
API_HOST=0.0.0.0
API_PORT=8000

# 日志级别
LOG_LEVEL=INFO

# 禁用热重载
API_RELOAD=false
```

## 🐳 Docker部署 (可选)

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

### 构建和运行
```bash
# 构建镜像
docker build -t alphaseeker-api .

# 运行容器
docker run -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  alphaseeker-api
```

## 📊 验证部署

### 1. 启动服务
```bash
python main.py
```

### 2. 运行测试
```bash
# 新终端窗口
python test_api.py
```

### 3. 检查接口
```bash
# 健康检查
curl http://localhost:8000/health

# 系统状态
curl http://localhost:8000/api/system/status

# LLM健康
curl http://localhost:8000/api/llm/health

# 获取交易对
curl http://localhost:8000/api/crypto/tickers
```

### 4. 测试分析
```bash
curl -X POST http://localhost:8000/api/crypto/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "lookback": 500
  }'
```

## 🔍 监控和维护

### 性能监控
```bash
# 查看性能指标
curl http://localhost:8000/api/system/performance

# 查看系统状态
curl http://localhost:8000/api/system/status
```

### 日志管理
```bash
# 查看实时日志
tail -f logs/api.log

# 查看错误日志
grep ERROR logs/api.log
```

### 资源监控
```bash
# 内存使用
ps aux | grep python

# CPU使用
top -p $(pgrep -f main.py)
```

## 🛠️ 故障排除

### 常见问题

#### 1. LLM连接失败
```bash
# 检查LLM服务是否运行
curl http://localhost:1234/v1/models

# 检查配置
echo $LLM_BASE_URL
echo $LLM_PROVIDER
echo $LLM_MODEL_NAME
```

#### 2. 数据获取失败
```bash
# 检查网络连接
ping api.binance.com

# 检查CCXT配置
# 查看日志中的CCXT相关错误
```

#### 3. 内存不足
```bash
# 减少并发任务数
export MAX_CONCURRENT_TASKS=4

# 减少批处理大小
export BATCH_SIZE=5

# 启用GC优化（自动）
# 系统会自动进行垃圾回收优化
```

#### 4. 端口被占用
```bash
# 查找占用端口的进程
lsof -i :8000

# 更改端口
export API_PORT=8001
```

### 调试模式
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 启用调试模式
export DEBUG=true

# 重新启动服务
python main.py
```

## 🔄 更新升级

### 依赖更新
```bash
# 激活虚拟环境
source venv/bin/activate

# 更新依赖
pip install --upgrade -r requirements.txt
```

### 配置迁移
```bash
# 备份现有配置
cp .env .env.backup

# 更新配置（如需要）
# 编辑.env文件
```

### 代码更新
```bash
# 拉取最新代码
git pull origin main

# 重启服务
python main.py
```

## 📋 检查清单

部署前检查：
- [ ] Python 3.9+ 已安装
- [ ] LLM服务已启动并可访问
- [ ] 网络连接正常
- [ ] .env配置文件已设置
- [ ] 端口8000未被占用

部署后验证：
- [ ] 服务正常启动
- [ ] 健康检查通过
- [ ] LLM健康检查通过
- [ ] 测试分析功能正常
- [ ] 监控接口可访问

## 🎯 性能建议

### 硬件要求
- **最低配置**: 4核CPU, 8GB RAM
- **推荐配置**: 8核CPU, 16GB RAM
- **高性能配置**: 16核CPU, 32GB RAM

### 网络要求
- 稳定的互联网连接
- 最低1Mbps带宽
- 低延迟（<100ms到交易所API）

### 存储要求
- 临时缓存：1GB
- 日志文件：500MB/天
- 可选历史数据：10GB+

## 📞 支持

如遇到部署问题：
1. 查看日志文件
2. 运行测试脚本
3. 检查系统状态
4. 参考故障排除指南

## 🔐 安全建议

1. **生产环境**：
   - 关闭调试模式
   - 使用HTTPS
   - 配置防火墙
   - 定期更新依赖

2. **LLM服务**：
   - 仅本地访问
   - 配置访问控制
   - 监控使用情况

3. **数据安全**：
   - 不在日志中记录敏感信息
   - 定期清理临时文件
   - 备份重要配置