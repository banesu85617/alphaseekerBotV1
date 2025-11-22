# AlphaSeeker 部署指南

## 概述

本指南详细说明如何在不同环境中部署和运行AlphaSeeker集成系统。

## 系统要求

### 最低要求
- Python 3.8+
- 内存: 4GB RAM
- 存储: 10GB 可用空间
- 网络: 稳定的互联网连接

### 推荐配置
- Python 3.9+
- 内存: 8GB+ RAM
- 存储: 50GB+ SSD
- CPU: 4核心以上
- 网络: 低延迟连接

## 依赖安装

### 1. Python环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 升级pip
pip install --upgrade pip
```

### 2. 依赖包安装

```bash
# 安装基础依赖
pip install fastapi uvicorn pydantic

# 安装ML依赖
pip install lightgbm scikit-learn pandas numpy scipy joblib

# 安装API依赖
pip install aiohttp httpx asyncio-throttle

# 安装验证器依赖
pip install pyyaml

# 安装可选依赖（如果需要）
pip install redis postgresql psycopg2-binary
```

### 3. 系统依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel

# macOS (使用Homebrew)
brew install gcc
```

## 配置设置

### 1. 环境变量配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

主要配置项：
```bash
# 服务器配置
ALPHASEEKER_HOST=0.0.0.0
ALPHASEEKER_PORT=8000

# 性能配置
ALPHASEEKER_MAX_CONCURRENT_TASKS=32
ALPHASEEKER_BATCH_SIZE=100

# LLM配置
ALPHASEEKER_LLM_PROVIDER=ollama
ALPHASEEKER_LLM_BASE_URL=http://localhost:11434
ALPHASEEKER_LLM_MODEL_NAME=llama2:13b

# 日志配置
ALPHASEEKER_LOG_LEVEL=INFO
```

### 2. 配置文件

```bash
# 创建配置目录
mkdir -p config

# 编辑主配置文件
nano config/main_config.yaml
```

### 3. 目录结构

创建必要的目录：
```bash
mkdir -p data models logs cache config
mkdir -p data/market_data data/training data/backtest
mkdir -p models/lightgbm models/features models/risk
```

## 启动方式

### 1. 开发环境

```bash
# 直接启动（调试模式）
python main_integration.py

# 或使用uvicorn
uvicorn main_integration:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### 2. 生产环境

```bash
# 使用Gunicorn + Uvicorn
pip install gunicorn

# 启动多个worker
gunicorn main_integration:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --timeout 30 \
    --keep-alive 2

# 或使用systemd服务
sudo systemctl enable alphaseeker
sudo systemctl start alphaseeker
```

### 3. Docker部署

创建`Dockerfile`：
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建目录
RUN mkdir -p data models logs cache config

# 设置环境变量
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main_integration:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：
```bash
# 构建镜像
docker build -t alphaseeker:latest .

# 运行容器
docker run -d \
    --name alphaseeker \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    alphaseeker:latest
```

### 4. Docker Compose部署

创建`docker-compose.yml`：
```yaml
version: '3.8'

services:
  alphaseeker:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
      - ./.env:/app/.env
    environment:
      - ALPHASEEKER_HOST=0.0.0.0
      - ALPHASEEKER_PORT=8000
    restart: unless-stopped
    depends_on:
      - redis
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # 可选：Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - alphaseeker
    restart: unless-stopped

volumes:
  redis_data:
```

启动：
```bash
docker-compose up -d
```

## 系统服务配置

### 1. systemd服务（Linux）

创建服务文件：`/etc/systemd/system/alphaseeker.service`

```ini
[Unit]
Description=AlphaSeeker Trading System
After=network.target

[Service]
Type=simple
User=alphaseeker
Group=alphaseeker
WorkingDirectory=/opt/alphaseeker
Environment=PATH=/opt/alphaseeker/venv/bin
ExecStart=/opt/alphaseeker/venv/bin/python main_integration.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=alphaseeker

# 资源限制
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

启用和启动：
```bash
# 创建用户
sudo useradd -r -s /bin/false alphaseeker

# 设置权限
sudo chown -R alphaseeker:alphaseeker /opt/alphaseeker
sudo chmod +x /opt/alphaseeker/main_integration.py

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable alphaseeker
sudo systemctl start alphaseeker

# 查看状态
sudo systemctl status alphaseeker
sudo journalctl -u alphaseeker -f
```

## 监控和日志

### 1. 日志配置

确保日志目录权限：
```bash
sudo mkdir -p /var/log/alphaseeker
sudo chown alphaseeker:alphaseeker /var/log/alphaseeker
```

### 2. 监控脚本

创建监控脚本：`monitor.py`
```python
#!/usr/bin/env python3
import requests
import time
import json
from datetime import datetime

def check_health():
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    while True:
        is_healthy, data = check_health()
        timestamp = datetime.now().isoformat()
        
        if is_healthy:
            print(f"[{timestamp}] ✅ 系统健康")
        else:
            print(f"[{timestamp}] ❌ 系统异常: {data}")
            
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    main()
```

### 3. 性能监控

使用以下命令监控系统性能：
```bash
# CPU和内存使用
top -p $(pgrep -f main_integration)

# 网络连接
netstat -tlnp | grep :8000

# 磁盘空间
df -h /opt/alphaseeker

# 日志监控
tail -f /var/log/alphaseeker/alphaseeker.log
```

## 故障排除

### 1. 常见问题

#### 端口被占用
```bash
# 查找占用端口的进程
lsof -i :8000
# 或
netstat -tlnp | grep :8000

# 终止进程
kill -9 <PID>
```

#### 权限问题
```bash
# 设置正确的文件权限
sudo chown -R alphaseeker:alphaseeker /opt/alphaseeker
sudo chmod +x /opt/alphaseeker/main_integration.py
```

#### 依赖缺失
```bash
# 检查Python包
pip list | grep -E "(fastapi|uvicorn|lightgbm)"

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

#### 内存不足
```bash
# 检查内存使用
free -h
# 或
htop

# 调整配置
# 在.env中减少ALPHASEEKER_MAX_CONCURRENT_TASKS
```

### 2. 调试模式

启动调试模式：
```bash
# 开发模式
export ALPHASEEKER_DEBUG=true
export ALPHASEEKER_LOG_LEVEL=DEBUG
python main_integration.py

# 使用pdb调试
python -m pdb main_integration.py
```

### 3. 日志分析

查看错误日志：
```bash
# 系统日志
sudo journalctl -u alphaseeker -f

# 应用日志
tail -f logs/alphaseeker.log

# 搜索错误
grep ERROR logs/alphaseeker.log | tail -20
```

## 安全配置

### 1. 防火墙配置

```bash
# Ubuntu/Debian (使用ufw)
sudo ufw allow 8000
sudo ufw enable

# CentOS/RHEL (使用firewalld)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### 2. SSL/TLS配置

使用Nginx反向代理配置SSL：
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.pem;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. API访问控制

在`.env`中设置：
```bash
# 限制访问源
ALPHASEEKER_CORS_ALLOWED_ORIGINS=https://your-domain.com

# 启用速率限制
ALPHASEEKER_RATE_LIMITING_ENABLED=true
ALPHASEEKER_RATE_LIMITING_REQUESTS_PER_MINUTE=60
```

## 性能优化

### 1. 系统优化

```bash
# 增加文件描述符限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 优化网络设置
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
sysctl -p
```

### 2. 应用优化

在配置中调整：
```yaml
performance:
  max_concurrent_tasks: 64    # 根据CPU核心数调整
  batch_size: 200            # 增大批处理大小
  enable_cache: true         # 启用缓存
```

### 3. 缓存配置

Redis缓存（可选）：
```bash
# 安装Redis
sudo apt-get install redis-server

# 在.env中配置
ALPHASEEKER_REDIS_URL=redis://localhost:6379/0
```

## 备份和恢复

### 1. 数据备份

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/alphaseeker_$DATE"

mkdir -p $BACKUP_DIR

# 备份数据
cp -r data/ $BACKUP_DIR/
cp -r models/ $BACKUP_DIR/
cp -r config/ $BACKUP_DIR/
cp .env $BACKUP_DIR/

# 压缩
tar -czf alphaseeker_backup_$DATE.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: alphaseeker_backup_$DATE.tar.gz"
```

### 2. 自动化备份

添加到crontab：
```bash
# 每天凌晨2点备份
0 2 * * * /path/to/backup.sh
```

## 升级指南

### 1. 版本升级

```bash
# 停止服务
sudo systemctl stop alphaseeker

# 备份现有数据
cp -r /opt/alphaseeker /opt/alphaseeker.backup.$(date +%Y%m%d)

# 更新代码
git pull origin main

# 更新依赖
pip install -r requirements.txt --upgrade

# 启动服务
sudo systemctl start alphaseeker
```

### 2. 数据库迁移

如果使用数据库，运行迁移脚本：
```bash
python manage.py migrate
```

## 联系支持

如需技术支持，请：
1. 查看日志文件
2. 收集系统信息
3. 提交Issue到GitHub仓库

---

**注意**: 在生产环境中部署前，请务必进行充分测试。