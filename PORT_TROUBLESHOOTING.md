# AlphaSeeker 端口问题解决指南

## 🔧 常见端口问题

### 1. 端口8000被占用
**症状**: 系统启动时提示"端口8000已被占用"

**解决方案**:
```bash
# 方法1: 自动修复
python3 fix_port_issues.py

# 方法2: 手动查找并终止进程
lsof -ti:8000 | xargs kill -9

# 方法3: 更改端口
python3 main_integration.py --port 8001
```

### 2. LLM服务端口11434无法访问
**症状**: LLM评估功能无法工作

**解决方案**:
```bash
# 启动Ollama服务
ollama serve

# 或者启动指定模型
ollama run llama2:13b

# 检查服务状态
curl http://localhost:11434/api/tags
```

### 3. 防火墙或网络配置问题
**症状**: 无法访问Web界面

**解决方案**:
```bash
# 检查防火墙状态
sudo ufw status
# 允许端口8000
sudo ufw allow 8000
# 允许端口11434
sudo ufw allow 11434
```

## 🚀 快速启动

### 选项1: 使用诊断工具
```bash
# 运行端口诊断
python3 port_diagnostic.py

# 启动系统
python3 port_diagnostic.py start
```

### 选项2: 使用修复工具
```bash
# 修复端口问题
python3 fix_port_issues.py

# 启动系统
python3 main_integration.py
```

### 选项3: 使用启动脚本
```bash
# 运行启动脚本
bash start_alpha.sh
```

## 📋 检查清单

在启动前请确认:

- [ ] Python 3.8+ 已安装
- [ ] 所有依赖包已安装 (`pip install -r requirements.txt`)
- [ ] 端口8000和11434未被占用
- [ ] Ollama服务正在运行 (如果使用本地LLM)
- [ ] 防火墙允许相应端口

## 🌐 访问地址

启动成功后，您可以访问:

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **性能监控**: http://localhost:8000/performance

## 🛠️ 常见错误解决

### 错误1: "Address already in use"
```bash
# 查找占用端口的进程
sudo lsof -i :8000
# 终止进程
sudo kill -9 <PID>
```

### 错误2: "Permission denied"
```bash
# 使用sudo运行 (不推荐)
sudo python3 main_integration.py
# 或者更改端口
python3 main_integration.py --port 8001
```

### 错误3: "ModuleNotFoundError"
```bash
# 重新安装依赖
pip install -r requirements.txt
# 或者指定安装
pip install fastapi uvicorn lightgbm
```

## 📞 获取帮助

如果问题仍然存在，请运行诊断工具并提供输出:

```bash
python3 port_diagnostic.py > diagnostic_output.txt
```

然后将 `diagnostic_output.txt` 的内容提供给技术支持。