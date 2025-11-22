#!/bin/bash

# AlphaSeeker 启动脚本
# ====================

echo "🚀 AlphaSeeker 启动脚本"
echo "=========================="

# 设置项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 检查Python环境
echo "🔍 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python3"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"

# 检查依赖包
echo "📦 检查依赖包..."
MISSING_PACKAGES=""
for package in fastapi uvicorn lightgbm pandas numpy scikit-learn requests yaml ccxt; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $package"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "⚠️  缺少依赖包:$MISSING_PACKAGES"
    echo "正在安装..."
    pip install fastapi uvicorn lightgbm pandas numpy scikit-learn requests pyyaml ccxt
fi

echo "✅ 依赖包检查完成"

# 检查端口占用
echo "🔍 检查端口状态..."
if lsof -i :8000 &> /dev/null; then
    echo "⚠️  端口8000已被占用"
    echo "占用进程:"
    lsof -i :8000
    echo ""
    echo "要停止占用进程，请运行: lsof -ti:8000 | xargs kill -9"
    exit 1
fi

echo "✅ 端口8000可用"

# 检查LLM服务
echo "🤖 检查LLM服务..."
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "✅ Ollama服务正常运行"
    AVAILABLE_MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    echo "可用模型: $AVAILABLE_MODELS"
else
    echo "⚠️  Ollama服务未运行"
    echo "请先启动Ollama: ollama serve"
    echo "或者运行: ./start_ollama.sh"
fi

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p data logs models cache config

echo "🚀 启动AlphaSeeker系统..."
echo "=========================="
echo "Web界面: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo "健康检查: http://localhost:8000/health"
echo ""
echo "按 Ctrl+C 停止系统"
echo "=========================="

# 启动系统
python3 main_integration.py