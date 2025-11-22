#!/bin/bash

# AlphaSeeker-API 启动脚本

set -e

echo "🚀 启动 AlphaSeeker-API..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python版本过低，需要 $required_version 或更高版本，当前版本：$python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📥 安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 检查环境配置
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，从示例文件创建..."
    cp .env.example .env
    echo "📝 请编辑 .env 文件配置您的LLM服务"
fi

# 检查LLM配置
echo "🔍 检查LLM配置..."
if grep -q "LLM_PROVIDER=your_provider" .env 2>/dev/null; then
    echo "⚠️  请在 .env 文件中配置您的LLM提供商和模型"
fi

# 启动服务
echo "🚀 启动服务..."
python main.py