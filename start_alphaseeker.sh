#!/bin/bash

# AlphaSeeker 启动脚本
cd "$(dirname "$0")"

echo "🚀 启动AlphaSeeker系统..."

# 设置Python路径
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 加载环境变量
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 启动系统
python3 main_integration.py
