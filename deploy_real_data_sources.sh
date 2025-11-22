#!/bin/bash

# AlphaSeeker 2.0 真实数据源部署脚本
# ================================

echo "🚀 AlphaSeeker 2.0 真实数据源部署开始..."
echo "================================================"

# 检查当前目录
if [ ! -f "main_integration_enhanced.py" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    echo "📍 当前目录: $(pwd)"
    echo "📋 确保 main_integration_enhanced.py 存在"
    exit 1
fi

# 备份原文件
echo "📦 备份原文件..."
if [ -f "main_integration.py" ]; then
    cp main_integration.py main_integration_backup_$(date +%Y%m%d_%H%M%S).py
    echo "✅ 原 main_integration.py 已备份"
fi

# 部署新版本
echo "🔄 部署新版本..."
if [ -f "main_integration.py" ]; then
    mv main_integration.py main_integration_original.py
fi

cp main_integration_enhanced.py main_integration.py
echo "✅ 新版本部署完成"

# 检查依赖
echo "🔍 检查依赖..."
python3 -c "import ccxt, aiohttp" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ 核心依赖已安装"
else
    echo "⚠️ 安装缺失依赖..."
    pip3 install ccxt aiohttp
fi

# 创建数据源目录
echo "📁 创建数据源目录..."
mkdir -p data_sources
echo "✅ 数据源目录已创建"

# 测试部署
echo "🧪 测试部署..."
echo "启动AlphaSeeker 2.0 ..."

# 在后台启动服务
python3 main_integration.py &
SERVER_PID=$!

echo "⏳ 等待服务启动..."
sleep 5

# 测试健康状态
echo "🔍 测试服务状态..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ 服务启动成功！"
    echo ""
    echo "🎉 AlphaSeeker 2.0 真实数据源部署完成！"
    echo ""
    echo "📍 访问地址: http://localhost:8000"
    echo "🔧 主要功能:"
    echo "   • 真实市场数据 (Binance → OKX → CoinGecko)"
    echo "   • 智能数据源切换"
    echo "   • 新币自动发现"
    echo "   • 保持原有UI不变"
    echo ""
    echo "💡 使用说明:"
    echo "   • 访问主页查看系统状态"
    echo "   • 进入扫描页面查看真实市场数据"
    echo "   • 使用分析功能查看具体币种详情"
    echo ""
    echo "⚡ 智能切换逻辑:"
    echo "   1. 优先使用 Binance 数据"
    echo "   2. 如果Binance无该币种，使用 OKX"
    echo "   3. 如果都失败，使用 CoinGecko 补充"
    echo "   4. 所有源都失败时，使用高质量备用数据"
    echo ""
    
    # 停止测试服务器
    kill $SERVER_PID 2>/dev/null
    
    echo "🗑️ 启动服务器:"
    echo "   ./start.sh"
    echo ""
    echo "🔧 或者手动启动:"
    echo "   python3 main_integration.py"
    
else
    echo "❌ 服务启动失败"
    echo "🔍 查看错误日志..."
    
    # 停止失败的服务器
    kill $SERVER_PID 2>/dev/null
    
    echo "💡 故障排除建议:"
    echo "   1. 检查端口8000是否被占用"
    echo "   2. 检查网络连接"
    echo "   3. 查看详细错误: python3 main_integration.py"
    exit 1
fi

echo ""
echo "✨ 部署完成！享受真实的市场数据体验！"