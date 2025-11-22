#!/bin/bash

# AlphaSeeker 增强版 - 快速部署脚本
# 完整功能 + 真实数据源

echo "🚀 AlphaSeeker 2.0 增强版部署脚本"
echo "=================================="
echo ""

# 检查当前目录
if [ ! -f "alphaseeker_enhanced_real.py" ]; then
    echo "❌ 错误：请在 alphaseekerBotV1 目录中运行此脚本"
    exit 1
fi

# 停止现有服务
echo "🔄 停止现有服务..."
pkill -f "alphaseeker.*real" 2>/dev/null || true
pkill -f "main_integration" 2>/dev/null || true
sleep 2

# 检查Python依赖
echo "🔍 检查Python依赖..."
python3 -c "import aiohttp, uvicorn" 2>/dev/null || {
    echo "📦 安装必要依赖..."
    pip3 install aiohttp uvicorn fastapi
}

# 创建日志目录
mkdir -p logs

# 启动增强版服务
echo "🌟 启动 AlphaSeeker 2.0 增强版..."
echo "📊 真实数据源 + 完整功能"

nohup python3 alphaseeker_enhanced_real.py > logs/alphaseeker_enhanced.log 2>&1 &
ENHANCED_PID=$!

echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 增强版启动成功！"
    echo ""
    echo "🌐 访问地址:"
    echo "   主页: http://localhost:8000"
    echo "   市场扫描: http://localhost:8000/scan"
    echo "   性能统计: http://localhost:8000/performance"
    echo "   BTC分析: http://localhost:8000/analyze/BTCUSDT"
    echo "   ETH分析: http://localhost:8000/analyze/ETHUSDT"
    echo "   系统健康: http://localhost:8000/health"
    echo ""
    echo "🔍 API接口:"
    echo "   扫描API: http://localhost:8000/api/v1/scan/market"
    echo "   分析API: http://localhost:8000/api/v1/signal/analyze?symbol=BTCUSDT"
    echo "   性能API: http://localhost:8000/api/v1/performance"
    echo ""
    echo "📋 功能对比:"
    echo "   ✅ 真实市场数据源 (CoinGecko API)"
    echo "   ✅ 市场扫描页面"
    echo "   ✅ 币种详细分析页面"
    echo "   ✅ 系统性能统计"
    echo "   ✅ 完整API接口"
    echo "   ✅ 实时健康监控"
    echo ""
    echo "🆔 进程ID: $ENHANCED_PID"
    echo "📄 日志文件: logs/alphaseeker_enhanced.log"
    echo ""
    echo "💡 提示：您现在拥有了完整功能的AlphaSeeker系统！"
else
    echo "❌ 服务启动失败，检查日志: logs/alphaseeker_enhanced.log"
    exit 1
fi

# 显示实时日志（可选）
read -p "是否查看实时日志？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📄 实时日志 (按Ctrl+C退出):"
    tail -f logs/alphaseeker_enhanced.log
fi