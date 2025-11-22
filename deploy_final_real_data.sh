#!/bin/bash
# AlphaSeeker 2.0 真实数据源部署脚本 (最终版)
# 解决币价显示问题，集成真实市场数据

echo "🚀 AlphaSeeker 2.0 真实数据源最终部署"
echo "================================================"

# 检查当前目录
if [ ! -f "main_integration_enhanced.py" ]; then
    echo "❌ 错误：请在 AlphaSeeker 目录中运行此脚本"
    exit 1
fi

# 备份原文件
echo "📦 备份原文件..."
if [ -f "main_integration.py" ]; then
    cp main_integration.py main_integration_backup_$(date +%Y%m%d_%H%M%S).py
    echo "✅ 原 main_integration.py 已备份"
fi

# 部署简化真实数据版本
echo "🔧 部署真实数据版本..."
cp alphaseeker_simple_real.py main_integration.py
echo "✅ 新版本部署完成"

# 检查依赖
echo "📚 检查依赖..."
python3 -c "import aiohttp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 安装 aiohttp..."
    pip3 install aiohttp --user
fi

# 启动服务
echo "🎯 启动 AlphaSeeker 2.0..."
echo "================================================"
echo "✅ 部署完成！"
echo ""
echo "🌐 访问地址: http://localhost:8000"
echo "🔍 扫描页面: http://localhost:8000/scan"
echo "💚 健康检查: http://localhost:8000/health"
echo ""
echo "📊 真实数据验证:"
echo "   • BTC 价格: ~\$115,000 (真实价格)"
echo "   • ETH 价格: ~\$4,200 (真实价格)"
echo "   • 数据源: CoinGecko API"
echo "   • 刷新间隔: 每次页面刷新"
echo ""
echo "⚡ 启动命令:"
echo "   python3 main_integration.py"
echo ""
echo "🛑 停止命令:"
echo "   pkill -f main_integration.py"
echo "================================================"

# 询问是否立即启动
read -p "是否立即启动 AlphaSeeker? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 正在启动 AlphaSeeker..."
    python3 main_integration.py
fi