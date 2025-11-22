#!/bin/bash

# AlphaSeeker 2.0 完整功能版部署脚本
# =====================================

echo "🚀 AlphaSeeker 2.0 完整功能版部署"
echo "================================="

# 检查当前目录
if [ ! -f "main_integration.py" ]; then
    echo "❌ 错误: 请在 AlphaSeeker 项目根目录中运行此脚本"
    exit 1
fi

# 备份原文件
echo "📦 创建备份..."
if [ -f "main_integration.py.backup" ]; then
    echo "⚠️ 备份文件已存在，跳过备份创建"
else
    cp main_integration.py main_integration.py.backup
    echo "✅ 备份创建完成: main_integration.py.backup"
fi

# 部署增强版
echo "🔄 部署增强版系统..."
if [ -f "main_integration_enhanced.py" ]; then
    cp main_integration_enhanced.py main_integration.py
    echo "✅ 增强版系统已部署"
else
    echo "❌ 错误: 找不到 main_integration_enhanced.py 文件"
    exit 1
fi

# 设置执行权限
chmod +x main_integration.py

echo ""
echo "🎉 部署完成！"
echo "=================="
echo "📋 新功能列表:"
echo "  ✅ 完整功能导航界面"
echo "  ✅ 市场扫描页面 (/scan)"
echo "  ✅ 币种分析页面 (/analyze/{symbol})"
echo "  ✅ 性能统计页面 (/performance)"
echo "  ✅ 实时数据更新"
echo "  ✅ 现代化用户界面"
echo "  ✅ 响应式设计"
echo ""
echo "🚀 使用说明:"
echo "1. 启动系统: ./start.sh"
echo "2. 访问主页: http://0.0.0.0:8000"
echo "3. 体验新功能:"
echo "   - 点击导航菜单访问各个功能"
echo "   - 使用快速分析功能"
echo "   - 查看实时市场数据"
echo ""
echo "🔄 回滚方法:"
echo "如需回滚到原版本，运行:"
echo "  cp main_integration.py.backup main_integration.py"
echo ""
echo "🌟 现在您的 AlphaSeeker 系统具有完整的功能界面了！"
