#!/usr/bin/env python3
"""
AlphaSeeker 快速启动指南
=====================

修复内容:
1. ✅ Lite版本市场扫描功能 - 修复异步处理错误
2. ✅ 完整版本配置错误 - 修复LLM_TIMEOUT浮点数解析

使用方法:
1. Lite版本: python3 alphaseeker_lite.py
2. 完整版本: ./start_alphaseeker.sh (修复配置后)
"""

def print_lite_guide():
    """打印Lite版本使用指南"""
    print("=" * 60)
    print("🚀 AlphaSeeker Lite 启动指南")
    print("=" * 60)
    print("1. 启动命令:")
    print("   cd /workspace/code")
    print("   python3 alphaseeker_lite.py")
    print()
    print("2. 访问地址:")
    print("   主页: http://localhost:8000")
    print("   市场扫描: http://localhost:8000/scan  (✅ 已修复)")
    print("   健康检查: http://localhost:8000/health")
    print("   性能统计: http://localhost:8000/performance")
    print("   分析示例: http://localhost:8000/analyze/BTCUSDT")
    print()

def print_full_guide():
    """打印完整版本使用指南"""
    print("=" * 60)
    print("🏗️ AlphaSeeker 完整版启动指南")
    print("=" * 60)
    print("修复内容:")
    print("✅ 修复了 LLM_TIMEOUT 配置解析错误")
    print("✅ 将 timeout 字段从 int 改为 float 类型")
    print()
    print("启动命令:")
    print("   cd /workspace/code")
    print("   ./start_alphaseeker.sh")
    print()
    print("⚠️ 注意: 如果仍遇到依赖问题，请先运行:")
    print("   python3 fix_environment.py")
    print()

def main():
    """主函数"""
    print("=" * 80)
    print("🎯 AlphaSeeker 启动指南")
    print("=" * 80)
    
    # 打印修复摘要
    print("📝 本次修复摘要:")
    print("1. ✅ Lite版本市场扫描功能 - 修复ERR_EMPTY_RESPONSE错误")
    print("2. ✅ 完整版本配置错误 - 修复LLM_TIMEOUT浮点数解析问题")
    print()
    
    # 打印Lite版本指南
    print_lite_guide()
    
    # 打印完整版本指南
    print_full_guide()
    
    print("=" * 80)
    print("💡 建议:")
    print("- 先测试Lite版本确认功能正常")
    print("- Lite版本无权限依赖，可立即使用")
    print("- 完整版本需安装依赖，适合生产环境")
    print("=" * 80)

if __name__ == "__main__":
    main()