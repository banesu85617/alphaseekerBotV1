#!/usr/bin/env python3
"""
双重验证机制基础测试
"""

import asyncio
import sys
import os
import traceback

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def simple_test():
    """简单测试函数"""
    try:
        print("开始简单测试...")
        
        # 直接导入类
        from validation.coordinator import SignalValidationCoordinator
        from validation.config import ValidationConfig
        from validation.coordinator import ValidationRequest, ValidationPriority
        
        print("✅ 类导入成功")
        
        # 创建配置
        config = ValidationConfig()
        print("✅ 配置创建成功")
        
        # 创建验证请求
        request = ValidationRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=45000.0,
            features={"feature1": 0.5},
            indicators={"rsi": 45.0},
            risk_context={"volatility": 0.02},
            priority=ValidationPriority.MEDIUM
        )
        print("✅ 验证请求创建成功")
        
        print("✅ 基础功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 简单测试失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    if success:
        print("🎉 基础测试通过！")
        sys.exit(0)
    else:
        print("💥 基础测试失败")
        sys.exit(1)