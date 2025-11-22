#!/usr/bin/env python3
"""
双重验证机制快速演示
验证LightGBM + LLM双层验证流程
"""

import asyncio
import sys
import os
from validation.coordinator import SignalValidationCoordinator, ValidationRequest, ValidationPriority
from validation.config import ValidationConfig

async def quick_demo():
    """快速演示双重验证机制"""
    print("🚀 双重验证机制快速演示")
    print("=" * 50)
    
    # 创建开发环境配置
    config = ValidationConfig()
    
    # 创建验证请求
    request = ValidationRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        current_price=45000.0,
        features={
            'mid_price': 45000.0,
            'spread': 2.5,
            'volatility_60s': 0.025,
            'volume_1m': 1250.5
        },
        indicators={
            'rsi': 45.2,
            'macd': -125.3,
            'adx': 28.5
        },
        risk_context={
            'volatility': 0.035,
            'var_95': 0.025
        },
        priority=ValidationPriority.MEDIUM
    )
    
    # 创建协调器并执行验证
    coordinator = SignalValidationCoordinator(config)
    await coordinator.initialize()
    
    print(f"📊 验证信号: {request.symbol}")
    print(f"💰 价格: ${request.current_price:,.2f}")
    print("🔄 执行双重验证...")
    
    result = await coordinator.validate_signal(request)
    
    print(f"✅ 验证完成!")
    print(f"📋 状态: {result.status.value}")
    print(f"🎯 综合评分: {result.combined_score:.3f}")
    
    if result.layer1_result:
        print(f"🧠 第一层: 标签={result.layer1_result.label}, 概率={result.layer1_result.probability:.3f}")
    
    if result.layer2_result:
        print(f"🤖 第二层: 方向={result.layer2_result.direction}, 置信度={result.layer2_result.confidence:.3f}")
    
    await coordinator.shutdown()
    
    print("🎉 双重验证机制演示成功!")
    return result

if __name__ == "__main__":
    asyncio.run(quick_demo())