#!/usr/bin/env python3
"""
双重验证机制测试脚本
验证核心功能的基本可用性
"""

import asyncio
import sys
import os

# 添加路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from validation import (
        SignalValidationCoordinator, 
        ValidationRequest,
        ValidationPriority,
        ValidationConfig,
        ValidationStatus,
        Layer1Result,
        Layer2Result
    )
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


async def test_basic_imports():
    """测试基础导入"""
    print("\n=== 测试基础导入 ===")
    
    try:
        # 测试配置创建
        config = ValidationConfig.create_development_config()
        print("✅ 配置创建成功")
        
        # 测试验证请求创建
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
        
        return True
        
    except Exception as e:
        print(f"❌ 基础导入测试失败: {e}")
        return False


async def test_coordinator_creation():
    """测试协调器创建"""
    print("\n=== 测试协调器创建 ===")
    
    try:
        config = ValidationConfig.create_development_config()
        
        # 测试协调器创建
        coordinator = SignalValidationCoordinator(config)
        print("✅ 协调器创建成功")
        
        # 测试初始化
        await coordinator.initialize()
        print("✅ 协调器初始化成功")
        
        # 测试关闭
        await coordinator.shutdown()
        print("✅ 协调器关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 协调器测试失败: {e}")
        return False


async def test_mock_validation():
    """测试模拟验证"""
    print("\n=== 测试模拟验证 ===")
    
    try:
        config = ValidationConfig.create_development_config()
        
        async with SignalValidationCoordinator(config) as coordinator:
            # 创建测试请求
            request = ValidationRequest(
                symbol="BTCUSDT",
                timeframe="1h",
                current_price=45000.0,
                features={
                    "mid_price": 45000.0,
                    "spread": 2.5,
                    "volatility_60s": 0.025,
                    "volume_1m": 1250.5
                },
                indicators={
                    "rsi": 45.2,
                    "macd": -125.3,
                    "adx": 28.5
                },
                risk_context={
                    "volatility": 0.035,
                    "var_95": 0.025
                },
                priority=ValidationPriority.MEDIUM
            )
            
            print("✅ 测试请求创建成功")
            
            # 执行验证（使用模拟实现）
            result = await coordinator.validate_signal(request)
            
            print(f"✅ 验证执行成功")
            print(f"   状态: {result.status.value}")
            print(f"   符号: {result.symbol}")
            print(f"   时间框架: {result.timeframe}")
            print(f"   处理时间: {result.total_processing_time:.3f}s")
            
            if result.layer1_result:
                print(f"   第一层: 标签={result.layer1_result.label}, "
                      f"概率={result.layer1_result.probability:.3f}")
            
            if result.layer2_result:
                print(f"   第二层: 方向={result.layer2_result.direction}, "
                      f"置信度={result.layer2_result.confidence:.3f}")
            
            return True
            
    except Exception as e:
        print(f"❌ 模拟验证测试失败: {e}")
        return False


async def test_performance_stats():
    """测试性能统计"""
    print("\n=== 测试性能统计 ===")
    
    try:
        config = ValidationConfig.create_development_config()
        
        async with SignalValidationCoordinator(config) as coordinator:
            # 获取初始统计
            stats = coordinator.get_performance_stats()
            print(f"✅ 初始性能统计: {stats}")
            
            # 检查监控器状态
            if hasattr(coordinator, 'monitor'):
                health = await coordinator.monitor.check_health_status()
                print(f"✅ 监控器健康状态: {health['status']}")
            
            return True
            
    except Exception as e:
        print(f"❌ 性能统计测试失败: {e}")
        return False


async def test_configuration():
    """测试配置功能"""
    print("\n=== 测试配置功能 ===")
    
    try:
        # 测试不同环境配置
        dev_config = ValidationConfig.create_development_config()
        test_config = ValidationConfig.create_test_config()
        
        print(f"✅ 开发环境配置: 并发数={dev_config.max_concurrent_tasks}")
        print(f"✅ 测试环境配置: 并发数={test_config.max_concurrent_tasks}")
        
        # 测试配置转换
        config_dict = dev_config.to_dict()
        new_config = ValidationConfig.from_dict(config_dict)
        print("✅ 配置序列化/反序列化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置功能测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始双重验证机制测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("基础导入", test_basic_imports),
        ("协调器创建", test_coordinator_creation),
        ("模拟验证", test_mock_validation),
        ("性能统计", test_performance_stats),
        ("配置功能", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 运行测试: {test_name}")
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️ 部分测试失败")
        return 1


if __name__ == "__main__":
    # 运行测试
    exit_code = asyncio.run(main())
    sys.exit(exit_code)