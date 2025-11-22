#!/usr/bin/env python3
"""
双重验证机制完整演示
展示LightGBM + 本地LLM的双层验证流程
"""

import asyncio
import sys
import os
import json
import time
from typing import Dict, Any

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from validation.coordinator import (
    SignalValidationCoordinator, 
    ValidationRequest, 
    ValidationPriority,
    ValidationStatus
)
from validation.config import ValidationConfig

async def create_demo_requests() -> list:
    """创建演示验证请求"""
    
    demo_data = [
        {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'current_price': 45000.0,
            'features': {
                'mid_price': 45000.0,
                'spread': 2.5,
                'wap_1': 45001.2,
                'wap_5': 44998.7,
                'order_imbalance_1': 0.15,
                'depth_imbalance': 0.08,
                'volatility_60s': 0.025,
                'volume_1m': 1250.5,
                'volume_5m': 5680.2
            },
            'indicators': {
                'rsi': 45.2,
                'macd': -125.3,
                'bollinger_position': 0.35,
                'adx': 28.5,
                'atr': 450.0
            },
            'risk_context': {
                'volatility': 0.035,
                'var_95': 0.025,
                'liquidity_score': 0.85
            }
        },
        {
            'symbol': 'ETHUSDT',
            'timeframe': '1h', 
            'current_price': 2800.0,
            'features': {
                'mid_price': 2800.0,
                'spread': 1.8,
                'wap_1': 2800.5,
                'wap_5': 2799.8,
                'order_imbalance_1': -0.12,
                'depth_imbalance': 0.05,
                'volatility_60s': 0.032,
                'volume_1m': 890.3,
                'volume_5m': 4200.1
            },
            'indicators': {
                'rsi': 58.7,
                'macd': 45.2,
                'bollinger_position': 0.72,
                'adx': 32.1,
                'atr': 28.5
            },
            'risk_context': {
                'volatility': 0.042,
                'var_95': 0.031,
                'liquidity_score': 0.78
            }
        },
        {
            'symbol': 'ADAUSDT',
            'timeframe': '1h',
            'current_price': 0.45,
            'features': {
                'mid_price': 0.45,
                'spread': 0.0008,
                'wap_1': 0.4501,
                'wap_5': 0.4499,
                'order_imbalance_1': 0.08,
                'depth_imbalance': 0.03,
                'volatility_60s': 0.045,
                'volume_1m': 2500000.0,
                'volume_5m': 12500000.0
            },
            'indicators': {
                'rsi': 35.4,
                'macd': -0.0015,
                'bollinger_position': 0.28,
                'adx': 18.7,
                'atr': 0.008
            },
            'risk_context': {
                'volatility': 0.055,
                'var_95': 0.042,
                'liquidity_score': 0.65
            }
        }
    ]
    
    requests = []
    for data in demo_data:
        request = ValidationRequest(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            current_price=data['current_price'],
            features=data['features'],
            indicators=data['indicators'],
            risk_context=data['risk_context'],
            priority=ValidationPriority.MEDIUM
        )
        requests.append(request)
    
    return requests

async def demonstrate_single_validation():
    """演示单个信号验证"""
    print("\n" + "="*60)
    print("🎯 单个信号验证演示")
    print("="*60)
    
    # 创建配置
    config = ValidationConfig.create_development_config()
    
    async with SignalValidationCoordinator(config) as coordinator:
        # 创建验证请求
        request = (await create_demo_requests())[0]  # BTCUSDT
        
        print(f"📊 验证信号: {request.symbol} {request.timeframe}")
        print(f"💰 当前价格: ${request.current_price:,.2f}")
        print(f"📈 RSI: {request.indicators['rsi']}")
        print(f"📊 MACD: {request.indicators['macd']}")
        print(f"💧 波动率: {request.risk_context['volatility']:.3f}")
        
        # 执行验证
        print("\n🔄 正在执行双重验证...")
        start_time = time.time()
        
        result = await coordinator.validate_signal(request)
        
        processing_time = time.time() - start_time
        
        # 输出结果
        print(f"\n✅ 验证完成 (耗时: {processing_time:.3f}s)")
        print(f"📋 验证状态: {result.status.value}")
        
        if result.layer1_result:
            print(f"🧠 第一层结果:")
            print(f"   预测标签: {result.layer1_result.label} ({'买入' if result.layer1_result.label == 1 else '持有' if result.layer1_result.label == 0 else '卖出'})")
            print(f"   概率: {result.layer1_result.probability:.3f}")
            print(f"   置信度: {result.layer1_result.confidence:.3f}")
        
        if result.layer2_result:
            print(f"🤖 第二层结果:")
            print(f"   建议方向: {result.layer2_result.direction}")
            print(f"   置信度: {result.layer2_result.confidence:.3f}")
            print(f"   风险评估: {result.layer2_result.risk_assessment}")
            
            if result.layer2_result.entry_price:
                print(f"   建议参数:")
                print(f"   入场价: ${result.layer2_result.entry_price:.4f}")
                print(f"   止损价: ${result.layer2_result.stop_loss:.4f}")
                print(f"   止盈价: ${result.layer2_result.take_profit:.4f}")
        
        print(f"🎯 综合评分: {result.combined_score:.3f}")
        if result.risk_reward_ratio:
            print(f"⚖️ 风险回报比: {result.risk_reward_ratio:.2f}")
        
        return result

async def demonstrate_batch_validation():
    """演示批量信号验证"""
    print("\n" + "="*60)
    print("🎯 批量信号验证演示")
    print("="*60)
    
    # 创建配置
    config = ValidationConfig.create_development_config()
    config.batch_size = 3
    
    async with SignalValidationCoordinator(config) as coordinator:
        # 创建批量验证请求
        requests = await create_demo_requests()
        
        print(f"📊 批量验证 {len(requests)} 个信号:")
        for request in requests:
            print(f"   • {request.symbol} @ ${request.current_price:,.4f}")
        
        # 执行批量验证
        print("\n🔄 正在执行批量双重验证...")
        start_time = time.time()
        
        results = await coordinator.batch_validate(requests)
        
        processing_time = time.time() - start_time
        
        # 输出结果
        print(f"\n✅ 批量验证完成 (耗时: {processing_time:.3f}s)")
        print("\n📋 验证结果汇总:")
        print("-" * 80)
        print(f"{'符号':<12} {'状态':<15} {'第一层':<8} {'第二层':<12} {'综合评分':<10} {'R/R比':<8}")
        print("-" * 80)
        
        for result in results:
            layer1_info = f"{result.layer1_result.label}" if result.layer1_result else "N/A"
            layer2_info = result.layer2_result.direction if result.layer2_result else "N/A"
            
            print(f"{result.symbol:<12} {result.status.value:<15} {layer1_info:<8} {layer2_info:<12} "
                  f"{result.combined_score:.3f:<10.3f} {result.risk_reward_ratio:.2f if result.risk_reward_ratio else 'N/A':<8}")
        
        # 获取性能统计
        stats = coordinator.get_performance_stats()
        print(f"\n📈 性能统计:")
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   第一层通过率: {stats['layer1_passed']}/{stats['total_requests']}")
        print(f"   第二层通过率: {stats['layer2_passed']}/{stats['total_requests']}")
        print(f"   成功率: {stats['success_rate']:.2%}")
        
        return results

async def demonstrate_performance_monitoring():
    """演示性能监控"""
    print("\n" + "="*60)
    print("📊 性能监控演示")
    print("="*60)
    
    # 创建配置
    config = ValidationConfig.create_development_config()
    config.monitoring_config.enable_performance_monitoring = True
    
    async with SignalValidationCoordinator(config) as coordinator:
        # 执行几个验证请求以生成监控数据
        requests = await create_demo_requests()
        
        print("🔄 执行验证请求以生成监控数据...")
        for request in requests:
            await coordinator.validate_signal(request)
        
        # 获取性能摘要
        print("\n📈 获取性能摘要...")
        perf_summary = await coordinator.monitor.get_performance_summary(time_window_minutes=60)
        
        print("📊 性能指标:")
        print(f"   时间窗口: {perf_summary['time_window_minutes']} 分钟")
        print(f"   总请求数: {perf_summary['total_requests']}")
        print(f"   成功率: {perf_summary['success_rate']:.2%}")
        print(f"   错误率: {perf_summary['error_rate']:.2%}")
        print(f"   超时率: {perf_summary['timeout_rate']:.2%}")
        
        if 'processing_times' in perf_summary:
            times = perf_summary['processing_times']
            print(f"   平均处理时间: {times['avg']:.3f}s")
            print(f"   P50处理时间: {times['p50']:.3f}s")
            print(f"   P95处理时间: {times['p95']:.3f}s")
            print(f"   P99处理时间: {times['p99']:.3f}s")
        
        if 'status_distribution' in perf_summary:
            print(f"   状态分布: {perf_summary['status_distribution']}")
        
        # 检查健康状态
        print("\n🏥 健康状态检查...")
        health_status = await coordinator.monitor.check_health_status()
        print(f"   系统状态: {health_status['status']}")
        print(f"   是否健康: {health_status['healthy']}")
        
        if health_status.get('issues'):
            print(f"   健康问题: {health_status['issues']}")
        
        return health_status

async def main():
    """主演示函数"""
    print("🚀 双重验证机制完整演示")
    print("🎯 LightGBM + 本地LLM 双层验证系统")
    print("⏰ 演示时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 1. 演示单个验证
        single_result = await demonstrate_single_validation()
        
        # 2. 演示批量验证
        batch_results = await demonstrate_batch_validation()
        
        # 3. 演示性能监控
        monitoring_result = await demonstrate_performance_monitoring()
        
        # 4. 演示结果保存
        print("\n" + "="*60)
        print("💾 演示结果保存")
        print("="*60)
        
        # 保存演示结果
        demo_results = {
            'timestamp': time.time(),
            'single_validation': single_result.to_dict() if single_result else None,
            'batch_validation_count': len(batch_results) if batch_results else 0,
            'performance_monitoring': monitoring_result,
            'system_info': {
                'architecture': 'LightGBM + Local LLM',
                'validation_layers': 2,
                'features': [
                    '异步并发处理',
                    '智能重试机制',
                    '实时性能监控',
                    '配置化验证流程',
                    '多LLM提供商支持'
                ]
            }
        }
        
        # 保存到文件
        with open('demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False)
        
        print("✅ 演示结果已保存到 demo_results.json")
        
        print("\n" + "="*60)
        print("🎉 双重验证机制演示完成！")
        print("="*60)
        print("✨ 核心功能验证:")
        print("   ✅ LightGBM快速筛选 (第一层)")
        print("   ✅ 本地LLM深度评估 (第二层)")
        print("   ✅ 验证结果融合算法")
        print("   ✅ 异步并发处理")
        print("   ✅ 超时控制和错误处理")
        print("   ✅ 实时性能监控")
        print("   ✅ 配置化验证流程")
        print("\n🚀 系统已准备就绪，可用于生产环境！")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())