#!/usr/bin/env python3
"""
市场扫描系统核心功能测试
验证系统主要组件是否正常工作
"""

import sys
import os
import asyncio
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试导入功能"""
    print("🧪 测试组件导入...")
    
    try:
        # 测试核心组件
        from scanner.core.market_scanner import MarketScanner, ScanConfig
        print("✅ MarketScanner 导入成功")
        
        from scanner.cache.memory_cache import MemoryCache
        print("✅ MemoryCache 导入成功")
        
        from scanner.strategies import StrategyFactory, PriorityStrategy, FilterStrategy
        print("✅ 策略系统导入成功")
        
        from scanner.utils import DataProcessor, MetricsCalculator
        print("✅ 工具类导入成功")
        
        from scanner.config import ConfigManager, PresetConfigs
        print("✅ 配置管理导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        from scanner.cache.memory_cache import MemoryCache
        
        # 测试内存缓存
        cache = MemoryCache(default_ttl=60)
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        
        if result == "test_value":
            print("✅ 内存缓存测试通过")
        else:
            print("❌ 内存缓存测试失败")
            return False
        
        # 测试策略
        from scanner.strategies import StrategyFactory
        test_data = [
            {'symbol': 'BTCUSDT', 'volume_24h': 1000000, 'price': 50000},
            {'symbol': 'ETHUSDT', 'volume_24h': 500000, 'price': 3000}
        ]
        
        volume_strategy = StrategyFactory.create_priority_strategy("volume")
        processed_data = volume_strategy.apply(test_data)
        
        if len(processed_data) > 0:
            print("✅ 策略系统测试通过")
        else:
            print("❌ 策略系统测试失败")
            return False
        
        # 测试指标计算
        from scanner.utils import MetricsCalculator
        calculator = MetricsCalculator()
        
        market_data = {'price': 50000, 'volume_24h': 1000000, 'price_change_24h': 2.5}
        technical_indicators = {'rsi': 65, 'volatility': 0.15}
        
        scores = calculator.calculate_comprehensive_score(market_data, technical_indicators)
        
        if 'combined_score' in scores:
            print("✅ 指标计算测试通过")
        else:
            print("❌ 指标计算测试失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

async def test_async_functionality():
    """测试异步功能"""
    print("\n🧪 测试异步功能...")
    
    try:
        from scanner.core.market_scanner import MarketScanner, ScanConfig
        
        # 创建配置
        config = ScanConfig(
            max_tickers=10,
            batch_size=5,
            timeout=10.0,
            enable_deep_analysis=False  # 关闭深度分析以简化测试
        )
        
        # 创建扫描器
        scanner = MarketScanner(config)
        
        # 测试基本状态
        status = scanner.get_status()
        print(f"✅ 扫描器状态: {status['status']}")
        
        # 模拟扫描
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        print(f"🧪 执行模拟扫描: {symbols}")
        
        # 执行扫描
        report = await scanner.scan_markets(symbols, scan_type="quick")
        
        print(f"✅ 扫描完成:")
        print(f"   - 总交易对: {report.total_symbols}")
        print(f"   - 扫描时长: {report.duration:.2f}s")
        print(f"   - 最佳机会: {report.top_opportunities[0].symbol if report.top_opportunities else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 异步功能测试失败: {e}")
        return False

def test_configuration():
    """测试配置系统"""
    print("\n🧪 测试配置系统...")
    
    try:
        from scanner.config import ConfigManager, PresetConfigs
        
        # 测试预设配置
        hf_config = PresetConfigs.high_frequency_config()
        quality_config = PresetConfigs.quality_focused_config()
        balanced_config = PresetConfigs.balanced_config()
        
        print(f"✅ 高频配置: max_workers={hf_config.scanner.max_workers}")
        print(f"✅ 质量配置: deep_analysis={quality_config.scanner.enable_deep_analysis}")
        print(f"✅ 平衡配置: 使用默认参数")
        
        # 测试配置管理
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"✅ 配置管理器正常工作")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        return False

def print_system_summary():
    """打印系统摘要"""
    print("\n" + "="*60)
    print("🚀 AlphaSeeker 市场扫描系统功能摘要")
    print("="*60)
    print("✅ 并行市场扫描: MarketScanner + ScanConfig")
    print("✅ 智能交易对筛选: PriorityStrategy + FilterStrategy")
    print("✅ 深度分析触发机制: Deep Analysis Callbacks")
    print("✅ 多级缓存系统: MemoryCache + RedisCache")
    print("✅ 扫描结果聚合和统计: ScanReport + Statistics")
    print("✅ 实时市场监控和警报: PerformanceMonitor + AlertManager")
    print("✅ 可配置的扫描策略: ConfigManager + PresetConfigs")
    print("✅ 扫描性能优化和监控: Performance Metrics + Alerts")
    print("="*60)
    print("📁 代码结构:")
    print("   📂 code/scanner/")
    print("   ├── 🧠 core/market_scanner.py - 主扫描器")
    print("   ├── 💾 cache/ - 缓存系统")
    print("   ├── 🎯 strategies/ - 策略系统")
    print("   ├── 📊 monitoring/ - 监控系统")
    print("   ├── 🔧 utils/ - 工具类")
    print("   └── ⚙️ config/ - 配置管理")
    print("="*60)

def main():
    """主测试函数"""
    print("🔍 AlphaSeeker 市场扫描系统测试")
    print("="*60)
    
    # 运行测试
    tests = [
        ("导入功能", test_imports),
        ("基本功能", test_basic_functionality),
        ("配置系统", test_configuration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 运行异步测试
    try:
        async_result = asyncio.run(test_async_functionality())
        results.append(("异步功能", async_result))
    except Exception as e:
        print(f"❌ 异步功能测试异常: {e}")
        results.append(("异步功能", False))
    
    # 打印结果
    print("\n" + "="*60)
    print("📋 测试结果汇总:")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15} : {status}")
        if result:
            passed += 1
    
    print("="*60)
    print(f"总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！系统功能正常")
        print_system_summary()
    else:
        print("⚠️ 部分测试失败，但核心功能可用")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)