#!/usr/bin/env python3
"""
市场扫描系统演示脚本
展示如何使用市场扫描和深度分析系统的各个功能
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# 导入扫描系统组件
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner import (
    create_scanner,
    create_preset_config,
    ConfigManager,
    ScanConfig,
    create_opportunity_alert,
    PresetConfigs
)
from scanner.cache import RedisCache, MemoryCache
from scanner.monitoring import AlertManager, create_performance_alert
from scanner.strategies import StrategyFactory
from scanner.utils import DataProcessor, MetricsCalculator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScannerDemo:
    """扫描系统演示类"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.data_processor = DataProcessor()
        self.metrics_calculator = MetricsCalculator()
        self.alert_manager = AlertManager()
        
    async def setup_demo(self):
        """设置演示环境"""
        logger.info("Setting up scanner demo environment...")
        
        # 创建配置目录
        os.makedirs("demo_configs", exist_ok=True)
        os.makedirs("demo_data", exist_ok=True)
        os.makedirs("demo_metrics", exist_ok=True)
        
        # 设置默认配置
        self.config = self.config_manager.load_config()
        
        # 创建内存缓存演示
        self.memory_cache = MemoryCache(default_ttl=300, max_size=1000)
        logger.info("Demo setup completed")
    
    async def demo_basic_scanning(self):
        """演示基本扫描功能"""
        logger.info("=== Basic Scanning Demo ===")
        
        # 创建扫描器
        scanner = create_scanner(self.config)
        
        # 模拟交易对列表
        symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
            "SOLUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT"
        ]
        
        logger.info(f"Starting scan for {len(symbols)} symbols...")
        
        # 执行扫描
        scan_start = datetime.now()
        report = await scanner.scan_markets(symbols, scan_type="quick")
        scan_duration = (datetime.now() - scan_start).total_seconds()
        
        # 展示结果
        logger.info(f"Scan completed in {scan_duration:.2f} seconds")
        logger.info(f"Total symbols processed: {report.total_symbols}")
        logger.info(f"Scan duration: {report.duration:.2f}s")
        logger.info(f"Performance: {report.performance_metrics.get('symbols_per_second', 0):.2f} symbols/sec")
        
        # 展示top机会
        logger.info("Top 5 opportunities:")
        for i, result in enumerate(report.top_opportunities[:5], 1):
            logger.info(f"  {i}. {result.symbol}: Score={result.score:.3f}, Confidence={result.confidence:.3f}")
        
        return report
    
    async def demo_advanced_strategies(self):
        """演示高级策略功能"""
        logger.info("=== Advanced Strategies Demo ===")
        
        # 创建不同的策略
        volume_priority = StrategyFactory.create_priority_strategy("volume")
        volatility_priority = StrategyFactory.create_priority_strategy("volatility")
        strict_filter = StrategyFactory.create_filter_strategy("strict")
        balanced_filter = StrategyFactory.create_filter_strategy("balanced")
        
        # 生成模拟数据
        test_data = await self._generate_test_data(50)
        
        # 应用不同策略
        strategies = [
            ("Volume Priority", volume_priority),
            ("Volatility Priority", volatility_priority),
            ("Strict Filter", strict_filter),
            ("Balanced Filter", balanced_filter)
        ]
        
        for strategy_name, strategy in strategies:
            logger.info(f"Applying {strategy_name}...")
            
            if hasattr(strategy, 'apply'):  # 确保是BaseStrategy的子类
                try:
                    processed_data = strategy.apply(test_data)
                    logger.info(f"{strategy_name}: {len(test_data)} -> {len(processed_data)} items")
                except Exception as e:
                    logger.error(f"Error applying {strategy_name}: {e}")
    
    async def demo_caching_system(self):
        """演示缓存系统"""
        logger.info("=== Caching System Demo ===")
        
        # 测试内存缓存
        logger.info("Testing Memory Cache...")
        
        # 设置缓存
        await self.memory_cache.set("test_key", {"data": "test_value", "timestamp": datetime.now()})
        
        # 获取缓存
        cached_value = self.memory_cache.get("test_key")
        logger.info(f"Cached value: {cached_value}")
        
        # 测试批量操作
        batch_data = {f"key_{i}": f"value_{i}" for i in range(10)}
        self.memory_cache.bulk_set(batch_data)
        
        batch_keys = list(batch_data.keys())
        bulk_results = self.memory_cache.bulk_get(batch_keys)
        logger.info(f"Bulk get results: {len(bulk_results)} items retrieved")
        
        # 获取统计信息
        cache_stats = self.memory_cache.get_stats()
        logger.info(f"Cache stats: Hit rate={cache_stats['hit_rate']:.2%}")
    
    async def demo_performance_monitoring(self):
        """演示性能监控"""
        logger.info("=== Performance Monitoring Demo ===")
        
        from scanner.monitoring import PerformanceMonitor, PerformanceMetrics
        
        # 创建性能监控器
        monitor = PerformanceMonitor()
        
        # 启动监控
        await monitor.start_monitoring(interval=5.0)
        
        # 模拟性能数据
        test_metrics = [
            PerformanceMetrics(
                scan_id=f"test_scan_{i}",
                timestamp=datetime.now(),
                duration=10.0 + i * 2,
                symbols_processed=100 + i * 10,
                symbols_per_second=10.0 + i,
                memory_usage_mb=512.0 + i * 50,
                cpu_usage_percent=25.0 + i * 5,
                cache_hit_rate=0.8 + i * 0.02,
                error_rate=0.02 - i * 0.002,
                throughput_mb=0.0,
                latency_p50=0.5 + i * 0.1,
                latency_p95=1.2 + i * 0.2,
                latency_p99=2.0 + i * 0.3
            )
            for i in range(5)
        ]
        
        # 记录指标
        for metrics in test_metrics:
            await monitor.record_metrics(metrics)
        
        # 获取统计信息
        await asyncio.sleep(1)  # 等待监控循环
        
        stats = monitor.get_statistics()
        logger.info(f"Performance statistics:")
        logger.info(f"  Total scans: {stats.get('total_scans', 0)}")
        logger.info(f"  Average duration: {stats.get('duration', {}).get('mean', 0):.2f}s")
        logger.info(f"  Average throughput: {stats.get('throughput', {}).get('mean', 0):.2f} symbols/sec")
        
        # 停止监控
        await monitor.stop_monitoring()
    
    async def demo_alert_system(self):
        """演示警报系统"""
        logger.info("=== Alert System Demo ===")
        
        # 添加自定义警报规则
        from scanner.monitoring.alert_manager import AlertRule, AlertSeverity
        
        # 高机会警报规则
        opportunity_rule = AlertRule(
            id="high_opportunity_demo",
            name="High Opportunity Demo",
            description="Demo rule for high opportunity detection",
            alert_type="opportunity",
            severity=AlertSeverity.WARNING,
            condition=lambda data: data.get('score', 0) > 0.8
        )
        
        # 性能警报规则
        performance_rule = AlertRule(
            id="performance_demo", 
            name="Performance Demo",
            description="Demo rule for performance issues",
            alert_type="performance",
            severity=AlertSeverity.ERROR,
            condition=lambda data: data.get('duration', 0) > 20.0
        )
        
        self.alert_manager.add_rule(opportunity_rule)
        self.alert_manager.add_rule(performance_rule)
        
        # 启动处理器
        await self.alert_manager.start_handlers()
        
        # 发送测试警报
        high_opportunity_alert = create_opportunity_alert(
            "BTCUSDT", 0.92, "Strong bullish signals detected"
        )
        await self.alert_manager.send_alert(high_opportunity_alert)
        
        performance_alert = create_performance_alert(
            "scan_duration", 25.0, 20.0
        )
        await self.alert_manager.send_alert(performance_alert)
        
        # 检查规则
        rule_data = {'score': 0.85, 'duration': 15.0}
        triggered_alerts = await self.alert_manager.check_rules(rule_data)
        logger.info(f"Triggered {len(triggered_alerts)} alerts from rules")
        
        # 获取统计信息
        alert_stats = self.alert_manager.get_statistics()
        logger.info(f"Alert statistics: {alert_stats['total_alerts']} total alerts")
        
        # 停止处理器
        await self.alert_manager.stop_handlers()
    
    async def demo_deep_analysis(self):
        """演示深度分析功能"""
        logger.info("=== Deep Analysis Demo ===")
        
        # 创建扫描器
        scanner = create_scanner(self.config)
        
        # 定义深度分析回调
        async def deep_analysis_callback(symbol: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
            """深度分析回调函数"""
            logger.info(f"Performing deep analysis for {symbol}")
            
            # 模拟深度分析
            market_data = metadata.get('market_data', {})
            
            analysis_data = {
                'pattern_recognition': await self._analyze_patterns(symbol),
                'volume_profile': await self._analyze_volume_profile(symbol),
                'order_flow': await self._analyze_order_flow(symbol),
                'correlation_analysis': await self._analyze_correlations(symbol),
                'sentiment_analysis': await self._analyze_sentiment(symbol),
                'technical_divergence': await self._analyze_divergence(symbol),
                'support_resistance': await self._analyze_support_resistance(symbol),
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis_data
        
        # 设置回调
        scanner.callbacks['deep_analysis_callback'] = deep_analysis_callback
        
        # 执行扫描
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        report = await scanner.scan_markets(symbols, scan_type="deep")
        
        # 展示深度分析结果
        logger.info("Deep analysis results:")
        for result in report.top_opportunities:
            if result.deep_analysis_required:
                logger.info(f"  {result.symbol} - Deep analysis completed")
                analysis_data = result.analysis_data
                if analysis_data:
                    logger.info(f"    Pattern recognition: {analysis_data.get('pattern_recognition', {}).get('pattern_type', 'unknown')}")
                    logger.info(f"    Volume profile: {analysis_data.get('volume_profile', {}).get('volume_trend', 'unknown')}")
                    logger.info(f"    Sentiment: {analysis_data.get('sentiment_analysis', {}).get('sentiment_label', 'unknown')}")
    
    async def demo_performance_optimization(self):
        """演示性能优化"""
        logger.info("=== Performance Optimization Demo ===")
        
        # 测试不同配置的性能
        configs = [
            ("High Frequency", PresetConfigs.high_frequency_config()),
            ("Quality Focused", PresetConfigs.quality_focused_config()),
            ("Balanced", PresetConfigs.balanced_config())
        ]
        
        symbols = [f"SYMBOL{i:03d}USDT" for i in range(100)]  # 100个测试交易对
        
        for config_name, config in configs:
            logger.info(f"Testing {config_name} configuration...")
            
            scanner = create_scanner(config)
            
            # 记录性能
            start_time = datetime.now()
            report = await scanner.scan_markets(symbols, scan_type="full")
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"{config_name} Results:")
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info(f"  Throughput: {len(symbols)/duration:.2f} symbols/sec")
            logger.info(f"  Memory usage: {report.performance_metrics.get('memory_usage', 0):.0f}MB")
            logger.info(f"  Processed: {report.analyzed_symbols} symbols")
    
    async def demo_configuration_management(self):
        """演示配置管理"""
        logger.info("=== Configuration Management Demo ===")
        
        # 保存当前配置
        config_path = "demo_configs/demo_config.json"
        self.config_manager.save_config(self.config, config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # 修改配置
        updates = {
            'scanner': {
                'max_workers': 15,
                'batch_size': 30,
                'enable_deep_analysis': True
            },
            'strategy': {
                'filter_strategy': 'strict',
                'priority_strategy': 'volume'
            },
            'monitoring': {
                'enable_monitoring': True,
                'max_scan_duration': 45.0
            }
        }
        
        self.config_manager.update_config(updates)
        updated_config = self.config_manager.get_config()
        
        logger.info("Configuration updated:")
        logger.info(f"  Max workers: {updated_config.scanner.max_workers}")
        logger.info(f"  Batch size: {updated_config.scanner.batch_size}")
        logger.info(f"  Filter strategy: {updated_config.strategy.filter_strategy}")
        
        # 加载保存的配置
        loaded_config = self.config_manager.load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    
    async def demo_real_time_monitoring(self):
        """演示实时监控"""
        logger.info("=== Real-time Monitoring Demo ===")
        
        from scanner.monitoring import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # 启动监控
        await monitor.start_monitoring(interval=2.0)
        
        # 模拟实时扫描
        scanner = create_scanner(self.config)
        
        # 连续执行多次扫描
        for i in range(3):
            logger.info(f"Starting scan cycle {i+1}/3...")
            
            symbols = [f"SYMBOL{j:03d}USDT" for j in range(20)]
            report = await scanner.scan_markets(symbols)
            
            logger.info(f"Scan {i+1} completed: {report.analyzed_symbols} symbols in {report.duration:.2f}s")
            
            # 短暂等待
            await asyncio.sleep(1)
        
        # 获取实时统计
        await asyncio.sleep(1)  # 等待监控系统处理
        stats = monitor.get_statistics()
        system_health = monitor.get_system_health()
        
        logger.info("Real-time monitoring results:")
        logger.info(f"  Total monitored scans: {stats.get('total_scans', 0)}")
        logger.info(f"  System health: {system_health.get('status', 'unknown')}")
        logger.info(f"  Memory usage: {system_health.get('memory_usage', 0):.1f}%")
        logger.info(f"  CPU usage: {system_health.get('cpu_usage', 0):.1f}%")
        
        # 停止监控
        await monitor.stop_monitoring()
    
    async def _generate_test_data(self, count: int) -> List[Dict[str, Any]]:
        """生成测试数据"""
        import random
        
        data = []
        for i in range(count):
            item = {
                'symbol': f"SYMBOL{i:03d}USDT",
                'volume_24h': random.uniform(100000, 10000000),
                'price': random.uniform(1, 100000),
                'price_change_24h': random.uniform(-10, 10),
                'market_cap': random.uniform(1000000, 1000000000),
                'bid_ask_spread': random.uniform(0.0001, 0.005),
                'volatility': random.uniform(0.01, 0.3),
                'volume_trend': random.uniform(-0.5, 0.5),
                'quality_score': random.uniform(0.3, 1.0)
            }
            data.append(item)
        
        return data
    
    async def _analyze_patterns(self, symbol: str) -> Dict[str, Any]:
        """分析价格模式"""
        return {
            'pattern_type': 'bullish_flag',
            'confidence': 0.75,
            'breakout_probability': 0.8
        }
    
    async def _analyze_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """分析成交量分布"""
        return {
            'volume_trend': 'increasing',
            'volume_strength': 0.8,
            'accumulation': True
        }
    
    async def _analyze_order_flow(self, symbol: str) -> Dict[str, Any]:
        """分析订单流"""
        return {
            'order_flow_momentum': 0.6,
            'imbalance_ratio': 1.2,
            'liquidity_score': 0.75
        }
    
    async def _analyze_correlations(self, symbol: str) -> Dict[str, Any]:
        """分析相关性"""
        return {
            'market_correlation': 0.85,
            'sector_correlation': 0.72,
            'correlation_strength': 'strong'
        }
    
    async def _analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """分析市场情绪"""
        return {
            'sentiment_label': 'bullish',
            'sentiment_score': 0.78,
            'fear_greed_index': 68
        }
    
    async def _analyze_divergence(self, symbol: str) -> Dict[str, Any]:
        """分析技术背离"""
        return {
            'rsi_divergence': False,
            'macd_divergence': False,
            'price_momentum_divergence': False,
            'divergence_strength': 'weak'
        }
    
    async def _analyze_support_resistance(self, symbol: str) -> Dict[str, Any]:
        """分析支撑阻力"""
        return {
            'support_levels': [49500, 49200],
            'resistance_levels': [50800, 51200],
            'breakout_level': 51000
        }


async def main():
    """主演示函数"""
    print("🚀 AlphaSeeker 市场扫描和深度分析系统演示")
    print("=" * 60)
    
    demo = ScannerDemo()
    
    try:
        # 设置演示环境
        await demo.setup_demo()
        
        # 运行各种演示
        demonstrations = [
            ("基本扫描功能", demo.demo_basic_scanning),
            ("高级策略系统", demo.demo_advanced_strategies),
            ("缓存系统", demo.demo_caching_system),
            ("性能监控", demo.demo_performance_monitoring),
            ("警报系统", demo.demo_alert_system),
            ("深度分析", demo.demo_deep_analysis),
            ("性能优化", demo.demo_performance_optimization),
            ("配置管理", demo.demo_configuration_management),
            ("实时监控", demo.demo_real_time_monitoring),
        ]
        
        for demo_name, demo_func in demonstrations:
            print(f"\n📋 {demo_name}")
            print("-" * 40)
            
            try:
                await demo_func()
                print(f"✅ {demo_name} 演示完成")
            except Exception as e:
                print(f"❌ {demo_name} 演示失败: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("📁 演示数据已保存到 demo_data/ 目录")
        print("📄 配置文件已保存到 demo_configs/ 目录")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ 演示失败: {e}")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())