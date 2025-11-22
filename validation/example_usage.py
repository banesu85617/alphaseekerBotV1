"""
双重验证机制使用示例
演示如何使用SignalValidationCoordinator进行信号验证
"""

import asyncio
import logging
from typing import Dict, Any
import time

# 导入验证模块
from validation.coordinator import SignalValidationCoordinator, ValidationRequest, ValidationPriority
from validation.config import ValidationConfig, load_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_request(symbol: str = "BTCUSDT", timeframe: str = "1h") -> ValidationRequest:
    """创建示例验证请求"""
    
    # 示例特征数据
    features = {
        'mid_price': 45000.0,
        'spread': 2.5,
        'wap_1': 45001.2,
        'wap_5': 44998.7,
        'order_imbalance_1': 0.15,
        'depth_imbalance': 0.08,
        'volatility_60s': 0.025,
        'volume_1m': 1250.5,
        'volume_5m': 5680.2
    }
    
    # 示例技术指标
    indicators = {
        'rsi': 45.2,
        'macd': -125.3,
        'bollinger_position': 0.35,
        'adx': 28.5,
        'atr': 450.0
    }
    
    # 示例风险上下文
    risk_context = {
        'volatility': 0.035,
        'var_95': 0.025,
        'liquidity_score': 0.85
    }
    
    return ValidationRequest(
        symbol=symbol,
        timeframe=timeframe,
        current_price=45002.5,
        features=features,
        indicators=indicators,
        risk_context=risk_context,
        priority=ValidationPriority.MEDIUM
    )


async def basic_validation_example():
    """基础验证示例"""
    logger.info("=== 基础验证示例 ===")
    
    # 创建配置
    config = ValidationConfig.create_development_config()
    
    # 创建协调器
    async with SignalValidationCoordinator(config) as coordinator:
        # 创建验证请求
        request = await create_sample_request("BTCUSDT", "1h")
        
        # 执行验证
        logger.info(f"开始验证: {request.symbol} {request.timeframe}")
        start_time = time.time()
        
        result = await coordinator.validate_signal(request)
        
        processing_time = time.time() - start_time
        
        # 输出结果
        logger.info(f"验证完成，耗时: {processing_time:.3f}s")
        logger.info(f"状态: {result.status.value}")
        logger.info(f"综合评分: {result.combined_score:.3f}")
        
        if result.layer1_result:
            logger.info(f"第一层结果: 标签={result.layer1_result.label}, "
                       f"概率={result.layer1_result.probability:.3f}")
        
        if result.layer2_result:
            logger.info(f"第二层结果: 方向={result.layer2_result.direction}, "
                       f"置信度={result.layer2_result.confidence:.3f}")
        
        return result


async def batch_validation_example():
    """批量验证示例"""
    logger.info("=== 批量验证示例 ===")
    
    # 创建配置
    config = ValidationConfig.create_development_config()
    config.batch_size = 5  # 小批次用于演示
    
    # 创建协调器
    async with SignalValidationCoordinator(config) as coordinator:
        # 创建多个验证请求
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        requests = []
        
        for symbol in symbols:
            request = await create_sample_request(symbol, "1h")
            requests.append(request)
        
        # 执行批量验证
        logger.info(f"开始批量验证 {len(requests)} 个信号")
        start_time = time.time()
        
        results = await coordinator.batch_validate(requests)
        
        processing_time = time.time() - start_time
        
        # 输出结果
        logger.info(f"批量验证完成，耗时: {processing_time:.3f}s")
        
        for i, result in enumerate(results):
            logger.info(f"{i+1}. {result.symbol}: {result.status.value}, "
                       f"评分={result.combined_score:.3f}")
        
        # 获取性能统计
        stats = coordinator.get_performance_stats()
        logger.info(f"性能统计: {stats}")
        
        return results


async def performance_monitoring_example():
    """性能监控示例"""
    logger.info("=== 性能监控示例 ===")
    
    # 创建配置
    config = ValidationConfig.create_development_config()
    config.monitoring_config.enable_performance_monitoring = True
    
    # 创建协调器
    async with SignalValidationCoordinator(config) as coordinator:
        # 执行多个验证请求以生成监控数据
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        for symbol in symbols:
            request = await create_sample_request(symbol, "1h")
            await coordinator.validate_signal(request)
        
        # 获取性能摘要
        perf_summary = await coordinator.monitor.get_performance_summary(time_window_minutes=60)
        logger.info(f"性能摘要: {perf_summary}")
        
        # 获取实时统计
        real_time_stats = await coordinator.monitor.get_real_time_stats()
        logger.info(f"实时统计: {real_time_stats}")
        
        # 检查健康状态
        health_status = await coordinator.monitor.check_health_status()
        logger.info(f"健康状态: {health_status}")
        
        return health_status


async def configuration_example():
    """配置示例"""
    logger.info("=== 配置示例 ===")
    
    # 从配置文件加载
    try:
        config = ValidationConfig.from_yaml("validation/config_example.yaml")
        logger.info("从YAML文件加载配置成功")
    except FileNotFoundError:
        logger.info("使用默认配置")
        config = ValidationConfig.create_development_config()
    
    # 创建不同环境的配置
    dev_config = ValidationConfig.create_development_config()
    prod_config = ValidationConfig.create_production_config()
    test_config = ValidationConfig.create_test_config()
    
    logger.info(f"开发环境配置: 并发数={dev_config.max_concurrent_tasks}")
    logger.info(f"生产环境配置: 并发数={prod_config.max_concurrent_tasks}")
    logger.info(f"测试环境配置: 并发数={test_config.max_concurrent_tasks}")
    
    # 保存配置
    config.save_to_yaml("validation/test_config.yaml")
    config.save_to_json("validation/test_config.json")
    logger.info("配置已保存到文件")
    
    return config


async def error_handling_example():
    """错误处理示例"""
    logger.info("=== 错误处理示例 ===")
    
    # 创建配置
    config = ValidationConfig.create_development_config()
    
    # 创建协调器
    async with SignalValidationCoordinator(config) as coordinator:
        # 创建包含错误数据的请求
        error_request = ValidationRequest(
            symbol="INVALID_SYMBOL",
            timeframe="1h",
            current_price=0.0,  # 无效价格
            features={},  # 空特征
            indicators={},  # 空指标
            risk_context={},  # 空风险上下文
            priority=ValidationPriority.LOW
        )
        
        try:
            # 尝试验证
            result = await coordinator.validate_signal(error_request)
            logger.info(f"错误处理结果: {result.status.value}")
            logger.info(f"错误信息: {result.metadata}")
            
        except Exception as e:
            logger.error(f"捕获异常: {str(e)}")
        
        return coordinator.get_performance_stats()


async def main():
    """主函数"""
    logger.info("双重验证机制演示程序启动")
    
    try:
        # 运行各种示例
        await basic_validation_example()
        await asyncio.sleep(1)
        
        await batch_validation_example()
        await asyncio.sleep(1)
        
        await performance_monitoring_example()
        await asyncio.sleep(1)
        
        await configuration_example()
        await asyncio.sleep(1)
        
        await error_handling_example()
        
        logger.info("所有示例运行完成")
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise


if __name__ == "__main__":
    # 运行演示程序
    asyncio.run(main())