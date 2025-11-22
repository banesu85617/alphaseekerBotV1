"""
使用示例文件

展示如何使用多策略信号处理管道进行各种任务。
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 导入管道组件
from .pipeline import MultiStrategyPipeline
from .types import (
    PipelineConfig, MarketData, TechnicalIndicators, RiskMetrics, MLFeatures,
    MLPrediction, BacktestResult, ScanRequest, SignalDirection, StrategyType
)
from .config_example import get_preset_config, print_config_summary

async def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 1. 创建配置
    config = get_preset_config("default")
    
    # 2. 初始化管道
    pipeline = MultiStrategyPipeline(config)
    
    # 3. 启动管道
    session_id = await pipeline.start()
    print(f"管道已启动，会话ID: {session_id}")
    
    try:
        # 4. 准备数据
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            price=45000.0,
            volume=1000000.0,
            data_freshness=5.0
        )
        
        technical_indicators = TechnicalIndicators(
            rsi=65.5,
            macd=120.5,
            macd_signal=110.2,
            adx=28.3,
            sma_50=44500.0,
            sma_200=42000.0
        )
        
        risk_metrics = RiskMetrics(
            garch_volatility=0.035,
            var_95=0.025,
            max_drawdown=-0.12,
            sharpe_ratio=1.8
        )
        
        ml_prediction = MLPrediction(
            label=1,  # 买入
            probability_scores={-1: 0.15, 0: 0.25, 1: 0.60},
            confidence=0.72,
            model_version="v1.2.3"
        )
        
        # 5. 处理单个符号
        result = await pipeline.process_single_symbol(
            symbol="BTCUSDT",
            market_data=market_data,
            technical_indicators=technical_indicators,
            risk_metrics=risk_metrics,
            ml_prediction=ml_prediction
        )
        
        print(f"处理结果:")
        print(f"  符号: {result.symbol}")
        print(f"  最终方向: {result.final_direction.value}")
        print(f"  综合评分: {result.final_score:.3f}")
        print(f"  置信度: {result.combined_confidence:.3f}")
        print(f"  风险回报比: {result.risk_reward_ratio:.2f}")
        print(f"  决策原因: {', '.join(result.decision_reason)}")
        
    finally:
        # 6. 停止管道
        await pipeline.stop()
        print("管道已停止")

async def example_batch_scan():
    """批量扫描示例"""
    print("\n=== 批量扫描示例 ===")
    
    # 1. 创建配置
    config = get_preset_config("high_performance")
    pipeline = MultiStrategyPipeline(config)
    
    await pipeline.start()
    
    try:
        # 2. 准备批量数据
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        
        symbol_data_map = {}
        for symbol in symbols:
            # 模拟市场数据
            symbol_data_map[symbol] = {
                "price": 40000 + hash(symbol) % 10000,
                "volume": 1000000 + hash(symbol) % 500000,
                "data_freshness": 10,
                "technical_indicators": {
                    "rsi": 50 + hash(symbol) % 40,
                    "macd": 100 + hash(symbol) % 200,
                    "macd_signal": 95 + hash(symbol) % 190,
                    "adx": 20 + hash(symbol) % 20,
                    "sma_50": 39000 + hash(symbol) % 1000,
                    "sma_200": 38000 + hash(symbol) % 2000
                },
                "risk_metrics": {
                    "garch_volatility": 0.02 + hash(symbol) % 100 / 10000,
                    "var_95": 0.015 + hash(symbol) % 50 / 10000,
                    "max_drawdown": -0.08 - hash(symbol) % 50 / 1000,
                    "sharpe_ratio": 1.0 + hash(symbol) % 200 / 100
                },
                "ml_prediction": {
                    "label": hash(symbol) % 3 - 1,  # -1, 0, 1
                    "probability_scores": {
                        -1: 0.2 + hash(symbol) % 30 / 100,
                        0: 0.2 + hash(symbol) % 30 / 100,
                        1: 0.6 - hash(symbol) % 60 / 100
                    },
                    "confidence": 0.5 + hash(symbol) % 50 / 100,
                    "model_version": "v1.2.3"
                }
            }
        
        # 3. 创建扫描请求
        scan_request = ScanRequest(
            symbols=symbols,
            max_symbols=5,
            top_n=3,
            filters={
                "min_confidence": 0.6,
                "min_score": 0.3,
                "min_risk_reward_ratio": 1.0,
                "allowed_directions": ["long", "short"]
            }
        )
        
        # 4. 执行批量扫描
        scan_result = await pipeline.batch_scan(scan_request, symbol_data_map)
        
        print(f"批量扫描结果:")
        print(f"  请求ID: {scan_result.request_id}")
        print(f"  处理时间: {scan_result.processing_time:.2f}秒")
        print(f"  总符号数: {scan_result.total_symbols}")
        print(f"  有效符号数: {scan_result.filtered_symbols}")
        print(f"  错误数: {len(scan_result.errors)}")
        
        print(f"\n前 {len(scan_result.results)} 个结果:")
        for i, result in enumerate(scan_result.results, 1):
            print(f"  {i}. {result.symbol}: {result.final_direction.value} "
                  f"(评分: {result.final_score:.3f}, 置信度: {result.combined_confidence:.3f})")
        
    finally:
        await pipeline.stop()

async def example_performance_monitoring():
    """性能监控示例"""
    print("\n=== 性能监控示例 ===")
    
    config = get_preset_config("high_performance")
    pipeline = MultiStrategyPipeline(config)
    
    await pipeline.start()
    
    try:
        # 模拟多次处理来产生性能数据
        symbols = [f"SYMBOL{i:03d}" for i in range(20)]
        
        for symbol in symbols:
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=100 + hash(symbol) % 1000,
                volume=100000 + hash(symbol) % 1000000,
                data_freshness=hash(symbol) % 30
            )
            
            technical_indicators = TechnicalIndicators(
                rsi=30 + hash(symbol) % 40,
                macd=hash(symbol) % 200 - 100,
                macd_signal=hash(symbol) % 180 - 90,
                adx=15 + hash(symbol) % 25
            )
            
            await pipeline.process_single_symbol(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators
            )
        
        # 获取性能指标
        metrics = await pipeline.get_performance_metrics()
        print(f"性能指标:")
        print(f"  总处理时间: {metrics.total_processing_time:.2f}秒")
        print(f"  吞吐量: {metrics.throughput:.2f} 信号/秒")
        print(f"  总体准确率: {metrics.accuracy_metrics.get('overall_accuracy', 0):.2%}")
        
        # 获取详细性能报告
        report = await pipeline.get_performance_report(timedelta(hours=1))
        print(f"\n性能报告:")
        print(f"  系统健康状态: {report['summary']['system_health']}")
        print(f"  当前吞吐量: {report['summary']['current_throughput']:.2f}")
        print(f"  当前准确率: {report['summary']['current_accuracy']:.2%}")
        print(f"  活跃告警数: {report['summary']['total_alerts']}")
        
        # 检查告警
        alerts = await pipeline.check_alerts()
        if alerts:
            print(f"\n检测到 {len(alerts)} 个告警:")
            for alert in alerts:
                print(f"  - {alert['type']}: {alert['message']}")
        else:
            print("\n暂无告警")
        
    finally:
        await pipeline.stop()

async def example_backtest_validation():
    """回测验证示例"""
    print("\n=== 回测验证示例 ===")
    
    config = get_preset_config("high_accuracy")
    pipeline = MultiStrategyPipeline(config)
    
    await pipeline.start()
    
    try:
        # 准备历史数据 (模拟)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'open': 40000 + pd.Series(range(len(dates))) % 2000,
            'high': 41000 + pd.Series(range(len(dates))) % 2000,
            'low': 39000 + pd.Series(range(len(dates))) % 2000,
            'close': 40000 + pd.Series(range(len(dates))) % 2000,
            'volume': 1000000 + pd.Series(range(len(dates))) % 500000
        })
        
        # 创建策略信号
        from .types import StrategySignal
        
        signal = StrategySignal(
            strategy_type=StrategyType.TECHNICAL_INDICATOR,
            direction=SignalDirection.LONG,
            confidence=0.8,
            score=0.75,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            market_data=MarketData(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                price=45000.0,
                volume=1000000.0
            ),
            technical_indicators=TechnicalIndicators(
                rsi=35.0,  # 超卖
                macd=150.0,
                macd_signal=120.0,
                adx=25.0
            )
        )
        
        # 执行回测验证
        backtest_result = await pipeline.validate_signal_backtest(
            signal=signal,
            historical_data=historical_data,
            validation_period=timedelta(days=30)
        )
        
        print(f"回测验证结果:")
        print(f"  综合评分: {backtest_result.score:.3f}")
        print(f"  总收益率: {backtest_result.total_return:.2%}")
        print(f"  夏普比率: {backtest_result.sharpe_ratio:.3f}")
        print(f"  最大回撤: {backtest_result.max_drawdown:.2%}")
        print(f"  胜率: {backtest_result.win_rate:.2%}")
        print(f"  盈亏比: {backtest_result.profit_factor:.2f}")
        print(f"  交易次数: {backtest_result.trade_count}")
        
    finally:
        await pipeline.stop()

async def example_config_customization():
    """配置自定义示例"""
    print("\n=== 配置自定义示例 ===")
    
    # 1. 使用预设配置
    preset_configs = ["default", "high_performance", "high_accuracy", "conservative", "aggressive"]
    
    for preset_name in preset_configs:
        config = get_preset_config(preset_name)
        print(f"\n{preset_name.upper()} 配置:")
        print_config_summary(config)
        
        # 创建管道并运行一次
        pipeline = MultiStrategyPipeline(config)
        
        await pipeline.start()
        
        # 获取管道状态
        status = pipeline.get_pipeline_status()
        print(f"管道状态: {status['is_running']}")
        print(f"配置已验证: {status['config']['max_concurrent_tasks']} 并发任务")
        
        await pipeline.stop()
    
    # 2. 自定义配置
    print("\n=== 自定义配置示例 ===")
    
    custom_config = PipelineConfig(
        max_concurrent_tasks=12,
        timeout_seconds=12,
        batch_size=75,
        ml_probability_threshold=0.7,
        ml_confidence_threshold=0.65,
        llm_confidence_threshold=0.7,
        min_risk_reward_ratio=1.2,
        max_position_size=0.08,
        max_leverage=8.0,
        max_symbols_per_scan=50,
        top_n_results=8
    )
    
    print("自定义配置:")
    print_config_summary(custom_config)

def example_data_format():
    """数据格式示例"""
    print("\n=== 数据格式示例 ===")
    
    # MarketData 示例
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=datetime(2023, 12, 1, 10, 30, 0),
        price=42500.0,
        volume=1500000.0,
        data_freshness=15.0
    )
    
    # TechnicalIndicators 示例
    technical_indicators = TechnicalIndicators(
        rsi=58.5,
        macd=125.3,
        macd_signal=118.7,
        bollinger_upper=43200.0,
        bollinger_middle=42500.0,
        bollinger_lower=41800.0,
        adx=22.1,
        atr=850.0,
        sma_20=42100.0,
        sma_50=41900.0,
        sma_200=40500.0,
        ema_12=42650.0,
        ema_26=42380.0
    )
    
    # RiskMetrics 示例
    risk_metrics = RiskMetrics(
        garch_volatility=0.032,
        var_95=0.028,
        expected_shortfall=0.035,
        max_drawdown=-0.095,
        sharpe_ratio=1.65,
        sortino_ratio=2.1,
        calmar_ratio=1.8
    )
    
    # MLFeatures 示例
    ml_features = MLFeatures(
        spread=15.2,
        order_imbalance=0.15,
        depth_imbalance=-0.08,
        wap_1=42505.5,
        wap_5=42498.2,
        volatility_60s=0.028,
        mid_price=42502.5,
        volume_features={
            "volume_1m": 150000.0,
            "volume_5m": 750000.0,
            "volume_1h": 1200000.0
        }
    )
    
    # MLPrediction 示例
    ml_prediction = MLPrediction(
        label=1,  # 买入
        probability_scores={-1: 0.18, 0: 0.22, 1: 0.60},
        confidence=0.73,
        model_version="lightgbm_v2.1.0",
        prediction_time=datetime(2023, 12, 1, 10, 30, 0)
    )
    
    # BacktestResult 示例
    backtest_result = BacktestResult(
        score=0.78,
        total_return=0.152,
        sharpe_ratio=1.85,
        max_drawdown=-0.065,
        win_rate=0.68,
        profit_factor=1.42,
        trade_count=24,
        backtest_period="30天",
        strategy_name="RSI策略",
        additional_metrics={
            "avg_trade_duration": 4.5,
            "max_consecutive_wins": 5,
            "max_consecutive_losses": 3
        }
    )
    
    print("数据结构示例:")
    print(f"MarketData: {market_data.symbol} @ {market_data.price}")
    print(f"TechnicalIndicators: RSI={technical_indicators.rsi}, MACD={technical_indicators.macd}")
    print(f"RiskMetrics: VaR={risk_metrics.var_95:.1%}, Sharpe={risk_metrics.sharpe_ratio:.2f}")
    print(f"MLFeatures: spread={ml_features.spread}, vol={ml_features.volatility_60s:.1%}")
    print(f"MLPrediction: label={ml_prediction.label}, confidence={ml_prediction.confidence:.2f}")
    print(f"BacktestResult: score={backtest_result.score:.3f}, return={backtest_result.total_return:.2%}")

async def run_all_examples():
    """运行所有示例"""
    print("多策略信号处理管道 - 使用示例")
    print("=" * 60)
    
    try:
        # 运行各种示例
        await example_basic_usage()
        await example_batch_scan()
        await example_performance_monitoring()
        await example_backtest_validation()
        await example_config_customization()
        example_data_format()
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行示例
    asyncio.run(run_all_examples())