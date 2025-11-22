"""
AlphaSeeker-Bot机器学习引擎示例脚本
演示完整的使用流程：数据预处理、模型训练、因子评估、推理和风险管理
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# 导入ML引擎
from ml_engine import create_ml_engine
from ml_engine.utils.helpers import Logger


def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    生成示例市场数据
    
    Args:
        n_samples: 样本数量
        
    Returns:
        示例市场数据DataFrame
    """
    print(f"生成 {n_samples} 个样本的示例数据...")
    
    # 时间序列
    timestamps = pd.date_range(
        start='2024-01-01', 
        periods=n_samples, 
        freq='1S'  # 每秒一个数据点
    )
    
    # 基础价格（模拟随机游走）
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.001, n_samples)
    close_prices = 50000 * np.exp(np.cumsum(price_changes))
    
    # 生成买卖价差
    base_spread = 0.01  # 1%的基准价差
    spread = base_spread * (1 + np.random.normal(0, 0.1, n_samples))
    spread = np.maximum(spread, 0.001)  # 最小价差
    
    # 买一和卖一价格
    bid_prices = close_prices - spread / 2
    ask_prices = close_prices + spread / 2
    
    # 成交量和订单簿深度
    base_volume = 100
    volume = base_volume * np.random.lognormal(0, 0.5, n_samples)
    
    # 订单簿深度（模拟买卖压力）
    bid_volume = volume * np.random.beta(2, 2, n_samples) * 2
    ask_volume = volume * np.random.beta(2, 2, n_samples) * 2
    
    # 高低价（基于价格变动）
    high_prices = np.maximum(close_prices, 
                           close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))))
    low_prices = np.minimum(close_prices,
                          close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))))
    
    # 组装DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'bid_volume': bid_volume,
        'ask_volume': ask_volume,
        'close': close_prices,
        'high': high_prices,
        'low': low_prices,
        'volume': volume
    })
    
    print(f"示例数据生成完成: {len(data)} 行")
    return data


def save_sample_data(data: pd.DataFrame, filepath: str):
    """
    保存示例数据
    
    Args:
        data: 市场数据
        filepath: 保存路径
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"示例数据已保存至: {filepath}")


def demo_feature_engineering(ml_engine, data):
    """
    演示特征工程功能
    
    Args:
        ml_engine: ML引擎实例
        data: 市场数据
    """
    print("\n" + "="*50)
    print("特征工程演示")
    print("="*50)
    
    # 特征工程
    from ml_engine.features.feature_engineer import MicrostructureFeatureEngineer
    
    feature_engineer = MicrostructureFeatureEngineer()
    features_df = feature_engineer.process_features(data)
    
    print(f"原始特征数量: {data.shape[1]}")
    print(f"工程后特征数量: {features_df.shape[1]}")
    
    # 显示部分特征
    print("\n部分生成特征:")
    feature_cols = [col for col in features_df.columns if not col.startswith('timestamp')]
    for col in feature_cols[:10]:
        print(f"  - {col}")
    
    if len(feature_cols) > 10:
        print(f"  ... 还有 {len(feature_cols) - 10} 个特征")


def demo_factor_evaluation(ml_engine, data):
    """
    演示因子评估功能
    
    Args:
        ml_engine: ML引擎实例
        data: 市场数据
    """
    print("\n" + "="*50)
    print("因子评估演示")
    print("="*50)
    
    # 执行因子评估
    factor_results = ml_engine.evaluate_factors(data)
    
    if factor_results["success"]:
        print("因子评估成功完成")
        print(f"评估因子数量: {len(factor_results['factor_results'])}")
        print(f"顶级因子: {factor_results['top_factors'][:5]}")
        
        # 显示评估报告
        print("\n评估报告摘要:")
        print(factor_results["report"][:300] + "...")
    else:
        print(f"因子评估失败: {factor_results['error']}")


def demo_model_training(ml_engine, data):
    """
    演示模型训练功能
    
    Args:
        ml_engine: ML引擎实例
        data: 市场数据
    """
    print("\n" + "="*50)
    print("模型训练演示")
    print("="*50)
    
    # 训练模型
    train_results = ml_engine.train_model(data, price_col="close")
    
    if train_results["success"]:
        print("模型训练成功完成!")
        print(f"模型保存路径: {train_results['model_path']}")
        
        # 显示训练结果
        results = train_results["results"]
        print(f"训练样本数: {results['train_size']}")
        print(f"验证样本数: {results['val_size']}")
        print(f"测试样本数: {results['test_size']}")
        print(f"特征数量: {len(results['feature_names'])}")
        
        # 显示模型性能
        eval_results = results["evaluation_results"]
        print(f"模型准确率: {eval_results.get('accuracy', 'N/A'):.4f}")
        print(f"加权F1分数: {eval_results.get('weighted_f1', 'N/A'):.4f}")
        
        return train_results["model_path"]
    else:
        print(f"模型训练失败: {train_results['error']}")
        return None


def demo_model_inference(ml_engine, model_path):
    """
    演示模型推理功能
    
    Args:
        ml_engine: ML引擎实例
        model_path: 模型路径
    """
    print("\n" + "="*50)
    print("模型推理演示")
    print("="*50)
    
    # 加载模型
    if ml_engine.load_model(model_path):
        print("模型加载成功")
    else:
        print("模型加载失败")
        return
    
    # 准备示例市场数据
    market_data = {
        "bid_price": 50000.0,
        "ask_price": 50001.0,
        "bid_volume": 15.5,
        "ask_volume": 12.3,
        "close": 50000.5,
        "volume": 150.0,
        "timestamp": time.time()
    }
    
    print("执行单次预测...")
    signal = ml_engine.predict(market_data)
    
    print(f"交易信号: {signal['signal_label']}")
    print(f"置信度: {signal['confidence']:.3f}")
    print(f"推理延迟: {signal['latency_ms']:.2f}ms")
    print(f"概率分布: {signal['probabilities']}")
    
    # 批量预测演示
    print("\n执行批量预测...")
    batch_data = []
    for i in range(10):
        data = market_data.copy()
        data["close"] += np.random.normal(0, 10)  # 添加随机变动
        data["timestamp"] = time.time()
        batch_data.append(data)
    
    batch_results = ml_engine.batch_predict(batch_data)
    print(f"批量预测完成: {len(batch_results)} 个结果")
    
    # 显示批量结果统计
    signals = [r.get('signal_label', 'ERROR') for r in batch_results]
    signal_counts = pd.Series(signals).value_counts()
    print(f"信号分布: {dict(signal_counts)}")


def demo_risk_management(ml_engine, model_path):
    """
    演示风险管理功能
    
    Args:
        ml_engine: ML引擎实例
        model_path: 模型路径
    """
    print("\n" + "="*50)
    print("风险管理演示")
    print("="*50)
    
    # 确保模型已加载
    if not ml_engine.inference_manager:
        ml_engine.load_model(model_path)
    
    # 模拟交易流程
    account_balance = 10000.0  # 1万USDT
    
    for i in range(5):
        print(f"\n--- 交易周期 {i+1} ---")
        
        # 模拟市场数据
        market_data = {
            "bid_price": 50000 + np.random.normal(0, 50),
            "ask_price": 50001 + np.random.normal(0, 50),
            "bid_volume": np.random.uniform(10, 20),
            "ask_volume": np.random.uniform(8, 18),
            "close": 50000 + np.random.normal(0, 50),
            "volume": np.random.uniform(100, 200),
            "volatility_20": np.random.uniform(0.01, 0.05),
            "timestamp": time.time()
        }
        
        # 预测信号
        signal = ml_engine.predict(market_data)
        print(f"预测信号: {signal['signal_label']} (置信度: {signal['confidence']:.3f})")
        
        # 风险管理
        risk_result = ml_engine.manage_risk(market_data, signal, account_balance)
        
        print(f"风控动作: {risk_result['action']}")
        if 'risk_level' in risk_result:
            print(f"风险等级: {risk_result['risk_level']}")
        if 'reason' in risk_result:
            print(f"处理原因: {risk_result['reason']}")
        
        # 更新账户余额（模拟）
        if risk_result['action'] == 'close_position':
            pnl = risk_result['result']['final_pnl']
            account_balance += pnl
            print(f"平仓盈亏: {pnl:.2f}, 账户余额: {account_balance:.2f}")
    
    # 显示最终风险状态
    final_risk_status = ml_engine.risk_manager.get_risk_status()
    print(f"\n最终风险状态:")
    print(f"  风险等级: {final_risk_status['risk_level']}")
    print(f"  当前回撤: {final_risk_status['metrics']['current_drawdown']:.4f}")
    print(f"  总盈亏: {final_risk_status['metrics']['total_pnl']:.2f}")


def demo_performance_monitoring(ml_engine):
    """
    演示性能监控功能
    
    Args:
        ml_engine: ML引擎实例
    """
    print("\n" + "="*50)
    print("性能监控演示")
    print("="*50)
    
    # 获取性能统计
    stats = ml_engine.get_performance_stats()
    
    print("推理性能统计:")
    if 'inference_manager' in stats:
        inf_stats = stats['inference_manager']
        print(f"  总预测次数: {inf_stats.get('total_predictions', 0)}")
        print(f"  平均延迟: {inf_stats.get('avg_latency_ms', 0):.2f}ms")
        print(f"  P95延迟: {inf_stats.get('p95_latency_ms', 0):.2f}ms")
        print(f"  P99延迟: {inf_stats.get('p99_latency_ms', 0):.2f}ms")
        print(f"  目标达成: {'是' if inf_stats.get('target_met') else '否'}")
    
    print("\n模型信息:")
    if 'model_info' in stats:
        model_info = stats['model_info']
        print(f"  状态: {model_info.get('status', 'unknown')}")
        print(f"  特征数量: {model_info.get('n_features', 0)}")
        print(f"  最佳迭代: {model_info.get('best_iteration', 'N/A')}")
    
    print("\n系统健康检查:")
    health = ml_engine.health_check()
    print(f"  总体状态: {health['overall_status']}")
    for component, status in health['components'].items():
        print(f"  {component}: {status['status']} - {status['message']}")


def main():
    """主演示函数"""
    print("AlphaSeeker-Bot机器学习引擎完整演示")
    print("="*60)
    
    # 设置日志
    Logger.setup_logger("demo", "INFO")
    
    # 创建ML引擎实例
    print("初始化ML引擎...")
    ml_engine = create_ml_engine(log_level="INFO")
    
    try:
        # 1. 生成和保存示例数据
        print("\n步骤1: 生成示例数据")
        sample_data = generate_sample_data(5000)  # 5000个样本
        data_path = "demo_data/sample_market_data.csv"
        save_sample_data(sample_data, data_path)
        
        # 2. 特征工程演示
        print("\n步骤2: 特征工程")
        demo_feature_engineering(ml_engine, sample_data)
        
        # 3. 因子评估演示
        print("\n步骤3: 因子评估")
        demo_factor_evaluation(ml_engine, sample_data)
        
        # 4. 模型训练演示
        print("\n步骤4: 模型训练")
        model_path = demo_model_training(ml_engine, sample_data)
        
        if model_path:
            # 5. 模型推理演示
            print("\n步骤5: 模型推理")
            demo_model_inference(ml_engine, model_path)
            
            # 6. 风险管理演示
            print("\n步骤6: 风险管理")
            demo_risk_management(ml_engine, model_path)
            
            # 7. 性能监控演示
            print("\n步骤7: 性能监控")
            demo_performance_monitoring(ml_engine)
            
            # 8. 导出模型摘要
            print("\n步骤8: 导出模型摘要")
            summary_path = "demo_data/model_summary.md"
            summary = ml_engine.export_model_summary(summary_path)
            print(f"模型摘要已保存至: {summary_path}")
            print("\n模型摘要预览:")
            print(summary[:500] + "...")
        
        print("\n" + "="*60)
        print("演示完成!")
        print("="*60)
        
        # 清理演示数据
        print("\n清理演示文件...")
        import shutil
        if Path("demo_data").exists():
            shutil.rmtree("demo_data")
        print("演示文件清理完成")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()