"""
AlphaSeeker-Bot机器学习引擎主入口
统一的ML引擎API接口，整合所有功能模块
"""

import logging
import time
from typing import Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

try:
    # 相对导入（当作为包使用时）
    from .core.model import AlphaSeekerModel
    from .features.feature_engineer import MicrostructureFeatureEngineer
    from .features.factor_evaluator import AlphaFactorEvaluator
    from .training.pipeline import MLTrainingPipeline
    from .prediction.inference import MLInferenceManager
    from .risk.manager import RiskManager
    from .utils.helpers import (
        DataProcessor, PerformanceMonitor, ModelValidator, 
        ConfigManager, Logger, save_model_metadata, format_model_performance_metrics
    )
    from .config.settings import *
except ImportError:
    # 绝对导入（当直接运行时）
    from core.model import AlphaSeekerModel
    from features.feature_engineer import MicrostructureFeatureEngineer
    from features.factor_evaluator import AlphaFactorEvaluator
    from training.pipeline import MLTrainingPipeline
    from prediction.inference import MLInferenceManager
    from risk.manager import RiskManager
    from utils.helpers import (
        DataProcessor, PerformanceMonitor, ModelValidator, 
        ConfigManager, Logger, save_model_metadata, format_model_performance_metrics
    )
    from config.settings import *


class AlphaSeekerMLEngine:
    """
    AlphaSeeker-Bot机器学习引擎主类
    
    提供完整的机器学习功能接口：
    - 数据预处理和特征工程
    - 因子评估和分级
    - 模型训练和验证
    - 高性能推理和信号生成
    - 风险管理和仓位控制
    - 性能监控和模型维护
    """
    
    def __init__(self, config_path: str = None, log_level: str = "INFO"):
        """
        初始化ML引擎
        
        Args:
            config_path: 配置文件路径
            log_level: 日志级别
        """
        # 配置管理
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # 日志设置
        self.logger = Logger.setup_logger(
            "alpha_seeker_ml", 
            log_level,
            "logs/ml_engine.log"
        )
        
        # 组件初始化
        self.feature_engineer = MicrostructureFeatureEngineer()
        self.factor_evaluator = AlphaFactorEvaluator()
        self.training_pipeline = MLTrainingPipeline()
        self.risk_manager = RiskManager()
        self.performance_monitor = PerformanceMonitor()
        self.data_processor = DataProcessor()
        
        # 推理管理器和模型
        self.inference_manager = None
        self.model = None
        
        # 状态跟踪
        self.is_initialized = False
        self.model_path = None
        self.last_training_time = None
        
        self.logger.info("AlphaSeeker ML Engine 已初始化")
    
    def load_data(self, data_path: str) -> bool:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            加载成功标志
        """
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
                df = pd.read_pickle(data_path)
            else:
                self.logger.error(f"不支持的文件格式: {data_path}")
                return False
                
            # 数据质量检查
            quality_report = self.data_processor.validate_data_quality(df)
            self.logger.info(f"数据质量分数: {quality_report['quality_score']:.3f}")
            
            if quality_report['quality_score'] < 0.8:
                self.logger.warning("数据质量较低，建议检查数据源")
                
            self.logger.info(f"数据加载成功: {len(df)} 行, {len(df.columns)} 列")
            return True
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def preprocess_data(self, raw_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            raw_data: 原始数据（文件路径或DataFrame）
            
        Returns:
            预处理后的数据
        """
        try:
            # 加载数据
            if isinstance(raw_data, str):
                if not self.load_data(raw_data):
                    raise ValueError("数据加载失败")
                return raw_data
            elif isinstance(raw_data, pd.DataFrame):
                df = raw_data.copy()
            else:
                raise ValueError("不支持的数据类型")
                
            # 清洗数据
            df_clean = self.data_processor.clean_market_data(df)
            
            # 格式化数据
            df_formatted = self.data_processor.format_market_data(df_clean)
            
            self.logger.info(f"数据预处理完成: {len(df_formatted)} 行")
            return df_formatted
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def train_model(self, data: Union[str, pd.DataFrame], 
                   price_col: str = 'close') -> Dict:
        """
        训练模型
        
        Args:
            data: 训练数据（文件路径或DataFrame）
            price_col: 价格列名
            
        Returns:
            训练结果字典
        """
        try:
            self.logger.info("开始模型训练")
            
            # 数据预处理
            if isinstance(data, str):
                df = self.preprocess_data(data)
            else:
                df = self.data_processor.clean_market_data(data)
                
            # 运行完整训练流水线
            results = self.training_pipeline.run_complete_pipeline(df, price_col)
            
            if results["success"]:
                # 保存模型
                model_path = f"models/trading_model_{int(time.time())}.joblib"
                Path("models").mkdir(exist_ok=True)
                
                self.training_pipeline.save_pipeline_results(results, model_path)
                save_model_metadata(model_path, {
                    "training_results": results,
                    "feature_names": results["feature_names"],
                    "model_config": results["model_info"]
                })
                
                # 更新状态
                self.model = results["model"]
                self.model_path = model_path
                self.last_training_time = time.time()
                
                # 生成性能报告
                if results["evaluation_results"]:
                    performance_report = format_model_performance_metrics(results["evaluation_results"])
                    self.logger.info("=== 模型性能报告 ===\n" + performance_report)
                
                self.logger.info(f"模型训练完成，已保存至: {model_path}")
                return {
                    "success": True,
                    "model_path": model_path,
                    "results": results
                }
            else:
                self.logger.error(f"模型训练失败: {results.get('error', '未知错误')}")
                return {
                    "success": False,
                    "error": results.get('error', '未知错误')
                }
                
        except Exception as e:
            self.logger.error(f"模型训练异常: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载成功标志
        """
        try:
            # 创建推理管理器
            self.inference_manager = MLInferenceManager(model_path, self.config)
            
            # 加载模型
            if self.inference_manager.load_model(model_path):
                self.model_path = model_path
                self.is_initialized = True
                
                self.logger.info(f"模型加载成功: {model_path}")
                return True
            else:
                self.logger.error(f"模型加载失败: {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"模型加载异常: {str(e)}")
            return False
    
    def predict(self, market_data: Dict, position: str = "FLAT") -> Dict:
        """
        预测交易信号
        
        Args:
            market_data: 市场数据字典
            position: 当前仓位
            
        Returns:
            预测结果字典
        """
        try:
            if not self.is_initialized or self.inference_manager is None:
                raise ValueError("模型尚未加载，请先调用load_model()")
                
            # 预测并生成信号
            result = self.inference_manager.predict_and_signal(market_data, position)
            
            # 记录性能指标
            if "performance_stats" in result:
                stats = result["performance_stats"]
                self.performance_monitor.record_prediction(
                    latency_ms=stats["total_latency_ms"],
                    confidence=result["confidence"],
                    signal_strength=result.get("signal_strength", 0),
                    prediction_time=result["timestamp"]
                )
                
            return result
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return {
                "error": str(e),
                "signal": 0,
                "signal_label": "ERROR"
            }
    
    def batch_predict(self, market_data_list: List[Dict]) -> List[Dict]:
        """
        批量预测
        
        Args:
            market_data_list: 市场数据列表
            
        Returns:
            批量预测结果列表
        """
        try:
            if not self.is_initialized or self.inference_manager is None:
                raise ValueError("模型尚未加载")
                
            results = []
            for i, market_data in enumerate(market_data_list):
                result = self.predict(market_data)
                result["batch_index"] = i
                results.append(result)
                
            return results
            
        except Exception as e:
            self.logger.error(f"批量预测失败: {str(e)}")
            return [{"error": str(e)} for _ in market_data_list]
    
    def manage_risk(self, market_data: Dict, signal: Dict, 
                   account_balance: float) -> Dict:
        """
        风险管理
        
        Args:
            market_data: 市场数据
            signal: 信号信息
            account_balance: 账户余额
            
        Returns:
            风险管理结果
        """
        try:
            current_price = market_data.get('close', market_data.get('last_price'))
            if current_price is None:
                raise ValueError("无法获取当前价格")
                
            # 更新风险指标
            current_equity = account_balance + self.risk_manager.risk_metrics.unrealized_pnl
            self.risk_manager.update_risk_metrics(current_equity)
            
            # 仓位管理
            if signal.get("signal") != 0:  # 有交易信号
                # 计算仓位大小
                position_size = self.risk_manager.calculate_position_size(
                    signal_strength=signal.get("confidence", 0.5),
                    account_balance=account_balance,
                    price=current_price,
                    volatility=market_data.get("volatility_20")
                )
                
                # 开仓检查
                can_open = self.risk_manager.can_open_position(
                    signal=signal["signal"],
                    current_price=current_price,
                    account_balance=account_balance,
                    position_size=position_size
                )
                
                if can_open["can_open"]:
                    # 执行开仓
                    open_result = self.risk_manager.open_position(
                        signal=signal["signal"],
                        current_price=current_price,
                        position_size=position_size,
                        account_balance=account_balance,
                        current_time=time.time(),
                        volatility=market_data.get("volatility_20")
                    )
                    
                    return {
                        "action": "open_position",
                        "result": open_result,
                        "risk_status": self.risk_manager.get_risk_status()
                    }
                else:
                    return {
                        "action": "reject_signal",
                        "reason": can_open["reason"],
                        "risk_level": can_open["risk_level"]
                    }
            else:
                # 检查平仓条件
                if self.risk_manager.current_position:
                    update_result = self.risk_manager.update_position(
                        current_price=current_price,
                        current_time=time.time(),
                        market_data=market_data
                    )
                    
                    if update_result["exit_triggered"]:
                        # 执行平仓
                        close_result = self.risk_manager.close_position(
                            exit_price=current_price,
                            current_time=time.time(),
                            exit_reason=update_result["exit_reason"]
                        )
                        
                        return {
                            "action": "close_position",
                            "result": close_result,
                            "risk_status": self.risk_manager.get_risk_status()
                        }
                        
            # 返回当前风险状态
            return {
                "action": "monitor",
                "risk_status": self.risk_manager.get_risk_status(),
                "position": self.risk_manager.current_position
            }
            
        except Exception as e:
            self.logger.error(f"风险管理失败: {str(e)}")
            return {
                "error": str(e),
                "action": "error"
            }
    
    def get_performance_stats(self) -> Dict:
        """
        获取性能统计
        
        Returns:
            性能统计字典
        """
        stats = {
            "inference_performance": self.performance_monitor.get_performance_summary(),
            "model_info": self.model.get_model_info() if self.model else {},
            "risk_metrics": self.risk_manager.get_risk_status()
        }
        
        # 添加推理管理器统计
        if self.inference_manager:
            stats["inference_manager"] = self.inference_manager.get_performance_stats()
            
        return stats
    
    def evaluate_factors(self, data: Union[str, pd.DataFrame]) -> Dict:
        """
        评估因子
        
        Args:
            data: 数据（文件路径或DataFrame）
            
        Returns:
            因子评估结果
        """
        try:
            # 数据预处理
            if isinstance(data, str):
                df = self.preprocess_data(data)
            else:
                df = self.data_processor.clean_market_data(data)
                
            # 特征工程
            features_df = self.feature_engineer.process_features(df)
            
            # 选择核心特征
            features_selected = self.feature_engineer.select_features(features_df)
            
            # 计算未来收益
            future_returns = df['close'].pct_change().shift(-300)  # 假设5分钟horizon
            
            # 评估因子
            factor_results = self.factor_evaluator.evaluate_multiple_factors(
                features_selected, future_returns
            )
            
            # 生成报告
            report = self.factor_evaluator.generate_factor_report(factor_results)
            
            self.logger.info("因子评估完成")
            
            return {
                "success": True,
                "factor_results": factor_results.to_dict('records'),
                "report": report,
                "top_factors": self.factor_evaluator.select_top_factors(
                    factor_results, top_n=10
                )
            }
            
        except Exception as e:
            self.logger.error(f"因子评估失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def health_check(self) -> Dict:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        health_status = {
            "overall_status": "OK",
            "components": {},
            "timestamp": time.time()
        }
        
        # 检查模型
        if self.model is None:
            health_status["overall_status"] = "WARNING"
            health_status["components"]["model"] = {
                "status": "NOT_LOADED", 
                "message": "模型未加载"
            }
        else:
            health_status["components"]["model"] = {
                "status": "OK", 
                "message": "模型已加载"
            }
            
        # 检查推理管理器
        if self.inference_manager is None:
            health_status["overall_status"] = "WARNING"
            health_status["components"]["inference_manager"] = {
                "status": "NOT_INITIALIZED", 
                "message": "推理管理器未初始化"
            }
        else:
            health_status["components"]["inference_manager"] = {
                "status": "OK", 
                "message": "推理管理器正常"
            }
            
        # 检查性能指标
        if len(self.performance_monitor.latency_history) > 0:
            recent_avg_latency = sum(self.performance_monitor.latency_history[-10:]) / 10
            if recent_avg_latency > INFERENCE_CONFIG["TARGET_LATENCY_MS"]:
                health_status["components"]["performance"] = {
                    "status": "WARNING", 
                    "message": f"平均延迟超标: {recent_avg_latency:.2f}ms"
                }
            else:
                health_status["components"]["performance"] = {
                    "status": "OK", 
                    "message": f"性能正常: {recent_avg_latency:.2f}ms"
                }
        else:
            health_status["components"]["performance"] = {
                "status": "UNKNOWN", 
                "message": "无性能数据"
            }
            
        return health_status
    
    def update_config(self, new_config: Dict):
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        self.config_manager.config = self.config
        
        # 重新初始化相关组件
        self.risk_manager = RiskManager(self.config.get("risk_config"))
        
        self.logger.info("配置已更新")
    
    def export_model_summary(self, output_path: str = None) -> str:
        """
        导出模型摘要
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            模型摘要字符串
        """
        summary = []
        summary.append("# AlphaSeeker-Bot ML Engine 模型摘要\n")
        
        # 基本信息
        summary.append("## 基本信息")
        summary.append(f"- 模型路径: {self.model_path or '未加载'}")
        summary.append(f"- 最后训练时间: {self.last_training_time or '未知'}")
        summary.append(f"- 初始化状态: {self.is_initialized}")
        summary.append("")
        
        # 性能统计
        stats = self.get_performance_stats()
        summary.append("## 性能统计")
        
        if "inference_manager" in stats:
            inf_stats = stats["inference_manager"]
            summary.append(f"- 总预测次数: {inf_stats.get('total_predictions', 'N/A')}")
            summary.append(f"- 平均延迟: {inf_stats.get('avg_latency_ms', 'N/A'):.2f}ms")
            summary.append(f"- P95延迟: {inf_stats.get('p95_latency_ms', 'N/A'):.2f}ms")
            summary.append(f"- 目标延迟达成: {inf_stats.get('target_met', 'N/A')}")
            
        summary.append("")
        
        # 风险状态
        if "risk_metrics" in stats:
            risk_status = stats["risk_metrics"]
            summary.append("## 风险状态")
            summary.append(f"- 风险等级: {risk_status.get('risk_level', 'N/A')}")
            summary.append(f"- 当前回撤: {risk_status.get('metrics', {}).get('current_drawdown', 0):.4f}")
            summary.append(f"- 最大回撤: {risk_status.get('metrics', {}).get('max_drawdown', 0):.4f}")
            summary.append(f"- 胜率: {risk_status.get('metrics', {}).get('win_rate', 0):.4f}")
            
        summary_text = "\n".join(summary)
        
        # 保存到文件
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
                
        return summary_text


# 便利函数
def create_ml_engine(config_path: str = None, log_level: str = "INFO") -> AlphaSeekerMLEngine:
    """
    创建ML引擎实例
    
    Args:
        config_path: 配置文件路径
        log_level: 日志级别
        
    Returns:
        ML引擎实例
    """
    return AlphaSeekerMLEngine(config_path, log_level)


# 示例使用
if __name__ == "__main__":
    # 创建ML引擎
    ml_engine = create_ml_engine(log_level="DEBUG")
    
    # 健康检查
    health = ml_engine.health_check()
    print(f"健康状态: {health['overall_status']}")
    
    # 加载示例数据（如果有）
    # ml_engine.train_model("sample_data.csv")
    # ml_engine.load_model("models/trading_model.joblib")
    
    # 示例预测
    # market_data = {
    #     "bid_price": 50000,
    #     "ask_price": 50001,
    #     "bid_volume": 10,
    #     "ask_volume": 8,
    #     "close": 50000.5,
    #     "volume": 100,
    #     "timestamp": time.time()
    # }
    # 
    # signal = ml_engine.predict(market_data)
    # print(f"交易信号: {signal['signal_label']}, 置信度: {signal['confidence']:.3f}")
    
    print("AlphaSeeker ML Engine 演示完成")