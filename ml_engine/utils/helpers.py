"""
AlphaSeeker-Bot机器学习工具函数模块
提供数据处理、性能监控和通用工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import (
    MONITORING_CONFIG, MODEL_CONFIG, TARGET_CONFIG, 
    FEATURE_CONFIG, FACTOR_CONFIG, RISK_CONFIG,
    INFERENCE_CONFIG, DATA_CONFIG
)


class DataProcessor:
    """
    数据处理器
    
    处理市场数据清洗、格式化和质量检查
    """
    
    @staticmethod
    def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗市场数据
        
        Args:
            df: 原始市场数据
            
        Returns:
            清洗后的数据
        """
        df = df.copy()
        
        # 确保时间列存在并格式化
        time_columns = ['timestamp', 'timestamp_utc', 'time']
        time_col = None
        
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break
                
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            
        # 数值列处理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # 处理无穷大值
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # 异常值检测和处理
            if df[col].notna().sum() > 0:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q99)
                
        # 缺失值处理
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 去重
        if time_col:
            df = df.drop_duplicates(subset=[time_col]).sort_values(time_col)
            
        return df
    
    @staticmethod
    def format_market_data(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
        """
        格式化市场数据到标准格式
        
        Args:
            df: 输入数据
            required_columns: 必需列列表
            
        Returns:
            格式化后的数据
        """
        df = df.copy()
        
        # 标准列映射
        column_mapping = {
            'bid_price': 'bid_price',
            'ask_price': 'ask_price', 
            'bid_volume': 'bid_volume',
            'ask_volume': 'ask_volume',
            'last_price': 'last_price',
            'volume': 'volume',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'open': 'open'
        }
        
        # 重命名列
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
                
        # 确保必需的列存在
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logging.warning(f"缺失必需列: {missing_cols}")
                
        return df
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict:
        """
        验证数据质量
        
        Args:
            df: 输入数据
            
        Returns:
            数据质量报告
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "datetime_columns": list(df.select_dtypes(include=['datetime64', 'object']).columns),
            "data_types": df.dtypes.to_dict(),
            "quality_score": 1.0
        }
        
        # 计算质量分数
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_cells = df.duplicated().sum()
        
        if total_cells > 0:
            quality_score = 1.0 - (missing_cells + duplicate_cells) / total_cells
            report["quality_score"] = max(0.0, quality_score)
            
        # 检查异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outlier_count = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
                if outlier_count > 0:
                    outlier_cols.append(col)
                    
        report["outlier_columns"] = outlier_cols
        
        return report


class PerformanceMonitor:
    """
    性能监控器
    
    监控ML模型的推理性能、延迟和准确性
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化性能监控器
        
        Args:
            config: 监控配置
        """
        self.config = config or MONITORING_CONFIG
        self.metrics_history = []
        self.latency_history = []
        self.accuracy_history = []
        self.prediction_count = 0
        
        # 重训触发条件
        self.retrain_triggers = self.config.get("RETRAIN_TRIGGERS", {})
        
        self.logger = logging.getLogger(__name__)
        
    def record_prediction(self, latency_ms: float, confidence: float, 
                         signal_strength: float, prediction_time: float):
        """
        记录预测信息
        
        Args:
            latency_ms: 推理延迟(毫秒)
            confidence: 预测置信度
            signal_strength: 信号强度
            prediction_time: 预测时间戳
        """
        metric = {
            "timestamp": prediction_time,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "signal_strength": signal_strength
        }
        
        self.metrics_history.append(metric)
        self.latency_history.append(latency_ms)
        self.prediction_count += 1
        
        # 限制历史记录长度
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]
        if len(self.latency_history) > 10000:
            self.latency_history = self.latency_history[-5000:]
            
    def record_accuracy(self, accuracy: float, timestamp: float):
        """
        记录准确性指标
        
        Args:
            accuracy: 准确性值
            timestamp: 时间戳
        """
        self.accuracy_history.append({
            "timestamp": timestamp,
            "accuracy": accuracy
        })
        
        # 限制历史记录长度
        if len(self.accuracy_history) > 1000:
            self.accuracy_history = self.accuracy_history[-500:]
            
    def get_latency_stats(self, window_size: int = 100) -> Dict:
        """
        获取延迟统计
        
        Args:
            window_size: 统计窗口大小
            
        Returns:
            延迟统计字典
        """
        recent_latencies = self.latency_history[-window_size:]
        
        if not recent_latencies:
            return {"error": "no_latency_data"}
            
        return {
            "count": len(recent_latencies),
            "mean_ms": np.mean(recent_latencies),
            "median_ms": np.median(recent_latencies),
            "std_ms": np.std(recent_latencies),
            "min_ms": np.min(recent_latencies),
            "max_ms": np.max(recent_latencies),
            "p95_ms": np.percentile(recent_latencies, 95),
            "p99_ms": np.percentile(recent_latencies, 99),
            "target_ms": self.config.get("TARGET_LATENCY_MS", 500),
            "target_met": np.mean(recent_latencies) <= self.config.get("TARGET_LATENCY_MS", 500)
        }
        
    def get_performance_summary(self) -> Dict:
        """
        获取性能摘要
        
        Returns:
            性能摘要字典
        """
        if not self.metrics_history:
            return {"error": "no_performance_data"}
            
        recent_metrics = self.metrics_history[-1000:]  # 最近1000次
        
        return {
            "total_predictions": self.prediction_count,
            "latency_stats": self.get_latency_stats(),
            "confidence_stats": {
                "mean": np.mean([m["confidence"] for m in recent_metrics]),
                "median": np.median([m["confidence"] for m in recent_metrics]),
                "min": np.min([m["confidence"] for m in recent_metrics]),
                "max": np.max([m["confidence"] for m in recent_metrics])
            },
            "signal_strength_stats": {
                "mean": np.mean([m["signal_strength"] for m in recent_metrics]),
                "median": np.median([m["signal_strength"] for m in recent_metrics])
            }
        }
    
    def check_retrain_conditions(self) -> Dict:
        """
        检查重训条件
        
        Returns:
            重训条件检查结果
        """
        should_retrain = False
        reasons = []
        
        # 时间触发
        if self.retrain_triggers.get("days"):
            last_prediction_time = self.metrics_history[-1]["timestamp"] if self.metrics_history else 0
            days_since_last = (time.time() - last_prediction_time) / (24 * 3600)
            if days_since_last >= self.retrain_triggers["days"]:
                should_retrain = True
                reasons.append(f"时间触发: {days_since_last:.1f}天未重训")
                
        # IC阈值触发
        if self.retrain_triggers.get("ic_threshold") and self.accuracy_history:
            recent_ic = np.mean([acc["accuracy"] for acc in self.accuracy_history[-10:]])
            if recent_ic < self.retrain_triggers["ic_threshold"]:
                should_retrain = True
                reasons.append(f"IC阈值触发: {recent_ic:.4f} < {self.retrain_triggers['ic_threshold']}")
                
        # 回撤触发
        if self.retrain_triggers.get("drawdown_threshold"):
            # 这里需要从外部传入当前回撤值
            pass  # 简化处理
            
        return {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "trigger_time": time.time()
        }


class ModelValidator:
    """
    模型验证器
    
    验证模型的有效性和稳定性
    """
    
    @staticmethod
    def validate_model_predictions(predictions: np.ndarray, 
                                 actual_labels: np.ndarray) -> Dict:
        """
        验证模型预测
        
        Args:
            predictions: 预测值
            actual_labels: 实际标签
            
        Returns:
            验证结果字典
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(actual_labels, predictions)
        report = classification_report(actual_labels, predictions, output_dict=True)
        cm = confusion_matrix(actual_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "prediction_distribution": {
                "predicted_counts": np.unique(predictions, return_counts=True)[1].tolist(),
                "actual_counts": np.unique(actual_labels, return_counts=True)[1].tolist()
            }
        }
    
    @staticmethod
    def validate_feature_consistency(train_features: pd.DataFrame,
                                   inference_features: pd.DataFrame) -> Dict:
        """
        验证特征一致性
        
        Args:
            train_features: 训练特征
            inference_features: 推理特征
            
        Returns:
            一致性验证结果
        """
        train_cols = set(train_features.columns)
        inference_cols = set(inference_features.columns)
        
        missing_features = train_cols - inference_cols
        extra_features = inference_cols - train_cols
        
        return {
            "consistent": len(missing_features) == 0 and len(extra_features) == 0,
            "missing_features": list(missing_features),
            "extra_features": list(extra_features),
            "common_features": list(train_cols & inference_cols),
            "train_feature_count": len(train_cols),
            "inference_feature_count": len(inference_cols)
        }


class ConfigManager:
    """
    配置管理器
    
    管理和加载ML引擎配置
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        
    def load_config(self, config_dict: Dict = None) -> Dict:
        """
        加载配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            加载的配置
        """
        if config_dict:
            self.config = config_dict
        elif self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # 使用默认配置
            self.config = {
                "model_config": MODEL_CONFIG,
                "target_config": TARGET_CONFIG,
                "feature_config": FEATURE_CONFIG,
                "risk_config": RISK_CONFIG,
                "inference_config": INFERENCE_CONFIG,
                "data_config": DATA_CONFIG,
                "monitoring_config": MONITORING_CONFIG
            }
            
        return self.config
    
    def save_config(self, config_path: str = None):
        """
        保存配置
        
        Args:
            config_path: 保存路径
        """
        save_path = config_path or self.config_path
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
                
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value


class Logger:
    """
    日志管理器
    
    提供统一的日志配置和管理
    """
    
    @staticmethod
    def setup_logger(name: str = "ml_engine", 
                    level: str = "INFO",
                    log_file: str = None) -> logging.Logger:
        """
        设置日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别
            log_file: 日志文件路径
            
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果指定）
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    @staticmethod
    def log_model_info(logger: logging.Logger, model_info: Dict):
        """
        记录模型信息
        
        Args:
            logger: 日志记录器
            model_info: 模型信息字典
        """
        logger.info("=== 模型信息 ===")
        for key, value in model_info.items():
            if key != "feature_names":  # 特征名可能很长
                logger.info(f"{key}: {value}")
                
        if "feature_names" in model_info:
            feature_count = len(model_info["feature_names"])
            logger.info(f"特征数量: {feature_count}")
            
    @staticmethod
    def log_training_summary(logger: logging.Logger, results: Dict):
        """
        记录训练摘要
        
        Args:
            logger: 日志记录器
            results: 训练结果字典
        """
        logger.info("=== 训练摘要 ===")
        
        if "success" in results:
            logger.info(f"训练成功: {results['success']}")
            
        if "evaluation_results" in results:
            eval_results = results["evaluation_results"]
            logger.info(f"准确率: {eval_results.get('accuracy', 'N/A'):.4f}")
            logger.info(f"F1分数: {eval_results.get('weighted_f1', 'N/A'):.4f}")
            
        if "model_info" in results:
            model_info = results["model_info"]
            logger.info(f"最佳迭代数: {model_info.get('best_iteration', 'N/A')}")
            logger.info(f"树的数量: {model_info.get('num_trees', 'N/A')}")


def save_model_metadata(model_path: str, metadata: Dict):
    """
    保存模型元数据
    
    Args:
        model_path: 模型路径
        metadata: 元数据字典
    """
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    
    # 添加保存时间
    metadata['saved_at'] = time.time()
    metadata['ml_engine_version'] = '1.0.0'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
        
    logging.info(f"模型元数据已保存: {metadata_path}")


def load_model_metadata(model_path: str) -> Dict:
    """
    加载模型元数据
    
    Args:
        model_path: 模型路径
        
    Returns:
        元数据字典
    """
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        logging.warning(f"模型元数据文件不存在: {metadata_path}")
        return {}


def format_model_performance_metrics(metrics: Dict) -> str:
    """
    格式化模型性能指标
    
    Args:
        metrics: 性能指标字典
        
    Returns:
        格式化的性能报告字符串
    """
    report = []
    
    report.append("# 模型性能报告\n")
    
    # 总体指标
    if "accuracy" in metrics:
        report.append(f"## 总体指标")
        report.append(f"- 准确率: {metrics['accuracy']:.4f}")
        report.append(f"- F1分数: {metrics.get('weighted_f1', 'N/A'):.4f}")
        report.append("")
        
    # 分类报告
    if "classification_report" in metrics:
        report.append("## 分类报告")
        report.append("```")
        
        for class_name, metrics_data in metrics["classification_report"].items():
            if isinstance(metrics_data, dict):
                report.append(f"{class_name}: precision={metrics_data['precision']:.4f}, "
                            f"recall={metrics_data['recall']:.4f}, f1={metrics_data['f1-score']:.4f}")
        
        report.append("```\n")
        
    # 混淆矩阵
    if "confusion_matrix" in metrics:
        report.append("## 混淆矩阵")
        report.append("```")
        cm = metrics["confusion_matrix"]
        for row in cm:
            report.append(" ".join(f"{val:6d}" for val in row))
        report.append("```")
        
    return "\n".join(report)