"""
AlphaSeeker-Bot机器学习预测和信号生成模块
实现高性能模型推理和交易信号生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from pathlib import Path

from ..core.model import AlphaSeekerModel
from ..features.feature_engineer import MicrostructureFeatureEngineer
from ..config.settings import INFERENCE_CONFIG, TARGET_CONFIG


class FastInferenceEngine:
    """
    快速推理引擎
    
    优化目标：0.5秒内完成推理
    """
    
    def __init__(self, model: AlphaSeekerModel, feature_engineer: MicrostructureFeatureEngineer):
        """
        初始化推理引擎
        
        Args:
            model: 训练好的模型
            feature_engineer: 特征工程器
        """
        self.model = model
        self.feature_engineer = feature_engineer
        
        # 推理优化参数
        self.target_latency = INFERENCE_CONFIG["TARGET_LATENCY_MS"]
        self.enable_caching = INFERENCE_CONFIG["ENABLE_CACHING"]
        self.batch_size = INFERENCE_CONFIG["BATCH_SIZE"]
        
        # 缓存
        self.feature_cache = {} if self.enable_caching else None
        self.last_feature_vector = None
        self.prediction_cache = {}
        
        self.logger = logging.getLogger(__name__)
        
    def preprocess_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        快速特征预处理
        
        Args:
            raw_data: 原始市场数据
            
        Returns:
            预处理后的特征DataFrame
        """
        start_time = time.time()
        
        # 特征工程（简化版，只生成核心特征）
        processed_data = self.feature_engineer.process_features(raw_data)
        
        # 选择训练时的特征
        if self.model.feature_names:
            available_features = [f for f in self.model.feature_names if f in processed_data.columns]
            processed_data = processed_data[available_features]
        else:
            # 使用配置中的核心特征
            core_features = self.feature_engineer.config.get("CORE_FEATURES", [])
            available_features = [f for f in core_features if f in processed_data.columns]
            processed_data = processed_data[available_features]
            
        # 填充缺失值
        processed_data = processed_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        processing_time = (time.time() - start_time) * 1000
        self.logger.debug(f"特征预处理耗时: {processing_time:.2f}ms")
        
        return processed_data
    
    def predict_single(self, raw_data: Dict) -> Dict:
        """
        单样本预测
        
        Args:
            raw_data: 原始市场数据字典
            
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        # 转换为DataFrame
        df = pd.DataFrame([raw_data])
        
        # 特征预处理
        features = self.preprocess_features(df)
        
        # 预测
        prediction, probabilities = self.model.predict(features, return_proba=True)
        
        signal = prediction[0]  # 取第一个预测结果
        prob_array = probabilities[0]  # 对应的概率
        
        # 映射到标签
        label_mapping = {0: "HOLD", 1: "BUY", -1: "SELL"}
        signal_label = label_mapping.get(signal, "UNKNOWN")
        
        total_time = (time.time() - start_time) * 1000
        
        # 检查延迟目标
        if total_time > self.target_latency:
            self.logger.warning(f"推理延迟超出目标: {total_time:.2f}ms > {self.target_latency}ms")
        else:
            self.logger.debug(f"推理延迟: {total_time:.2f}ms")
            
        return {
            "signal": signal,
            "signal_label": signal_label,
            "probabilities": {
                "BUY": prob_array[1],    # 类别1对应买入
                "HOLD": prob_array[0],   # 类别0对应持有
                "SELL": prob_array[2]    # 类别2对应卖出
            },
            "confidence": max(prob_array),
            "latency_ms": total_time,
            "timestamp": time.time()
        }
    
    def predict_batch(self, raw_data_list: List[Dict], 
                     batch_size: int = None) -> List[Dict]:
        """
        批量预测
        
        Args:
            raw_data_list: 原始市场数据列表
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        results = []
        
        for i in range(0, len(raw_data_list), batch_size):
            batch = raw_data_list[i:i + batch_size]
            
            # 批量处理
            df_batch = pd.DataFrame(batch)
            features = self.preprocess_features(df_batch)
            
            # 批量预测
            predictions, probabilities = self.model.predict(features, return_proba=True)
            
            # 处理结果
            for j, (pred, probs) in enumerate(zip(predictions, probabilities)):
                signal = pred
                prob_array = probs
                
                label_mapping = {0: "HOLD", 1: "BUY", -1: "SELL"}
                signal_label = label_mapping.get(signal, "UNKNOWN")
                
                results.append({
                    "signal": signal,
                    "signal_label": signal_label,
                    "probabilities": {
                        "BUY": prob_array[1],
                        "HOLD": prob_array[0], 
                        "SELL": prob_array[2]
                    },
                    "confidence": max(prob_array),
                    "batch_index": i + j
                })
                
        return results


class SignalGenerator:
    """
    交易信号生成器
    
    基于模型预测和阈值规则生成交易信号
    """
    
    def __init__(self, inference_engine: FastInferenceEngine, 
                 config: Dict = None):
        """
        初始化信号生成器
        
        Args:
            inference_engine: 推理引擎
            config: 配置字典
        """
        self.inference_engine = inference_engine
        self.config = config or {}
        
        # 信号阈值配置
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.min_hold_probability = self.config.get("min_hold_probability", 0.4)
        
        # 信号历史
        self.signal_history = []
        self.last_signal_time = None
        
        self.logger = logging.getLogger(__name__)
        
    def generate_signal(self, market_data: Dict, 
                       position: str = "FLAT") -> Dict:
        """
        生成交易信号
        
        Args:
            market_data: 市场数据
            position: 当前仓位 ("LONG", "SHORT", "FLAT")
            
        Returns:
            交易信号字典
        """
        # 模型预测
        prediction = self.inference_engine.predict_single(market_data)
        
        # 信号过滤和增强
        signal = self._filter_signal(prediction, position)
        
        # 添加时间戳
        signal["timestamp"] = time.time()
        signal["market_data_timestamp"] = market_data.get("timestamp", time.time())
        
        # 更新信号历史
        self.signal_history.append(signal)
        self.last_signal_time = signal["timestamp"]
        
        # 限制历史记录长度
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
            
        return signal
    
    def _filter_signal(self, prediction: Dict, position: str) -> Dict:
        """
        信号过滤和增强
        
        Args:
            prediction: 模型预测结果
            position: 当前仓位
            
        Returns:
            过滤后的信号
        """
        signal = prediction.copy()
        
        # 基于置信度的信号过滤
        if prediction["confidence"] < self.confidence_threshold:
            # 置信度不够，保持持有
            signal["signal"] = 0
            signal["signal_label"] = "HOLD"
            signal["filter_reason"] = "low_confidence"
            
        elif position == "LONG" and prediction["signal"] == -1:
            # 当前多头仓位但预测卖出，可能触发平仓
            signal["signal"] = -1
            signal["signal_label"] = "CLOSE_LONG"
            
        elif position == "SHORT" and prediction["signal"] == 1:
            # 当前空头仓位但预测买入，可能触发平仓
            signal["signal"] = 1
            signal["signal_label"] = "CLOSE_SHORT"
            
        elif position == "FLAT":
            # 扁平仓位，可以开仓
            if prediction["signal"] == 0:
                signal["filter_reason"] = "flat_position_hold"
                
        # 计算信号强度
        probs = prediction["probabilities"]
        signal_strength = max(probs.values()) - min(probs.values())
        signal["signal_strength"] = signal_strength
        
        return signal
    
    def get_signal_statistics(self) -> Dict:
        """
        获取信号统计信息
        
        Returns:
            信号统计字典
        """
        if not self.signal_history:
            return {"error": "no_signal_history"}
            
        signals = [s["signal"] for s in self.signal_history]
        
        return {
            "total_signals": len(self.signal_history),
            "signal_distribution": {
                "BUY": signals.count(1),
                "SELL": signals.count(-1), 
                "HOLD": signals.count(0)
            },
            "avg_confidence": np.mean([s["confidence"] for s in self.signal_history]),
            "avg_latency_ms": np.mean([s["latency_ms"] for s in self.signal_history]),
            "last_signal_time": self.last_signal_time
        }


class MLInferenceManager:
    """
    机器学习推理管理器
    
    统一管理模型加载、推理和信号生成
    """
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        初始化推理管理器
        
        Args:
            model_path: 模型文件路径
            config: 配置字典
        """
        self.model_path = model_path
        self.config = config or {}
        
        # 组件
        self.model = None
        self.feature_engineer = None
        self.inference_engine = None
        self.signal_generator = None
        
        # 性能监控
        self.inference_times = []
        self.prediction_count = 0
        
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path: str = None) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载成功标志
        """
        try:
            if model_path is None:
                model_path = self.model_path
                
            if model_path is None:
                raise ValueError("未提供模型路径")
                
            # 加载模型数据
            import joblib
            model_data = joblib.load(model_path)
            
            # 创建模型实例
            self.model = AlphaSeekerModel()
            self.model.model = model_data["model"]
            self.model.feature_names = model_data["feature_names"]
            self.model.config = model_data.get("config", {})
            self.model.label_mapping = model_data.get("label_mapping", TARGET_CONFIG["LABELS"])
            self.model.is_fitted = True
            
            # 创建特征工程器
            self.feature_engineer = MicrostructureFeatureEngineer()
            
            # 创建推理引擎
            self.inference_engine = FastInferenceEngine(self.model, self.feature_engineer)
            
            # 创建信号生成器
            inference_config = self.config.get("inference", {})
            self.signal_generator = SignalGenerator(self.inference_engine, inference_config)
            
            self.logger.info(f"模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def predict_and_signal(self, market_data: Dict, position: str = "FLAT") -> Dict:
        """
        预测并生成信号
        
        Args:
            market_data: 市场数据
            position: 当前仓位
            
        Returns:
            预测和信号结果
        """
        start_time = time.time()
        
        try:
            if self.signal_generator is None:
                raise ValueError("模型尚未加载")
                
            # 生成信号
            result = self.signal_generator.generate_signal(market_data, position)
            
            # 更新统计信息
            total_time = (time.time() - start_time) * 1000
            self.inference_times.append(total_time)
            self.prediction_count += 1
            
            # 限制统计信息长度
            if len(self.inference_times) > 1000:
                self.inference_times = self.inference_times[-1000:]
                
            # 添加性能统计
            result["performance_stats"] = {
                "total_latency_ms": total_time,
                "avg_latency_ms": np.mean(self.inference_times[-100:]),  # 最近100次平均
                "prediction_count": self.prediction_count
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return {
                "error": str(e),
                "signal": 0,
                "signal_label": "ERROR",
                "latency_ms": (time.time() - start_time) * 1000
            }
    
    def get_performance_stats(self) -> Dict:
        """
        获取性能统计
        
        Returns:
            性能统计字典
        """
        if not self.inference_times:
            return {"error": "no_inference_data"}
            
        recent_times = self.inference_times[-100:]  # 最近100次
        
        return {
            "total_predictions": self.prediction_count,
            "avg_latency_ms": np.mean(recent_times),
            "median_latency_ms": np.median(recent_times),
            "p95_latency_ms": np.percentile(recent_times, 95),
            "p99_latency_ms": np.percentile(recent_times, 99),
            "min_latency_ms": np.min(recent_times),
            "max_latency_ms": np.max(recent_times),
            "target_latency_ms": INFERENCE_CONFIG["TARGET_LATENCY_MS"],
            "target_met": np.mean(recent_times) <= INFERENCE_CONFIG["TARGET_LATENCY_MS"]
        }
    
    def health_check(self) -> Dict:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        status = {
            "overall_status": "OK",
            "components": {},
            "timestamp": time.time()
        }
        
        # 检查模型
        if self.model is None or not self.model.is_fitted:
            status["overall_status"] = "ERROR"
            status["components"]["model"] = {"status": "ERROR", "message": "模型未加载"}
        else:
            status["components"]["model"] = {"status": "OK", "message": "模型正常"}
            
        # 检查特征工程器
        if self.feature_engineer is None:
            status["overall_status"] = "ERROR" if status["overall_status"] == "OK" else status["overall_status"]
            status["components"]["feature_engineer"] = {"status": "ERROR", "message": "特征工程器未初始化"}
        else:
            status["components"]["feature_engineer"] = {"status": "OK", "message": "特征工程器正常"}
            
        # 检查推理引擎
        if self.inference_engine is None:
            status["overall_status"] = "ERROR" if status["overall_status"] == "OK" else status["overall_status"]
            status["components"]["inference_engine"] = {"status": "ERROR", "message": "推理引擎未初始化"}
        else:
            status["components"]["inference_engine"] = {"status": "OK", "message": "推理引擎正常"}
            
        # 检查性能
        if self.inference_times:
            avg_latency = np.mean(self.inference_times[-10:])  # 最近10次
            if avg_latency > INFERENCE_CONFIG["TARGET_LATENCY_MS"]:
                status["components"]["performance"] = {
                    "status": "WARNING", 
                    "message": f"平均延迟超标: {avg_latency:.2f}ms"
                }
            else:
                status["components"]["performance"] = {
                    "status": "OK", 
                    "message": f"平均延迟正常: {avg_latency:.2f}ms"
                }
        else:
            status["components"]["performance"] = {"status": "UNKNOWN", "message": "无推理数据"}
            
        return status