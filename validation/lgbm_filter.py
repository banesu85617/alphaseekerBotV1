"""
LightGBM快速筛选器
实现第一层的快速分类筛选逻辑
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Using mock implementation.")

logger = logging.getLogger(__name__)


@dataclass
class LightGBMConfig:
    """LightGBM配置"""
    model_path: str = "models/lightgbm_model.txt"
    probability_threshold: float = 0.65
    confidence_threshold: float = 0.6
    require_direction: bool = True
    batch_size: int = 100
    prediction_timeout: float = 2.0
    
    # 特征工程配置
    feature_normalization: bool = True
    outlier_handling: bool = True
    missing_value_strategy: str = "median"


class LightGBMFilter:
    """
    LightGBM快速筛选器
    
    负责第一层的快速分类，输出标签和概率分布
    """
    
    def __init__(self, config: LightGBMConfig):
        """
        初始化LightGBM过滤器
        
        Args:
            config: LightGBM配置对象
        """
        self.config = config
        self.model = None
        self.feature_names = []
        self.is_initialized = False
        
        # 特征预处理器
        self.feature_preprocessors = {
            'normalization': self._normalize_features,
            'outlier_handling': self._handle_outliers,
            'missing_values': self._handle_missing_values
        }

    async def initialize(self) -> None:
        """初始化LightGBM模型"""
        logger.info("初始化LightGBM过滤器...")
        
        try:
            if LIGHTGBM_AVAILABLE:
                # 加载预训练模型
                self.model = lgb.Booster(model_file=self.config.model_path)
                self.feature_names = self.model.feature_name()
                logger.info(f"LightGBM模型加载成功，特征数量: {len(self.feature_names)}")
            else:
                # 使用模拟实现
                logger.warning("使用LightGBM模拟实现")
                self.model = MockLightGBMModel()
            
            self.is_initialized = True
            logger.info("LightGBM过滤器初始化完成")
            
        except Exception as e:
            logger.error(f"LightGBM初始化失败: {str(e)}")
            # 回退到模拟实现
            self.model = MockLightGBMModel()
            self.is_initialized = True
            logger.info("使用模拟LightGBM实现")

    async def validate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行验证（分类预测）
        
        Args:
            features: 特征字典
            
        Returns:
            预测结果: {'label': int, 'probability': float, 'confidence': float}
        """
        if not self.is_initialized:
            raise RuntimeError("LightGBM过滤器未初始化")
        
        start_time = time.time()
        
        try:
            # 特征预处理
            processed_features = await self._preprocess_features(features)
            
            # 转换为模型输入格式
            feature_vector = self._prepare_feature_vector(processed_features)
            
            # 执行预测
            prediction_result = await self._predict(feature_vector)
            
            processing_time = time.time() - start_time
            logger.debug(f"LightGBM预测完成，耗时: {processing_time:.3f}s")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"LightGBM验证失败: {str(e)}")
            # 返回安全的默认结果
            return {
                'label': 0,
                'probability': 0.0,
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    async def _preprocess_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """特征预处理"""
        processed = features.copy()
        
        # 处理数值特征
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                numeric_features[key] = value
        
        # 应用预处理管道
        if self.config.feature_normalization:
            processed.update(self.feature_preprocessors['normalization'](numeric_features))
        
        if self.config.outlier_handling:
            processed.update(self.feature_preprocessors['outlier_handling'](numeric_features))
        
        if self.config.missing_value_strategy:
            processed.update(self.feature_preprocessors['missing_values'](processed))
        
        return processed

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """特征标准化"""
        # 简化的标准化实现
        # 在实际应用中，应该使用训练数据统计信息
        normalized = {}
        for key, value in features.items():
            if abs(value) > 1e6:  # 大数值标准化
                normalized[f"{key}_normalized"] = np.log1p(abs(value)) * np.sign(value)
            else:
                normalized[key] = value
        
        return normalized

    def _handle_outliers(self, features: Dict[str, float]) -> Dict[str, float]:
        """异常值处理"""
        # 简化的异常值处理
        handled = {}
        for key, value in features.items():
            if abs(value) > 3 * np.std([value]):  # 简单的3-sigma规则
                handled[f"{key}_clipped"] = np.clip(value, -3, 3)
            else:
                handled[key] = value
        
        return handled

    def _handle_missing_values(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """缺失值处理"""
        handled = features.copy()
        
        for key, value in features.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if self.config.missing_value_strategy == "median":
                    # 使用中位数填充（这里简化为0）
                    handled[key] = 0.0
                elif self.config.missing_value_strategy == "mean":
                    # 使用均值填充
                    handled[key] = 0.0
                else:
                    handled[key] = 0.0
        
        return handled

    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """准备特征向量"""
        # 确保特征顺序与模型期望一致
        feature_vector = []
        
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)
            if isinstance(value, (int, float)):
                feature_vector.append(value)
            else:
                feature_vector.append(0.0)  # 默认值
        
        return np.array(feature_vector, dtype=np.float32)

    async def _predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """执行预测"""
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        # 检查预测超时
        start_time = time.time()
        
        if LIGHTGBM_AVAILABLE and hasattr(self.model, 'predict'):
            # 真实LightGBM预测
            prediction = self.model.predict(feature_vector.reshape(1, -1), 
                                          pred_leaf=False, 
                                          num_iteration=self.model.best_iteration)
            
            # 获取概率分布
            if len(prediction.shape) == 2:
                probabilities = prediction[0]  # 多类分类
            else:
                probabilities = prediction  # 二分类
            
            # 获取预测标签（概率最高的类别）
            predicted_class = int(np.argmax(probabilities))
            max_probability = float(np.max(probabilities))
            
            # 计算置信度（归一化的最大概率）
            confidence = max_probability / np.sum(probabilities) if np.sum(probabilities) > 0 else 0.0
            
        else:
            # 模拟预测结果
            prediction_result = self.model.predict(feature_vector)
            predicted_class = prediction_result['label']
            max_probability = prediction_result['probability']
            confidence = prediction_result['confidence']
        
        processing_time = time.time() - start_time
        
        return {
            'label': predicted_class,
            'probability': max_probability,
            'confidence': confidence,
            'processing_time': processing_time,
            'feature_importance': self._get_feature_importance() if LIGHTGBM_AVAILABLE else None
        }

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if not LIGHTGBM_AVAILABLE or self.model is None:
            return None
        
        try:
            importance = self.model.feature_importance(importance_type='gain')
            feature_importance = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importance)
            }
            return feature_importance
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {str(e)}")
            return None

    async def batch_validate(self, feature_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量验证"""
        logger.info(f"开始批量验证 {len(feature_list)} 个样本")
        
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(feature_list), batch_size):
            batch = feature_list[i:i + batch_size]
            
            # 并行处理批次
            tasks = [self.validate(features) for features in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"批量验证任务 {i+j} 异常: {str(result)}")
                    results.append({
                        'label': 0,
                        'probability': 0.0,
                        'confidence': 0.0,
                        'error': str(result)
                    })
                else:
                    results.append(result)
        
        logger.info(f"批量验证完成，处理 {len(results)} 个结果")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        info = {
            'status': 'ready',
            'feature_count': len(self.feature_names),
            'lightgbm_available': LIGHTGBM_AVAILABLE
        }
        
        if LIGHTGBM_AVAILABLE and self.model is not None:
            info.update({
                'model_type': 'lightgbm',
                'num_trees': getattr(self.model, 'num_trees', 'unknown'),
                'best_iteration': getattr(self.model, 'best_iteration', 'unknown')
            })
        else:
            info.update({
                'model_type': 'mock',
                'description': 'Mock implementation for development'
            })
        
        return info

    async def shutdown(self) -> None:
        """关闭过滤器"""
        logger.info("正在关闭LightGBM过滤器...")
        self.model = None
        self.is_initialized = False
        logger.info("LightGBM过滤器已关闭")


class MockLightGBMModel:
    """模拟LightGBM模型，用于开发测试"""
    
    def __init__(self):
        self.feature_count = 20  # 模拟特征数量
    
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """模拟预测"""
        # 基于特征向量的简单规则生成预测
        if len(feature_vector) == 0:
            return {
                'label': 0,
                'probability': 0.5,
                'confidence': 0.5
            }
        
        # 简单的启发式预测
        feature_sum = np.sum(feature_vector)
        
        if feature_sum > 0.5:
            label = 1  # 买入
            probability = min(0.9, 0.6 + abs(feature_sum) * 0.1)
        elif feature_sum < -0.5:
            label = -1  # 卖出
            probability = min(0.9, 0.6 + abs(feature_sum) * 0.1)
        else:
            label = 0  # 持有
            probability = 0.6
        
        confidence = min(0.95, probability * 0.9)
        
        return {
            'label': label,
            'probability': probability,
            'confidence': confidence
        }