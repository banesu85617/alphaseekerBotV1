"""
AlphaSeeker-Bot机器学习引擎核心LightGBM模型
基于分析文档中的模型架构和训练流程
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from pathlib import Path

from ..config.settings import MODEL_CONFIG, TARGET_CONFIG, INFERENCE_CONFIG


class AlphaSeekerModel:
    """
    AlphaSeeker-Bot LightGBM多分类模型
    
    基于分析文档实现：
    - 三分类任务(买入/持有/卖出)
    - 类别平衡处理
    - 时间序列切分
    - 特征重要性分析
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化模型
        
        Args:
            config: 模型配置字典，默认使用MODEL_CONFIG
        """
        self.config = config or MODEL_CONFIG
        self.model = None
        self.feature_names = None
        self.label_mapping = TARGET_CONFIG["LABELS"]
        self.is_fitted = False
        
        # 推理优化
        self._setup_inference_config()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def _setup_inference_config(self):
        """设置推理性能优化配置"""
        # 优化LightGBM推理参数
        self.inference_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'num_leaves': 15,  # 减少叶子数以提升推理速度
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'max_depth': 4,  # 减少深度
            'verbose': -1,
            'n_jobs': INFERENCE_CONFIG["NUM_THREADS"],
            'boost_from_average': True,
            'force_col_wise': True,  # 加速训练
        }
    
    def _prepare_target(self, future_returns: pd.Series) -> pd.Series:
        """
        基于未来收益构造目标变量
        
        Args:
            future_returns: 未来收益序列
            
        Returns:
            三分类标签序列 (1:买入, 0:持有, -1:卖出)
        """
        threshold = TARGET_CONFIG["PROFIT_THRESHOLD"]
        
        # 基于阈值构造三分类标签
        labels = pd.Series(0, index=future_returns.index)  # 默认持有
        
        labels[future_returns > threshold] = 1   # 买入
        labels[future_returns < -threshold] = -1  # 卖出
        
        return labels.astype(int)
    
    def _validate_data(self, X: pd.DataFrame, y: pd.Series = None) -> bool:
        """
        验证输入数据
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            验证结果
        """
        # 检查特征
        if X.empty or X.isnull().all().all():
            self.logger.error("特征矩阵为空或全为缺失值")
            return False
            
        # 检查特征名称一致性
        if self.feature_names is not None:
            if not set(X.columns).issubset(set(self.feature_names)):
                missing_features = set(self.feature_names) - set(X.columns)
                self.logger.warning(f"缺失特征: {missing_features}")
                # 为缺失特征填充0
                for feature in missing_features:
                    X[feature] = 0
                    
        # 检查标签分布（如果有y）
        if y is not None:
            unique_labels = set(y.unique())
            expected_labels = set(TARGET_CONFIG["LABELS"].keys())
            if not unique_labels.issubset(expected_labels):
                self.logger.error(f"无效的标签值: {unique_labels - expected_labels}")
                return False
                
        return True
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            feature_names: List[str] = None) -> 'AlphaSeekerModel':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征  
            y_val: 验证标签
            feature_names: 特征名称列表
            
        Returns:
            自身
        """
        # 数据验证
        if not self._validate_data(X, y):
            raise ValueError("训练数据验证失败")
            
        # 存储特征名称
        if feature_names is None:
            self.feature_names = list(X.columns)
        else:
            self.feature_names = feature_names
            
        # 确保X包含所有预期特征
        if not set(self.feature_names).issubset(set(X.columns)):
            missing_features = set(self.feature_names) - set(X.columns)
            self.logger.warning(f"添加缺失特征: {missing_features}")
            for feature in missing_features:
                X[feature] = 0
                
        # 重新排列特征顺序
        X = X[self.feature_names]
        
        # 构造目标变量
        if y.name != 'target':
            y = self._prepare_target(y)
            
        # 计算类别权重
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        self.logger.info(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        self.logger.info(f"类别权重: {class_weight_dict}")
        
        # 准备训练数据
        train_data = lgb.Dataset(
            X, 
            label=y,
            feature_name=self.feature_names,
            categorical_feature=[]
        )
        
        # 准备验证数据
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            if not self._validate_data(X_val, y_val):
                raise ValueError("验证数据验证失败")
                
            y_val = self._prepare_target(y_val)
            X_val = X_val[self.feature_names]
            
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
            
        # 合并配置参数
        params = {**self.config, **self.inference_params}
        
        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.is_fitted = True
        self.logger.info("模型训练完成")
        
        return self
    
    def predict(self, X: pd.DataFrame, 
               return_proba: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测
        
        Args:
            X: 预测特征
            return_proba: 是否返回概率
            
        Returns:
            预测标签或(预测标签, 预测概率)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        # 数据验证
        if not self._validate_data(X):
            raise ValueError("预测数据验证失败")
            
        # 确保特征顺序一致
        if self.feature_names is not None:
            if not set(X.columns).issubset(set(self.feature_names)):
                missing_features = set(self.feature_names) - set(X.columns)
                self.logger.warning(f"预测时补充缺失特征: {missing_features}")
                for feature in missing_features:
                    X[feature] = 0
                    
            X = X.reindex(columns=self.feature_names, fill_value=0)
        
        # 预测
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 映射回原始标签（-1, 0, 1）
        # LightGBM返回的类别是0, 1, 2，需要映射为-1, 0, 1
        mapped_predictions = predicted_classes - 1
        
        if return_proba:
            return mapped_predictions, predictions
        else:
            return mapped_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 预测特征
            
        Returns:
            预测概率矩阵
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        probabilities = self.model.predict(X, num_iteration=self.model.best_iteration)
        return probabilities
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ('gain', 'split', 'cover')
            
        Returns:
            特征重要性DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        importance = self.model.feature_importance(importance_type=importance_type)
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        模型评估
        
        Args:
            X: 评估特征
            y: 评估标签
            
        Returns:
            评估结果字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        # 数据验证
        if not self._validate_data(X, y):
            raise ValueError("评估数据验证失败")
            
        # 构造真实标签
        y_true = self._prepare_target(y)
        
        # 预测
        y_pred = self.predict(X)
        
        # 计算评估指标
        report = classification_report(
            y_true, y_pred, 
            target_names=list(self.label_mapping.values()),
            output_dict=True
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 整理结果
        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'macro_f1': report['macro avg']['f1-score']
        }
        
        return results
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        保存模型
        
        Args:
            filepath: 模型保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'label_mapping': self.label_mapping,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"模型已保存至: {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'AlphaSeekerModel':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            自身
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.label_mapping = model_data['label_mapping']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"模型已从 {filepath} 加载")
        return self
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
            
        info = {
            "status": "fitted",
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "num_trees": self.model.num_trees(),
            "label_mapping": self.label_mapping,
            "feature_names": self.feature_names[:10] if self.feature_names else None  # 只显示前10个特征名
        }
        
        return info