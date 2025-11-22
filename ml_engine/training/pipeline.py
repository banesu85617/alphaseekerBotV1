"""
AlphaSeeker-Bot机器学习模型训练模块
整合特征工程、因子评估和LightGBM模型训练的完整流程
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

from ..core.model import AlphaSeekerModel
from ..features.feature_engineer import MicrostructureFeatureEngineer
from ..features.factor_evaluator import AlphaFactorEvaluator
from ..config.settings import DATA_CONFIG, TARGET_CONFIG


class MLTrainingPipeline:
    """
    机器学习训练流水线
    
    基于分析文档实现：
    - 时间序列数据切分
    - 特征工程和筛选
    - 因子评估和分级
    - LightGBM模型训练
    - 模型评估和验证
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化训练流水线
        
        Args:
            config: 训练配置
        """
        self.config = config or DATA_CONFIG
        self.feature_engineer = MicrostructureFeatureEngineer()
        self.factor_evaluator = AlphaFactorEvaluator()
        self.model = None
        
        # 数据切分参数
        self.train_test_split = self.config.get("TRAIN_TEST_SPLIT", 0.8)
        self.val_split = self.config.get("VAL_SPLIT", 0.2)
        self.shuffle = self.config.get("SHUFFLE", False)
        
        # 数据质量参数
        self.min_data_points = self.config.get("MIN_DATA_POINTS", 1000)
        self.max_missing_ratio = self.config.get("MAX_MISSING_RATIO", 0.1)
        self.outlier_threshold = self.config.get("OUTLIER_THRESHOLD", 3.0)
        
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据准备和质量检查
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            清洗后的数据DataFrame
        """
        self.logger.info(f"开始数据准备，原始数据形状: {df.shape}")
        
        # 检查数据量
        if len(df) < self.min_data_points:
            raise ValueError(f"数据点数量不足: {len(df)} < {self.min_data_points}")
            
        # 按时间排序
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif 'timestamp_utc' in df.columns:
            df = df.sort_values('timestamp_utc').reset_index(drop=True)
            
        # 检查缺失值
        missing_ratio = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratio[missing_ratio > self.max_missing_ratio].index.tolist()
        
        if high_missing_cols:
            self.logger.warning(f"高缺失率列({len(high_missing_cols)}个): {high_missing_cols[:10]}")
            
        # 处理缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(0)
                
        # 处理异常值
        if self.outlier_threshold > 0:
            df = self.feature_engineer.handle_outliers(df, method='clip', threshold=self.outlier_threshold)
            
        self.logger.info(f"数据准备完成，处理后数据形状: {df.shape}")
        
        return df
    
    def create_targets(self, df: pd.DataFrame, price_col: str = 'close') -> Tuple[pd.Series, pd.Series]:
        """
        创建未来收益和目标变量
        
        Args:
            df: 价格数据DataFrame
            price_col: 价格列名
            
        Returns:
            (未来收益序列, 目标变量序列)
        """
        if price_col not in df.columns:
            raise ValueError(f"价格列 '{price_col}' 不存在于数据中")
            
        # 计算未来收益
        horizon = TARGET_CONFIG["HORIZON_SECONDS"]
        
        if 'timestamp_utc' in df.columns:
            # 基于时间窗口计算未来收益
            time_diff = pd.to_datetime(df['timestamp_utc']).diff().dt.total_seconds()
            cumulative_time = time_diff.cumsum()
            
            # 找到目标时间窗口对应的索引
            target_indices = []
            for i, time_elapsed in enumerate(cumulative_time):
                if time_elapsed >= horizon:
                    target_indices.append(i)
                else:
                    target_indices.append(len(df) - 1)  # 如果不够horizon，使用最后一个有效价格
        else:
            # 基于数据点数量(假设每秒一个数据点)
            target_indices = df.index + min(horizon, len(df) - 1)
            target_indices = np.clip(target_indices, 0, len(df) - 1)
            
        # 计算未来收益
        future_prices = df[price_col].iloc[target_indices].values
        current_prices = df[price_col].values
        future_returns = (future_prices - current_prices) / current_prices
        
        future_returns = pd.Series(future_returns, index=df.index)
        
        # 创建目标变量
        threshold = TARGET_CONFIG["PROFIT_THRESHOLD"]
        targets = pd.Series(0, index=df.index)
        
        targets[future_returns > threshold] = 1   # 买入
        targets[future_returns < -threshold] = -1  # 卖出
        
        # 确保有足够的有效数据点
        valid_targets = targets.dropna()
        if len(valid_targets) < self.min_data_points // 2:
            raise ValueError(f"有效目标变量数量不足: {len(valid_targets)}")
            
        self.logger.info(f"目标变量分布: {dict(valid_targets.value_counts().sort_index())}")
        
        return future_returns, targets
    
    def split_data(self, df: pd.DataFrame, targets: pd.Series) -> Tuple[Dict, pd.Series, Dict, pd.Series]:
        """
        时间序列数据切分
        
        Args:
            df: 特征DataFrame
            targets: 目标变量
            
        Returns:
            (训练集特征, 训练集标签, 测试集特征, 测试集标签)
        """
        # 计算切分点
        split_point = int(len(df) * self.train_test_split)
        
        # 训练集
        train_df = df.iloc[:split_point]
        train_targets = targets.iloc[:split_point]
        
        # 测试集
        test_df = df.iloc[split_point:]
        test_targets = targets.iloc[split_point:]
        
        # 验证集切分(从训练集中)
        val_point = int(len(train_df) * (1 - self.val_split))
        
        train_final_df = train_df.iloc[:val_point]
        train_final_targets = train_targets.iloc[:val_point]
        
        val_df = train_df.iloc[val_point:]
        val_targets = train_targets.iloc[val_point:]
        
        # 检查标签覆盖
        all_labels = set(targets.unique())
        train_labels = set(train_final_targets.unique())
        test_labels = set(test_targets.unique())
        val_labels = set(val_targets.unique())
        
        if not test_labels.issubset(all_labels):
            self.logger.warning("测试集标签覆盖不完整，启用随机切分")
            # 如果测试集标签覆盖不完整，进行随机切分
            indices = np.arange(len(df))
            np.random.seed(42)
            np.random.shuffle(indices)
            
            train_size = int(len(indices) * self.train_test_split)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            train_df = df.iloc[train_indices]
            train_targets = targets.iloc[train_indices]
            test_df = df.iloc[test_indices]
            test_targets = targets.iloc[test_indices]
            
            # 重新划分验证集
            val_point = int(len(train_df) * (1 - self.val_split))
            train_final_df = train_df.iloc[:val_point]
            train_final_targets = train_targets.iloc[:val_point]
            val_df = train_df.iloc[val_point:]
            val_targets = train_targets.iloc[val_point:]
            
        self.logger.info(f"数据切分完成:")
        self.logger.info(f"  训练集: {len(train_final_df)} 样本")
        self.logger.info(f"  验证集: {len(val_df)} 样本")
        self.logger.info(f"  测试集: {len(test_df)} 样本")
        
        return (
            train_final_df, train_final_targets,
            test_df, test_targets,
            val_df, val_targets
        )
    
    def feature_selection(self, train_df: pd.DataFrame, train_targets: pd.Series,
                         val_df: pd.DataFrame, val_targets: pd.Series,
                         test_df: pd.DataFrame, test_targets: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        特征选择和评估
        
        Args:
            train_df: 训练特征
            train_targets: 训练目标
            val_df: 验证特征
            val_targets: 验证目标
            test_df: 测试特征
            test_targets: 测试目标
            
        Returns:
            (筛选后的训练特征, 验证特征, 测试特征)
        """
        self.logger.info("开始特征选择和评估")
        
        # 1. 基础特征工程
        train_processed = self.feature_engineer.process_features(train_df)
        val_processed = self.feature_engineer.process_features(val_df)
        test_processed = self.feature_engineer.process_features(test_df)
        
        # 2. 选择核心特征
        core_features = self.feature_engineer.select_features(train_processed)
        
        # 确保所有数据集有相同的特征
        common_features = list(core_features.columns)
        
        train_features = train_processed[common_features]
        val_features = val_processed[common_features]
        test_features = test_processed[common_features]
        
        # 3. 因子评估（仅在训练集上进行）
        try:
            # 计算未来收益（用于因子评估）
            _, train_returns = self.create_targets(train_df, 'close')
            
            # 评估因子
            factor_results = self.factor_evaluator.evaluate_multiple_factors(
                train_features, train_returns
            )
            
            # 选择顶级因子
            selected_factors = self.factor_evaluator.select_top_factors(
                factor_results, top_n=30, min_grade='B'
            )
            
            # 重新选择特征
            if len(selected_factors) > 0:
                available_factors = [f for f in selected_factors if f in train_features.columns]
                if len(available_factors) > 0:
                    train_features = train_features[available_factors]
                    val_features = val_features[available_factors]
                    test_features = test_features[available_factors]
                    
                    self.logger.info(f"因子评估后选择 {len(available_factors)} 个特征")
                    
                    # 生成因子评估报告
                    report = self.factor_evaluator.generate_factor_report(factor_results)
                    self.logger.info("因子评估报告:\n" + report)
            
        except Exception as e:
            self.logger.warning(f"因子评估失败: {str(e)}，使用所有可用特征")
            
        # 4. 特征标准化
        numeric_cols = train_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 在训练集上拟合并转换
            mean_val = train_features[col].mean()
            std_val = train_features[col].std()
            
            if std_val > 1e-8:  # 避免除零
                train_features[col] = (train_features[col] - mean_val) / std_val
                val_features[col] = (val_features[col] - mean_val) / std_val
                test_features[col] = (test_features[col] - mean_val) / std_val
            else:
                train_features[col] = 0
                val_features[col] = 0  
                test_features[col] = 0
                
        # 移除无穷大和NaN
        train_features = train_features.replace([np.inf, -np.inf], 0).fillna(0)
        val_features = val_features.replace([np.inf, -np.inf], 0).fillna(0)
        test_features = test_features.replace([np.inf, -np.inf], 0).fillna(0)
        
        self.logger.info(f"特征选择完成，最终特征数量: {train_features.shape[1]}")
        
        return train_features, val_features, test_features
    
    def train_model(self, train_features: pd.DataFrame, train_targets: pd.Series,
                   val_features: pd.DataFrame, val_targets: pd.Series) -> AlphaSeekerModel:
        """
        训练LightGBM模型
        
        Args:
            train_features: 训练特征
            train_targets: 训练目标
            val_features: 验证特征  
            val_targets: 验证目标
            
        Returns:
            训练好的模型
        """
        self.logger.info("开始训练LightGBM模型")
        
        # 创建模型
        self.model = AlphaSeekerModel()
        
        # 训练模型
        self.model.fit(
            train_features, train_targets,
            val_features, val_targets,
            feature_names=list(train_features.columns)
        )
        
        # 获取特征重要性
        feature_importance = self.model.get_feature_importance()
        self.logger.info("前10个重要特征:")
        for _, row in feature_importance.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
        self.logger.info("模型训练完成")
        
        return self.model
    
    def evaluate_model(self, test_features: pd.DataFrame, test_targets: pd.Series) -> Dict:
        """
        模型评估
        
        Args:
            test_features: 测试特征
            test_targets: 测试目标
            
        Returns:
            评估结果字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        self.logger.info("开始模型评估")
        
        # 计算预测
        predictions = self.model.predict(test_features)
        
        # 模型评估
        results = self.model.evaluate(test_features, test_targets)
        
        # 计算额外指标
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(test_targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_targets, predictions, average='weighted'
        )
        
        # 添加到结果中
        results['accuracy'] = accuracy
        results['weighted_precision'] = precision
        results['weighted_recall'] = recall
        results['weighted_f1'] = f1
        
        self.logger.info(f"模型评估完成:")
        self.logger.info(f"  准确率: {accuracy:.4f}")
        self.logger.info(f"  权重F1: {f1:.4f}")
        
        return results
    
    def run_complete_pipeline(self, df: pd.DataFrame, price_col: str = 'close') -> Dict:
        """
        运行完整的训练流水线
        
        Args:
            df: 原始数据DataFrame
            price_col: 价格列名
            
        Returns:
            流水线结果字典
        """
        self.logger.info("开始完整的ML训练流水线")
        
        try:
            # 1. 数据准备
            df_clean = self.prepare_data(df)
            
            # 2. 创建目标变量
            future_returns, targets = self.create_targets(df_clean, price_col)
            
            # 3. 数据切分
            (
                train_features, train_targets,
                test_features, test_targets,
                val_features, val_targets
            ) = self.split_data(df_clean, targets)
            
            # 4. 特征选择
            (
                train_features_final, 
                val_features_final, 
                test_features_final
            ) = self.feature_selection(
                train_features, train_targets,
                val_features, val_targets,
                test_features, test_targets
            )
            
            # 5. 模型训练
            model = self.train_model(
                train_features_final, train_targets,
                val_features_final, val_targets
            )
            
            # 6. 模型评估
            results = self.evaluate_model(test_features_final, test_targets)
            
            # 7. 组装最终结果
            pipeline_results = {
                "success": True,
                "model": model,
                "feature_names": list(train_features_final.columns),
                "train_size": len(train_features_final),
                "val_size": len(val_features_final),
                "test_size": len(test_features_final),
                "evaluation_results": results,
                "model_info": model.get_model_info()
            }
            
            self.logger.info("ML训练流水线完成")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"训练流水线失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": None,
                "feature_names": [],
                "evaluation_results": {}
            }
    
    def save_pipeline_results(self, results: Dict, filepath: str) -> None:
        """
        保存训练结果
        
        Args:
            results: 流水线结果
            filepath: 保存路径
        """
        import joblib
        
        # 保存模型和元数据
        if results["success"] and results["model"] is not None:
            model_data = {
                "model": results["model"],
                "feature_names": results["feature_names"],
                "model_info": results["model_info"],
                "evaluation_results": results["evaluation_results"],
                "pipeline_config": self.config
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"训练结果已保存至: {filepath}")
        else:
            self.logger.error("无法保存失败的训练结果")