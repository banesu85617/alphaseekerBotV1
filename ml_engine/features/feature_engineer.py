"""
AlphaSeeker-Bot特征工程模块
基于分析文档中的60余项微结构特征实现
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..config.settings import FEATURE_CONFIG


class MicrostructureFeatureEngineer:
    """
    微结构特征工程器
    
    基于分析文档实现：
    - 订单簿不平衡特征
    - 价差和WAP特征  
    - 波动率特征
    - 成交量特征
    - 订单流特征
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化特征工程器
        
        Args:
            config: 特征配置
        """
        self.config = config or FEATURE_CONFIG
        self.scalers = {}
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建价格相关特征
        
        Args:
            df: 包含订单簿数据的DataFrame
            
        Returns:
            添加价格特征后的DataFrame
        """
        df = df.copy()
        
        # 中间价
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
            
        # 买卖价差
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            df['spread'] = df['ask_price'] - df['bid_price']
            df['spread_pct'] = df['spread'] / df['mid_price']
            
        # 加权平均价 (WAP)
        if all(col in df.columns for col in ['bid_price', 'bid_volume', 'ask_price', 'ask_volume']):
            total_volume = df['bid_volume'] + df['ask_volume']
            df['wap_1'] = np.where(
                total_volume > 0,
                (df['bid_price'] * df['ask_volume'] + df['ask_price'] * df['bid_volume']) / total_volume,
                df['mid_price']
            )
            
        # 多时间框架WAP
        for window in [5, 10, 20, 60]:
            if 'wap_1' in df.columns:
                df[f'wap_{window}'] = df['wap_1'].rolling(window=window, min_periods=1).mean()
                
        return df
    
    def create_order_book_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建订单簿特征
        
        Args:
            df: 包含订单簿数据的DataFrame
            
        Returns:
            添加订单簿特征后的DataFrame
        """
        df = df.copy()
        
        # 基础订单簿数据
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            df['total_volume'] = df['bid_volume'] + df['ask_volume']
            df['bid_ask_ratio'] = df['bid_volume'] / (df['ask_volume'] + 1e-8)
            
        # 订单不平衡
        if 'total_volume' in df.columns:
            df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / df['total_volume']
            df['order_imbalance_abs'] = np.abs(df['order_imbalance'])
            
        # 深度不平衡 (基于前N档数据)
        depth_levels = ['bid_volume_5', 'bid_volume_10', 'ask_volume_5', 'ask_volume_10']
        if all(col in df.columns for col in depth_levels):
            df['depth_imbalance'] = (
                (df['bid_volume_5'] + df['bid_volume_10']) - 
                (df['ask_volume_5'] + df['ask_volume_10'])
            ) / (
                (df['bid_volume_5'] + df['bid_volume_10']) + 
                (df['ask_volume_5'] + df['ask_volume_10']) + 1e-8
            )
            
        # 订单簿强度
        if 'order_imbalance' in df.columns:
            df['order_book_strength'] = df['order_imbalance'] * df['spread_pct']
            
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建成交量特征
        
        Args:
            df: 包含成交量数据的DataFrame
            
        Returns:
            添加成交量特征后的DataFrame
        """
        df = df.copy()
        
        # 基础成交量
        if 'volume' in df.columns:
            df['volume_log'] = np.log1p(df['volume'])
            df['volume_sqrt'] = np.sqrt(df['volume'])
            
        # 成交量滚动统计
        for window in [5, 10, 20]:
            if 'volume' in df.columns:
                df[f'volume_mean_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window=window, min_periods=1).std()
                df[f'volume_cv_{window}'] = df[f'volume_std_{window}'] / (df[f'volume_mean_{window}'] + 1e-8)
                
        # 成交量趋势
        if 'volume' in df.columns:
            df['volume_trend'] = df['volume'].rolling(window=5, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # 成交量变化率
            df['volume_change'] = df['volume'].pct_change()
            df['volume_change_log'] = np.log1p(df['volume_change'])
            
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建波动率特征
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            添加波动率特征后的DataFrame
        """
        df = df.copy()
        
        price_col = 'mid_price' if 'mid_price' in df.columns else 'price'
        
        if price_col in df.columns:
            # 价格收益率
            df['returns'] = df[price_col].pct_change()
            df['returns_log'] = np.log1p(df['returns'])
            
            # 波动率（多时间框架）
            for window in [10, 20, 60]:
                df[f'volatility_{window}'] = df['returns'].rolling(window=window, min_periods=1).std()
                df[f'volatility_log_{window}'] = np.log1p(df[f'volatility_{window}'])
                
            # 已实现波动率
            df['realized_volatility'] = np.sqrt(
                df['returns'].rolling(window=60, min_periods=1).apply(lambda x: np.sum(x**2))
            )
            
            # 价格范围波动率
            if 'high' in df.columns and 'low' in df.columns:
                df['price_range'] = (df['high'] - df['low']) / df[price_col]
                df['range_volatility'] = df['price_range'].rolling(window=20, min_periods=1).std()
                
        return df
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建微结构特征
        
        Args:
            df: 原始DataFrame
            
        Returns:
            添加微结构特征后的DataFrame
        """
        df = df.copy()
        
        # 交易强度
        if 'volume' in df.columns and 'spread' in df.columns:
            df['trade_intensity'] = df['volume'] / (df['spread'] + 1e-8)
            
        # 订单流不平衡
        if 'order_imbalance' in df.columns and 'volume' in df.columns:
            df['order_flow_imbalance'] = df['order_imbalance'] * df['volume']
            
        # 价格影响
        if 'spread' in df.columns and 'volume' in df.columns:
            df['price_impact'] = df['spread'] / (df['volume'] + 1e-8)
            
        # 有效价差
        if 'mid_price' in df.columns and 'last_price' in df.columns:
            df['effective_spread'] = 2 * np.abs(df['last_price'] - df['mid_price']) / df['mid_price']
            
        # Amihud流动性
        if 'returns' in df.columns and 'volume' in df.columns:
            df['amihud_illiquidity'] = np.abs(df['returns']) / (df['volume'] + 1e-8)
            
        # Kyle's Lambda
        if 'order_imbalance' in df.columns and 'returns' in df.columns:
            df['kyle_lambda'] = df['returns'] / (np.abs(df['order_imbalance']) + 1e-8)
            
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                              feature_cols: List[str]) -> pd.DataFrame:
        """
        创建滚动统计特征
        
        Args:
            df: 输入DataFrame
            feature_cols: 需要创建滚动特征的列
            
        Returns:
            添加滚动特征后的DataFrame
        """
        df = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            # 滚动均值、标准差、偏度、峰度
            for window in [10, 20, 60]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_rolling_skew_{window}'] = df[col].rolling(window=window, min_periods=1).skew()
                df[f'{col}_rolling_kurt_{window}'] = df[col].rolling(window=window, min_periods=1).kurt()
                
        return df
    
    def create_lagged_features(self, df: pd.DataFrame, 
                             feature_cols: List[str], 
                             lags: List[int] = None) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 输入DataFrame
            feature_cols: 需要滞后处理的列
            lags: 滞后阶数列表
            
        Returns:
            添加滞后特征后的DataFrame
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10]
            
        df = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return df
    
    def create_diff_features(self, df: pd.DataFrame, 
                           feature_cols: List[str]) -> pd.DataFrame:
        """
        创建差分特征（避免前瞻偏差）
        
        Args:
            df: 输入DataFrame
            feature_cols: 需要差分的列
            
        Returns:
            添加差分特征后的DataFrame
        """
        df = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            # 一阶差分
            df[f'{col}_diff'] = df[col].diff()
            
            # 二阶差分
            df[f'{col}_diff2'] = df[col].diff().diff()
            
        return df
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        完整的特征工程流程
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            完整的特征DataFrame
        """
        df = df.copy()
        self.logger.info(f"开始特征工程，输入数据形状: {df.shape}")
        
        # 1. 基础价格特征
        df = self.create_price_features(df)
        self.logger.info("完成价格特征创建")
        
        # 2. 订单簿特征
        df = self.create_order_book_features(df)
        self.logger.info("完成订单簿特征创建")
        
        # 3. 成交量特征
        df = self.create_volume_features(df)
        self.logger.info("完成成交量特征创建")
        
        # 4. 波动率特征
        df = self.create_volatility_features(df)
        self.logger.info("完成波动率特征创建")
        
        # 5. 微结构特征
        df = self.create_microstructure_features(df)
        self.logger.info("完成微结构特征创建")
        
        # 6. 滞后特征（避免数据泄露）
        if self.config.get("LAG_FEATURES"):
            lagged_cols = ['spread', 'order_imbalance', 'volume', 'volatility_20']
            df = self.create_lagged_features(df, lagged_cols)
            self.logger.info("完成滞后特征创建")
        
        # 7. 差分特征（提高平稳性）
        if self.config.get("DIFF_FEATURES"):
            diff_cols = ['mid_price', 'wap_1', 'spread']
            df = self.create_diff_features(df, diff_cols)
            self.logger.info("完成差分特征创建")
        
        # 8. 滚动统计特征
        core_cols = ['spread', 'order_imbalance', 'volume', 'volatility_20']
        available_cols = [col for col in core_cols if col in df.columns]
        if available_cols:
            df = self.create_rolling_features(df, available_cols)
            self.logger.info("完成滚动统计特征创建")
        
        self.logger.info(f"特征工程完成，最终特征数量: {df.shape[1]}")
        
        return df
    
    def select_features(self, df: pd.DataFrame, 
                       feature_list: List[str] = None) -> pd.DataFrame:
        """
        特征选择
        
        Args:
            df: 特征DataFrame
            feature_list: 指定特征列表，如果为None则使用配置中的核心特征
            
        Returns:
            选择的特征DataFrame
        """
        if feature_list is None:
            feature_list = self.config.get("CORE_FEATURES", [])
            
        # 选择可用特征
        available_features = [col for col in feature_list if col in df.columns]
        
        if len(available_features) < len(feature_list):
            missing_features = set(feature_list) - set(available_features)
            self.logger.warning(f"缺失特征: {missing_features}")
            
        self.logger.info(f"选择特征数量: {len(available_features)}")
        
        return df[available_features]
    
    def handle_outliers(self, df: pd.DataFrame, 
                       method: str = 'clip',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        异常值处理
        
        Args:
            df: 输入DataFrame
            method: 处理方法 ('clip', 'remove', 'winsorize')
            threshold: 异常值阈值
            
        Returns:
            处理后的DataFrame
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'clip':
            # 截断法
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
        elif method == 'winsorize':
            # Winsorization
            from scipy.stats import mstats
            for col in numeric_cols:
                df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])
                
        elif method == 'remove':
            # 移除异常值
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].fillna(0)))
                df = df[z_scores < threshold]
                
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                         method: str = 'standard') -> pd.DataFrame:
        """
        特征标准化
        
        Args:
            df: 输入DataFrame
            method: 标准化方法 ('standard', 'robust', 'minmax')
            
        Returns:
            标准化后的DataFrame
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        return self.feature_names
    
    def set_feature_names(self, feature_names: List[str]):
        """
        设置特征名称
        
        Args:
            feature_names: 特征名称列表
        """
        self.feature_names = feature_names