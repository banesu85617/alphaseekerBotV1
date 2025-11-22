"""
技术指标计算模块
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple
import logging
from ..core.exceptions import IndicatorError

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def compute_sma(series: pd.Series, window: int) -> pd.Series:
        """简单移动平均"""
        try:
            return series.rolling(window=window, min_periods=window).mean()
        except Exception as e:
            logger.error(f"SMA calculation error: {e}")
            raise IndicatorError(f"SMA calculation failed: {e}")
    
    @staticmethod
    def compute_ema(series: pd.Series, window: int) -> pd.Series:
        """指数移动平均"""
        try:
            return series.ewm(span=window, adjust=False, min_periods=window).mean()
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            raise IndicatorError(f"EMA calculation failed: {e}")
    
    @staticmethod
    def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """相对强弱指标"""
        try:
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).fillna(0)
            loss = -delta.where(delta < 0, 0.0).fillna(0)
            
            avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
            avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return rsi.fillna(50)  # 中性值填充
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            raise IndicatorError(f"RSI calculation failed: {e}")
    
    @staticmethod
    def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """真实波动幅度均值"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.ewm(com=window - 1, min_periods=window).mean()
            
            return atr
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            raise IndicatorError(f"ATR calculation failed: {e}")
    
    @staticmethod
    def compute_bollinger_bands(
        series: pd.Series, 
        window: int = 20, 
        num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        try:
            sma = series.rolling(window=window, min_periods=window).mean()
            std = series.rolling(window=window, min_periods=window).std()
            
            upper_band = sma + (num_std * std)
            lower_band = sma - (num_std * std)
            
            return upper_band, sma, lower_band
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            raise IndicatorError(f"Bollinger Bands calculation failed: {e}")
    
    @staticmethod
    def compute_macd(
        series: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标"""
        try:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            raise IndicatorError(f"MACD calculation failed: {e}")
    
    @staticmethod
    def compute_stochastic(
        df: pd.DataFrame, 
        k_window: int = 14, 
        d_window: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """随机指标"""
        try:
            lowest_low = df['low'].rolling(window=k_window, min_periods=k_window).min()
            highest_high = df['high'].rolling(window=k_window, min_periods=k_window).max()
            
            k_percent = 100 * ((df['close'] - lowest_low) / 
                              (highest_high - lowest_low).replace(0, np.nan))
            
            d_percent = k_percent.rolling(window=d_window, min_periods=d_window).mean()
            
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"Stochastic calculation error: {e}")
            raise IndicatorError(f"Stochastic calculation failed: {e}")
    
    @staticmethod
    def compute_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """威廉指标"""
        try:
            highest_high = df['high'].rolling(window=window, min_periods=window).max()
            lowest_low = df['low'].rolling(window=window, min_periods=window).min()
            
            range_val = (highest_high - lowest_low).replace(0, np.nan)
            williams_r = -100 * ((highest_high - df['close']) / range_val)
            
            return williams_r
        except Exception as e:
            logger.error(f"Williams %R calculation error: {e}")
            raise IndicatorError(f"Williams %R calculation failed: {e}")
    
    @staticmethod
    def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """平均趋向指标"""
        try:
            # 计算真实范围
            df_temp = df.copy()
            df_temp['H-L'] = df_temp['high'] - df_temp['low']
            df_temp['H-PC'] = abs(df_temp['high'] - df_temp['close'].shift(1))
            df_temp['L-PC'] = abs(df_temp['low'] - df_temp['close'].shift(1))
            df_temp['TR'] = df_temp[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            
            # 计算方向性移动
            df_temp['DM_plus'] = np.where(
                (df_temp['high'] - df_temp['high'].shift(1)) > 
                (df_temp['low'].shift(1) - df_temp['low']), 
                df_temp['high'] - df_temp['high'].shift(1), 0
            )
            df_temp['DM_plus'] = np.where(df_temp['DM_plus'] < 0, 0, df_temp['DM_plus'])
            
            df_temp['DM_minus'] = np.where(
                (df_temp['low'].shift(1) - df_temp['low']) > 
                (df_temp['high'] - df_temp['high'].shift(1)), 
                df_temp['low'].shift(1) - df_temp['low'], 0
            )
            df_temp['DM_minus'] = np.where(df_temp['DM_minus'] < 0, 0, df_temp['DM_minus'])
            
            # 平滑计算
            alpha = 1 / window
            tr_smooth = df_temp['TR'].ewm(alpha=alpha, adjust=False, min_periods=window).mean()
            dm_plus_smooth = df_temp['DM_plus'].ewm(alpha=alpha, adjust=False, min_periods=window).mean()
            dm_minus_smooth = df_temp['DM_minus'].ewm(alpha=alpha, adjust=False, min_periods=window).mean()
            
            # 计算方向性指标
            di_plus = 100 * (dm_plus_smooth / tr_smooth.replace(0, np.nan)).fillna(0)
            di_minus = 100 * (dm_minus_smooth / tr_smooth.replace(0, np.nan)).fillna(0)
            
            # 计算ADX
            dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan))
            adx = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
            
            return adx
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            raise IndicatorError(f"ADX calculation failed: {e}")
    
    @staticmethod
    def compute_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """商品通道指数"""
        try:
            # 典型价格
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # 移动平均
            sma_tp = typical_price.rolling(window=window, min_periods=window).mean()
            
            # 平均绝对离差
            mad = typical_price.rolling(window=window, min_periods=window).apply(
                lambda x: np.nanmean(np.abs(x - np.nanmean(x))), raw=True
            )
            
            # CCI计算
            cci = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))
            
            return cci
        except Exception as e:
            logger.error(f"CCI calculation error: {e}")
            raise IndicatorError(f"CCI calculation failed: {e}")
    
    @staticmethod
    def compute_obv(df: pd.DataFrame) -> pd.Series:
        """能量潮指标"""
        try:
            price_change = df['close'].diff()
            volume_change = df['volume'] * np.sign(price_change)
            
            # 第一个值设为0
            obv = volume_change.fillna(0).cumsum()
            
            return obv
        except Exception as e:
            logger.error(f"OBV calculation error: {e}")
            raise IndicatorError(f"OBV calculation failed: {e}")
    
    @staticmethod
    def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """动量指标"""
        try:
            return series - series.shift(period)
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            raise IndicatorError(f"Momentum calculation failed: {e}")
    
    @staticmethod
    def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """应用所有技术指标"""
        try:
            df_indicators = df.copy()
            df_len = len(df_indicators)
            
            # 确保有足够的数据
            min_length = 200  # 最少需要200个数据点
            
            if df_len < min_length:
                logger.warning(f"Data length ({df_len}) is less than recommended minimum ({min_length})")
            
            # 计算基础指标
            if df_len >= 50:
                df_indicators['SMA_50'] = TechnicalIndicators.compute_sma(df_indicators['close'], 50)
            
            if df_len >= 200:
                df_indicators['SMA_200'] = TechnicalIndicators.compute_sma(df_indicators['close'], 200)
            
            if df_len >= 12:
                df_indicators['EMA_12'] = TechnicalIndicators.compute_ema(df_indicators['close'], 12)
            
            if df_len >= 26:
                df_indicators['EMA_26'] = TechnicalIndicators.compute_ema(df_indicators['close'], 26)
            
            # MACD
            if df_len >= 26:
                macd, signal, histogram = TechnicalIndicators.compute_macd(df_indicators['close'])
                df_indicators['MACD'] = macd
                df_indicators['Signal_Line'] = signal
                df_indicators['MACD_Histogram'] = histogram
            
            # RSI
            if df_len >= 15:
                df_indicators['RSI'] = TechnicalIndicators.compute_rsi(df_indicators['close'])
            
            # ATR
            if df_len >= 15:
                df_indicators['ATR'] = TechnicalIndicators.compute_atr(df_indicators)
            
            # 布林带
            if df_len >= 21:
                upper, middle, lower = TechnicalIndicators.compute_bollinger_bands(df_indicators['close'])
                df_indicators['Bollinger_Upper'] = upper
                df_indicators['Bollinger_Middle'] = middle
                df_indicators['Bollinger_Lower'] = lower
            
            # 动量
            if df_len >= 11:
                df_indicators['Momentum'] = TechnicalIndicators.compute_momentum(df_indicators['close'])
            
            # 随机指标
            if df_len >= 17:
                stoch_k, stoch_d = TechnicalIndicators.compute_stochastic(df_indicators)
                df_indicators['Stochastic_K'] = stoch_k
                df_indicators['Stochastic_D'] = stoch_d
            
            # 威廉指标
            if df_len >= 15:
                df_indicators['Williams_R'] = TechnicalIndicators.compute_williams_r(df_indicators)
            
            # ADX
            if df_len >= 28:
                df_indicators['ADX'] = TechnicalIndicators.compute_adx(df_indicators)
            
            # CCI
            if df_len >= 21:
                df_indicators['CCI'] = TechnicalIndicators.compute_cci(df_indicators)
            
            # OBV
            if df_len >= 2:
                df_indicators['OBV'] = TechnicalIndicators.compute_obv(df_indicators)
            
            # 收益率
            df_indicators['returns'] = df_indicators['close'].pct_change()
            
            logger.debug(f"Applied all technical indicators to DataFrame")
            return df_indicators
            
        except Exception as e:
            logger.error(f"Failed to apply technical indicators: {e}")
            raise IndicatorError(f"Technical indicators application failed: {e}")
    
    @staticmethod
    def get_latest_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """获取最新的指标值"""
        try:
            latest = df.iloc[-1]
            indicators = {}
            
            # 技术指标列
            indicator_cols = [
                'RSI', 'ATR', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                'MACD', 'Signal_Line', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower',
                'Momentum', 'Stochastic_K', 'Stochastic_D', 'Williams_R', 'ADX', 'CCI', 'OBV'
            ]
            
            for col in indicator_cols:
                if col in df.columns:
                    value = latest.get(col)
                    if pd.notna(value) and np.isfinite(value):
                        indicators[col] = float(value)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to get latest indicators: {e}")
            raise IndicatorError(f"Latest indicators retrieval failed: {e}")