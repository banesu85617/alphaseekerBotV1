"""
数据处理工具类
提供市场数据处理和计算功能
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """技术指标结果"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_position: Optional[float] = None
    atr: Optional[float] = None
    adx: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None
    momentum: Optional[float] = None
    roc: Optional[float] = None
    volatility: Optional[float] = None
    volume_sma: Optional[float] = None
    price_trend: Optional[float] = None


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.data_cache: dict[str, list[dict[str, Any]]] = {}
        
    def normalize_symbol(self, symbol: str) -> str:
        """标准化交易对名称"""
        return symbol.upper().replace('-', '').replace('_', '')
    
    async def fetch_market_data(self, symbol: str, exchange: str = "binance") -> Optional[dict[str, Any]]:
        """
        获取市场数据
        
        Args:
            symbol: 交易对
            exchange: 交易所
            
        Returns:
            市场数据字典
        """
        try:
            # 模拟实现 - 实际应该调用交易API
            # 这里可以集成CCXT或其他数据源
            
            normalized_symbol = self.normalize_symbol(symbol)
            
            # 生成模拟数据
            import random
            base_price = 50000 if "BTC" in normalized_symbol else 3000 if "ETH" in normalized_symbol else 100
            
            market_data = {
                'symbol': normalized_symbol,
                'exchange': exchange,
                'price': base_price + random.uniform(-0.1, 0.1) * base_price,
                'volume_24h': random.uniform(100000, 10000000),
                'price_change_24h': random.uniform(-10, 10),
                'high_24h': base_price * 1.05,
                'low_24h': base_price * 0.95,
                'bid': base_price * 0.999,
                'ask': base_price * 1.001,
                'timestamp': datetime.now().isoformat(),
                'market_cap': random.uniform(1000000, 1000000000),
                'circulating_supply': random.uniform(1000000, 100000000),
                'max_supply': random.uniform(10000000, 1000000000),
                'volume_trend': random.uniform(-0.5, 0.5),
                'order_book_depth': random.uniform(500000, 5000000)
            }
            
            # 计算衍生指标
            market_data['bid_ask_spread'] = (market_data['ask'] - market_data['bid']) / market_data['price']
            market_data['liquidity_score'] = self._calculate_liquidity_score(market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def batch_fetch_market_data(self, symbols: List[str], exchange: str = "binance") -> dict[str, dict[str, Any]]:
        """
        批量获取市场数据
        
        Args:
            symbols: 交易对列表
            exchange: 交易所
            
        Returns:
            市场数据字典
        """
        logger.info(f"Batch fetching market data for {len(symbols)} symbols")
        
        # 并行获取数据
        tasks = [self.fetch_market_data(symbol, exchange) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        market_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, dict):
                market_data[symbol] = result
            else:
                logger.warning(f"Failed to fetch data for {symbol}: {result}")
        
        logger.info(f"Successfully fetched data for {len(market_data)} symbols")
        return market_data
    
    def calculate_technical_indicators(self, ohlcv_data: pd.DataFrame) -> TechnicalIndicators:
        """
        计算技术指标
        
        Args:
            ohlcv_data: OHLCV数据
            
        Returns:
            技术指标结果
        """
        try:
            indicators = TechnicalIndicators()
            
            if ohlcv_data.empty or len(ohlcv_data) < 50:
                logger.warning("Insufficient data for technical indicators")
                return indicators
            
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            volume = ohlcv_data['volume']
            
            # RSI
            indicators.rsi = self._calculate_rsi(close, 14)
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            indicators.macd = ema_12 - ema_26
            indicators.macd_signal = indicators.macd.ewm(span=9).mean()
            indicators.macd_histogram = indicators.macd - indicators.macd_signal
            indicators.ema_12 = ema_12.iloc[-1]
            indicators.ema_26 = ema_26.iloc[-1]
            
            # 布林带
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators.bollinger_upper = (sma_20 + 2 * std_20).iloc[-1]
            indicators.bollinger_middle = sma_20.iloc[-1]
            indicators.bollinger_lower = (sma_20 - 2 * std_20).iloc[-1]
            
            # 布林带位置
            current_price = close.iloc[-1]
            indicators.bollinger_position = (current_price - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower)
            
            # 移动平均线
            indicators.sma_20 = sma_20.iloc[-1]
            indicators.sma_50 = close.rolling(window=50).mean().iloc[-1]
            
            # ATR
            indicators.atr = self._calculate_atr(high, low, close, 14)
            
            # ADX (简化计算)
            indicators.adx = self._calculate_adx(high, low, close, 14)
            
            # 随机指标
            stochastic_k, stochastic_d = self._calculate_stochastic(high, low, close, 14, 3)
            indicators.stochastic_k = stochastic_k
            indicators.stochastic_d = stochastic_d
            
            # Williams %R
            indicators.williams_r = self._calculate_williams_r(high, low, close, 14)
            
            # CCI
            indicators.cci = self._calculate_cci(high, low, close, 20)
            
            # 动量指标
            indicators.momentum = close.iloc[-1] - close.iloc[-10] if len(close) >= 10 else 0
            indicators.roc = (close.iloc[-1] - close.iloc[-12]) / close.iloc[-12] * 100 if len(close) >= 12 else 0
            
            # 波动率
            returns = close.pct_change().dropna()
            indicators.volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            # 成交量平均
            indicators.volume_sma = volume.rolling(window=20).mean().iloc[-1]
            
            # 趋势强度
            indicators.price_trend = self._calculate_trend_strength(close, 20)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not rsi.empty else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """计算ATR"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = pd.Series(tr).rolling(window=period).mean()
            
            return atr.iloc[-1] if not atr.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """计算ADX (简化版本)"""
        try:
            # 简化的ADX计算
            high_diff = high.diff()
            low_diff = low.diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_dm_avg = pd.Series(plus_dm).rolling(window=period).mean()
            minus_dm_avg = pd.Series(minus_dm).rolling(window=period).mean()
            
            # 简化ADX计算
            dx = np.abs(plus_dm_avg - minus_dm_avg) / (plus_dm_avg + minus_dm_avg) * 100
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not adx.empty else 25.0
            
        except Exception:
            return 25.0
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """计算随机指标"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return k_percent.iloc[-1] if not k_percent.empty else 50.0, d_percent.iloc[-1] if not d_percent.empty else 50.0
            
        except Exception:
            return 50.0, 50.0
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """计算Williams %R"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            wr = -100 * (highest_high - close) / (highest_high - lowest_low)
            
            return wr.iloc[-1] if not wr.empty else -50.0
            
        except Exception:
            return -50.0
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
        """计算CCI"""
        try:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            
            return cci.iloc[-1] if not cci.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> float:
        """计算趋势强度"""
        try:
            if len(prices) < period:
                return 0.0
            
            # 计算线性回归斜率
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # 归一化斜率
            trend_strength = slope / prices.mean() if prices.mean() != 0 else 0
            
            return min(1.0, max(-1.0, trend_strength * 100))
            
        except Exception:
            return 0.0
    
    def _calculate_liquidity_score(self, market_data: dict[str, Any]) -> float:
        """计算流动性评分"""
        try:
            volume_score = min(1.0, market_data.get('volume_24h', 0) / 10000000)  # 1000万作为基准
            spread_score = 1.0 - min(1.0, market_data.get('bid_ask_spread', 0) / 0.01)  # 1%作为基准
            depth_score = min(1.0, market_data.get('order_book_depth', 0) / 1000000)  # 100万作为基准
            
            return (volume_score + spread_score + depth_score) / 3
            
        except Exception:
            return 0.5
    
    async def analyze_patterns(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """分析价格模式"""
        try:
            patterns = {
                'is_bullish': False,
                'is_bearish': False,
                'is_consolidating': False,
                'breakout_probability': 0.0,
                'support_levels': [],
                'resistance_levels': [],
                'pattern_type': 'unknown'
            }
            
            price = market_data.get('price', 0)
            volatility = market_data.get('volatility', 0)
            volume_24h = market_data.get('volume_24h', 0)
            
            # 简单的模式识别逻辑
            price_change = market_data.get('price_change_24h', 0)
            
            if price_change > 3:
                patterns['is_bullish'] = True
                patterns['pattern_type'] = 'bullish_momentum'
            elif price_change < -3:
                patterns['is_bearish'] = True
                patterns['pattern_type'] = 'bearish_momentum'
            elif abs(price_change) < 1:
                patterns['is_consolidating'] = True
                patterns['pattern_type'] = 'consolidation'
            
            # 突破概率计算
            if volatility < 0.05:  # 低波动率
                patterns['breakout_probability'] = 0.7
            elif volatility > 0.15:  # 高波动率
                patterns['breakout_probability'] = 0.3
            else:
                patterns['breakout_probability'] = 0.5
            
            # 支撑阻力位
            high_24h = market_data.get('high_24h', price)
            low_24h = market_data.get('low_24h', price)
            
            patterns['support_levels'] = [low_24h * 0.99, low_24h * 0.97]
            patterns['resistance_levels'] = [high_24h * 1.01, high_24h * 1.03]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns for {symbol}: {e}")
            return {}
    
    async def analyze_volume_profile(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """分析成交量分布"""
        try:
            volume_profile = {
                'volume_trend': 'neutral',
                'volume_strength': 0.5,
                'volume_surge': False,
                'accumulation': False,
                'distribution': False
            }
            
            volume_24h = market_data.get('volume_24h', 0)
            volume_trend = market_data.get('volume_trend', 0)
            
            # 成交量趋势
            if volume_trend > 0.2:
                volume_profile['volume_trend'] = 'increasing'
            elif volume_trend < -0.2:
                volume_profile['volume_trend'] = 'decreasing'
            
            # 成交量强度
            volume_profile['volume_strength'] = min(1.0, volume_24h / 5000000)  # 500万作为基准
            
            # 成交量激增
            volume_profile['volume_surge'] = volume_trend > 0.5
            
            # 吸筹派发
            price_change = market_data.get('price_change_24h', 0)
            if volume_trend > 0.3 and abs(price_change) < 2:
                volume_profile['accumulation'] = True
            elif volume_trend > 0.3 and abs(price_change) > 5:
                volume_profile['distribution'] = True
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile for {symbol}: {e}")
            return {}
    
    async def analyze_sentiment(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """分析市场情绪"""
        try:
            sentiment = {
                'sentiment_score': 0.5,
                'sentiment_label': 'neutral',
                'fear_greed_index': 50,
                'social_mentions': 0,
                'news_sentiment': 0.5
            }
            
            price_change = market_data.get('price_change_24h', 0)
            volatility = market_data.get('volatility', 0)
            volume_trend = market_data.get('volume_trend', 0)
            
            # 基于价格变化的情绪
            if price_change > 5:
                sentiment['sentiment_score'] = 0.8
                sentiment['sentiment_label'] = 'very_bullish'
            elif price_change > 2:
                sentiment['sentiment_score'] = 0.65
                sentiment['sentiment_label'] = 'bullish'
            elif price_change < -5:
                sentiment['sentiment_score'] = 0.2
                sentiment['sentiment_label'] = 'very_bearish'
            elif price_change < -2:
                sentiment['sentiment_score'] = 0.35
                sentiment['sentiment_label'] = 'bearish'
            
            # 恐惧贪婪指数（简化）
            if volatility > 0.2:
                sentiment['fear_greed_index'] = 20  # 恐慌
            elif volatility < 0.05:
                sentiment['fear_greed_index'] = 80  # 贪婪
            else:
                sentiment['fear_greed_index'] = 50  # 中性
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {'sentiment_score': 0.5, 'sentiment_label': 'neutral'}
    
    def calculate_correlation(self, symbols: List[str], market_data: dict[str, dict[str, Any]]) -> dict[str, dict[float]]:
        """计算相关性"""
        try:
            if len(symbols) < 2:
                return {}
            
            # 提取价格数据
            prices = {}
            for symbol in symbols:
                if symbol in market_data:
                    prices[symbol] = market_data[symbol].get('price_change_24h', 0)
            
            if len(prices) < 2:
                return {}
            
            # 计算相关性矩阵
            correlations = {}
            symbol_list = list(prices.keys())
            
            for i, symbol1 in enumerate(symbol_list):
                correlations[symbol1] = {}
                for j, symbol2 in enumerate(symbol_list):
                    if i == j:
                        correlations[symbol1][symbol2] = 1.0
                    else:
                        # 简化的相关性计算
                        corr = self._calculate_simple_correlation(
                            prices[symbol1], prices[symbol2]
                        )
                        correlations[symbol1][symbol2] = corr
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}
    
    def _calculate_simple_correlation(self, x: float, y: float) -> float:
        """计算简化相关性"""
        # 简化的相关性计算，基于值的接近程度
        if x == 0 and y == 0:
            return 1.0
        elif x == 0 or y == 0:
            return 0.0
        else:
            # 计算符号一致性
            sign_correlation = 1.0 if (x > 0) == (y > 0) else -1.0
            
            # 计算大小相关性
            magnitude_correlation = 1.0 / (1.0 + abs(abs(x) - abs(y)))
            
            return sign_correlation * magnitude_correlation


# 示例使用
if __name__ == "__main__":
    async def test_data_processor():
        processor = DataProcessor()
        
        # 测试单个市场数据获取
        market_data = await processor.fetch_market_data("BTCUSDT")
        print(f"Market data: {market_data}")
        
        # 测试批量获取
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        batch_data = await processor.batch_fetch_market_data(symbols)
        print(f"Batch data: {len(batch_data)} items")
        
        # 测试模式分析
        patterns = await processor.analyze_patterns("BTCUSDT", market_data)
        print(f"Patterns: {patterns}")
        
        # 测试成交量分析
        volume_profile = await processor.analyze_volume_profile("BTCUSDT", market_data)
        print(f"Volume profile: {volume_profile}")
        
        # 测试情绪分析
        sentiment = await processor.analyze_sentiment("BTCUSDT", market_data)
        print(f"Sentiment: {sentiment}")
        
        # 测试相关性分析
        correlations = processor.calculate_correlation(symbols, batch_data)
        print(f"Correlations: {correlations}")
    
    # 运行测试
    import asyncio
    asyncio.run(test_data_processor())