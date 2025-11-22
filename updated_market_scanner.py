#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新后的MarketScanner模块 - 兼容有无yfinance两种情况
请将此文件内容复制到: scanner/market_scanner.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Tuple
import logging
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入yfinance，如果失败则使用模拟数据
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.info("✅ yfinance可用，将使用真实数据源")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("⚠️ yfinance不可用，将使用模拟数据进行测试")

class ScanMode:
    """扫描模式枚举"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    SCHEDULED = "scheduled"

def get_default_scan_config():
    """获取默认扫描配置"""
    return {
        'name': 'default_scan',
        'mode': ScanMode.REAL_TIME,
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'scan_interval': 300,
        'technical_indicators': ['sma', 'rsi', 'macd', 'bollinger']
    }

def generate_mock_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """生成模拟股票数据用于测试"""
    np.random.seed(hash(symbol) % 2**32)  # 为每个symbol生成一致的数据
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='D')
    
    # 生成基础价格
    base_price = {'AAPL': 150, 'GOOGL': 2800, 'MSFT': 350, 'TSLA': 200}.get(symbol, 100)
    
    # 生成随机游走价格
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for i in range(1, len(dates)):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1))  # 确保价格为正
    
    df = pd.DataFrame({
        'date': dates,
        'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'high': [p * np.random.uniform(1.01, 1.05) for p in prices],
        'low': [p * np.random.uniform(0.95, 0.99) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in range(len(dates))]
    })
    
    df.set_index('date', inplace=True)
    return df

def get_stock_data(symbol: str) -> pd.DataFrame:
    """获取股票数据，支持真实和模拟数据"""
    if YFINANCE_AVAILABLE:
        try:
            # 使用yfinance获取真实数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")
            if df.empty:
                logger.warning(f"{symbol} 数据为空，使用模拟数据")
                return generate_mock_data(symbol)
            return df
        except Exception as e:
            logger.warning(f"获取{symbol}真实数据失败: {e}，使用模拟数据")
            return generate_mock_data(symbol)
    else:
        # 使用模拟数据
        logger.info(f"为{symbol}生成模拟数据")
        return generate_mock_data(symbol)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    df = df.copy()
    
    # SMA - 简单移动平均线
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # EMA - 指数移动平均线
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI - 相对强弱指标
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD - 移动平均收敛散度
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 布林带
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    return df

def calculate_confidence(df: pd.DataFrame, signal_type: str) -> float:
    """计算信号置信度"""
    try:
        current = df.iloc[-1]
        
        base_confidence = 0.5
        
        # RSI置信度
        if signal_type == 'bullish' and current['rsi'] < 30:
            base_confidence += 0.2
        elif signal_type == 'bearish' and current['rsi'] > 70:
            base_confidence += 0.2
            
        # MACD置信度
        if signal_type == 'bullish' and current['macd'] > current['macd_signal']:
            base_confidence += 0.15
        elif signal_type == 'bearish' and current['macd'] < current['macd_signal']:
            base_confidence += 0.15
            
        # 价格位置置信度
        if signal_type == 'bullish' and current['close'] < current['bb_lower'] * 1.02:
            base_confidence += 0.1
        elif signal_type == 'bearish' and current['close'] > current['bb_upper'] * 0.98:
            base_confidence += 0.1
            
        # 成交量确认
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        if current['volume'] > avg_volume * 1.5:
            base_confidence += 0.1
            
        return min(max(base_confidence, 0.1), 0.95)
    except Exception as e:
        logger.warning(f"置信度计算错误: {e}")
        return 0.5

def generate_signals(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """生成交易信号"""
    signals = []
    
    try:
        if len(df) < 50:
            logger.warning(f"{symbol} 数据不足，无法生成可靠信号")
            return signals
            
        current = df.iloc[-1]
        
        # 看涨信号检测
        bullish_signals = []
        if current['close'] > current['sma_20']:
            bullish_signals.append('价格突破SMA20')
        if current['rsi'] < 40:
            bullish_signals.append('RSI超卖')
        if current['macd'] > current['macd_signal'] and len(df) > 1:
            prev_macd = df.iloc[-2]['macd']
            prev_signal = df.iloc[-2]['macd_signal']
            if prev_macd <= prev_signal:
                bullish_signals.append('MACD金叉')
        if current['close'] < current['bb_lower'] * 1.02:
            bullish_signals.append('布林带下轨反弹')
            
        if bullish_signals:
            confidence = calculate_confidence(df, 'bullish')
            
            signals.append({
                'symbol': symbol,
                'type': 'BUY',
                'price': float(current['close']),
                'confidence': confidence,
                'reasons': bullish_signals,
                'timestamp': datetime.now(),
                'volume_ratio': float(current['volume'] / df['volume'].rolling(20).mean().iloc[-1])
            })
        
        # 看跌信号检测
        bearish_signals = []
        if current['close'] < current['sma_20']:
            bearish_signals.append('价格跌破SMA20')
        if current['rsi'] > 60:
            bearish_signals.append('RSI超买')
        if current['macd'] < current['macd_signal'] and len(df) > 1:
            prev_macd = df.iloc[-2]['macd']
            prev_signal = df.iloc[-2]['macd_signal']
            if prev_macd >= prev_signal:
                bearish_signals.append('MACD死叉')
        if current['close'] > current['bb_upper'] * 0.98:
            bearish_signals.append('布林带上轨阻力')
            
        if bearish_signals:
            confidence = calculate_confidence(df, 'bearish')
            
            signals.append({
                'symbol': symbol,
                'type': 'SELL',
                'price': float(current['close']),
                'confidence': confidence,
                'reasons': bearish_signals,
                'timestamp': datetime.now(),
                'volume_ratio': float(current['volume'] / df['volume'].rolling(20).mean().iloc[-1])
            })
            
    except Exception as e:
        logger.error(f"生成{symbol}信号时出错: {e}")
        
    return signals

class MarketScanner:
    """市场扫描器 - 执行技术分析和信号生成"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_default_scan_config()
        self._data_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_duration = timedelta(seconds=60)
        self._is_scanning = False
        
        data_source = "真实数据源 (yfinance)" if YFINANCE_AVAILABLE else "模拟数据"
        logger.info(f"🚀 MarketScanner初始化完成，数据源: {data_source}")
    
    def get_status(self) -> Dict:
        """获取扫描器状态"""
        return {
            'is_scanning': self._is_scanning,
            'cache_size': len(self._data_cache),
            'config': self.config,
            'data_source': 'yfinance' if YFINANCE_AVAILABLE else 'mock_data'
        }
    
    def _get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        if symbol in self._data_cache:
            df, timestamp = self._data_cache[symbol]
            if datetime.now() - timestamp < self._cache_duration:
                logger.debug(f"使用{symbol}的缓存数据")
                return df
        return None
    
    def _set_cached_data(self, symbol: str, df: pd.DataFrame) -> None:
        """设置缓存数据"""
        self._data_cache[symbol] = (df, datetime.now())
    
    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """获取股票数据（带缓存）"""
        # 先检查缓存
        cached_data = self._get_cached_data(symbol)
        if cached_data is not None:
            return cached_data
            
        # 获取新数据
        df = get_stock_data(symbol)
        
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        # 缓存数据
        self._set_cached_data(symbol, df)
        
        return df
    
    def scan_single(self, symbol: str) -> List[Dict]:
        """扫描单个股票"""
        try:
            logger.info(f"🔍 正在扫描 {symbol}...")
            df = self.get_stock_data(symbol)
            
            if df.empty:
                logger.warning(f"{symbol} 数据为空")
                return []
                
            signals = generate_signals(df, symbol)
            logger.info(f"{symbol} 生成 {len(signals)} 个信号")
            return signals
            
        except Exception as e:
            logger.error(f"扫描 {symbol} 失败: {e}")
            return []
    
    def scan_symbols(self, symbols: List[str]) -> List[Dict]:
        """扫描多个股票"""
        logger.info(f"🚀 开始扫描 {len(symbols)} 个股票...")
        all_signals = []
        
        for symbol in symbols:
            signals = self.scan_single(symbol)
            all_signals.extend(signals)
            
        logger.info(f"✅ 扫描完成，共生成 {len(all_signals)} 个信号")
        return all_signals

# 全局实例
_scanner_instance = None

def get_market_scanner(config: Optional[Dict] = None) -> MarketScanner:
    """获取MarketScanner实例（单例模式）"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = MarketScanner(config)
    return _scanner_instance

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试MarketScanner...")
    scanner = get_market_scanner()
    print(f"📊 状态: {scanner.get_status()}")
    
    # 测试扫描
    test_symbols = ["AAPL", "GOOGL", "MSFT"]
    signals = scanner.scan_symbols(test_symbols)
    
    print(f"\n📈 扫描结果 ({len(signals)} 个信号):")
    for signal in signals:
        print(f"  {signal['symbol']}: {signal['type']} @ ${signal['price']:.2f} (置信度: {signal['confidence']:.2f})")
        print(f"    原因: {', '.join(signal['reasons'])}")
    
    print("\n✅ MarketScanner测试完成!")