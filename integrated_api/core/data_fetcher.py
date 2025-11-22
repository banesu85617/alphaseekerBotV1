"""
数据获取模块
"""

import pandas as pd
import ccxt
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..core.exceptions import DataError, NetworkError, RateLimitError
from ..config.settings import settings

logger = logging.getLogger(__name__)


class DataFetcher:
    """数据获取器"""
    
    def __init__(self):
        self.exchange = None
        self._init_exchange()
    
    def _init_exchange(self):
        """初始化交易所连接"""
        try:
            # 创建CCXT实例
            exchange_config = {
                'enableRateLimit': settings.data.enable_rate_limit,
                'rateLimit': settings.data.rate_limit,
                'timeout': settings.data.timeout,
                'options': {'adjustForTimeDifference': True}
            }
            
            self.exchange = getattr(ccxt, settings.data.exchange_name)(exchange_config)
            logger.info(f"Initialized {settings.data.exchange_name} exchange connection")
            
            # 加载市场数据
            self._load_markets()
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise DataError(f"Exchange initialization failed: {e}")
    
    def _load_markets(self):
        """加载市场数据"""
        try:
            self.exchange.load_markets()
            logger.info(f"Loaded {len(self.exchange.markets)} markets")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise DataError(f"Market loading failed: {e}")
    
    def get_available_tickers(self) -> List[str]:
        """获取可用的交易对"""
        try:
            tickers = []
            for symbol, market in self.exchange.markets.items():
                # 只返回永续合约（USDT本位）
                if (market.get('contract', False) and 
                    market.get('quote', '') == 'USDT' and
                    market.get('type', '') == 'future'):
                    tickers.append(symbol)
            
            return sorted(tickers)
        except Exception as e:
            logger.error(f"Failed to get tickers: {e}")
            raise DataError(f"Ticker retrieval failed: {e}")
    
    def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """获取OHLCV数据"""
        logger.debug(f"Fetching OHLCV for {symbol} - timeframe: {timeframe}, limit: {limit}")
        
        try:
            # 验证symbol
            if symbol not in self.exchange.markets:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # 获取数据
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )
            
            if not ohlcv:
                logger.warning(f"No OHLCV data returned for {symbol}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 数据清洗和格式化
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 转换为数值类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除无效数据
            df.dropna(subset=numeric_columns, inplace=True)
            
            # 验证数据质量
            if df.empty:
                logger.warning(f"DataFrame became empty after cleaning for {symbol}")
                return df
            
            # 添加衍生数据
            df['returns'] = df['close'].pct_change()
            df['hl2'] = (df['high'] + df['low']) / 2
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            logger.debug(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error for {symbol}: {e}")
            raise NetworkError(f"Network error fetching {symbol}: {e}")
        
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded for {symbol}: {e}")
            raise RateLimitError(f"Rate limit exceeded: {e}")
        
        except ccxt.BadSymbol as e:
            logger.error(f"Invalid symbol {symbol}: {e}")
            raise ValueError(f"Invalid symbol '{symbol}'")
        
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {e}")
            raise DataError(f"Data fetch failed for {symbol}: {e}")
    
    def get_multiple_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 1000,
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """批量获取OHLCV数据"""
        logger.info(f"Fetching OHLCV for {len(symbols)} symbols with {max_workers} workers")
        
        async def fetch_single_symbol(symbol: str) -> tuple:
            """异步获取单个交易对数据"""
            try:
                df = self.get_ohlcv_data(symbol, timeframe, limit)
                return symbol, df
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        async def fetch_all():
            """异步获取所有数据"""
            semaphore = asyncio.Semaphore(max_workers)
            
            async def bounded_fetch(symbol: str):
                async with semaphore:
                    return await fetch_single_symbol(symbol)
            
            tasks = [bounded_fetch(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return dict(results)
        
        try:
            # 在线程池中运行异步代码
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(fetch_all())
            loop.close()
            
            # 统计结果
            successful = sum(1 for df in results.values() if not df.empty)
            failed = len(symbols) - successful
            
            logger.info(f"Batch fetch completed: {successful} successful, {failed} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            raise DataError(f"Batch data fetch failed: {e}")
    
    def get_recent_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """获取最新ticker数据"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'timestamp': ticker['timestamp'],
                'last': ticker['last'],
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'volume': ticker.get('baseVolume'),
                'percentage': ticker.get('percentage'),
                'change': ticker.get('change')
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise DataError(f"Ticker fetch failed for {symbol}: {e}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证交易对是否有效"""
        try:
            return symbol in self.exchange.markets
        except Exception:
            return False
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """获取交易对信息"""
        try:
            if symbol not in self.exchange.markets:
                raise ValueError(f"Symbol {symbol} not found")
            
            market = self.exchange.markets[symbol]
            
            return {
                'symbol': symbol,
                'type': market.get('type'),
                'spot': market.get('spot'),
                'future': market.get('future'),
                'contract': market.get('contract'),
                'active': market.get('active'),
                'quote': market.get('quote'),
                'base': market.get('base'),
                'precision': market.get('precision'),
                'limits': market.get('limits')
            }
            
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            raise DataError(f"Symbol info retrieval failed for {symbol}: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            if self.exchange:
                # CCXT不提供显式的关闭方法
                pass
        except Exception:
            pass


class DataCache:
    """数据缓存"""
    
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: pd.DataFrame):
        """设置缓存数据"""
        if len(self.cache) >= settings.data.max_cache_size:
            # 简单的LRU实现：删除最旧的条目
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (data.copy(), datetime.now().timestamp())
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


# 全局数据获取器实例
data_fetcher = DataFetcher()
data_cache = DataCache(ttl=settings.data.cache_ttl)