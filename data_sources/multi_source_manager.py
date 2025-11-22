"""
多数据源管理器
支持Binance、OKX、CoinGecko的智能切换
"""

import asyncio
import aiohttp
import ccxt
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """数据源类型"""
    BINANCE = "binance"
    OKX = "okx"
    COINGECKO = "coingecko"


@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    high_24h: float
    low_24h: float
    timestamp: str
    source: str
    exchange: str


class RateLimiter:
    """API速率限制器"""
    
    def __init__(self):
        self.call_counts = {}  # {source: count}
        self.reset_times = {}  # {source: reset_time}
        
    async def acquire(self, source: DataSource, limit: int = 30, window: int = 60) -> bool:
        """获取API调用权限"""
        now = datetime.now()
        source_name = source.value
        
        # 初始化重置时间
        if source_name not in self.reset_times:
            self.reset_times[source_name] = now
            self.call_counts[source_name] = 0
        
        # 检查是否需要重置计数器
        if now >= self.reset_times[source_name]:
            self.call_counts[source_name] = 0
            self.reset_times[source_name] = now + timedelta(seconds=window)
        
        # 检查是否超过限制
        if self.call_counts[source_name] >= limit:
            wait_time = (self.reset_times[source_name] - now).total_seconds()
            logger.warning(f"{source_name} API限制，需等待 {wait_time:.1f} 秒")
            await asyncio.sleep(wait_time + 0.1)
            return await self.acquire(source, limit, window)
        
        self.call_counts[source_name] += 1
        return True


class MultiSourceManager:
    """多数据源管理器"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.exchanges = {}
        self._init_exchanges()
        self.coingecko_session = None
        
    def _init_exchanges(self):
        """初始化交易所连接"""
        try:
            # Binance连接 - 暂时禁用由于地理限制
            # self.exchanges[DataSource.BINANCE] = ccxt.binanceusdm({
            #     'enableRateLimit': True,
            #     'rateLimit': 200,
            #     'timeout': 30000,
            #     'options': {'adjustForTimeDifference': True}
            # })
            
            # OKX连接 (主要数据源)
            self.exchanges[DataSource.OKX] = ccxt.okx({
                'enableRateLimit': True,
                'rateLimit': 200,
                'timeout': 30000,
                'options': {'adjustForTimeDifference': True}
            })
            
            logger.info("交易所连接初始化完成")
            
        except Exception as e:
            logger.error(f"交易所连接初始化失败: {e}")
    
    async def init_coingecko(self):
        """初始化CoinGecko HTTP会话"""
        if self.coingecko_session is None:
            self.coingecko_session = aiohttp.ClientSession()
            logger.info("CoinGecko HTTP会话初始化完成")
    
    async def close(self):
        """关闭资源"""
        if self.coingecko_session:
            await self.coingecko_session.close()
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        智能获取市场数据：尝试OKX → CoinGecko → Binance (备用)
        """
        # 1. 优先尝试OKX (无地理限制)
        data = await self._try_okx(symbol)
        if data:
            return data
            
        # 2. 尝试CoinGecko (免费API)
        data = await self._try_coingecko(symbol)
        if data:
            return data
            
        # 3. 最后尝试Binance (地理限制的备用)
        data = await self._try_binance(symbol)
        if data:
            return data
        
        logger.error(f"所有数据源都无法获取 {symbol} 的数据")
        return None
    
    async def _try_binance(self, symbol: str) -> Optional[MarketData]:
        """尝试从Binance获取数据"""
        try:
            # 检查Binance是否已初始化 (由于地理限制可能未初始化)
            if DataSource.BINANCE not in self.exchanges:
                logger.info("Binance数据源已禁用 (地理限制)")
                return None
                
            await self.rate_limiter.acquire(DataSource.BINANCE)
            
            exchange = self.exchanges[DataSource.BINANCE]
            if not hasattr(exchange, 'markets') or not exchange.markets:
                exchange.load_markets()
            
            # 标准化交易对格式
            std_symbol = self._normalize_symbol(symbol)
            
            if std_symbol in exchange.markets:
                ticker = exchange.fetch_ticker(std_symbol)
                
                return MarketData(
                    symbol=std_symbol,
                    price=ticker['last'],
                    volume_24h=ticker['quoteVolume'],
                    price_change_24h=ticker['percentage'] or 0,
                    high_24h=ticker['high'],
                    low_24h=ticker['low'],
                    timestamp=datetime.now().isoformat(),
                    source="binance",
                    exchange="Binance"
                )
        except Exception as e:
            logger.warning(f"Binance数据获取失败: {e}")
        
        return None
    
    async def _try_okx(self, symbol: str) -> Optional[MarketData]:
        """尝试从OKX获取数据"""
        try:
            await self.rate_limiter.acquire(DataSource.OKX)
            
            exchange = self.exchanges[DataSource.OKX]
            if not hasattr(exchange, 'markets') or not exchange.markets:
                exchange.load_markets()
            
            # 标准化交易对格式
            std_symbol = self._normalize_symbol(symbol)
            
            if std_symbol in exchange.markets:
                ticker = exchange.fetch_ticker(std_symbol)
                
                return MarketData(
                    symbol=std_symbol,
                    price=ticker['last'],
                    volume_24h=ticker['quoteVolume'],
                    price_change_24h=ticker['percentage'] or 0,
                    high_24h=ticker['high'],
                    low_24h=ticker['low'],
                    timestamp=datetime.now().isoformat(),
                    source="okx",
                    exchange="OKX"
                )
        except Exception as e:
            logger.warning(f"OKX数据获取失败: {e}")
        
        return None
    
    async def _try_coingecko(self, symbol: str) -> Optional[MarketData]:
        """尝试从CoinGecko获取数据"""
        try:
            await self.rate_limiter.acquire(DataSource.COINGECKO, limit=25, window=60)  # CoinGecko免费版限制
            await self.init_coingecko()
            
            # 简化的CoinGecko获取方法
            data = await self._coingecko_simple_lookup(symbol)
            if data:
                return data
                    
        except Exception as e:
            logger.warning(f"CoinGecko数据获取失败: {e}")
        
        return None
    
    async def _coingecko_simple_lookup(self, symbol: str) -> Optional[MarketData]:
        """简化的CoinGecko价格获取"""
        try:
            # 符号映射
            symbol_mapping = {
                'BTCUSDT': 'bitcoin',
                'BTC': 'bitcoin', 
                'ETHUSDT': 'ethereum',
                'ETH': 'ethereum',
                'ADAUSDT': 'cardano',
                'ADA': 'cardano',
                'SOLUSDT': 'solana',
                'SOL': 'solana'
            }
            
            # 获取代币ID
            token_id = symbol_mapping.get(symbol.upper())
            if not token_id:
                return None
            
            # 使用简单价格API
            price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
            
            async with self.coingecko_session.get(price_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if token_id in data:
                        token_data = data[token_id]
                        current_price = token_data.get('usd', 0)
                        change_24h = token_data.get('usd_24h_change', 0)
                        volume_24h = token_data.get('usd_24h_vol', 0)
                        
                        if current_price > 0:
                            return MarketData(
                                symbol=symbol,
                                price=current_price,
                                volume_24h=volume_24h,
                                price_change_24h=change_24h,
                                high_24h=current_price * 1.1,  # 估算值
                                low_24h=current_price * 0.9,   # 估算值
                                timestamp=datetime.now().isoformat(),
                                source="coingecko",
                                exchange="CoinGecko"
                            )
                            
        except Exception as e:
            logger.warning(f"CoinGecko简单查找失败: {e}")
            
        return None
    
    async def _coingecko_symbol_lookup(self, symbol: str) -> Optional[MarketData]:
        """CoinGecko符号查找"""
        try:
            # 使用CoinGecko API搜索代币
            search_url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
            
            async with self.coingecko_session.get(search_url) as response:
                if response.status == 200:
                    search_data = await response.json()
                    
                    if search_data.get('coins') and len(search_data['coins']) > 0:
                        # 获取第一个匹配的结果
                        coin = search_data['coins'][0]
                        coin_id = coin['id']
                        
                        # 获取详细价格数据
                        price_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                        async with self.coingecko_session.get(price_url) as price_response:
                            if price_response.status == 200:
                                price_data = await price_response.json()
                                
                                market_data = price_data.get('market_data', {})
                                current_price = market_data.get('current_price', {}).get('usd', 0)
                                
                                if current_price > 0:
                                    return MarketData(
                                        symbol=coin['symbol'].upper() + 'USDT',
                                        price=current_price,
                                        volume_24h=market_data.get('total_volume', {}).get('usd', 0),
                                        price_change_24h=market_data.get('price_change_percentage_24h', 0),
                                        high_24h=market_data.get('high_24h', {}).get('usd', 0),
                                        low_24h=market_data.get('low_24h', {}).get('usd', 0),
                                        timestamp=datetime.now().isoformat(),
                                        source="coingecko",
                                        exchange="CoinGecko"
                                    )
        except Exception as e:
            logger.warning(f"CoinGecko符号查找失败 {symbol}: {e}")
        
        return None
    
    def _normalize_symbol(self, symbol: str) -> str:
        """标准化交易对格式"""
        return symbol.upper().replace('-', '').replace('_', '')
    
    async def get_available_symbols(self) -> List[str]:
        """获取可用交易对列表"""
        all_symbols = set()
        
        # 从Binance获取
        try:
            await self.rate_limiter.acquire(DataSource.BINANCE)
            binance = self.exchanges[DataSource.BINANCE]
            binance.load_markets()
            for symbol in binance.markets:
                if symbol.endswith('/USDT'):  # 只取USDT交易对
                    all_symbols.add(symbol)
        except Exception as e:
            logger.warning(f"Binance币种获取失败: {e}")
        
        # 从OKX获取
        try:
            await self.rate_limiter.acquire(DataSource.OKX)
            okx = self.exchanges[DataSource.OKX]
            okx.load_markets()
            for symbol in okx.markets:
                if symbol.endswith('/USDT'):
                    all_symbols.add(symbol)
        except Exception as e:
            logger.warning(f"OKX币种获取失败: {e}")
        
        return sorted(list(all_symbols))
    
    async def get_new_coins_from_coingecko(self, days: int = 7) -> List[str]:
        """从CoinGecko获取最近新币"""
        try:
            await self.rate_limiter.acquire(DataSource.COINGECKO, limit=5, window=60)
            await self.init_coingecko()
            
            # 获取新币列表
            new_coins_url = f"https://api.coingecko.com/api/v3/coins/list/new"
            
            async with self.coingecko_session.get(new_coins_url) as response:
                if response.status == 200:
                    coins = await response.json()
                    
                    # 过滤最近几天的新币
                    recent_coins = []
                    for coin in coins[:50]:  # 限制数量避免超时
                        recent_coins.append(coin['symbol'].upper() + 'USDT')
                    
                    return recent_coins
                    
        except Exception as e:
            logger.warning(f"CoinGecko新币获取失败: {e}")
        
        return []
    
    async def batch_get_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """批量获取市场数据"""
        results = {}
        tasks = []
        
        for symbol in symbols:
            task = asyncio.create_task(self.get_market_data(symbol))
            tasks.append((symbol, task))
        
        # 并发执行
        for symbol, task in tasks:
            try:
                data = await task
                if data:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"批量获取 {symbol} 数据失败: {e}")
        
        return results


# 全局数据源管理器实例
data_source_manager = MultiSourceManager()