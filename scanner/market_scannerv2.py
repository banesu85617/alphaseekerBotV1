"""
AlphaSeeker 市场扫描器 (Market Scanner)

提供实时市场数据扫描和分析功能，支持：
- 多数据源集成
- 技术指标计算
- 信号生成和筛选
- 实时数据更新
- 配置化管理
"""

import os
import sys
import json
import asyncio
import aiohttp
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

# 导入配置模块
from .config import ConfigManager, ScanConfig, DataSource, ScanMode
from pipeline import StrategyType, StrategySignal

class ScannerStatus(Enum):
    """扫描器状态枚举"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class MarketData:
    """市场数据类"""
    
    def __init__(self, symbol: str, price: float, volume: int = 0, timestamp: datetime = None):
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.timestamp = timestamp or datetime.now()
        self.change = 0.0
        self.change_percent = 0.0

@dataclass
class ScanResult:
    """扫描结果"""
    symbol: str
    signal_type: StrategyType
    confidence: float
    price: float
    volume: int
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MarketScanner:
    """市场扫描器主类"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # 获取扫描配置
        self.scan_config = self.config_manager.get_scan_config()
        
        # 状态管理
        self.status = ScannerStatus.STOPPED
        self.last_scan_time = None
        self.scan_results: List[ScanResult] = []
        
        # 数据缓存
        self.market_data_cache = {}
        self.last_cache_time = {}
        
        # 回调函数
        self.on_result_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        self.logger.info(f"市场扫描器初始化完成: {self.scan_config.name}")
    
    def add_result_callback(self, callback: Callable[[ScanResult], None]):
        """添加结果回调函数"""
        self.on_result_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """添加错误回调函数"""
        self.on_error_callbacks.append(callback)
    
    def _notify_result(self, result: ScanResult):
        """通知结果"""
        for callback in self.on_result_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")
    
    def _notify_error(self, error: Exception):
        """通知错误"""
        for callback in self.on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"错误回调函数执行失败: {e}")
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # 检查缓存
            cache_key = f"{symbol}_{period}"
            if (cache_key in self.market_data_cache and 
                (datetime.now() - self.last_cache_time.get(cache_key, datetime.min)).seconds < 60):
                return self.market_data_cache[cache_key]
            
            # 从yfinance获取数据
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data is not None and not data.empty:
                # 缓存数据
                self.market_data_cache[cache_key] = data
                self.last_cache_time[cache_key] = datetime.now()
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败 {symbol}: {e}")
            self._notify_error(e)
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标"""
        try:
            indicators = {}
            
            # 移动平均线
            indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
            indicators['ema_12'] = data['Close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            indicators['rsi'] = indicators['rsi'].iloc[-1]
            
            # MACD
            macd_line = indicators['ema_12'] - indicators['ema_26']
            signal_line = macd_line.ewm(span=9).mean()
            indicators['macd'] = macd_line.iloc[-1] if hasattr(macd_line, 'iloc') else macd_line
            indicators['macd_signal'] = signal_line.iloc[-1] if hasattr(signal_line, 'iloc') else signal_line
            
            # 布林带
            sma_20 = indicators['sma_20']
            std_20 = data['Close'].rolling(window=20).std().iloc[-1]
            indicators['bollinger_upper'] = sma_20 + (std_20 * 2)
            indicators['bollinger_lower'] = sma_20 - (std_20 * 2)
            
            # 成交量指标
            indicators['volume_sma'] = data['Volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = data['Volume'].iloc[-1] / indicators['volume_sma']
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            return {}
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, float]) -> Optional[ScanResult]:
        """生成交易信号"""
        try:
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            signal_score = 0.0
            signal_types = []
            
            # 移动平均线信号
            if current_price > indicators['sma_20'] > indicators['sma_50']:
                signal_score += 0.3
                signal_types.append("bullish_ma")
            elif current_price < indicators['sma_20'] < indicators['sma_50']:
                signal_score -= 0.3
                signal_types.append("bearish_ma")
            
            # RSI信号
            if indicators['rsi'] < 30:
                signal_score += 0.4
                signal_types.append("oversold_rsi")
            elif indicators['rsi'] > 70:
                signal_score -= 0.4
                signal_types.append("overbought_rsi")
            
            # MACD信号
            if indicators['macd'] > indicators['macd_signal']:
                signal_score += 0.2
                signal_types.append("bullish_macd")
            else:
                signal_score -= 0.2
                signal_types.append("bearish_macd")
            
            # 布林带信号
            if current_price <= indicators['bollinger_lower']:
                signal_score += 0.3
                signal_types.append("bollinger_bounce")
            elif current_price >= indicators['bollinger_upper']:
                signal_score -= 0.3
                signal_types.append("bollinger_resistance")
            
            # 成交量确认
            if indicators['volume_ratio'] > 1.5:
                signal_score *= 1.2  # 成交量放大信号
            
            # 转换为0-1置信度
            confidence = max(0, min(1, (signal_score + 1) / 2))
            
            # 只返回置信度超过阈值的信号
            if confidence >= self.scan_config.confidence_threshold:
                # 确定主要信号类型
                if signal_score > 0.2:
                    signal_type = StrategyType.ALPHA_SEEKING
                elif signal_score < -0.2:
                    signal_type = StrategyType.DAY_TRADING
                else:
                    signal_type = StrategyType.SWING_TRADING
                
                return ScanResult(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    volume=current_volume,
                    indicators=indicators,
                    metadata={'signal_types': signal_types, 'signal_score': signal_score}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"生成信号失败 {symbol}: {e}")
            return None
    
    async def scan_symbol(self, symbol: str) -> Optional[ScanResult]:
        """扫描单个股票"""
        try:
            # 获取数据
            data = self.get_stock_data(symbol)
            if data is None or data.empty:
                return None
            
            # 计算技术指标
            indicators = self.calculate_technical_indicators(data)
            if not indicators:
                return None
            
            # 生成信号
            result = self.generate_signal(symbol, data, indicators)
            return result
            
        except Exception as e:
            self.logger.error(f"扫描股票 {symbol} 失败: {e}")
            self._notify_error(e)
            return None
    
    async def scan_markets(self, symbols: Optional[List[str]] = None) -> List[ScanResult]:
        """扫描市场"""
        try:
            # 设置状态
            self.status = ScannerStatus.RUNNING
            self.last_scan_time = datetime.now()
            
            # 使用配置中的股票列表
            if symbols is None:
                symbols = self.scan_config.symbols[:self.scan_config.max_symbols]
            
            self.logger.info(f"开始扫描 {len(symbols)} 只股票")
            
            # 并发扫描
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                tasks = []
                for symbol in symbols:
                    task = asyncio.create_task(self.scan_symbol(symbol))
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            valid_results = []
            for result in results:
                if isinstance(result, ScanResult):
                    valid_results.append(result)
                    self._notify_result(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"扫描任务异常: {result}")
            
            self.scan_results = valid_results
            self.status = ScannerStatus.STOPPED
            
            self.logger.info(f"扫描完成，找到 {len(valid_results)} 个有效信号")
            return valid_results
            
        except Exception as e:
            self.logger.error(f"市场扫描失败: {e}")
            self.status = ScannerStatus.ERROR
            self._notify_error(e)
            return []
    
    def start_scan(self, symbols: Optional[List[str]] = None) -> List[ScanResult]:
        """同步启动扫描"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.scan_markets(symbols))
            loop.close()
            return results
        except Exception as e:
            self.logger.error(f"启动扫描失败: {e}")
            return []
    
    def get_latest_results(self) -> List[ScanResult]:
        """获取最新扫描结果"""
        return self.scan_results.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """获取扫描器状态"""
        return {
            'status': self.status.value,
            'last_scan_time': self.last_scan_time,
            'symbols_count': len(self.scan_config.symbols),
            'results_count': len(self.scan_results),
            'config_name': self.scan_config.name
        }
    
    def stop_scan(self):
        """停止扫描"""
        self.status = ScannerStatus.STOPPED
        self.logger.info("扫描已停止")
    
    def update_config(self, **kwargs) -> bool:
        """更新配置"""
        try:
            success = self.config_manager.update_scan_config(**kwargs)
            if success:
                self.scan_config = self.config_manager.get_scan_config()
                self.logger.info(f"配置已更新: {kwargs}")
            return success
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")
            return False

# 创建默认扫描器实例
_default_scanner = None

def get_market_scanner(config_manager: Optional[ConfigManager] = None) -> MarketScanner:
    """获取默认市场扫描器实例"""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = MarketScanner(config_manager)
    return _default_scanner

__version__ = "1.0.0"
__author__ = "AlphaSeeker Team"

__all__ = [
    "MarketScanner",
    "ScanResult", 
    "MarketData",
    "ScannerStatus",
    "get_market_scanner"
]
