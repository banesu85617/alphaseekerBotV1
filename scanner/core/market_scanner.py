"""
市场扫描和深度分析系统
高效的市场扫描和深度分析功能，支持数百交易对的并行处理

主要功能：
1. 并行市场扫描
2. 智能交易对筛选和优先级排序
3. 深度分析触发机制
4. 多级缓存系统
5. 扫描结果聚合和统计
6. 实时市场监控和警报
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
import threading
from enum import Enum

# 缓存系统
from ..cache.redis_cache import RedisCache
from ..cache.memory_cache import MemoryCache

# 策略系统
from ..strategies.scan_strategies import BaseStrategy, PriorityStrategy, FilterStrategy

# 监控系统
from ..monitoring.performance_monitor import PerformanceMonitor
from ..monitoring.alert_manager import AlertManager

# 工具类
from ..utils.data_processor import DataProcessor
from ..utils.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class ScanStatus(Enum):
    """扫描状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ScanConfig:
    """扫描配置"""
    # 基础配置
    max_tickers: int = 100
    batch_size: int = 20
    max_workers: int = 10
    timeout: float = 30.0
    
    # 过滤配置
    min_volume: float = 1000000.0
    min_market_cap: float = 10000000.0
    allowed_symbols: List[str] = field(default_factory=list)
    excluded_symbols: List[str] = field(default_factory=list)
    
    # 深度分析配置
    enable_deep_analysis: bool = True
    deep_analysis_threshold: float = 0.7
    max_deep_analysis_pairs: int = 5
    
    # 缓存配置
    cache_ttl: int = 60  # 秒
    enable_redis: bool = True
    
    # 监控配置
    enable_monitoring: bool = True
    alert_threshold: float = 0.8
    
    # 策略配置
    priority_strategy: str = "volume_volume"  # volume_volume, volatility_volume, custom
    filter_strategy: str = "strict"  # strict, balanced, permissive


@dataclass
class ScanResult:
    """扫描结果"""
    symbol: str
    timestamp: datetime
    score: float
    confidence: float
    volume: float
    price_change_24h: float
    volatility: float
    technical_score: float
    sentiment_score: float
    risk_score: float
    deep_analysis_required: bool = False
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanReport:
    """扫描报告"""
    scan_id: str
    timestamp: datetime
    duration: float
    total_symbols: int
    analyzed_symbols: int
    filtered_symbols: int
    top_opportunities: List[ScanResult]
    statistics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class MarketScanner:
    """市场扫描器主类"""
    
    def __init__(
        self,
        config: ScanConfig,
        redis_client: Optional[Any] = None,
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        self.config = config
        self.redis_client = redis_client
        self.callbacks = callbacks or {}
        
        # 初始化缓存系统
        self.memory_cache = MemoryCache(default_ttl=config.cache_ttl)
        self.redis_cache = RedisCache(redis_client) if redis_client else None
        
        # 初始化策略
        self.priority_strategy = self._load_priority_strategy()
        self.filter_strategy = self._load_filter_strategy()
        
        # 初始化监控组件
        self.performance_monitor = PerformanceMonitor() if config.enable_monitoring else None
        self.alert_manager = AlertManager() if config.enable_monitoring else None
        
        # 初始化工具类
        self.data_processor = DataProcessor()
        self.metrics_calculator = MetricsCalculator()
        
        # 状态管理
        self.status = ScanStatus.IDLE
        self.current_scan_id: Optional[str] = None
        self.scan_history: List[ScanReport] = []
        self.active_scans: Dict[str, asyncio.Task] = {}
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # 统计信息
        self.total_scans = 0
        self.total_symbols_processed = 0
        self.avg_scan_duration = 0.0
        self.last_scan_start_time: Optional[datetime] = None
        
        logger.info(f"MarketScanner initialized with config: {config}")
    
    def _load_priority_strategy(self) -> PriorityStrategy:
        """加载优先级策略"""
        strategy_map = {
            "volume_volume": "volume",
            "volatility_volume": "volatility",
            "custom": "custom"
        }
        return PriorityStrategy(strategy_map.get(self.config.priority_strategy, "volume"))
    
    def _load_filter_strategy(self) -> FilterStrategy:
        """加载过滤策略"""
        strategy_map = {
            "strict": "strict",
            "balanced": "balanced", 
            "permissive": "permissive"
        }
        return FilterStrategy(strategy_map.get(self.config.filter_strategy, "balanced"))
    
    async def scan_markets(
        self,
        symbols: List[str],
        scan_type: str = "full",
        custom_filters: Optional[Dict[str, Any]] = None
    ) -> ScanReport:
        """
        执行市场扫描
        
        Args:
            symbols: 交易对列表
            scan_type: 扫描类型 (full, quick, deep)
            custom_filters: 自定义过滤器
            
        Returns:
            ScanReport: 扫描报告
        """
        scan_id = f"scan_{int(time.time())}_{len(symbols)}"
        self.current_scan_id = scan_id
        self.status = ScanStatus.RUNNING
        self.last_scan_start_time = datetime.now()
        
        logger.info(f"Starting market scan {scan_id} for {len(symbols)} symbols")
        
        try:
            with self.performance_monitor.track_scan(scan_id) if self.performance_monitor else nullcontext():
                # 1. 预处理和过滤
                filtered_symbols = await self._preprocess_and_filter(symbols, custom_filters)
                
                # 2. 并行扫描
                scan_results = await self._parallel_scan(filtered_symbols, scan_type)
                
                # 3. 深度分析触发
                if self.config.enable_deep_analysis:
                    scan_results = await self._trigger_deep_analysis(scan_results)
                
                # 4. 结果聚合和排序
                final_results = await self._aggregate_and_rank(scan_results)
                
                # 5. 生成报告
                report = await self._generate_report(scan_id, final_results)
                
                # 6. 更新统计信息
                await self._update_statistics(report)
                
                # 7. 缓存结果
                await self._cache_results(report)
                
                # 8. 发送警报
                if self.alert_manager:
                    await self._check_and_send_alerts(report)
                
                self.status = ScanStatus.COMPLETED
                logger.info(f"Scan {scan_id} completed successfully")
                return report
                
        except Exception as e:
            self.status = ScanStatus.ERROR
            logger.error(f"Scan {scan_id} failed: {str(e)}")
            raise
        finally:
            self.current_scan_id = None
    
    async def _preprocess_and_filter(
        self,
        symbols: List[str],
        custom_filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """预处理和过滤交易对"""
        start_time = time.time()
        
        try:
            # 1. 应用配置过滤器
            filtered_symbols = self.filter_strategy.filter_symbols(
                symbols,
                allowed=self.config.allowed_symbols,
                excluded=self.config.excluded_symbols
            )
            
            # 2. 应用自定义过滤器
            if custom_filters:
                filtered_symbols = await self._apply_custom_filters(filtered_symbols, custom_filters)
            
            # 3. 按优先级排序
            filtered_symbols = await self._sort_by_priority(filtered_symbols)
            
            # 4. 限制数量
            if len(filtered_symbols) > self.config.max_tickers:
                filtered_symbols = filtered_symbols[:self.config.max_tickers]
            
            logger.info(f"Filtered {len(symbols)} symbols to {len(filtered_symbols)} in {time.time() - start_time:.2f}s")
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return symbols  # 返回原始列表作为备选方案
    
    async def _parallel_scan(self, symbols: List[str], scan_type: str) -> List[ScanResult]:
        """并行扫描"""
        start_time = time.time()
        results = []
        
        # 分批处理
        batches = [symbols[i:i + self.config.batch_size] for i in range(0, len(symbols), self.config.batch_size)]
        
        logger.info(f"Processing {len(symbols)} symbols in {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            batch_tasks = []
            
            # 为每个symbol创建异步任务
            for symbol in batch:
                task = asyncio.create_task(
                    self._scan_single_symbol(symbol, scan_type)
                )
                batch_tasks.append((symbol, task))
            
            # 等待批次完成
            completed_tasks = []
            for symbol, task in batch_tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=self.config.timeout)
                    if result:
                        completed_tasks.append(result)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout scanning {symbol}")
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {str(e)}")
            
            results.extend(completed_tasks)
            logger.info(f"Completed batch {batch_idx + 1}/{len(batches)}")
        
        logger.info(f"Parallel scan completed in {time.time() - start_time:.2f}s for {len(results)} results")
        return results
    
    async def _scan_single_symbol(self, symbol: str, scan_type: str) -> Optional[ScanResult]:
        """扫描单个交易对"""
        try:
            # 检查缓存
            cache_key = f"scan:{symbol}:{scan_type}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # 获取市场数据
            market_data = await self._fetch_market_data(symbol)
            if not market_data:
                return None
            
            # 计算指标
            technical_indicators = await self._calculate_technical_indicators(market_data)
            
            # 计算评分
            scores = await self._calculate_scores(market_data, technical_indicators)
            
            # 构建结果
            result = ScanResult(
                symbol=symbol,
                timestamp=datetime.now(),
                score=scores['combined_score'],
                confidence=scores['confidence'],
                volume=market_data.get('volume', 0),
                price_change_24h=market_data.get('price_change_24h', 0),
                volatility=technical_indicators.get('volatility', 0),
                technical_score=scores['technical_score'],
                sentiment_score=scores['sentiment_score'],
                risk_score=scores['risk_score'],
                deep_analysis_required=scores['combined_score'] > self.config.deep_analysis_threshold,
                metadata={
                    'scan_type': scan_type,
                    'timestamp': datetime.now().isoformat(),
                    'market_data': market_data,
                    'technical_indicators': technical_indicators
                }
            )
            
            # 缓存结果
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {str(e)}")
            return None
    
    async def _trigger_deep_analysis(self, scan_results: List[ScanResult]) -> List[ScanResult]:
        """触发深度分析"""
        # 筛选需要深度分析的交易对
        candidates = [
            result for result in scan_results 
            if result.deep_analysis_required
        ]
        
        # 按评分排序，限制数量
        candidates.sort(key=lambda x: x.score, reverse=True)
        candidates = candidates[:self.config.max_deep_analysis_pairs]
        
        logger.info(f"Triggering deep analysis for {len(candidates)} high-priority pairs")
        
        # 并行执行深度分析
        if candidates:
            tasks = [
                self._perform_deep_analysis(result)
                for result in candidates
            ]
            
            deep_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 更新结果
            for i, (result, deep_result) in enumerate(zip(candidates, deep_results)):
                if isinstance(deep_result, Exception):
                    logger.error(f"Deep analysis failed for {result.symbol}: {deep_result}")
                else:
                    # 合并深度分析结果
                    result.analysis_data.update(deep_result)
                    result.score = self._adjust_score_with_deep_analysis(result.score, deep_result)
        
        return scan_results
    
    async def _perform_deep_analysis(self, result: ScanResult) -> Dict[str, Any]:
        """执行深度分析"""
        try:
            # 调用深度分析服务
            if 'deep_analysis_callback' in self.callbacks:
                analysis_data = await self.callbacks['deep_analysis_callback'](result.symbol, result.metadata)
                return analysis_data
            else:
                # 默认深度分析逻辑
                analysis_data = {
                    'pattern_recognition': await self._analyze_patterns(result.symbol),
                    'volume_profile': await self._analyze_volume_profile(result.symbol),
                    'order_flow': await self._analyze_order_flow(result.symbol),
                    'correlation_analysis': await self._analyze_correlations(result.symbol),
                    'sentiment_analysis': await self._analyze_sentiment(result.symbol),
                    'timestamp': datetime.now().isoformat()
                }
                return analysis_data
                
        except Exception as e:
            logger.error(f"Deep analysis failed for {result.symbol}: {str(e)}")
            return {'error': str(e)}
    
    async def _aggregate_and_rank(self, scan_results: List[ScanResult]) -> List[ScanResult]:
        """聚合和排序结果"""
        if not scan_results:
            return []
        
        try:
            # 按综合评分排序
            scan_results.sort(key=lambda x: x.score, reverse=True)
            
            # 计算统计信息
            scores = [r.score for r in scan_results]
            statistics = {
                'mean_score': np.mean(scores),
                'median_score': np.median(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'top_quartile': np.percentile(scores, 75),
                'bottom_quartile': np.percentile(scores, 25)
            }
            
            # 应用统计调整
            for result in scan_results:
                result.metadata['statistics'] = statistics
            
            logger.info(f"Aggregated and ranked {len(scan_results)} results")
            return scan_results
            
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            return scan_results
    
    async def _generate_report(self, scan_id: str, results: List[ScanResult]) -> ScanReport:
        """生成扫描报告"""
        end_time = datetime.now()
        duration = (end_time - self.last_scan_start_time).total_seconds() if self.last_scan_start_time else 0
        
        # 计算统计信息
        statistics = {
            'total_symbols': len(results),
            'high_opportunities': len([r for r in results if r.score > 0.8]),
            'medium_opportunities': len([r for r in results if 0.6 <= r.score <= 0.8]),
            'low_opportunities': len([r for r in results if r.score < 0.6]),
            'avg_volume': np.mean([r.volume for r in results]) if results else 0,
            'avg_volatility': np.mean([r.volatility for r in results]) if results else 0,
            'top_symbols': [r.symbol for r in results[:10]]
        }
        
        # 性能指标
        performance_metrics = {
            'duration': duration,
            'symbols_per_second': len(results) / duration if duration > 0 else 0,
            'cache_hit_rate': await self._calculate_cache_hit_rate(),
            'memory_usage': await self._get_memory_usage(),
        }
        
        # 警报
        alerts = await self._generate_alerts(results, statistics)
        
        # 创建报告
        report = ScanReport(
            scan_id=scan_id,
            timestamp=end_time,
            duration=duration,
            total_symbols=len(results),
            analyzed_symbols=len(results),
            filtered_symbols=0,  # TODO: 计算实际过滤数量
            top_opportunities=results[:10],  # Top 10
            statistics=statistics,
            performance_metrics=performance_metrics,
            alerts=alerts
        )
        
        return report
    
    async def _update_statistics(self, report: ScanReport):
        """更新统计信息"""
        self.total_scans += 1
        self.total_symbols_processed += report.analyzed_symbols
        
        # 更新平均扫描时间
        if self.avg_scan_duration == 0:
            self.avg_scan_duration = report.duration
        else:
            self.avg_scan_duration = (self.avg_scan_duration + report.duration) / 2
        
        # 保存到历史记录
        self.scan_history.append(report)
        
        # 限制历史记录数量
        if len(self.scan_history) > 100:
            self.scan_history = self.scan_history[-100:]
    
    # 辅助方法实现
    async def _get_cached_result(self, cache_key: str) -> Optional[ScanResult]:
        """获取缓存结果"""
        # 内存缓存
        cached = self.memory_cache.get(cache_key)
        if cached:
            return cached
        
        # Redis缓存
        if self.redis_cache:
            cached = await self.redis_cache.get(cache_key)
            if cached:
                return ScanResult(**cached)
        
        return None
    
    async def _cache_result(self, cache_key: str, result: ScanResult):
        """缓存结果"""
        # 内存缓存
        self.memory_cache.set(cache_key, result)
        
        # Redis缓存
        if self.redis_cache:
            await self.redis_cache.set(cache_key, result.__dict__)
    
    async def _cache_results(self, report: ScanReport):
        """缓存扫描结果"""
        cache_key = f"report:{report.scan_id}"
        
        # 缓存到Redis
        if self.redis_cache:
            await self.redis_cache.set(cache_key, report.__dict__, ttl=300)  # 5分钟
    
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        # 模拟实现 - 实际应该调用交易API
        try:
            # 这里应该调用真实的API
            return {
                'symbol': symbol,
                'price': 50000.0,  # 模拟价格
                'volume': 1000000.0,  # 模拟成交量
                'price_change_24h': 2.5,  # 模拟24h涨跌
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {str(e)}")
            return None
    
    async def _calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算技术指标"""
        # 模拟实现 - 实际应该计算真实的技术指标
        return {
            'rsi': 65.0,
            'macd': 0.5,
            'bollinger_position': 0.7,
            'volatility': 0.15,
            'trend_strength': 0.8
        }
    
    async def _calculate_scores(self, market_data: Dict[str, Any], technical_indicators: Dict[str, Any]) -> Dict[str, float]:
        """计算各种评分"""
        # 技术评分
        technical_score = min(1.0, technical_indicators.get('trend_strength', 0.5))
        
        # 情绪评分（基于成交量和价格变化）
        volume_factor = min(1.0, market_data.get('volume', 0) / 10000000.0)
        price_factor = abs(market_data.get('price_change_24h', 0)) / 10.0  # 归一化
        sentiment_score = min(1.0, (volume_factor + price_factor) / 2)
        
        # 风险评分（基于波动性）
        risk_score = 1.0 - min(1.0, technical_indicators.get('volatility', 0.2))
        
        # 综合评分
        combined_score = (technical_score * 0.4 + sentiment_score * 0.3 + risk_score * 0.3)
        confidence = min(1.0, (technical_score + sentiment_score + risk_score) / 3)
        
        return {
            'technical_score': technical_score,
            'sentiment_score': sentiment_score,
            'risk_score': risk_score,
            'combined_score': combined_score,
            'confidence': confidence
        }
    
    async def _check_and_send_alerts(self, report: ScanReport):
        """检查和发送警报"""
        if not self.alert_manager:
            return
        
        # 检查高机会警报
        high_opportunities = [r for r in report.top_opportunities if r.score > self.config.alert_threshold]
        
        if high_opportunities:
            alert = {
                'type': 'high_opportunities',
                'timestamp': datetime.now().isoformat(),
                'message': f"Found {len(high_opportunities)} high-potential opportunities",
                'data': {
                    'symbols': [r.symbol for r in high_opportunities],
                    'scores': [r.score for r in high_opportunities],
                    'scan_id': report.scan_id
                }
            }
            await self.alert_manager.send_alert(alert)
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'status': self.status.value,
            'current_scan_id': self.current_scan_id,
            'total_scans': self.total_scans,
            'total_symbols_processed': self.total_symbols_processed,
            'avg_scan_duration': self.avg_scan_duration,
            'active_scans': len(self.active_scans),
            'scan_history_count': len(self.scan_history)
        }
    
    def get_scan_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取扫描历史"""
        history = self.scan_history[-limit:] if limit > 0 else self.scan_history
        return [report.__dict__ for report in history]


# 辅助上下文管理器
from contextlib import nullcontext

if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test_scanner():
        config = ScanConfig()
        scanner = MarketScanner(config)
        
        # 模拟交易对列表
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
        
        # 执行扫描
        report = await scanner.scan_markets(symbols)
        print(f"Scan completed: {report.scan_id}")
        print(f"Duration: {report.duration:.2f}s")
        print(f"Total symbols: {report.total_symbols}")
        print(f"Top opportunities: {[r.symbol for r in report.top_opportunities]}")
    
    asyncio.run(test_scanner())