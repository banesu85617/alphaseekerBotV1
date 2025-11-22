"""
验证机制工具模块
提供超时管理、重试管理等通用工具
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class TimeoutConfig:
    """超时配置"""
    default_timeout: float = 5.0
    layer1_timeout: float = 2.0
    layer2_timeout: float = 5.0
    total_timeout: float = 10.0


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_on_timeout: bool = True
    retry_on_error: bool = True


class TimeoutManager:
    """超时管理器"""
    
    def __init__(self, timeout_config: TimeoutConfig):
        """
        初始化超时管理器
        
        Args:
            timeout_config: 超时配置
        """
        self.config = timeout_config
    
    async def timeout_context(self, timeout_duration: float):
        """超时上下文管理器"""
        return TimeoutContext(timeout_duration)


class TimeoutContext:
    """超时上下文"""
    
    def __init__(self, timeout_duration: float):
        self.timeout_duration = timeout_duration
        self.task = None
    
    async def __aenter__(self):
        # 创建超时任务
        self.task = asyncio.create_task(asyncio.sleep(self.timeout_duration))
        return self.task
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


class RetryManager:
    """重试管理器"""
    
    def __init__(self, retry_config: RetryConfig):
        """
        初始化重试管理器
        
        Args:
            retry_config: 重试配置
        """
        self.config = retry_config
    
    async def retry(
        self, 
        func: Callable, 
        *args, 
        max_retries: Optional[int] = None, 
        retry_delay: Optional[float] = None,
        **kwargs
    ) -> T:
        """
        执行带重试的函数调用
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            max_retries: 最大重试次数（覆盖配置）
            retry_delay: 重试延迟（覆盖配置）
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 重试失败后抛出最后一次异常
        """
        max_retries = max_retries or self.config.max_retries
        retry_delay = retry_delay or self.config.retry_delay
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # 执行函数
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 成功返回结果
                if attempt > 0:
                    logger.info(f"函数执行成功，经过 {attempt} 次重试")
                return result
                
            except Exception as e:
                last_exception = e
                
                # 判断是否应该重试
                if not self._should_retry(e, attempt, max_retries):
                    logger.warning(f"函数执行失败，不进行重试: {str(e)}")
                    break
                
                # 如果是最后一次尝试
                if attempt == max_retries:
                    logger.error(f"函数执行失败，已达到最大重试次数 {max_retries}: {str(e)}")
                    break
                
                # 计算延迟时间
                delay = self._calculate_delay(attempt, retry_delay)
                
                logger.warning(f"函数执行失败，第 {attempt + 1} 次重试，{delay:.2f}秒后重试: {str(e)}")
                
                # 等待重试延迟
                await asyncio.sleep(delay)
        
        # 所有重试都失败，抛出最后一个异常
        raise last_exception

    def _should_retry(self, exception: Exception, attempt: int, max_retries: int) -> bool:
        """判断是否应该重试"""
        if attempt >= max_retries:
            return False
        
        # 根据异常类型判断
        if isinstance(exception, asyncio.TimeoutError):
            return self.config.retry_on_timeout
        elif isinstance(exception, Exception):
            return self.config.retry_on_error
        
        return False

    def _calculate_delay(self, attempt: int, base_delay: float) -> float:
        """计算重试延迟"""
        if self.config.retry_strategy == RetryStrategy.FIXED_DELAY:
            return base_delay
        
        elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay * (self.config.backoff_factor ** attempt)
        
        elif self.config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return base_delay * (1 + attempt)
        
        else:
            return base_delay

    async def batch_retry(
        self, 
        func: Callable, 
        items: List[Any], 
        max_workers: int = 5,
        **kwargs
    ) -> List[T]:
        """
        批量重试执行
        
        Args:
            func: 要执行的函数
            items: 要处理的项目列表
            max_workers: 最大并发数
            **kwargs: 关键字参数
            
        Returns:
            执行结果列表
        """
        logger.info(f"开始批量重试执行，处理 {len(items)} 个项目")
        
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_item(item):
            async with semaphore:
                return await self.retry(func, item, **kwargs)
        
        # 并发处理
        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"项目 {i} 处理失败: {str(result)}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results


class AsyncCache:
    """异步缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300):
        """
        初始化异步缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.cache:
            return None
        
        # 检查TTL
        if time.time() - self.timestamps[key] > self.ttl:
            await self.delete(key)
            return None
        
        return self.cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        # 检查缓存大小限制
        if len(self.cache) >= self.max_size and key not in self.cache:
            # 删除最旧的条目
            await self._evict_oldest()
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    async def delete(self, key: str) -> None:
        """删除缓存条目"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    async def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.timestamps.clear()
    
    async def _evict_oldest(self) -> None:
        """删除最旧的条目"""
        if not self.timestamps:
            return
        
        oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
        await self.delete(oldest_key)
    
    async def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    async def keys(self) -> List[str]:
        """获取所有缓存键"""
        return list(self.cache.keys())


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        初始化速率限制器
        
        Args:
            max_calls: 时间窗口内最大调用次数
            time_window: 时间窗口（秒）
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self) -> bool:
        """
        获取调用许可
        
        Returns:
            是否获得许可
        """
        now = time.time()
        
        # 清理过期调用记录
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        # 检查是否超过限制
        if len(self.calls) >= self.max_calls:
            return False
        
        # 记录本次调用
        self.calls.append(now)
        return True
    
    async def wait_for_slot(self) -> None:
        """等待获得调用槽位"""
        while not await self.acquire():
            # 计算下次可调用时间
            if self.calls:
                oldest_call = min(self.calls)
                wait_time = self.time_window - (time.time() - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0.1)


class CircuitBreaker:
    """熔断器"""
    
    def __init__(
        self, 
        failure_threshold: int = 5, 
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        初始化熔断器
        
        Args:
            failure_threshold: 失败阈值
            timeout: 熔断超时时间
            expected_exception: 预期的异常类型
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器调用函数
        
        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            # 执行函数
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 调用成功，重置熔断器
            self._on_success()
            return result
            
        except self.expected_exception as e:
            # 调用失败
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置熔断器"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self) -> None:
        """处理成功调用"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self) -> None:
        """处理失败调用"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
    
    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'failure_threshold': self.failure_threshold,
            'timeout': self.timeout
        }


class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """验证交易对符号格式"""
        if not symbol:
            return False
        
        # 基本格式检查（字母数字和特殊字符）
        import re
        pattern = r'^[A-Z0-9]+[A-Z0-9_]*$'
        return bool(re.match(pattern, symbol))
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """验证时间周期格式"""
        valid_timeframes = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]
        return timeframe in valid_timeframes
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """验证价格值"""
        return price > 0 and price < float('inf') and not price != price
    
    @staticmethod
    def calculate_risk_reward_ratio(
        entry: float, 
        stop_loss: float, 
        take_profit: float,
        direction: str
    ) -> Optional[float]:
        """计算风险回报比"""
        if not all(x is not None and x > 0 for x in [entry, stop_loss, take_profit]):
            return None
        
        if direction == 'long':
            risk = entry - stop_loss
            reward = take_profit - entry
        elif direction == 'short':
            risk = stop_loss - entry
            reward = entry - take_profit
        else:
            return None
        
        if risk <= 0:
            return None
        
        return reward / risk
    
    @staticmethod
    def format_processing_time(seconds: float) -> str:
        """格式化处理时间"""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """清理文件名"""
        import re
        # 移除或替换不安全的字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 限制长度
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 5):
        """
        初始化批量处理器
        
        Args:
            batch_size: 批处理大小
            max_workers: 最大工作协程数
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    async def process_batch(
        self, 
        items: List[Any], 
        processor_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        批处理项目
        
        Args:
            items: 要处理的项目列表
            processor_func: 处理函数
            **kwargs: 关键字参数
            
        Returns:
            处理结果列表
        """
        logger.info(f"开始批处理 {len(items)} 个项目")
        
        results = []
        
        # 分批处理
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # 并发处理当前批次
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def process_item(item):
                async with semaphore:
                    try:
                        if asyncio.iscoroutinefunction(processor_func):
                            return await processor_func(item, **kwargs)
                        else:
                            return processor_func(item, **kwargs)
                    except Exception as e:
                        logger.error(f"处理项目失败: {str(e)}")
                        return None
            
            batch_tasks = [process_item(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理异常结果
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(None)
                else:
                    results.append(result)
            
            logger.info(f"批处理进度: {min(i + self.batch_size, len(items))}/{len(items)}")
        
        return results