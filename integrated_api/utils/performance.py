"""
性能优化工具
"""

import asyncio
import time
import gc
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from functools import wraps
from ..config.settings import settings

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能优化工具类"""
    
    @staticmethod
    def time_execution(func: Callable) -> Callable:
        """执行时间装饰器"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    @staticmethod
    def memory_monitor(func: Callable) -> Callable:
        """内存监控装饰器"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = await func(*args, **kwargs)
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                
                if memory_diff > 10:  # 如果内存增长超过10MB
                    logger.warning(f"{func.__name__} memory usage: {memory_before:.1f}MB -> {memory_after:.1f}MB (+{memory_diff:.1f}MB)")
                
                return result
            except Exception as e:
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                logger.error(f"{func.__name__} memory usage at error: {memory_after:.1f}MB")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                
                if memory_diff > 10:  # 如果内存增长超过10MB
                    logger.warning(f"{func.__name__} memory usage: {memory_before:.1f}MB -> {memory_after:.1f}MB (+{memory_diff:.1f}MB)")
                
                return result
            except Exception as e:
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                logger.error(f"{func.__name__} memory usage at error: {memory_after:.1f}MB")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    @staticmethod
    async def batch_process(
        items: List[Any],
        process_func: Callable[[Any], Awaitable[Any]],
        batch_size: int = 10,
        max_concurrent: int = 5,
        timeout: float = 30.0
    ) -> List[Any]:
        """批量处理"""
        if not items:
            return []
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item_with_timeout(item):
            async with semaphore:
                try:
                    return await asyncio.wait_for(process_func(item), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout processing item: {item}")
                    return None
                except Exception as e:
                    logger.error(f"Error processing item {item}: {e}")
                    return None
        
        # 分批处理
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            tasks = [process_item_with_timeout(item) for item in batch]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    results.append(None)
                else:
                    results.append(result)
        
        return results
    
    @staticmethod
    def optimize_gc():
        """优化垃圾回收"""
        try:
            # 获取当前垃圾回收阈值
            thresholds = gc.get_threshold()
            logger.debug(f"Current GC thresholds: {thresholds}")
            
            # 设置更积极的垃圾回收
            gc.set_threshold(settings.performance.gc_threshold, 25, 10)
            
            # 强制垃圾回收
            collected = gc.collect()
            logger.debug(f"GC: collected {collected} objects")
            
        except Exception as e:
            logger.warning(f"GC optimization failed: {e}")
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """获取内存统计"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # 常驻内存
                "vms_mb": memory_info.vms / 1024 / 1024,  # 虚拟内存
                "percent": process.memory_percent(),       # 内存使用百分比
                "available_mb": psutil.virtual_memory().available / 1024 / 1024  # 可用内存
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    @staticmethod
    def get_cpu_stats() -> Dict[str, Any]:
        """获取CPU统计"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get CPU stats: {e}")
            return {}
    
    @staticmethod
    async def timeout_wrapper(coro: Awaitable, timeout: float):
        """超时包装器"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            raise
    
    @staticmethod
    def rate_limiter(calls_per_second: float):
        """速率限制装饰器"""
        min_interval = 1.0 / calls_per_second
        last_called = [0.0]
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed
                if left_to_wait > 0:
                    await asyncio.sleep(left_to_wait)
                
                last_called[0] = time.time()
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def cache_result(ttl: int = 300):
        """简单缓存装饰器"""
        cache = {}
        
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = str(args) + str(sorted(kwargs.items()))
                
                if key in cache:
                    value, timestamp = cache[key]
                    if time.time() - timestamp < ttl:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return value
                
                result = await func(*args, **kwargs)
                cache[key] = (result, time.time())
                
                # 清理过期缓存
                current_time = time.time()
                expired_keys = [k for k, (_, ts) in cache.items() if current_time - ts > ttl]
                for k in expired_keys:
                    del cache[k]
                
                return result
            
            return async_wrapper
        return decorator
    
    @staticmethod
    def get_performance_summary() -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "memory": PerformanceOptimizer.get_memory_stats(),
            "cpu": PerformanceOptimizer.get_cpu_stats(),
            "gc_thresholds": gc.get_threshold(),
            "config": {
                "max_concurrent_tasks": settings.performance.max_concurrent_tasks,
                "batch_processing": settings.performance.batch_processing,
                "batch_size": settings.performance.batch_size,
                "cache_enabled": settings.performance.enable_cache
            }
        }


class AsyncPool:
    """异步线程池"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = None
    
    async def __aenter__(self):
        import concurrent.futures
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def run_in_executor(self, func: Callable, *args):
        """在线程池中运行函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)