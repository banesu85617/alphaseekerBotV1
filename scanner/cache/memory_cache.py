"""
内存缓存实现
提供快速的数据缓存功能，适用于短期数据存储
"""

import time
import threading
from typing import Any, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheItem:
    """缓存项"""
    value: Any
    timestamp: float
    ttl: float = 300.0  # 默认TTL: 5分钟
    
    def is_expired(self) -> bool:
        """检查是否已过期"""
        return time.time() - self.timestamp > self.ttl


class MemoryCache:
    """内存缓存类"""
    
    def __init__(self, default_ttl: float = 300.0, max_size: int = 10000):
        """
        初始化内存缓存
        
        Args:
            default_ttl: 默认生存时间(秒)
            max_size: 最大缓存项数量
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, CacheItem] = {}
        self._lock = threading.RLock()
        self._access_order: list = []  # 用于LRU eviction
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果不存在或已过期则返回None
        """
        with self._lock:
            item = self._cache.get(key)
            
            if item is None:
                self._stats['misses'] += 1
                return None
            
            # 检查是否过期
            if item.is_expired():
                self._delete_key(key)
                self._stats['misses'] += 1
                return None
            
            # 更新访问顺序(LRU)
            self._update_access_order(key)
            self._stats['hits'] += 1
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间(秒)，None则使用默认值
        """
        with self._lock:
            actual_ttl = ttl or self.default_ttl
            
            # 检查是否需要驱逐
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # 设置缓存项
            self._cache[key] = CacheItem(
                value=value,
                timestamp=time.time(),
                ttl=actual_ttl
            )
            
            # 更新访问顺序
            self._update_access_order(key)
            self._stats['sets'] += 1
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            True如果删除成功，False如果键不存在
        """
        with self._lock:
            return self._delete_key(key)
    
    def _delete_key(self, key: str) -> bool:
        """内部删除方法"""
        if key in self._cache:
            del self._cache[key]
            self._remove_from_access_order(key)
            self._stats['deletes'] += 1
            return True
        return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            True如果存在且未过期
        """
        with self._lock:
            item = self._cache.get(key)
            return item is not None and not item.is_expired()
    
    def get_ttl(self, key: str) -> Optional[float]:
        """
        获取键的剩余TTL
        
        Args:
            key: 缓存键
            
        Returns:
            剩余TTL(秒)，如果键不存在或已过期则返回None
        """
        with self._lock:
            item = self._cache.get(key)
            if item is None or item.is_expired():
                return None
            return max(0, item.ttl - (time.time() - item.timestamp))
    
    def expire(self, key: str, ttl: float) -> bool:
        """
        设置键的TTL
        
        Args:
            key: 缓存键
            ttl: 新的TTL(秒)
            
        Returns:
            True如果设置成功
        """
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return False
            
            item.ttl = ttl
            item.timestamp = time.time()
            return True
    
    def bulk_get(self, keys: list) -> Dict[str, Any]:
        """
        批量获取缓存值
        
        Args:
            keys: 缓存键列表
            
        Returns:
            缓存值字典
        """
        results = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    def bulk_set(self, items: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """
        批量设置缓存值
        
        Args:
            items: 键值对字典
            ttl: 生存时间(秒)
        """
        for key, value in items.items():
            self.set(key, value, ttl)
    
    def cleanup_expired(self) -> int:
        """
        清理过期的缓存项
        
        Returns:
            清理的项数量
        """
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, item in self._cache.items():
                if item.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._delete_key(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes'],
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'utilization': len(self._cache) / self.max_size
            }
    
    def _update_access_order(self, key: str):
        """更新访问顺序(LRU)"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _remove_from_access_order(self, key: str):
        """从访问顺序中移除"""
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _evict_lru(self):
        """驱逐最少最近使用的项"""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats['evictions'] += 1
    
    def get_keys(self, pattern: str = "*") -> list:
        """
        获取匹配的键列表
        
        Args:
            pattern: 匹配模式，支持*通配符
            
        Returns:
            匹配的键列表
        """
        with self._lock:
            import fnmatch
            return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        获取内存使用情况
        
        Returns:
            内存使用信息
        """
        import sys
        
        with self._lock:
            total_size = 0
            item_sizes = []
            
            for key, item in self._cache.items():
                size = sys.getsizeof(item.value) + sys.getsizeof(key) + sys.getsizeof(item)
                total_size += size
                item_sizes.append(size)
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'item_count': len(self._cache),
                'avg_item_size_bytes': total_size / len(self._cache) if self._cache else 0,
                'max_item_size_bytes': max(item_sizes) if item_sizes else 0,
                'min_item_size_bytes': min(item_sizes) if item_sizes else 0
            }


# 缓存装饰器
def cache_result(ttl: float = 300.0):
    """缓存结果装饰器"""
    def decorator(func):
        cache = MemoryCache(ttl)
        
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            result = cache.get(key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = cache.get_stats
        return wrapper
    
    return decorator


# 示例使用
if __name__ == "__main__":
    # 测试代码
    cache = MemoryCache(default_ttl=60, max_size=100)
    
    # 设置缓存
    cache.set("test_key", {"data": "test_value"}, ttl=120)
    
    # 获取缓存
    result = cache.get("test_key")
    print(f"Retrieved: {result}")
    
    # 检查统计信息
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    
    # 批量操作
    cache.bulk_set({
        "key1": "value1",
        "key2": "value2", 
        "key3": "value3"
    })
    
    bulk_results = cache.bulk_get(["key1", "key2", "key3"])
    print(f"Bulk results: {bulk_results}")
    
    # 清理过期项
    expired_count = cache.cleanup_expired()
    print(f"Cleaned up {expired_count} expired items")