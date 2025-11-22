"""
Redis缓存实现
提供分布式缓存功能，支持数据持久化和跨进程共享
"""

import json
import time
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Redis = None
import pickle

logger = logging.getLogger(__name__)

# 处理Redis不可用的情况
if REDIS_AVAILABLE:
    Redis = redis.Redis
    AsyncRedis = Redis
else:
    Redis = None
    AsyncRedis = None


class RedisCache:
    """Redis缓存类"""
    
    def __init__(
        self, 
        redis_client: Optional[Redis] = None,
        default_ttl: int = 300,
        key_prefix: str = "scanner:"
    ):
        """
        初始化Redis缓存
        
        Args:
            redis_client: Redis客户端实例
            default_ttl: 默认生存时间(秒)
            key_prefix: 键前缀
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, RedisCache will be disabled")
            self.redis_client = None
            self.default_ttl = default_ttl
            self.key_prefix = key_prefix
            self._connected = False
            return
            
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._connected = False
        
        # 如果没有提供客户端，创建默认连接
        if redis_client is None:
            self._create_default_connection()
    
    def _create_default_connection(self):
        """创建默认Redis连接"""
        if not REDIS_AVAILABLE:
            return
            
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=False,  # 保持原始字节数据
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self._connected = True
        except Exception as e:
            logger.warning(f"Failed to create Redis connection: {e}")
            self._connected = False
    
    async def connect(self) -> bool:
        """连接Redis"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                self._connected = True
                logger.info("Connected to Redis successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
        return False
    
    def is_connected(self) -> bool:
        """检查Redis连接状态"""
        return self._connected and self.redis_client is not None
    
    def _normalize_key(self, key: str) -> str:
        """规范化键名"""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果不存在或已过期则返回None
        """
        if not self.is_connected():
            logger.warning("Redis not connected, returning None")
            return None
        
        try:
            normalized_key = self._normalize_key(key)
            data = await self.redis_client.get(normalized_key)
            
            if data is None:
                return None
            
            # 尝试JSON解码
            try:
                return json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 如果JSON解码失败，尝试pickle解码
                try:
                    return pickle.loads(data)
                except Exception:
                    # 如果都失败，尝试作为字符串处理
                    return data.decode('utf-8')
                    
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间(秒)，None则使用默认值
            
        Returns:
            True如果设置成功
        """
        if not self.is_connected():
            logger.warning("Redis not connected, cannot set cache")
            return False
        
        try:
            normalized_key = self._normalize_key(key)
            actual_ttl = ttl or self.default_ttl
            
            # 尝试JSON序列化
            try:
                serialized_value = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                # 如果JSON序列化失败，使用pickle
                serialized_value = pickle.dumps(value)
            
            result = await self.redis_client.setex(
                normalized_key, 
                actual_ttl, 
                serialized_value
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            True如果删除成功
        """
        if not self.is_connected():
            return False
        
        try:
            normalized_key = self._normalize_key(key)
            result = await self.redis_client.delete(normalized_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            True如果存在
        """
        if not self.is_connected():
            return False
        
        try:
            normalized_key = self._normalize_key(key)
            result = await self.redis_client.exists(normalized_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        设置键的TTL
        
        Args:
            key: 缓存键
            ttl: TTL(秒)
            
        Returns:
            True如果设置成功
        """
        if not self.is_connected():
            return False
        
        try:
            normalized_key = self._normalize_key(key)
            result = await self.redis_client.expire(normalized_key, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> Optional[int]:
        """
        获取键的TTL
        
        Args:
            key: 缓存键
            
        Returns:
            TTL(秒)，如果键不存在则返回None
        """
        if not self.is_connected():
            return None
        
        try:
            normalized_key = self._normalize_key(key)
            ttl = await self.redis_client.ttl(normalized_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            return None
    
    async def clear_prefix(self, prefix: str = None) -> int:
        """
        清空前缀匹配的键
        
        Args:
            prefix: 前缀，None则使用默认前缀
            
        Returns:
            删除的键数量
        """
        if not self.is_connected():
            return 0
        
        try:
            actual_prefix = prefix or self.key_prefix
            pattern = f"{actual_prefix}*"
            
            # 获取匹配的键
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                result = await self.redis_client.delete(*keys)
                logger.info(f"Cleared {result} keys with prefix {actual_prefix}")
                return result
            
            return 0
            
        except Exception as e:
            logger.error(f"Redis CLEAR PREFIX error: {e}")
            return 0
    
    async def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存值
        
        Args:
            keys: 缓存键列表
            
        Returns:
            缓存值字典
        """
        if not self.is_connected():
            return {}
        
        try:
            normalized_keys = [self._normalize_key(key) for key in keys]
            results = await self.redis_client.mget(normalized_keys)
            
            result_dict = {}
            for key, data in zip(keys, results):
                if data is not None:
                    try:
                        # 尝试JSON解码
                        try:
                            result_dict[key] = json.loads(data)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # 使用pickle解码
                            result_dict[key] = pickle.loads(data)
                    except Exception:
                        # 作为字符串处理
                        result_dict[key] = data.decode('utf-8')
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Redis BULK GET error: {e}")
            return {}
    
    async def bulk_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """
        批量设置缓存值
        
        Args:
            items: 键值对字典
            ttl: TTL(秒)
            
        Returns:
            设置结果字典 {key: success}
        """
        if not self.is_connected():
            return {key: False for key in items.keys()}
        
        try:
            pipeline = self.redis_client.pipeline()
            actual_ttl = ttl or self.default_ttl
            
            results = {}
            
            for key, value in items.items():
                try:
                    normalized_key = self._normalize_key(key)
                    
                    # 尝试JSON序列化
                    try:
                        serialized_value = json.dumps(value, ensure_ascii=False)
                    except (TypeError, ValueError):
                        serialized_value = pickle.dumps(value)
                    
                    pipeline.setex(normalized_key, actual_ttl, serialized_value)
                    results[key] = True
                    
                except Exception as e:
                    logger.error(f"Error preparing key {key} for bulk set: {e}")
                    results[key] = False
            
            # 执行管道
            await pipeline.execute()
            return results
            
        except Exception as e:
            logger.error(f"Redis BULK SET error: {e}")
            return {key: False for key in items.keys()}
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        if not self.is_connected():
            return {'connected': False}
        
        try:
            # 获取Redis信息
            info = await self.redis_client.info()
            
            # 统计前缀匹配的键数量
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            # 获取键的详细信息
            key_details = []
            if keys:
                pipeline = self.redis_client.pipeline()
                for key in keys[:100]:  # 限制检查数量以避免性能问题
                    pipeline.ttl(key)
                ttls = await pipeline.execute()
                
                for key, ttl in zip(keys[:100], ttls):
                    key_details.append({
                        'key': key.decode('utf-8'),
                        'ttl': ttl if ttl > 0 else -1
                    })
            
            return {
                'connected': True,
                'redis_version': info.get('redis_version', 'unknown'),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0),
                'total_keys': len(keys),
                'checked_keys': len(key_details),
                'expired_keys': sum(1 for kd in key_details if kd['ttl'] == -1),
                'keys_with_ttl': sum(1 for kd in key_details if kd['ttl'] > 0),
                'key_samples': key_details,
                'key_prefix': self.key_prefix
            }
            
        except Exception as e:
            logger.error(f"Redis STATS error: {e}")
            return {'connected': False, 'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        if not self.is_connected():
            return {'status': 'disconnected', 'message': 'Redis client not initialized'}
        
        try:
            # 测试连接
            start_time = time.time()
            await self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000  # ms
            
            # 获取基本统计信息
            info = await self.redis_client.info('server')
            
            return {
                'status': 'healthy',
                'ping_time_ms': round(ping_time, 2),
                'redis_version': info.get('redis_version', 'unknown'),
                'uptime_seconds': info.get('uptime_in_seconds', 0),
                'connected': True
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Redis connection closed")
    
    # 高级功能
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """
        增加数值
        
        Args:
            key: 缓存键
            amount: 增加量
            ttl: TTL(秒)
            
        Returns:
            增加后的值
        """
        if not self.is_connected():
            return None
        
        try:
            normalized_key = self._normalize_key(key)
            
            if ttl is not None:
                # 使用Lua脚本确保原子性
                script = """
                if redis.call("EXISTS", KEYS[1]) == 0 then
                    redis.call("SETEX", KEYS[1], ARGV[2], ARGV[1])
                    return tonumber(ARGV[1])
                else
                    return redis.call("INCRBY", KEYS[1], ARGV[1])
                end
                """
                result = await self.redis_client.eval(
                    script, 1, normalized_key, str(amount), str(ttl)
                )
            else:
                result = await self.redis_client.incrby(normalized_key, amount)
            
            return int(result)
            
        except Exception as e:
            logger.error(f"Redis INCREMENT error for key {key}: {e}")
            return None
    
    async def set_nx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        仅在键不存在时设置
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: TTL(秒)
            
        Returns:
            True如果设置成功
        """
        if not self.is_connected():
            return False
        
        try:
            normalized_key = self._normalize_key(key)
            actual_ttl = ttl or self.default_ttl
            
            # 尝试JSON序列化
            try:
                serialized_value = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                serialized_value = pickle.dumps(value)
            
            if actual_ttl:
                result = await self.redis_client.setnx(normalized_key, serialized_value)
                if result:
                    await self.redis_client.expire(normalized_key, actual_ttl)
                return bool(result)
            else:
                return await self.redis_client.setnx(normalized_key, serialized_value)
                
        except Exception as e:
            logger.error(f"Redis SETNX error for key {key}: {e}")
            return False
    
    async def get_set(self, key: str, value: Any) -> Optional[Any]:
        """
        获取并设置新值
        
        Args:
            key: 缓存键
            value: 新值
            
        Returns:
            旧值
        """
        if not self.is_connected():
            return None
        
        try:
            normalized_key = self._normalize_key(key)
            
            # 获取旧值
            old_value = await self.get(key)
            
            # 设置新值
            await self.set(key, value)
            
            return old_value
            
        except Exception as e:
            logger.error(f"Redis GETSET error for key {key}: {e}")
            return None


# 异步缓存装饰器
def async_cache_result(ttl: int = 300, redis_cache: Optional[RedisCache] = None):
    """异步缓存结果装饰器"""
    def decorator(func):
        cache = redis_cache or RedisCache() if REDIS_AVAILABLE else None
        
        async def wrapper(*args, **kwargs):
            if cache is None:
                # 如果没有Redis缓存，直接执行函数
                return await func(*args, **kwargs)
            
            # 创建缓存键
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            result = await cache.get(key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl)
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear_prefix if cache else lambda: None
        wrapper.cache_info = cache.get_stats if cache else lambda: {}
        return wrapper
    
    return decorator


# 示例使用
if __name__ == "__main__":
    import asyncio
    
    async def test_redis_cache():
        # 创建Redis缓存实例
        cache = RedisCache(default_ttl=60)
        
        # 测试连接
        if await cache.connect():
            print("Connected to Redis")
            
            # 设置缓存
            success = await cache.set("test_key", {"data": "test_value", "timestamp": datetime.now().isoformat()})
            print(f"Set cache: {success}")
            
            # 获取缓存
            result = await cache.get("test_key")
            print(f"Retrieved: {result}")
            
            # 批量操作
            await cache.bulk_set({
                "key1": "value1",
                "key2": [1, 2, 3],
                "key3": {"nested": "data"}
            })
            
            bulk_results = await cache.bulk_get(["key1", "key2", "key3"])
            print(f"Bulk results: {bulk_results}")
            
            # 获取统计信息
            stats = await cache.get_stats()
            print(f"Stats: {stats}")
            
            # 健康检查
            health = await cache.health_check()
            print(f"Health: {health}")
            
        else:
            print("Failed to connect to Redis")
        
        # 关闭连接
        await cache.close()
    
    # 运行测试
    asyncio.run(test_redis_cache())