"""
性能监控系统
监控扫描性能、延迟和系统资源使用情况
"""

import time
import asyncio
import threading
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    scan_id: str
    timestamp: datetime
    duration: float
    symbols_processed: int
    symbols_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    error_rate: float
    throughput_mb: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanTiming:
    """扫描时间分解"""
    scan_id: str
    preprocessing_time: float
    parallel_scan_time: float
    deep_analysis_time: float
    aggregation_time: float
    caching_time: float
    alerting_time: float
    total_time: float


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 1000):
        """
        初始化性能监控器
        
        Args:
            max_history: 最大历史记录数量
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.timing_history: deque = deque(maxlen=max_history)
        
        # 实时监控
        self.real_time_monitor = RealTimeMonitor()
        self.alerts_enabled = True
        self.alert_thresholds = {
            'max_duration': 30.0,      # 最大扫描时间(秒)
            'min_throughput': 10.0,    # 最小吞吐量(交易对/秒)
            'max_memory_usage': 1024.0, # 最大内存使用(MB)
            'max_error_rate': 0.1,     # 最大错误率
            'max_latency_p95': 5.0     # 最大95分位延迟(秒)
        }
        
        # 统计计算器
        self.statistics_calculator = StatisticsCalculator()
        
        # 回调函数
        self.performance_callbacks: List[Callable] = []
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        logger.info("PerformanceMonitor initialized")
    
    def track_scan(self, scan_id: str):
        """创建扫描跟踪上下文管理器"""
        return ScanTracker(self, scan_id)
    
    async def start_monitoring(self, interval: float = 10.0):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Performance monitoring started with interval {interval}s")
    
    async def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def record_metrics(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        self.metrics_history.append(metrics)
        
        # 实时监控检查
        await self.real_time_monitor.check_metrics(metrics, self.alert_thresholds)
        
        # 调用回调函数
        for callback in self.performance_callbacks:
            try:
                await callback(metrics)
            except Exception as e:
                logger.error(f"Error in performance callback: {e}")
    
    async def record_timing(self, timing: ScanTiming):
        """记录时间分解"""
        self.timing_history.append(timing)
    
    def add_performance_callback(self, callback: Callable):
        """添加性能回调函数"""
        self.performance_callbacks.append(callback)
    
    def get_statistics(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.metrics_history:
            return {}
        
        # 过滤时间范围
        if time_range:
            cutoff_time = datetime.now() - time_range
            filtered_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        else:
            filtered_metrics = list(self.metrics_history)
        
        if not filtered_metrics:
            return {}
        
        # 计算统计信息
        return self.statistics_calculator.calculate_statistics(filtered_metrics)
    
    def get_recent_performance(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近性能数据"""
        recent_metrics = list(self.metrics_history)[-limit:]
        return [self._metrics_to_dict(m) for m in recent_metrics]
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        return self.real_time_monitor.get_system_health()
    
    def update_alert_thresholds(self, new_thresholds: Dict[str, float]):
        """更新警报阈值"""
        self.alert_thresholds.update(new_thresholds)
        logger.info(f"Alert thresholds updated: {new_thresholds}")
    
    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """转换指标为字典"""
        return {
            'scan_id': metrics.scan_id,
            'timestamp': metrics.timestamp.isoformat(),
            'duration': metrics.duration,
            'symbols_processed': metrics.symbols_processed,
            'symbols_per_second': metrics.symbols_per_second,
            'memory_usage_mb': metrics.memory_usage_mb,
            'cpu_usage_percent': metrics.cpu_usage_percent,
            'cache_hit_rate': metrics.cache_hit_rate,
            'error_rate': metrics.error_rate,
            'throughput_mb': metrics.throughput_mb,
            'latency_p50': metrics.latency_p50,
            'latency_p95': metrics.latency_p95,
            'latency_p99': metrics.latency_p99,
            'metadata': metrics.metadata
        }
    
    async def _monitoring_loop(self, interval: float):
        """监控循环"""
        try:
            while self.is_monitoring:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                
                # 记录指标
                metrics = PerformanceMetrics(
                    scan_id=f"system_monitor_{int(time.time())}",
                    timestamp=datetime.now(),
                    duration=interval,
                    symbols_processed=0,
                    symbols_per_second=0,
                    memory_usage_mb=system_metrics['memory_mb'],
                    cpu_usage_percent=system_metrics['cpu_percent'],
                    cache_hit_rate=0.0,
                    error_rate=0.0,
                    throughput_mb=0.0,
                    latency_p50=0.0,
                    latency_p95=0.0,
                    latency_p99=0.0,
                    metadata={'type': 'system_monitor', 'system': system_metrics}
                )
                
                await self.record_metrics(metrics)
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        try:
            process = psutil.Process()
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_io': dict(psutil.net_io_counters()._asdict()),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}


class ScanTracker:
    """扫描跟踪器"""
    
    def __init__(self, monitor: PerformanceMonitor, scan_id: str):
        self.monitor = monitor
        self.scan_id = scan_id
        self.start_time: Optional[datetime] = None
        self.timing_data: Dict[str, float] = {}
        self.symbols_processed = 0
        self.errors = 0
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            # 计算指标
            symbols_per_second = self.symbols_processed / duration if duration > 0 else 0
            error_rate = self.errors / max(1, self.symbols_processed)
            
            # 获取系统资源
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            
            # 创建性能指标
            metrics = PerformanceMetrics(
                scan_id=self.scan_id,
                timestamp=self.start_time,
                duration=duration,
                symbols_processed=self.symbols_processed,
                symbols_per_second=symbols_per_second,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                cache_hit_rate=0.0,  # 需要从实际缓存系统获取
                error_rate=error_rate,
                throughput_mb=0.0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
                metadata=self.timing_data
            )
            
            # 记录指标
            asyncio.create_task(self.monitor.record_metrics(metrics))
            
            # 记录时间分解
            if self.start_time:
                timing = ScanTiming(
                    scan_id=self.scan_id,
                    preprocessing_time=self.timing_data.get('preprocessing', 0),
                    parallel_scan_time=self.timing_data.get('parallel_scan', 0),
                    deep_analysis_time=self.timing_data.get('deep_analysis', 0),
                    aggregation_time=self.timing_data.get('aggregation', 0),
                    caching_time=self.timing_data.get('caching', 0),
                    alerting_time=self.timing_data.get('alerting', 0),
                    total_time=duration
                )
                asyncio.create_task(self.monitor.record_timing(timing))
    
    def add_symbol_processed(self, count: int = 1):
        """添加已处理的交易对数量"""
        self.symbols_processed += count
    
    def add_error(self, count: int = 1):
        """添加错误数量"""
        self.errors += count
    
    def record_timing(self, phase: str, duration: float):
        """记录阶段时间"""
        self.timing_data[phase] = duration


class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self):
        self.current_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alert_history: List[Dict[str, Any]] = []
    
    async def check_metrics(self, metrics: PerformanceMetrics, thresholds: Dict[str, float]):
        """检查指标并触发警报"""
        alerts = []
        
        # 检查持续时间
        if metrics.duration > thresholds['max_duration']:
            alerts.append({
                'type': 'duration_high',
                'severity': 'warning',
                'message': f"Scan duration {metrics.duration:.2f}s exceeds threshold {thresholds['max_duration']}s",
                'value': metrics.duration,
                'threshold': thresholds['max_duration'],
                'scan_id': metrics.scan_id,
                'timestamp': datetime.now()
            })
        
        # 检查吞吐量
        if metrics.symbols_per_second < thresholds['min_throughput']:
            alerts.append({
                'type': 'throughput_low',
                'severity': 'warning',
                'message': f"Scan throughput {metrics.symbols_per_second:.2f} symbols/s below threshold {thresholds['min_throughput']}/s",
                'value': metrics.symbols_per_second,
                'threshold': thresholds['min_throughput'],
                'scan_id': metrics.scan_id,
                'timestamp': datetime.now()
            })
        
        # 检查内存使用
        if metrics.memory_usage_mb > thresholds['max_memory_usage']:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"Memory usage {metrics.memory_usage_mb:.2f}MB exceeds threshold {thresholds['max_memory_usage']}MB",
                'value': metrics.memory_usage_mb,
                'threshold': thresholds['max_memory_usage'],
                'scan_id': metrics.scan_id,
                'timestamp': datetime.now()
            })
        
        # 检查错误率
        if metrics.error_rate > thresholds['max_error_rate']:
            alerts.append({
                'type': 'error_rate_high',
                'severity': 'error',
                'message': f"Error rate {metrics.error_rate:.2%} exceeds threshold {thresholds['max_error_rate']:.2%}",
                'value': metrics.error_rate,
                'threshold': thresholds['max_error_rate'],
                'scan_id': metrics.scan_id,
                'timestamp': datetime.now()
            })
        
        # 检查延迟
        if metrics.latency_p95 > thresholds['max_latency_p95']:
            alerts.append({
                'type': 'latency_high',
                'severity': 'warning',
                'message': f"95th percentile latency {metrics.latency_p95:.2f}s exceeds threshold {thresholds['max_latency_p95']}s",
                'value': metrics.latency_p95,
                'threshold': thresholds['max_latency_p95'],
                'scan_id': metrics.scan_id,
                'timestamp': datetime.now()
            })
        
        # 处理警报
        for alert in alerts:
            await self._process_alert(alert)
    
    async def _process_alert(self, alert: Dict[str, Any]):
        """处理警报"""
        alert_type = alert['type']
        
        # 添加到当前警报
        self.current_alerts[alert_type].append(alert)
        
        # 添加到历史记录
        self.alert_history.append(alert)
        
        # 限制历史记录数量
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # 记录日志
        logger.warning(f"Performance alert: {alert['message']}")
        
        # 如果有告警回调，可以在这里调用
        
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            return {
                'status': 'healthy',
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_status': 'connected',  # 简化处理
                'active_alerts': {k: len(v) for k, v in self.current_alerts.items()},
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近警报"""
        return self.alert_history[-limit:]
    
    def clear_alerts(self, alert_type: Optional[str] = None):
        """清除警报"""
        if alert_type:
            self.current_alerts[alert_type].clear()
        else:
            self.current_alerts.clear()


class StatisticsCalculator:
    """统计计算器"""
    
    @staticmethod
    def calculate_statistics(metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """计算统计信息"""
        if not metrics:
            return {}
        
        # 提取数值列表
        durations = [m.duration for m in metrics]
        throughputs = [m.symbols_per_second for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        cpu_usage = [m.cpu_usage_percent for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        latencies_p50 = [m.latency_p50 for m in metrics if m.latency_p50 > 0]
        latencies_p95 = [m.latency_p95 for m in metrics if m.latency_p95 > 0]
        latencies_p99 = [m.latency_p99 for m in metrics if m.latency_p99 > 0]
        
        stats = {
            'total_scans': len(metrics),
            'time_range': {
                'start': min(m.timestamp for m in metrics).isoformat(),
                'end': max(m.timestamp for m in metrics).isoformat()
            },
            'duration': StatisticsCalculator._calculate_distribution_stats(durations),
            'throughput': StatisticsCalculator._calculate_distribution_stats(throughputs),
            'memory_usage': StatisticsCalculator._calculate_distribution_stats(memory_usage),
            'cpu_usage': StatisticsCalculator._calculate_distribution_stats(cpu_usage),
            'error_rate': StatisticsCalculator._calculate_distribution_stats(error_rates),
            'latency_p50': StatisticsCalculator._calculate_distribution_stats(latencies_p50) if latencies_p50 else {},
            'latency_p95': StatisticsCalculator._calculate_distribution_stats(latencies_p95) if latencies_p95 else {},
            'latency_p99': StatisticsCalculator._calculate_distribution_stats(latencies_p99) if latencies_p99 else {}
        }
        
        return stats
    
    @staticmethod
    def _calculate_distribution_stats(values: List[float]) -> Dict[str, float]:
        """计算分布统计"""
        if not values:
            return {}
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'p25': StatisticsCalculator._percentile(values, 25),
            'p75': StatisticsCalculator._percentile(values, 75),
            'p90': StatisticsCalculator._percentile(values, 90),
            'p95': StatisticsCalculator._percentile(values, 95),
            'p99': StatisticsCalculator._percentile(values, 99)
        }
    
    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (p / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


# 示例使用
if __name__ == "__main__":
    async def test_performance_monitor():
        # 创建监控器
        monitor = PerformanceMonitor()
        
        # 启动监控
        await monitor.start_monitoring(interval=5.0)
        
        # 模拟扫描性能记录
        metrics = PerformanceMetrics(
            scan_id="test_scan_001",
            timestamp=datetime.now(),
            duration=15.5,
            symbols_processed=100,
            symbols_per_second=6.45,
            memory_usage_mb=512.0,
            cpu_usage_percent=25.0,
            cache_hit_rate=0.85,
            error_rate=0.02,
            throughput_mb=0.0,
            latency_p50=0.5,
            latency_p95=1.2,
            latency_p99=2.0
        )
        
        await monitor.record_metrics(metrics)
        
        # 获取统计信息
        stats = monitor.get_statistics()
        print("Statistics:", json.dumps(stats, indent=2, default=str))
        
        # 获取系统健康状态
        health = monitor.get_system_health()
        print("System Health:", json.dumps(health, indent=2))
        
        # 停止监控
        await monitor.stop_monitoring()
    
    # 运行测试
    import asyncio
    asyncio.run(test_performance_monitor())