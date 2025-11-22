"""
验证性能监控
提供实时性能指标收集、统计和分析功能
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

if TYPE_CHECKING:
    from .coordinator import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    processing_time: float
    layer1_time: float
    layer2_time: float
    success: bool
    combined_score: Optional[float]
    status: str
    symbol: str
    timeframe: str
    request_id: str


@dataclass
class AggregatedMetrics:
    """聚合指标数据类"""
    period_start: float
    period_end: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    timeout_requests: int
    avg_processing_time: float
    p50_processing_time: float
    p95_processing_time: float
    p99_processing_time: float
    avg_layer1_time: float
    avg_layer2_time: float
    success_rate: float
    error_rate: float
    timeout_rate: float
    avg_combined_score: float
    most_common_status: str
    symbol_distribution: Dict[str, int]
    error_summary: Dict[str, int]


class ValidationMonitor:
    """
    验证性能监控器
    
    负责收集、聚合和分析验证性能指标
    """
    
    def __init__(self, monitoring_config: Any):
        """
        初始化监控器
        
        Args:
            monitoring_config: 监控配置对象
        """
        self.config = monitoring_config
        self.metrics_history = deque(maxlen=10000)  # 保留最近10000条记录
        self.symbol_metrics = defaultdict(lambda: {
            'requests': 0, 'successes': 0, 'total_time': 0.0
        })
        self.status_distribution = defaultdict(int)
        self.error_summary = defaultdict(int)
        
        # 实时统计
        self.current_minute_stats = {
            'start_time': time.time(),
            'requests': 0,
            'successes': 0,
            'errors': 0,
            'total_time': 0.0
        }
        
        # 预警阈值
        self.alert_thresholds = self.config.alert_thresholds
        self.is_initialized = False

    async def initialize(self) -> None:
        """初始化监控器"""
        logger.info("初始化验证性能监控器...")
        
        try:
            # 启动监控任务
            self._start_monitoring_tasks()
            
            self.is_initialized = True
            logger.info("验证性能监控器初始化完成")
            
        except Exception as e:
            logger.error(f"监控器初始化失败: {str(e)}")
            raise

    def _start_monitoring_tasks(self) -> None:
        """启动监控任务"""
        # 启动清理任务
        asyncio.create_task(self._cleanup_old_metrics())
        
        # 启动统计重置任务
        asyncio.create_task(self._reset_minute_stats())
        
        # 启动预警检查任务
        asyncio.create_task(self._check_alerts())

    async def record_validation_performance(
        self, 
        request_id: str, 
        processing_time: float, 
        result: "ValidationResult"
    ) -> None:
        """记录验证性能"""
        if not self.is_initialized:
            return
        
        try:
            # 创建性能指标
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                processing_time=processing_time,
                layer1_time=result.metadata.get('layer1_processing_time', 0.0) if result.metadata else 0.0,
                layer2_time=result.metadata.get('layer2_processing_time', 0.0) if result.metadata else 0.0,
                success=result.status in [ValidationStatus.LAYER1_PASSED, ValidationStatus.LAYER2_PASSED],
                combined_score=result.combined_score,
                status=result.status.value,
                symbol=result.symbol,
                timeframe=result.timeframe,
                request_id=request_id
            )
            
            # 添加到历史记录
            self.metrics_history.append(metrics)
            
            # 更新统计
            self._update_statistics(metrics)
            
        except Exception as e:
            logger.error(f"记录性能指标失败: {str(e)}")

    def _update_statistics(self, metrics: PerformanceMetrics) -> None:
        """更新统计信息"""
        # 更新分钟统计
        self.current_minute_stats['requests'] += 1
        self.current_minute_stats['total_time'] += metrics.processing_time
        
        if metrics.success:
            self.current_minute_stats['successes'] += 1
        else:
            self.current_minute_stats['errors'] += 1
        
        # 更新符号统计
        symbol_stats = self.symbol_metrics[metrics.symbol]
        symbol_stats['requests'] += 1
        symbol_stats['total_time'] += metrics.processing_time
        
        if metrics.success:
            symbol_stats['successes'] += 1
        
        # 更新状态分布
        self.status_distribution[metrics.status] += 1
        
        # 更新错误统计
        if not metrics.success:
            if metrics.status == ValidationStatus.TIMEOUT.value:
                self.error_summary['timeout'] += 1
            elif metrics.status == ValidationStatus.ERROR.value:
                self.error_summary['error'] += 1
            elif metrics.status == ValidationStatus.LAYER1_FAILED.value:
                self.error_summary['layer1_failed'] += 1
            elif metrics.status == ValidationStatus.HOLD.value:
                self.error_summary['hold'] += 1

    async def get_performance_summary(
        self, 
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.is_initialized:
            return {'error': 'Monitor not initialized'}
        
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        # 筛选时间窗口内的数据
        window_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not window_metrics:
            return {
                'time_window_minutes': time_window_minutes,
                'total_requests': 0,
                'message': 'No data in time window'
            }
        
        # 计算聚合指标
        aggregated = self._calculate_aggregated_metrics(window_metrics)
        
        return {
            'time_window_minutes': time_window_minutes,
            'period_start': aggregated.period_start,
            'period_end': aggregated.period_end,
            'total_requests': aggregated.total_requests,
            'successful_requests': aggregated.successful_requests,
            'failed_requests': aggregated.failed_requests,
            'timeout_requests': aggregated.timeout_requests,
            'success_rate': aggregated.success_rate,
            'error_rate': aggregated.error_rate,
            'timeout_rate': aggregated.timeout_rate,
            'processing_times': {
                'avg': aggregated.avg_processing_time,
                'p50': aggregated.p50_processing_time,
                'p95': aggregated.p95_processing_time,
                'p99': aggregated.p99_processing_time
            },
            'layer_times': {
                'avg_layer1': aggregated.avg_layer1_time,
                'avg_layer2': aggregated.avg_layer2_time
            },
            'avg_combined_score': aggregated.avg_combined_score,
            'status_distribution': dict(self.status_distribution),
            'error_summary': dict(self.error_summary),
            'top_symbols': self._get_top_symbols(10)
        }

    def _calculate_aggregated_metrics(self, metrics: List[PerformanceMetrics]) -> AggregatedMetrics:
        """计算聚合指标"""
        if not metrics:
            # 返回空指标
            return AggregatedMetrics(
                period_start=time.time(),
                period_end=time.time(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                timeout_requests=0,
                avg_processing_time=0.0,
                p50_processing_time=0.0,
                p95_processing_time=0.0,
                p99_processing_time=0.0,
                avg_layer1_time=0.0,
                avg_layer2_time=0.0,
                success_rate=0.0,
                error_rate=0.0,
                timeout_rate=0.0,
                avg_combined_score=0.0,
                most_common_status="none",
                symbol_distribution={},
                error_summary={}
            )
        
        timestamps = [m.timestamp for m in metrics]
        processing_times = [m.processing_time for m in metrics]
        layer1_times = [m.layer1_time for m in metrics]
        layer2_times = [m.layer2_time for m in metrics]
        combined_scores = [m.combined_score for m in metrics if m.combined_score is not None]
        
        total_requests = len(metrics)
        successful_requests = sum(1 for m in metrics if m.success)
        failed_requests = total_requests - successful_requests
        timeout_requests = sum(1 for m in metrics if m.status == ValidationStatus.TIMEOUT.value)
        
        return AggregatedMetrics(
            period_start=min(timestamps),
            period_end=max(timestamps),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            timeout_requests=timeout_requests,
            avg_processing_time=sum(processing_times) / len(processing_times),
            p50_processing_time=self._calculate_percentile(processing_times, 50),
            p95_processing_time=self._calculate_percentile(processing_times, 95),
            p99_processing_time=self._calculate_percentile(processing_times, 99),
            avg_layer1_time=sum(layer1_times) / len(layer1_times) if layer1_times else 0.0,
            avg_layer2_time=sum(layer2_times) / len(layer2_times) if layer2_times else 0.0,
            success_rate=successful_requests / total_requests if total_requests > 0 else 0.0,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0.0,
            timeout_rate=timeout_requests / total_requests if total_requests > 0 else 0.0,
            avg_combined_score=sum(combined_scores) / len(combined_scores) if combined_scores else 0.0,
            most_common_status=max(self.status_distribution.items(), key=lambda x: x[1])[0],
            symbol_distribution=self._get_symbol_distribution(),
            error_summary=dict(self.error_summary)
        )

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def _get_symbol_distribution(self) -> Dict[str, int]:
        """获取符号分布"""
        return {symbol: stats['requests'] for symbol, stats in self.symbol_metrics.items()}

    def _get_top_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最活跃的符号"""
        symbols = sorted(
            self.symbol_metrics.items(),
            key=lambda x: x[1]['requests'],
            reverse=True
        )[:limit]
        
        return [
            {
                'symbol': symbol,
                'requests': stats['requests'],
                'successes': stats['successes'],
                'success_rate': stats['successes'] / stats['requests'] if stats['requests'] > 0 else 0.0,
                'avg_processing_time': stats['total_time'] / stats['requests'] if stats['requests'] > 0 else 0.0
            }
            for symbol, stats in symbols
        ]

    async def get_real_time_stats(self) -> Dict[str, Any]:
        """获取实时统计"""
        if not self.is_initialized:
            return {'error': 'Monitor not initialized'}
        
        current_time = time.time()
        elapsed_minutes = (current_time - self.current_minute_stats['start_time']) / 60
        
        requests = self.current_minute_stats['requests']
        successes = self.current_minute_stats['successes']
        errors = self.current_minute_stats['errors']
        total_time = self.current_minute_stats['total_time']
        
        return {
            'timestamp': current_time,
            'elapsed_minutes': elapsed_minutes,
            'requests_per_minute': requests / max(elapsed_minutes, 1/60),
            'successes_per_minute': successes / max(elapsed_minutes, 1/60),
            'errors_per_minute': errors / max(elapsed_minutes, 1/60),
            'avg_processing_time': total_time / requests if requests > 0 else 0.0,
            'current_success_rate': successes / requests if requests > 0 else 0.0,
            'total_requests': requests,
            'successful_requests': successes,
            'error_requests': errors
        }

    async def check_health_status(self) -> Dict[str, Any]:
        """检查健康状态"""
        if not self.is_initialized:
            return {'status': 'not_initialized', 'healthy': False}
        
        # 获取最近一分钟的统计
        recent_stats = await self.get_real_time_stats()
        
        # 检查阈值
        health_issues = []
        
        if recent_stats.get('avg_processing_time', 0) > self.alert_thresholds.get('avg_processing_time', 5.0):
            health_issues.append(f"Average processing time too high: {recent_stats['avg_processing_time']:.2f}s")
        
        error_rate = 1 - recent_stats.get('current_success_rate', 1.0)
        if error_rate > self.alert_thresholds.get('error_rate', 0.1):
            health_issues.append(f"Error rate too high: {error_rate:.2%}")
        
        # 计算超时率（简化版）
        timeout_rate = recent_stats.get('error_requests', 0) / max(recent_stats.get('total_requests', 1), 1)
        if timeout_rate > self.alert_thresholds.get('timeout_rate', 0.05):
            health_issues.append(f"Timeout rate too high: {timeout_rate:.2%}")
        
        return {
            'status': 'healthy' if not health_issues else 'degraded',
            'healthy': len(health_issues) == 0,
            'issues': health_issues,
            'metrics': recent_stats,
            'alert_thresholds': self.alert_thresholds,
            'monitoring_enabled': self.config.enable_performance_monitoring
        }

    async def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """获取趋势分析"""
        if not self.is_initialized:
            return {'error': 'Monitor not initialized'}
        
        cutoff_time = time.time() - (hours * 3600)
        
        # 获取指定时间窗口内的数据
        window_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not window_metrics:
            return {'message': 'No data available for trend analysis'}
        
        # 按小时分组
        hourly_data = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'total_time': 0.0,
            'errors': 0
        })
        
        for metric in window_metrics:
            hour_key = int(metric.timestamp // 3600)
            hour_data = hourly_data[hour_key]
            hour_data['requests'] += 1
            hour_data['total_time'] += metric.processing_time
            
            if metric.success:
                hour_data['successes'] += 1
            else:
                hour_data['errors'] += 1
        
        # 计算趋势
        hourly_summary = []
        for hour_key in sorted(hourly_data.keys()):
            data = hourly_data[hour_key]
            avg_time = data['total_time'] / data['requests'] if data['requests'] > 0 else 0.0
            success_rate = data['successes'] / data['requests'] if data['requests'] > 0 else 0.0
            
            hourly_summary.append({
                'hour': datetime.fromtimestamp(hour_key * 3600).isoformat(),
                'requests': data['requests'],
                'success_rate': success_rate,
                'avg_processing_time': avg_time,
                'errors': data['errors']
            })
        
        return {
            'analysis_period_hours': hours,
            'total_hours_with_data': len(hourly_summary),
            'hourly_data': hourly_summary,
            'overall_trend': self._calculate_trend(hourly_summary)
        }

    def _calculate_trend(self, hourly_summary: List[Dict[str, Any]]) -> str:
        """计算整体趋势"""
        if len(hourly_summary) < 2:
            return "insufficient_data"
        
        # 比较前一半和后一半的平均性能
        mid_point = len(hourly_summary) // 2
        first_half = hourly_summary[:mid_point]
        second_half = hourly_summary[mid_point:]
        
        first_half_success = sum(d['success_rate'] for d in first_half) / len(first_half)
        second_half_success = sum(d['success_rate'] for d in second_half) / len(second_half)
        
        first_half_time = sum(d['avg_processing_time'] for d in first_half) / len(first_half)
        second_half_time = sum(d['avg_processing_time'] for d in second_half) / len(second_half)
        
        # 判断趋势
        success_improvement = second_half_success - first_half_success
        time_improvement = first_half_time - second_half_time
        
        if success_improvement > 0.05 and time_improvement > 0.5:
            return "improving"
        elif success_improvement < -0.05 or time_improvement < -0.5:
            return "degrading"
        else:
            return "stable"

    async def _cleanup_old_metrics(self) -> None:
        """清理旧的指标数据"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                
                if not self.metrics_history:
                    continue
                
                # 保留最近24小时的数据
                cutoff_time = time.time() - (24 * 3600)
                
                # 移除旧数据
                while self.metrics_history and self.metrics_history[0].timestamp < cutoff_time:
                    self.metrics_history.popleft()
                
                # 清理符号统计（只保留最近活跃的符号）
                cutoff_symbol_time = time.time() - (6 * 3600)  # 6小时
                inactive_symbols = []
                for symbol, stats in self.symbol_metrics.items():
                    # 简化清理逻辑，实际应该跟踪最后活跃时间
                    if stats['requests'] < 5:  # 请求量太少的符号
                        inactive_symbols.append(symbol)
                
                for symbol in inactive_symbols:
                    del self.symbol_metrics[symbol]
                
                logger.debug(f"清理完成，保留 {len(self.metrics_history)} 条记录")
                
            except Exception as e:
                logger.error(f"清理旧指标失败: {str(e)}")

    async def _reset_minute_stats(self) -> None:
        """重置分钟统计"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟重置一次
                
                self.current_minute_stats = {
                    'start_time': time.time(),
                    'requests': 0,
                    'successes': 0,
                    'errors': 0,
                    'total_time': 0.0
                }
                
            except Exception as e:
                logger.error(f"重置分钟统计失败: {str(e)}")

    async def _check_alerts(self) -> None:
        """检查预警"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                health_status = await self.check_health_status()
                
                if not health_status['healthy'] and health_status['issues']:
                    logger.warning(f"检测到健康问题: {health_status['issues']}")
                    
                    # 这里可以添加发送告警的逻辑
                    # 例如：发送邮件、Slack消息等
                
            except Exception as e:
                logger.error(f"预警检查失败: {str(e)}")

    async def export_metrics(self, file_path: str, hours: int = 24) -> None:
        """导出指标数据"""
        if not self.is_initialized:
            raise RuntimeError("Monitor not initialized")
        
        cutoff_time = time.time() - (hours * 3600)
        
        window_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        export_data = {
            'export_time': time.time(),
            'time_window_hours': hours,
            'metrics_count': len(window_metrics),
            'metrics': [
                {
                    'timestamp': m.timestamp,
                    'processing_time': m.processing_time,
                    'layer1_time': m.layer1_time,
                    'layer2_time': m.layer2_time,
                    'success': m.success,
                    'combined_score': m.combined_score,
                    'status': m.status,
                    'symbol': m.symbol,
                    'timeframe': m.timeframe,
                    'request_id': m.request_id
                }
                for m in window_metrics
            ]
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"指标数据已导出到 {file_path}")
            
        except Exception as e:
            logger.error(f"导出指标数据失败: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """关闭监控器"""
        logger.info("正在关闭验证性能监控器...")
        self.is_initialized = False
        logger.info("验证性能监控器已关闭")