"""
性能监控器模块

负责监控多策略信号处理管道的性能，包括：
- 处理时延监控
- 吞吐量统计
- 缓存命中率
- 错误率分析
- 策略贡献度评估
- 实时性能指标
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .types import (
    PerformanceMetrics, StrategyType, SignalDirection, PipelineConfig,
    FusionResult, StrategySignal, PipelineError
)

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._metrics_history = deque(maxlen=10000)  # 保存最近10000次指标
        self._latency_breakdown = defaultdict(list)
        self._cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0})
        self._error_stats = defaultdict(int)
        self._strategy_performance = defaultdict(lambda: {
            "total_signals": 0,
            "successful_signals": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        })
        self._throughput_history = deque(maxlen=1000)
        self._start_time = datetime.now()
        self._last_checkpoint = datetime.now()
        
    async def record_pipeline_start(self) -> str:
        """记录管道开始执行"""
        self._current_session_id = f"session_{int(time.time())}"
        self._session_start = time.time()
        
        logger.info(f"开始记录性能指标，会话ID: {self._current_session_id}")
        return self._current_session_id
    
    async def record_latency_breakdown(
        self, 
        stage: str, 
        duration: float, 
        session_id: Optional[str] = None
    ):
        """记录延迟分解"""
        self._latency_breakdown[stage].append({
            "timestamp": datetime.now(),
            "duration": duration,
            "session_id": session_id or self._current_session_id
        })
        
        # 保持最近1000个记录
        if len(self._latency_breakdown[stage]) > 1000:
            self._latency_breakdown[stage] = self._latency_breakdown[stage][-1000:]
    
    async def record_cache_operation(
        self, 
        cache_type: str, 
        hit: bool, 
        session_id: Optional[str] = None
    ):
        """记录缓存操作"""
        cache_key = f"{cache_type}_{session_id or self._current_session_id}"
        
        if hit:
            self._cache_stats[cache_key]["hits"] += 1
        else:
            self._cache_stats[cache_key]["misses"] += 1
    
    async def record_error(
        self, 
        error_type: str, 
        error_message: str, 
        session_id: Optional[str] = None
    ):
        """记录错误"""
        error_key = f"{error_type}_{session_id or self._current_session_id}"
        self._error_stats[error_key] += 1
        
        logger.warning(f"记录错误: {error_type} - {error_message}")
    
    async def record_strategy_performance(
        self,
        strategy_type: StrategyType,
        signal: StrategySignal,
        fusion_result: FusionResult,
        actual_outcome: Optional[SignalDirection] = None
    ):
        """记录策略性能"""
        perf_data = self._strategy_performance[strategy_type]
        perf_data["total_signals"] += 1
        
        # 如果有实际结果，更新性能指标
        if actual_outcome is not None:
            if actual_outcome == signal.direction:
                perf_data["successful_signals"] += 1
            
            # 计算准确率
            accuracy = perf_data["successful_signals"] / perf_data["total_signals"]
            perf_data["accuracy"] = accuracy
        
        logger.debug(f"记录策略 {strategy_type.value} 性能: "
                    f"准确率 {perf_data['accuracy']:.3f}")
    
    async def record_throughput(self, signals_processed: int, time_taken: float):
        """记录吞吐量"""
        throughput = signals_processed / max(time_taken, 0.001)  # 防止除零
        self._throughput_history.append({
            "timestamp": datetime.now(),
            "throughput": throughput,
            "signals_processed": signals_processed,
            "time_taken": time_taken
        })
    
    async def generate_performance_metrics(
        self, 
        session_id: Optional[str] = None
    ) -> PerformanceMetrics:
        """生成性能指标"""
        current_time = datetime.now()
        
        # 计算总体处理时间
        total_time = (current_time - self._start_time).total_seconds()
        
        # 计算延迟分解统计
        latency_breakdown = {}
        for stage, records in self._latency_breakdown.items():
            if records:
                durations = [r["duration"] for r in records]
                latency_breakdown[stage] = {
                    "mean": np.mean(durations),
                    "p50": np.percentile(durations, 50),
                    "p95": np.percentile(durations, 95),
                    "p99": np.percentile(durations, 99),
                    "max": np.max(durations),
                    "min": np.min(durations)
                }
        
        # 计算缓存命中率
        cache_hit_rates = {}
        total_hits = 0
        total_requests = 0
        
        for cache_key, stats in self._cache_stats.items():
            hits = stats["hits"]
            misses = stats["misses"]
            total_cache_requests = hits + misses
            
            if total_cache_requests > 0:
                hit_rate = hits / total_cache_requests
                cache_hit_rates[cache_key] = hit_rate
                total_hits += hits
                total_requests += total_cache_requests
        
        overall_cache_hit_rate = total_hits / max(total_requests, 1)
        
        # 计算错误率
        error_rates = {}
        total_errors = sum(self._error_stats.values())
        
        for error_key, count in self._error_stats.items():
            error_rates[error_key] = count
        
        # 计算策略贡献度
        strategy_contribution = {}
        total_strategy_signals = sum(
            perf["total_signals"] for perf in self._strategy_performance.values()
        )
        
        for strategy_type, perf_data in self._strategy_performance.items():
            if total_strategy_signals > 0:
                contribution = perf_data["total_signals"] / total_strategy_signals
                strategy_contribution[strategy_type] = contribution
        
        # 计算吞吐量
        recent_throughput = 0.0
        if self._throughput_history:
            recent_throughputs = [t["throughput"] for t in list(self._throughput_history)[-10:]]
            recent_throughput = np.mean(recent_throughputs)
        
        metrics = PerformanceMetrics(
            total_processing_time=total_time,
            throughput=recent_throughput,
            accuracy_metrics=self._calculate_accuracy_metrics(),
            latency_breakdown=latency_breakdown,
            cache_hit_rates=cache_hit_rates,
            error_rates=error_rates,
            strategy_contribution=strategy_contribution,
            timestamp=current_time
        )
        
        # 保存指标历史
        self._metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """计算准确率指标"""
        metrics = {}
        
        # 计算总体准确率
        total_successful = sum(
            perf["successful_signals"] for perf in self._strategy_performance.values()
        )
        total_signals = sum(
            perf["total_signals"] for perf in self._strategy_performance.values()
        )
        
        overall_accuracy = total_successful / max(total_signals, 1)
        metrics["overall_accuracy"] = overall_accuracy
        
        # 计算各策略的准确率
        for strategy_type, perf_data in self._strategy_performance.items():
            strategy_accuracy = perf_data["accuracy"]
            metrics[f"{strategy_type.value}_accuracy"] = strategy_accuracy
        
        return metrics
    
    async def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """检查性能告警"""
        alerts = []
        
        try:
            # 生成当前性能指标
            current_metrics = await self.generate_performance_metrics()
            
            # 检查延迟告警
            latency_alerts = await self._check_latency_alerts(current_metrics)
            alerts.extend(latency_alerts)
            
            # 检查吞吐量告警
            throughput_alerts = await self._check_throughput_alerts(current_metrics)
            alerts.extend(throughput_alerts)
            
            # 检查错误率告警
            error_alerts = await self._check_error_rate_alerts(current_metrics)
            alerts.extend(error_alerts)
            
            # 检查缓存命中率告警
            cache_alerts = await self._check_cache_alerts(current_metrics)
            alerts.extend(cache_alerts)
            
            # 检查准确率告警
            accuracy_alerts = await self._check_accuracy_alerts(current_metrics)
            alerts.extend(accuracy_alerts)
            
            if alerts:
                logger.warning(f"检测到 {len(alerts)} 个性能告警")
            
            return alerts
            
        except Exception as e:
            logger.error(f"检查性能告警时出错: {e}")
            return [{"type": "system_error", "message": str(e), "severity": "high"}]
    
    async def _check_latency_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """检查延迟告警"""
        alerts = []
        
        # 检查总体延迟
        if metrics.total_processing_time > self.config.timeout_seconds:
            alerts.append({
                "type": "latency",
                "metric": "total_processing_time",
                "value": metrics.total_processing_time,
                "threshold": self.config.timeout_seconds,
                "severity": "high",
                "message": f"总处理时间 {metrics.total_processing_time:.2f}s 超过阈值 {self.config.timeout_seconds}s"
            })
        
        # 检查各阶段延迟
        for stage, stats in metrics.latency_breakdown.items():
            if stats["p95"] > 2.0:  # 95分位数超过2秒
                alerts.append({
                    "type": "latency",
                    "metric": f"{stage}_p95",
                    "value": stats["p95"],
                    "threshold": 2.0,
                    "severity": "medium",
                    "message": f"{stage} 阶段95分位数延迟 {stats['p95']:.2f}s 较高"
                })
        
        return alerts
    
    async def _check_throughput_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """检查吞吐量告警"""
        alerts = []
        
        # 检查吞吐量下限
        if metrics.throughput < 10.0:  # 每秒少于10个信号
            alerts.append({
                "type": "throughput",
                "metric": "signals_per_second",
                "value": metrics.throughput,
                "threshold": 10.0,
                "severity": "medium",
                "message": f"吞吐量 {metrics.throughput:.2f} 信号/秒 低于预期"
            })
        
        return alerts
    
    async def _check_error_rate_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """检查错误率告警"""
        alerts = []
        
        total_errors = sum(metrics.error_rates.values())
        if total_errors > 50:  # 错误数过多
            alerts.append({
                "type": "error_rate",
                "metric": "total_errors",
                "value": total_errors,
                "threshold": 50,
                "severity": "high",
                "message": f"总错误数 {total_errors} 超过阈值"
            })
        
        # 检查特定错误类型
        for error_type, count in metrics.error_rates.items():
            if count > 20:  # 单个错误类型过多
                alerts.append({
                    "type": "error_rate",
                    "metric": error_type,
                    "value": count,
                    "threshold": 20,
                    "severity": "medium",
                    "message": f"{error_type} 错误数 {count} 超过阈值"
                })
        
        return alerts
    
    async def _check_cache_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """检查缓存命中率告警"""
        alerts = []
        
        # 检查总体缓存命中率
        overall_hit_rate = sum(metrics.cache_hit_rates.values()) / max(len(metrics.cache_hit_rates), 1)
        if overall_hit_rate < 0.7:  # 命中率低于70%
            alerts.append({
                "type": "cache",
                "metric": "overall_hit_rate",
                "value": overall_hit_rate,
                "threshold": 0.7,
                "severity": "medium",
                "message": f"缓存命中率 {overall_hit_rate:.2%} 低于阈值 70%"
            })
        
        return alerts
    
    async def _check_accuracy_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """检查准确率告警"""
        alerts = []
        
        # 检查总体准确率
        overall_accuracy = metrics.accuracy_metrics.get("overall_accuracy", 0.0)
        if overall_accuracy < 0.6:  # 准确率低于60%
            alerts.append({
                "type": "accuracy",
                "metric": "overall_accuracy",
                "value": overall_accuracy,
                "threshold": 0.6,
                "severity": "high",
                "message": f"总体准确率 {overall_accuracy:.2%} 低于阈值 60%"
            })
        
        return alerts
    
    async def get_performance_trend(
        self, 
        metric_name: str, 
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """获取性能趋势"""
        current_time = datetime.now()
        window_start = current_time - time_window
        
        # 过滤时间窗口内的指标
        window_metrics = [
            m for m in self._metrics_history 
            if m.timestamp >= window_start
        ]
        
        if not window_metrics:
            return {"status": "no_data", "time_window": str(time_window)}
        
        # 提取指标值
        if metric_name == "throughput":
            values = [m.throughput for m in window_metrics]
        elif metric_name == "accuracy":
            values = [m.accuracy_metrics.get("overall_accuracy", 0.0) for m in window_metrics]
        else:
            return {"status": "invalid_metric", "metric": metric_name}
        
        if not values:
            return {"status": "no_values", "metric": metric_name}
        
        # 计算趋势统计
        trend_stats = {
            "metric": metric_name,
            "time_window": str(time_window),
            "data_points": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "latest": values[-1],
            "trend": "stable"
        }
        
        # 计算趋势方向
        if len(values) >= 10:
            first_half = np.mean(values[:len(values)//2])
            second_half = np.mean(values[len(values)//2:])
            
            if second_half > first_half * 1.05:
                trend_stats["trend"] = "increasing"
            elif second_half < first_half * 0.95:
                trend_stats["trend"] = "decreasing"
        
        return trend_stats
    
    async def generate_performance_report(
        self, 
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """生成性能报告"""
        current_metrics = await self.generate_performance_metrics()
        alerts = await self.check_performance_alerts()
        
        # 获取关键趋势
        throughput_trend = await self.get_performance_trend("throughput", time_window)
        accuracy_trend = await self.get_performance_trend("accuracy", time_window)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "time_window": str(time_window),
            "summary": {
                "total_uptime": (datetime.now() - self._start_time).total_seconds(),
                "current_throughput": current_metrics.throughput,
                "current_accuracy": current_metrics.accuracy_metrics.get("overall_accuracy", 0.0),
                "total_alerts": len(alerts),
                "system_health": self._calculate_system_health(current_metrics, alerts)
            },
            "performance_metrics": {
                "throughput": current_metrics.throughput,
                "latency_breakdown": current_metrics.latency_breakdown,
                "cache_hit_rates": current_metrics.cache_hit_rates,
                "error_rates": current_metrics.error_rates,
                "strategy_contribution": current_metrics.strategy_contribution
            },
            "trends": {
                "throughput_trend": throughput_trend,
                "accuracy_trend": accuracy_trend
            },
            "alerts": alerts,
            "recommendations": await self._generate_recommendations(current_metrics, alerts)
        }
        
        return report
    
    def _calculate_system_health(
        self, 
        metrics: PerformanceMetrics, 
        alerts: List[Dict[str, Any]]
    ) -> str:
        """计算系统健康状态"""
        high_severity_alerts = [a for a in alerts if a.get("severity") == "high"]
        
        if len(high_severity_alerts) > 0:
            return "poor"
        elif len(alerts) > 5:
            return "fair"
        elif len(alerts) > 0:
            return "good"
        else:
            return "excellent"
    
    async def _generate_recommendations(
        self, 
        metrics: PerformanceMetrics, 
        alerts: List[Dict[str, Any]]
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于告警生成建议
        for alert in alerts:
            alert_type = alert.get("type")
            
            if alert_type == "latency":
                recommendations.append("考虑增加并行处理能力或优化算法效率")
            elif alert_type == "throughput":
                recommendations.append("建议增加批处理大小或优化资源分配")
            elif alert_type == "error_rate":
                recommendations.append("需要检查错误源头，增强异常处理机制")
            elif alert_type == "cache":
                recommendations.append("考虑增加缓存容量或优化缓存策略")
            elif alert_type == "accuracy":
                recommendations.append("建议重新训练模型或调整信号融合权重")
        
        # 基于性能指标生成建议
        if metrics.throughput < 20:
            recommendations.append("当前吞吐量较低，建议优化管道配置")
        
        if metrics.accuracy_metrics.get("overall_accuracy", 1.0) < 0.7:
            recommendations.append("准确率有待提升，建议调整策略融合算法")
        
        # 去重并限制数量
        recommendations = list(set(recommendations))[:5]
        
        return recommendations
    
    async def export_metrics(
        self, 
        format: str = "json",
        time_window: Optional[timedelta] = None
    ) -> str:
        """导出指标数据"""
        if not time_window:
            time_window = timedelta(hours=24)
        
        current_time = datetime.now()
        window_start = current_time - time_window
        
        # 过滤时间窗口内的数据
        window_metrics = [
            m for m in self._metrics_history 
            if m.timestamp >= window_start
        ]
        
        if format == "json":
            import json
            
            export_data = {
                "export_timestamp": current_time.isoformat(),
                "time_window": str(time_window),
                "data_points": len(window_metrics),
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "throughput": m.throughput,
                        "total_processing_time": m.total_processing_time,
                        "accuracy_metrics": m.accuracy_metrics,
                        "latency_breakdown": m.latency_breakdown,
                        "cache_hit_rates": m.cache_hit_rates,
                        "error_rates": m.error_rates,
                        "strategy_contribution": {
                            k.value: v for k, v in m.strategy_contribution.items()
                        }
                    }
                    for m in window_metrics
                ]
            }
            
            return json.dumps(export_data, indent=2, default=str)
        
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def reset_metrics(self):
        """重置所有指标"""
        self._metrics_history.clear()
        self._latency_breakdown.clear()
        self._cache_stats.clear()
        self._error_stats.clear()
        self._strategy_performance.clear()
        self._throughput_history.clear()
        self._start_time = datetime.now()
        self._last_checkpoint = datetime.now()
        
        logger.info("性能监控器指标已重置")