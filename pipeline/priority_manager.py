"""
优先级管理器模块

负责信号的优先级排序和管理，包括：
- 动态优先级计算
- 资源分配优化
- 实时优先级调整
- 批量处理调度
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .types import (
    StrategySignal, SignalDirection, StrategyType, MarketData,
    PipelineConfig, FusionResult, PerformanceMetrics
)

logger = logging.getLogger(__name__)

class PriorityManager:
    """优先级管理器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._priority_queue = []
        self._processing_history = []
        self._priority_factors = self._initialize_priority_factors()
        self._resource_constraints = self._initialize_resource_constraints()
        
    def _initialize_priority_factors(self) -> Dict[str, float]:
        """初始化优先级因子权重"""
        return {
            "data_freshness": 0.25,      # 数据新鲜度
            "confidence": 0.20,          # 置信度
            "volatility": 0.15,          # 波动率水平
            "liquidity": 0.15,           # 流动性
            "performance": 0.10,         # 历史绩效
            "risk_level": 0.10,          # 风险等级
            "strategy_type": 0.05        # 策略类型优先级
        }
    
    def _initialize_resource_constraints(self) -> Dict[str, Any]:
        """初始化资源约束"""
        return {
            "max_concurrent_symbols": 50,
            "max_concurrent_signals": 200,
            "processing_time_limit": 2.0,  # 秒
            "memory_limit_mb": 1024,
            "cpu_limit_percent": 80
        }
    
    async def prioritize_signals(
        self, 
        signals: List[StrategySignal],
        market_context: Optional[Dict] = None
    ) -> List[StrategySignal]:
        """对信号进行优先级排序"""
        if not signals:
            return signals
        
        try:
            logger.info(f"开始对 {len(signals)} 个信号进行优先级排序")
            
            # 1. 计算每个信号的优先级分数
            prioritized_signals = []
            for signal in signals:
                priority_score = await self._calculate_priority_score(signal, market_context or {})
                prioritized_signals.append((signal, priority_score))
            
            # 2. 按优先级排序
            prioritized_signals.sort(key=lambda x: x[1], reverse=True)
            
            # 3. 应用资源约束
            filtered_signals = self._apply_resource_constraints(
                [signal for signal, _ in prioritized_signals]
            )
            
            # 4. 限制处理数量
            limited_signals = filtered_signals[:self.config.max_symbols_per_scan]
            
            logger.info(f"优先级排序完成，保留 {len(limited_signals)} 个高优先级信号")
            return limited_signals
            
        except Exception as e:
            logger.error(f"信号优先级排序时出错: {e}")
            return signals  # 出错时返回原始信号列表
    
    async def _calculate_priority_score(
        self, 
        signal: StrategySignal, 
        market_context: Dict
    ) -> float:
        """计算信号优先级分数"""
        score = 0.0
        
        # 1. 数据新鲜度分数 (0-1)
        freshness_score = self._calculate_freshness_score(signal)
        score += freshness_score * self._priority_factors["data_freshness"]
        
        # 2. 置信度分数 (0-1)
        confidence_score = signal.confidence
        score += confidence_score * self._priority_factors["confidence"]
        
        # 3. 波动率分数 (0-1)
        volatility_score = self._calculate_volatility_score(signal, market_context)
        score += volatility_score * self._priority_factors["volatility"]
        
        # 4. 流动性分数 (0-1)
        liquidity_score = self._calculate_liquidity_score(signal, market_context)
        score += liquidity_score * self._priority_factors["liquidity"]
        
        # 5. 绩效分数 (0-1)
        performance_score = self._calculate_performance_score(signal)
        score += performance_score * self._priority_factors["performance"]
        
        # 6. 风险等级分数 (0-1)
        risk_score = self._calculate_risk_score(signal)
        score += risk_score * self._priority_factors["risk_level"]
        
        # 7. 策略类型分数 (0-1)
        strategy_score = self._calculate_strategy_type_score(signal)
        score += strategy_score * self._priority_factors["strategy_type"]
        
        return min(1.0, max(0.0, score))  # 限制在0-1范围内
    
    def _calculate_freshness_score(self, signal: StrategySignal) -> float:
        """计算数据新鲜度分数"""
        data_age = (datetime.now() - signal.market_data.timestamp).total_seconds()
        
        if data_age <= 5:      # 5秒内
            return 1.0
        elif data_age <= 30:   # 30秒内
            return 1.0 - (data_age - 5) / 25 * 0.2
        elif data_age <= 60:   # 1分钟内
            return 0.8 - (data_age - 30) / 30 * 0.3
        else:                  # 超过1分钟
            return max(0.0, 0.5 - (data_age - 60) / 60 * 0.5)
    
    def _calculate_volatility_score(self, signal: StrategySignal, market_context: Dict) -> float:
        """计算波动率分数"""
        # 从风险指标中获取波动率
        volatility = None
        if signal.risk_metrics and signal.risk_metrics.garch_volatility:
            volatility = signal.risk_metrics.garch_volatility
        
        # 从市场上下文中获取波动率
        if not volatility and "volatility" in market_context:
            volatility = market_context["volatility"]
        
        if volatility is None:
            return 0.5  # 中性分数
        
        # 适中的波动率得分最高 (0.02-0.04)
        if 0.02 <= volatility <= 0.04:
            return 1.0
        elif volatility < 0.02:  # 低波动率
            return 0.7
        elif volatility < 0.06:  # 中高波动率
            return 0.8 - (volatility - 0.04) / 0.02 * 0.3
        else:  # 过高波动率
            return max(0.0, 0.5 - (volatility - 0.06) / 0.04 * 0.5)
    
    def _calculate_liquidity_score(self, signal: StrategySignal, market_context: Dict) -> float:
        """计算流动性分数"""
        # 从成交量中推断流动性
        volume = signal.market_data.volume
        if volume <= 0:
            return 0.0
        
        # 成交量标准化 (使用对数)
        log_volume = np.log10(volume)
        if log_volume >= 8:      # 高流动性
            return 1.0
        elif log_volume >= 7:    # 中等流动性
            return 0.8
        elif log_volume >= 6:    # 一般流动性
            return 0.6
        else:                    # 低流动性
            return 0.3
    
    def _calculate_performance_score(self, signal: StrategySignal) -> float:
        """计算绩效分数"""
        score = 0.0
        
        # 基于信号评分的绩效
        score += min(1.0, signal.score * 2)  # 信号分数权重
        
        # 基于置信度的绩效
        score += signal.confidence * 0.3
        
        # 如果有回测结果，增加权重
        if signal.backtest_result:
            bt_score = signal.backtest_result.score
            win_rate = signal.backtest_result.win_rate
            score += bt_score * 0.4 + win_rate * 0.2
        
        return min(1.0, score)
    
    def _calculate_risk_score(self, signal: StrategySignal) -> float:
        """计算风险等级分数 (风险越低分数越高)"""
        if signal.risk_metrics:
            risk_metrics = signal.risk_metrics
            
            # VaR风险
            if risk_metrics.var_95:
                if risk_metrics.var_95 > 0.05:  # 高风险
                    return 0.2
                elif risk_metrics.var_95 > 0.03:  # 中等风险
                    return 0.6
                else:  # 低风险
                    return 0.9
            
            # 最大回撤风险
            if risk_metrics.max_drawdown:
                max_dd = abs(risk_metrics.max_drawdown)
                if max_dd > 0.2:  # 高回撤
                    return 0.3
                elif max_dd > 0.1:  # 中等回撤
                    return 0.7
                else:  # 低回撤
                    return 0.9
        
        # 没有风险指标时的默认分数
        return 0.5
    
    def _calculate_strategy_type_score(self, signal: StrategySignal) -> float:
        """计算策略类型分数"""
        strategy_scores = {
            StrategyType.ML_PREDICTION: 1.0,           # ML预测最高优先级
            StrategyType.TECHNICAL_INDICATOR: 0.8,     # 技术指标次高
            StrategyType.BACKTEST_REFERENCE: 0.6,      # 回测参考中等
            StrategyType.RISK_MODEL: 0.4               # 风险模型较低
        }
        
        return strategy_scores.get(signal.strategy_type, 0.5)
    
    def _apply_resource_constraints(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """应用资源约束"""
        # 限制并发处理的符号数量
        max_symbols = self._resource_constraints["max_concurrent_symbols"]
        unique_symbols = {}
        
        for signal in signals:
            symbol = signal.symbol
            if symbol not in unique_symbols:
                unique_symbols[symbol] = []
            unique_symbols[symbol].append(signal)
        
        # 对每个符号，只保留优先级最高的信号
        filtered_signals = []
        for symbol, symbol_signals in unique_symbols.items():
            if len(symbol_signals) > 1:
                # 选择评分最高的信号
                best_signal = max(symbol_signals, key=lambda x: x.score * x.confidence)
                filtered_signals.append(best_signal)
            else:
                filtered_signals.append(symbol_signals[0])
        
        # 限制总数量
        if len(filtered_signals) > max_symbols:
            # 按优先级排序后截取
            filtered_signals.sort(key=lambda x: x.score * x.confidence, reverse=True)
            filtered_signals = filtered_signals[:max_symbols]
        
        return filtered_signals
    
    async def schedule_batch_processing(
        self,
        signals: List[StrategySignal],
        fusion_results: Optional[List[FusionResult]] = None
    ) -> Dict[str, Any]:
        """安排批量处理"""
        if not signals:
            return {"status": "no_signals", "batches": []}
        
        try:
            # 1. 按优先级分组
            priority_groups = self._group_by_priority(signals)
            
            # 2. 创建处理批次
            batches = self._create_processing_batches(priority_groups)
            
            # 3. 分配资源
            resource_allocation = self._allocate_resources(batches)
            
            # 4. 预测处理时间
            estimated_time = self._estimate_processing_time(batches)
            
            scheduling_info = {
                "status": "scheduled",
                "total_signals": len(signals),
                "total_batches": len(batches),
                "batches": batches,
                "resource_allocation": resource_allocation,
                "estimated_processing_time": estimated_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"批量处理调度完成: {len(batches)} 个批次，"
                       f"预计处理时间 {estimated_time:.2f} 秒")
            
            return scheduling_info
            
        except Exception as e:
            logger.error(f"批量处理调度时出错: {e}")
            return {"status": "error", "error": str(e)}
    
    def _group_by_priority(self, signals: List[StrategySignal]) -> Dict[str, List[StrategySignal]]:
        """按优先级分组信号"""
        groups = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        for signal in signals:
            priority_score = signal.score * signal.confidence
            if priority_score >= 0.8:
                groups["high"].append(signal)
            elif priority_score >= 0.5:
                groups["medium"].append(signal)
            else:
                groups["low"].append(signal)
        
        return groups
    
    def _create_processing_batches(self, priority_groups: Dict[str, List[StrategySignal]]) -> List[Dict]:
        """创建处理批次"""
        batch_size = self.config.batch_size
        batches = []
        batch_id = 1
        
        for priority, signals in priority_groups.items():
            if not signals:
                continue
            
            # 按优先级处理，高优先级先处理
            for i in range(0, len(signals), batch_size):
                batch_signals = signals[i:i + batch_size]
                
                batch = {
                    "batch_id": f"{priority}_{batch_id}",
                    "priority": priority,
                    "signals": len(batch_signals),
                    "symbols": list(set(s.signal for s in batch_signals)),
                    "estimated_time": len(batch_signals) * 0.1,  # 假设每个信号0.1秒
                    "status": "pending"
                }
                batches.append(batch)
                batch_id += 1
        
        # 按优先级排序批次
        priority_order = {"high": 0, "medium": 1, "low": 2}
        batches.sort(key=lambda x: priority_order[x["priority"]])
        
        return batches
    
    def _allocate_resources(self, batches: List[Dict]) -> Dict[str, Any]:
        """分配资源"""
        allocation = {
            "cpu_allocation": {},
            "memory_allocation": {},
            "processing_slots": {},
            "total_resources_used": 0
        }
        
        for batch in batches:
            batch_id = batch["batch_id"]
            signal_count = batch["signals"]
            
            # CPU分配 (假设每个信号需要2%的CPU)
            cpu_percent = min(20, signal_count * 2)
            allocation["cpu_allocation"][batch_id] = f"{cpu_percent}%"
            
            # 内存分配 (假设每个信号需要5MB内存)
            memory_mb = signal_count * 5
            allocation["memory_allocation"][batch_id] = f"{memory_mb}MB"
            
            # 处理槽位
            allocation["processing_slots"][batch_id] = min(4, max(1, signal_count // 10))
        
        return allocation
    
    def _estimate_processing_time(self, batches: List[Dict]) -> float:
        """估算处理时间"""
        total_time = 0.0
        
        for batch in batches:
            # 基于批次大小和优先级估算时间
            base_time = batch["estimated_time"]
            
            # 优先级调整
            if batch["priority"] == "high":
                multiplier = 1.0
            elif batch["priority"] == "medium":
                multiplier = 1.2
            else:  # low
                multiplier = 1.5
            
            total_time += base_time * multiplier
        
        return total_time
    
    async def update_priority_factors(self, new_factors: Dict[str, float]):
        """更新优先级因子权重"""
        # 验证权重总和
        total_weight = sum(new_factors.values())
        if abs(total_weight - 1.0) > 0.01:
            # 归一化权重
            new_factors = {k: v / total_weight for k, v in new_factors.items()}
        
        self._priority_factors.update(new_factors)
        logger.info(f"优先级因子权重已更新: {new_factors}")
    
    async def optimize_for_throughput(self, target_throughput: float) -> Dict[str, Any]:
        """为吞吐量优化"""
        optimization_suggestions = {
            "batch_size": self.config.batch_size,
            "concurrent_tasks": self.config.max_concurrent_tasks,
            "cache_ttl": self.config.cache_ttl,
            "priority_factors": self._priority_factors.copy(),
            "resource_constraints": self._resource_constraints.copy()
        }
        
        # 根据目标吞吐量调整参数
        if target_throughput > 100:  # 高吞吐量
            optimization_suggestions["batch_size"] = min(200, self.config.batch_size * 2)
            optimization_suggestions["concurrent_tasks"] = min(32, self.config.max_concurrent_tasks * 2)
            # 减少缓存时间以提高数据新鲜度
            for key in optimization_suggestions["cache_ttl"]:
                optimization_suggestions["cache_ttl"][key] = max(30, self.config.cache_ttl[key] // 2)
        
        elif target_throughput < 50:  # 低吞吐量，更注重质量
            optimization_suggestions["batch_size"] = max(50, self.config.batch_size // 2)
            optimization_suggestions["concurrent_tasks"] = max(8, self.config.max_concurrent_tasks // 2)
            # 增加缓存时间以减少重复计算
            for key in optimization_suggestions["cache_ttl"]:
                optimization_suggestions["cache_ttl"][key] = self.config.cache_ttl[key] * 2
        
        return optimization_suggestions
    
    def get_priority_statistics(self) -> Dict[str, Any]:
        """获取优先级统计信息"""
        if not self._processing_history:
            return {"status": "no_data"}
        
        recent_history = self._processing_history[-1000:]  # 最近1000次处理
        
        return {
            "total_processed": len(self._processing_history),
            "recent_processed": len(recent_history),
            "average_batch_size": np.mean([h.get("batch_size", 0) for h in recent_history]),
            "priority_distribution": self._calculate_priority_distribution(recent_history),
            "processing_efficiency": self._calculate_processing_efficiency(recent_history),
            "current_factors": self._priority_factors.copy(),
            "resource_utilization": self._calculate_resource_utilization()
        }
    
    def _calculate_priority_distribution(self, history: List[Dict]) -> Dict[str, int]:
        """计算优先级分布"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for entry in history:
            priority = entry.get("priority", "unknown")
            if priority in distribution:
                distribution[priority] += 1
        
        return distribution
    
    def _calculate_processing_efficiency(self, history: List[Dict]) -> Dict[str, float]:
        """计算处理效率"""
        if not history:
            return {"throughput": 0.0, "latency": 0.0}
        
        # 计算吞吐量 (信号/秒)
        total_signals = sum(h.get("signals_processed", 0) for h in history)
        total_time = sum(h.get("processing_time", 0) for h in history)
        throughput = total_signals / max(total_time, 0.1)
        
        # 计算平均延迟
        latencies = [h.get("processing_time", 0) for h in history if h.get("processing_time", 0) > 0]
        avg_latency = np.mean(latencies) if latencies else 0.0
        
        return {"throughput": throughput, "latency": avg_latency}
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """计算资源利用率"""
        # 这里简化处理，实际中应该基于真实监控数据
        return {
            "cpu_utilization": 0.65,  # 65% CPU利用率
            "memory_utilization": 0.58,  # 58% 内存利用率
            "io_utilization": 0.42   # 42% IO利用率
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._processing_history.clear()
        logger.info("优先级管理器统计信息已重置")