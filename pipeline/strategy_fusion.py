"""
策略融合器模块

负责将多种策略信号进行融合，实现：
- 动态权重调整
- 冲突解决机制
- 综合评分计算
- 策略贡献度分析
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .types import (
    StrategySignal, StrategyType, SignalDirection, FusionResult,
    PipelineConfig, RiskLevel, ConfidenceLevel,
    PipelineError
)

logger = logging.getLogger(__name__)

class StrategyFusion:
    """策略融合器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._fusion_history = []
        self._strategy_performance = self._initialize_performance_tracker()
        self._conflict_resolution_rules = self._initialize_conflict_rules()
        self._dynamic_weights = config.strategy_weights.copy()
        
    def _initialize_performance_tracker(self) -> Dict[StrategyType, Dict]:
        """初始化性能跟踪器"""
        return {
            strategy_type: {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "total_signals": 0,
                "successful_signals": 0,
                "last_updated": datetime.now()
            }
            for strategy_type in StrategyType
        }
    
    def _initialize_conflict_rules(self) -> Dict[str, float]:
        """初始化冲突解决规则权重"""
        return {
            "same_direction_boost": 1.2,      # 相同方向信号增强
            "opposite_direction_penalty": 0.5, # 相反方向信号惩罚
            "high_confidence_priority": 1.3,   # 高置信度优先级
            "ml_signal_priority": 1.4,        # ML信号优先级
            "technical_signal_priority": 1.1,  # 技术信号优先级
            "risk_signal_override": 0.3        # 风险信号覆盖
        }
    
    async def fuse_signals(
        self, 
        signals: List[StrategySignal],
        market_context: Optional[Dict] = None
    ) -> FusionResult:
        """融合策略信号"""
        if not signals:
            raise PipelineError("没有信号可以融合")
        
        symbol = signals[0].symbol
        start_time = datetime.now()
        
        try:
            logger.info(f"开始融合 {len(signals)} 个信号 for {symbol}")
            
            # 1. 信号预处理和验证
            validated_signals = await self._preprocess_signals(signals)
            
            # 2. 检测冲突并解决
            resolved_signals = await self._resolve_conflicts(validated_signals)
            
            # 3. 计算动态权重
            dynamic_weights = await self._calculate_dynamic_weights(resolved_signals, market_context)
            
            # 4. 计算综合评分
            fusion_result = await self._calculate_combined_score(resolved_signals, dynamic_weights)
            
            # 5. 执行风险回报检验
            final_result = await self._apply_risk_reward_filter(fusion_result)
            
            # 6. 更新性能跟踪
            self._update_performance_tracking(resolved_signals, final_result)
            
            # 7. 记录融合历史
            self._fusion_history.append({
                "timestamp": start_time,
                "symbol": symbol,
                "input_signals": len(signals),
                "output_signal": final_result,
                "processing_time": (datetime.now() - start_time).total_seconds()
            })
            
            logger.info(f"信号融合完成，最终方向: {final_result.final_direction}, "
                       f"综合评分: {final_result.final_score:.3f}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"信号融合过程中出错: {e}")
            # 返回默认的HOLD信号
            return FusionResult(
                symbol=symbol,
                final_direction=SignalDirection.HOLD,
                final_score=0.0,
                combined_confidence=0.0,
                risk_reward_ratio=0.0,
                decision_reason=[f"融合失败: {str(e)}"]
            )
    
    async def _preprocess_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """预处理信号"""
        # 按置信度排序
        sorted_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
        
        # 移除重复或相似的信号
        unique_signals = self._deduplicate_signals(sorted_signals)
        
        # 验证信号
        validated_signals = []
        for signal in unique_signals:
            if self._is_signal_valid(signal):
                validated_signals.append(signal)
        
        return validated_signals
    
    def _deduplicate_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """去除重复信号"""
        seen_signatures = set()
        unique_signals = []
        
        for signal in signals:
            # 创建信号签名
            signature = (
                signal.strategy_type,
                signal.direction,
                round(signal.confidence, 2),  # 精度到0.01
                int(signal.timestamp.timestamp() // 60)  # 精确到分钟
            )
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_signals.append(signal)
        
        return unique_signals
    
    def _is_signal_valid(self, signal: StrategySignal) -> bool:
        """检查信号有效性"""
        # 基本字段检查
        if not all([signal.symbol, signal.direction, signal.market_data]):
            return False
        
        # 置信度范围检查
        if not (0 <= signal.confidence <= 1):
            return False
        
        # 时间戳检查
        signal_age = (datetime.now() - signal.timestamp).total_seconds()
        if signal_age > 300:  # 5分钟过期
            return False
        
        # 数据新鲜度检查
        if signal.market_data.data_freshness > 60:  # 1分钟以上认为不新鲜
            return False
        
        return True
    
    async def _resolve_conflicts(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """解决信号冲突"""
        if len(signals) <= 1:
            return signals
        
        # 按方向分组信号
        direction_groups = {
            SignalDirection.LONG: [],
            SignalDirection.SHORT: [],
            SignalDirection.HOLD: []
        }
        
        for signal in signals:
            direction_groups[signal.direction].append(signal)
        
        # 解决方向冲突
        resolved_signals = await self._resolve_direction_conflicts(direction_groups)
        
        # 解决强度冲突
        final_signals = await self._resolve_intensity_conflicts(resolved_signals)
        
        return final_signals
    
    async def _resolve_direction_conflicts(
        self, 
        direction_groups: Dict[SignalDirection, List[StrategySignal]]
    ) -> List[StrategySignal]:
        """解决方向冲突"""
        non_hold_groups = {
            k: v for k, v in direction_groups.items() 
            if k != SignalDirection.HOLD and v
        }
        
        if not non_hold_groups:
            # 只有HOLD信号
            return direction_groups[SignalDirection.HOLD]
        
        if len(non_hold_groups) == 1:
            # 只有一个非HOLD方向
            dominant_direction = list(non_hold_groups.keys())[0]
            return direction_groups[dominant_direction] + direction_groups[SignalDirection.HOLD]
        
        # 有多个非HOLD方向冲突
        return await self._handle_multi_direction_conflict(non_hold_groups)
    
    async def _handle_multi_direction_conflict(
        self, 
        direction_groups: Dict[SignalDirection, List[StrategySignal]]
    ) -> List[StrategySignal]:
        """处理多方向冲突"""
        long_signals = direction_groups.get(SignalDirection.LONG, [])
        short_signals = direction_groups.get(SignalDirection.SHORT, [])
        
        # 计算各方向的综合强度
        long_strength = sum(signal.score * signal.confidence for signal in long_signals)
        short_strength = sum(signal.score * signal.confidence for signal in short_signals)
        
        # 根据强度和优先级决定
        if long_strength > short_strength * 1.2:  # 强优势
            return long_signals
        elif short_strength > long_strength * 1.2:  # 强优势
            return short_signals
        else:
            # 势均力敌，降级为HOLD
            return [SignalDirection.HOLD]
    
    async def _resolve_intensity_conflicts(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """解决强度冲突"""
        if len(signals) <= 1:
            return signals
        
        # 按强度排序
        sorted_signals = sorted(signals, key=lambda x: x.score * x.confidence, reverse=True)
        
        # 保留最强的几个信号，过滤掉过弱的信号
        threshold = sorted_signals[0].score * sorted_signals[0].confidence * 0.3
        filtered_signals = [
            signal for signal in sorted_signals
            if signal.score * signal.confidence >= threshold
        ]
        
        return filtered_signals
    
    async def _calculate_dynamic_weights(
        self, 
        signals: List[StrategySignal], 
        market_context: Optional[Dict] = None
    ) -> Dict[StrategyType, float]:
        """计算动态权重"""
        base_weights = self.config.strategy_weights.copy()
        performance_weights = self._get_performance_weights()
        market_weights = self._get_market_adjustment_weights(market_context or {})
        
        # 组合权重
        final_weights = {}
        for strategy_type in StrategyType:
            base_weight = base_weights.get(strategy_type, 0.0)
            perf_weight = performance_weights.get(strategy_type, 1.0)
            market_weight = market_weights.get(strategy_type, 1.0)
            
            final_weights[strategy_type] = base_weight * perf_weight * market_weight
        
        # 归一化权重
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        
        self._dynamic_weights = final_weights
        return final_weights
    
    def _get_performance_weights(self) -> Dict[StrategyType, float]:
        """基于历史性能调整权重"""
        weights = {}
        for strategy_type, perf_data in self._strategy_performance.items():
            # 基于F1分数调整权重
            f1_score = perf_data.get("f1_score", 0.5)
            accuracy = perf_data.get("accuracy", 0.5)
            
            # 权重范围: 0.5 - 2.0
            weight = 0.5 + (f1_score + accuracy) / 2
            weights[strategy_type] = min(2.0, max(0.5, weight))
        
        return weights
    
    def _get_market_adjustment_weights(self, market_context: Dict) -> Dict[StrategyType, float]:
        """基于市场环境调整权重"""
        weights = {strategy_type: 1.0 for strategy_type in StrategyType}
        
        # 波动率调整
        volatility = market_context.get("volatility", 0.02)
        if volatility > 0.05:  # 高波动率
            weights[StrategyType.RISK_MODEL] *= 1.5  # 增强风险模型权重
            weights[StrategyType.ML_PREDICTION] *= 0.8  # 降低ML权重
        
        # 趋势强度调整
        trend_strength = market_context.get("trend_strength", 0.0)
        if trend_strength > 0.7:  # 强趋势
            weights[StrategyType.TECHNICAL_INDICATOR] *= 1.3  # 增强技术指标权重
        
        return weights
    
    async def _calculate_combined_score(
        self, 
        signals: List[StrategySignal], 
        dynamic_weights: Dict[StrategyType, float]
    ) -> FusionResult:
        """计算综合评分"""
        if not signals:
            raise PipelineError("没有有效信号用于计算")
        
        symbol = signals[0].symbol
        
        # 按策略类型分组信号
        strategy_groups = {}
        for signal in signals:
            strategy_type = signal.strategy_type
            if strategy_type not in strategy_groups:
                strategy_groups[strategy_type] = []
            strategy_groups[strategy_type].append(signal)
        
        # 计算各策略类型的最佳信号
        component_scores = {}
        confidence_breakdown = {}
        decision_reasons = []
        
        for strategy_type, signals_in_group in strategy_groups.items():
            # 选择该策略类型中评分最高的信号
            best_signal = max(signals_in_group, key=lambda x: x.score * x.confidence)
            
            # 计算加权分数
            weight = dynamic_weights.get(strategy_type, 0.0)
            weighted_score = best_signal.score * weight
            component_scores[strategy_type] = weighted_score
            
            # 记录置信度分解
            confidence_breakdown[strategy_type.value] = best_signal.confidence
            
            # 收集决策原因
            decision_reasons.append(
                f"{strategy_type.value}: 方向={best_signal.direction.value}, "
                f"置信度={best_signal.confidence:.3f}, 权重={weight:.3f}"
            )
        
        # 计算最终评分
        total_score = sum(component_scores.values())
        
        # 计算最终置信度 (加权平均)
        total_confidence = 0.0
        total_weight = 0.0
        for strategy_type, signals_in_group in strategy_groups.items():
            weight = dynamic_weights.get(strategy_type, 0.0)
            best_signal = max(signals_in_group, key=lambda x: x.score * x.confidence)
            total_confidence += best_signal.confidence * weight
            total_weight += weight
        
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        # 确定最终方向
        final_direction = self._determine_final_direction(signals, component_scores)
        
        # 计算风险回报比
        risk_reward_ratio = self._calculate_risk_reward_ratio(signals)
        
        return FusionResult(
            symbol=symbol,
            final_direction=final_direction,
            final_score=total_score,
            combined_confidence=final_confidence,
            risk_reward_ratio=risk_reward_ratio,
            component_scores=component_scores,
            confidence_breakdown=confidence_breakdown,
            decision_reason=decision_reasons,
            metadata={"dynamic_weights": dynamic_weights}
        )
    
    def _determine_final_direction(
        self, 
        signals: List[StrategySignal],
        component_scores: Dict[StrategyType, float]
    ) -> SignalDirection:
        """确定最终交易方向"""
        # 按方向汇总分数
        direction_scores = {
            SignalDirection.LONG: 0.0,
            SignalDirection.SHORT: 0.0,
            SignalDirection.HOLD: 0.0
        }
        
        for signal in signals:
            direction_scores[signal.direction] += signal.score * signal.confidence
        
        # 找到得分最高的方向
        max_direction = max(direction_scores.items(), key=lambda x: x[1])
        
        # 应用置信度阈值
        if max_direction[1] < 0.3:  # 最低阈值
            return SignalDirection.HOLD
        
        return max_direction[0]
    
    def _calculate_risk_reward_ratio(self, signals: List[StrategySignal]) -> float:
        """计算风险回报比"""
        # 这里简化处理，实际中应该基于具体的止损止盈设置
        # 优先使用LLM评估中的参数
        for signal in signals:
            if signal.llm_assessment:
                llm = signal.llm_assessment
                if llm.stop_loss > 0 and llm.take_profit > 0:
                    risk = abs(llm.optimal_entry - llm.stop_loss)
                    reward = abs(llm.take_profit - llm.optimal_entry)
                    if risk > 0:
                        return reward / risk
        
        # 默认风险回报比
        return 1.5
    
    async def _apply_risk_reward_filter(self, fusion_result: FusionResult) -> FusionResult:
        """应用风险回报过滤器"""
        min_rr = self.config.min_risk_reward_ratio
        
        if fusion_result.risk_reward_ratio < min_rr:
            # 风险回报比不达标，降级为HOLD
            return FusionResult(
                symbol=fusion_result.symbol,
                final_direction=SignalDirection.HOLD,
                final_score=fusion_result.final_score * 0.5,  # 降低评分
                combined_confidence=fusion_result.combined_confidence * 0.7,
                risk_reward_ratio=fusion_result.risk_reward_ratio,
                component_scores=fusion_result.component_scores,
                confidence_breakdown=fusion_result.confidence_breakdown,
                decision_reason=fusion_result.decision_reason + [
                    f"风险回报比 {fusion_result.risk_reward_ratio:.2f} 低于阈值 {min_rr}"
                ]
            )
        
        return fusion_result
    
    def _update_performance_tracking(
        self, 
        signals: List[StrategySignal], 
        fusion_result: FusionResult
    ):
        """更新性能跟踪"""
        for signal in signals:
            strategy_type = signal.strategy_type
            perf_data = self._strategy_performance[strategy_type]
            
            # 更新计数
            perf_data["total_signals"] += 1
            
            # 这里简化处理，实际中需要根据实际交易结果更新
            # 假设融合结果反映了信号的实际效果
            if fusion_result.final_direction == signal.direction:
                perf_data["successful_signals"] += 1
    
    def get_fusion_statistics(self) -> Dict:
        """获取融合统计信息"""
        if not self._fusion_history:
            return {}
        
        recent_history = self._fusion_history[-100:]  # 最近100次
        
        return {
            "total_fusions": len(self._fusion_history),
            "recent_fusions": len(recent_history),
            "average_processing_time": np.mean([h["processing_time"] for h in recent_history]),
            "direction_distribution": self._get_direction_distribution(recent_history),
            "strategy_weights": self._dynamic_weights.copy(),
            "performance_metrics": self._strategy_performance.copy()
        }
    
    def _get_direction_distribution(self, history: List[Dict]) -> Dict[str, int]:
        """获取方向分布统计"""
        distribution = {"long": 0, "short": 0, "hold": 0}
        
        for entry in history:
            direction = entry["output_signal"].final_direction.value
            distribution[direction] += 1
        
        return distribution
    
    def reset_performance_tracking(self):
        """重置性能跟踪"""
        self._strategy_performance = self._initialize_performance_tracker()
        logger.info("性能跟踪已重置")
    
    def update_conflict_resolution_rules(self, rules: Dict[str, float]):
        """更新冲突解决规则"""
        self._conflict_resolution_rules.update(rules)
        logger.info("冲突解决规则已更新")