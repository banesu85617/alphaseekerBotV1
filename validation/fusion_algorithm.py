"""
验证结果融合算法
实现两层验证结果的综合评分与决策融合
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """融合策略枚举"""
    EQUAL_WEIGHT = "equal_weight"
    ADAPTIVE_WEIGHT = "adaptive_weight"
    PERFORMANCE_BASED = "performance_based"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class FusionConfig:
    """融合配置"""
    strategy: FusionStrategy = FusionStrategy.EQUAL_WEIGHT
    
    # 权重配置
    layer1_weight: float = 0.3
    layer2_weight: float = 0.4
    technical_indicators_weight: float = 0.15
    risk_context_weight: float = 0.15
    
    # 阈值配置
    risk_reward_threshold: float = 1.0
    confidence_threshold: float = 0.65
    minimum_score_threshold: float = 0.5
    
    # 自适应权重配置
    performance_history_window: int = 100
    adaptation_rate: float = 0.05
    
    # 动态权重调整
    enable_dynamic_weights: bool = True
    volatility_adjustment: bool = True
    liquidity_adjustment: bool = False


@dataclass
class ScoreComponent:
    """评分组件"""
    name: str
    value: float
    weight: float
    contribution: float
    metadata: Dict[str, Any] = None


class ValidationFusion:
    """
    验证结果融合器
    
    负责将两层验证结果融合为综合评分
    """
    
    def __init__(self, config: FusionConfig):
        """
        初始化融合器
        
        Args:
            config: 融合配置对象
        """
        self.config = config
        self.performance_history = []
        self.adaptive_weights = {
            'layer1': config.layer1_weight,
            'layer2': config.layer2_weight,
            'technical': config.technical_indicators_weight,
            'risk': config.risk_context_weight
        }
        self.is_initialized = False

    async def initialize(self) -> None:
        """初始化融合器"""
        logger.info("初始化验证结果融合器...")
        
        try:
            # 初始化性能历史记录
            if self.config.strategy == FusionStrategy.PERFORMANCE_BASED:
                await self._load_performance_history()
            
            self.is_initialized = True
            logger.info("融合器初始化完成")
            
        except Exception as e:
            logger.error(f"融合器初始化失败: {str(e)}")
            raise

    async def calculate_combined_score(self, fusion_input: Dict[str, Any]) -> float:
        """
        计算综合评分
        
        Args:
            fusion_input: 融合输入数据
            
        Returns:
            综合评分 (0-1)
        """
        if not self.is_initialized:
            raise RuntimeError("融合器未初始化")
        
        start_time = time.time()
        
        try:
            # 计算各组件评分
            score_components = await self._calculate_score_components(fusion_input)
            
            # 根据策略融合评分
            combined_score = await self._fuse_scores(score_components)
            
            # 应用阈值过滤
            final_score = self._apply_thresholds(combined_score)
            
            # 更新性能历史
            if self.config.enable_dynamic_weights:
                await self._update_performance_history(fusion_input, final_score)
            
            processing_time = time.time() - start_time
            logger.debug(f"综合评分计算完成: {final_score:.3f}, 耗时: {processing_time:.3f}s")
            
            return final_score
            
        except Exception as e:
            logger.error(f"综合评分计算失败: {str(e)}")
            return 0.0

    async def _calculate_score_components(self, fusion_input: Dict[str, Any]) -> List[ScoreComponent]:
        """计算评分组件"""
        components = []
        
        # 第一层评分
        layer1_score = await self._calculate_layer1_score(fusion_input)
        components.append(layer1_score)
        
        # 第二层评分
        layer2_score = await self._calculate_layer2_score(fusion_input)
        components.append(layer2_score)
        
        # 技术指标评分
        technical_score = await self._calculate_technical_score(fusion_input)
        components.append(technical_score)
        
        # 风险上下文评分
        risk_score = await self._calculate_risk_score(fusion_input)
        components.append(risk_score)
        
        return components

    async def _calculate_layer1_score(self, fusion_input: Dict[str, Any]) -> ScoreComponent:
        """计算第一层评分"""
        layer1_result = fusion_input.get('layer1_result', {})
        
        # 获取概率和置信度
        probability = layer1_result.get('probability', 0.5)
        confidence = layer1_result.get('confidence', 0.5)
        label = layer1_result.get('label', 0)
        
        # 计算评分：概率 * 置信度 * 方向权重
        direction_weight = 1.0 if label != 0 else 0.5  # 对非持有信号给予更高权重
        
        # 对于买入/卖出信号，给予额外奖励
        if label == 1:  # 买入
            direction_bonus = 0.1
        elif label == -1:  # 卖出
            direction_bonus = 0.1
        else:  # 持有
            direction_bonus = 0.0
        
        base_score = probability * confidence * direction_weight + direction_bonus
        
        # 考虑技术指标一致性
        technical_consistency = await self._check_technical_consistency(fusion_input, layer1_result)
        technical_bonus = technical_consistency * 0.1
        
        final_score = min(1.0, base_score + technical_bonus)
        
        return ScoreComponent(
            name="layer1_score",
            value=final_score,
            weight=self.adaptive_weights['layer1'],
            contribution=final_score * self.adaptive_weights['layer1'],
            metadata={
                'probability': probability,
                'confidence': confidence,
                'label': label,
                'technical_consistency': technical_consistency
            }
        )

    async def _calculate_layer2_score(self, fusion_input: Dict[str, Any]) -> ScoreComponent:
        """计算第二层评分"""
        layer2_result = fusion_input.get('layer2_result', {})
        
        # 基础评分：置信度
        confidence = layer2_result.get('confidence', 0.5)
        base_score = confidence
        
        # 参数完整性奖励
        entry = layer2_result.get('entry_price')
        stop_loss = layer2_result.get('stop_loss')
        take_profit = layer2_result.get('take_profit')
        
        if all(x is not None for x in [entry, stop_loss, take_profit]):
            completeness_bonus = 0.1
        else:
            completeness_bonus = 0.0
        
        # R/R比奖励
        risk_reward = fusion_input.get('risk_reward_ratio', 0.0)
        if risk_reward is not None:
            if risk_reward >= 2.0:
                rr_bonus = 0.15
            elif risk_reward >= 1.5:
                rr_bonus = 0.1
            elif risk_reward >= 1.0:
                rr_bonus = 0.05
            else:
                rr_bonus = 0.0
        else:
            rr_bonus = 0.0
        
        # 方向一致性奖励
        direction_consistency = await self._check_direction_consistency(fusion_input)
        consistency_bonus = direction_consistency * 0.1
        
        final_score = min(1.0, base_score + completeness_bonus + rr_bonus + consistency_bonus)
        
        return ScoreComponent(
            name="layer2_score",
            value=final_score,
            weight=self.adaptive_weights['layer2'],
            contribution=final_score * self.adaptive_weights['layer2'],
            metadata={
                'confidence': confidence,
                'risk_reward_ratio': risk_reward,
                'has_complete_params': all(x is not None for x in [entry, stop_loss, take_profit]),
                'direction_consistency': direction_consistency
            }
        )

    async def _calculate_technical_score(self, fusion_input: Dict[str, Any]) -> ScoreComponent:
        """计算技术指标评分"""
        technical_indicators = fusion_input.get('technical_indicators', {})
        
        score_factors = []
        
        # RSI评分
        rsi = technical_indicators.get('rsi')
        if rsi is not None:
            # RSI在30-70之间为中性，过低为超卖(利好)，过高为超买(利空)
            if 30 <= rsi <= 70:
                rsi_score = 0.5
            elif rsi < 30:
                rsi_score = 0.8  # 超卖，可能反弹
            else:
                rsi_score = 0.8  # 超买，可能下跌
            score_factors.append(rsi_score)
        
        # MACD评分
        macd = technical_indicators.get('macd')
        if macd is not None:
            # MACD金叉/死叉的简化评分
            macd_score = 0.6 if abs(macd) > 0.001 else 0.4
            score_factors.append(macd_score)
        
        # 布林带评分
        bb_position = technical_indicators.get('bollinger_position')
        if bb_position is not None:
            # 接近下轨为利好，接近上轨为利空
            bb_score = 0.7 if bb_position < 0.3 else 0.3 if bb_position > 0.7 else 0.5
            score_factors.append(bb_score)
        
        # ADX评分（趋势强度）
        adx = technical_indicators.get('adx')
        if adx is not None:
            # ADX > 20 表示趋势明显
            adx_score = 0.7 if adx > 20 else 0.4
            score_factors.append(adx_score)
        
        # 计算平均技术评分
        if score_factors:
            technical_score = np.mean(score_factors)
        else:
            technical_score = 0.5
        
        return ScoreComponent(
            name="technical_score",
            value=technical_score,
            weight=self.adaptive_weights['technical'],
            contribution=technical_score * self.adaptive_weights['technical'],
            metadata={
                'rsi': technical_indicators.get('rsi'),
                'macd': technical_indicators.get('macd'),
                'bollinger_position': technical_indicators.get('bollinger_position'),
                'adx': technical_indicators.get('adx'),
                'factor_scores': score_factors
            }
        )

    async def _calculate_risk_score(self, fusion_input: Dict[str, Any]) -> ScoreComponent:
        """计算风险上下文评分"""
        risk_context = fusion_input.get('risk_context', {})
        
        score_factors = []
        
        # 波动率评分
        volatility = risk_context.get('volatility')
        if volatility is not None:
            # 适中的波动率最好，过高或过低都不理想
            if 0.1 <= volatility <= 0.3:
                volatility_score = 0.8
            elif volatility < 0.1:
                volatility_score = 0.6  # 波动率太低，机会较少
            else:
                volatility_score = 0.4  # 波动率太高，风险较大
            score_factors.append(volatility_score)
        
        # VaR评分
        var_95 = risk_context.get('var_95')
        if var_95 is not None:
            # VaR绝对值较小表示风险较低
            var_score = min(1.0, 1.0 / (abs(var_95) + 0.01))
            score_factors.append(var_score)
        
        # 计算平均风险评分
        if score_factors:
            risk_score = np.mean(score_factors)
        else:
            risk_score = 0.5
        
        # 如果配置了波动率调整，进行额外调整
        if self.config.volatility_adjustment and volatility is not None:
            # 根据市场波动率调整风险评分权重
            vol_adjustment = 1.0
            if volatility > 0.5:  # 高波动环境
                vol_adjustment = 1.2
            elif volatility < 0.05:  # 低波动环境
                vol_adjustment = 0.8
            
            risk_score = min(1.0, risk_score * vol_adjustment)
        
        return ScoreComponent(
            name="risk_score",
            value=risk_score,
            weight=self.adaptive_weights['risk'],
            contribution=risk_score * self.adaptive_weights['risk'],
            metadata={
                'volatility': risk_context.get('volatility'),
                'var_95': risk_context.get('var_95'),
                'vol_adjustment': vol_adjustment if self.config.volatility_adjustment else 1.0
            }
        )

    async def _fuse_scores(self, components: List[ScoreComponent]) -> float:
        """融合评分"""
        if self.config.strategy == FusionStrategy.EQUAL_WEIGHT:
            return await self._equal_weight_fusion(components)
        elif self.config.strategy == FusionStrategy.ADAPTIVE_WEIGHT:
            return await self._adaptive_weight_fusion(components)
        elif self.config.strategy == FusionStrategy.PERFORMANCE_BASED:
            return await self._performance_based_fusion(components)
        elif self.config.strategy == FusionStrategy.CONFIDENCE_WEIGHTED:
            return await self._confidence_weighted_fusion(components)
        else:
            return await self._equal_weight_fusion(components)

    async def _equal_weight_fusion(self, components: List[ScoreComponent]) -> float:
        """等权重融合"""
        if not components:
            return 0.0
        
        return sum(comp.contribution for comp in components)

    async def _adaptive_weight_fusion(self, components: List[ScoreComponent]) -> float:
        """自适应权重融合"""
        # 根据组件质量动态调整权重
        adapted_score = 0.0
        
        for component in components:
            # 质量因子：组件值越高，权重越大
            quality_factor = component.value
            
            # 自适应权重
            adaptive_weight = component.weight * (0.5 + 0.5 * quality_factor)
            
            contribution = component.value * adaptive_weight
            adapted_score += contribution
        
        return min(1.0, adapted_score)

    async def _performance_based_fusion(self, components: List[ScoreComponent]) -> float:
        """基于历史性能的融合"""
        if not self.performance_history:
            return await self._equal_weight_fusion(components)
        
        # 计算各组件的历史表现
        performance_weights = {}
        
        for component in components:
            # 获取该组件的历史表现
            history_key = component.name
            recent_performance = self.performance_history[-self.config.performance_history_window:]
            
            if recent_performance:
                # 计算该组件对最终结果的影响
                successful_outcomes = [
                    1 for outcome in recent_performance 
                    if outcome.get('component_scores', {}).get(history_key, 0) > 0.6
                ]
                success_rate = len(successful_outcomes) / len(recent_performance)
                
                # 性能权重
                performance_weights[history_key] = success_rate
            else:
                performance_weights[history_key] = 0.5
        
        # 应用性能权重
        adapted_score = 0.0
        for component in components:
            performance_factor = performance_weights.get(component.name, 0.5)
            adapted_weight = component.weight * (0.5 + 0.5 * performance_factor)
            contribution = component.value * adapted_weight
            adapted_score += contribution
        
        return min(1.0, adapted_score)

    async def _confidence_weighted_fusion(self, components: List[ScoreComponent]) -> float:
        """置信度加权融合"""
        # 根据组件的置信度加权
        total_confidence_weight = 0.0
        weighted_score = 0.0
        
        for component in components:
            # 从元数据中提取置信度
            confidence = component.metadata.get('confidence', component.value)
            
            # 置信度权重
            confidence_weight = component.weight * confidence
            
            weighted_score += component.value * confidence_weight
            total_confidence_weight += confidence_weight
        
        if total_confidence_weight > 0:
            return min(1.0, weighted_score / total_confidence_weight)
        else:
            return await self._equal_weight_fusion(components)

    def _apply_thresholds(self, score: float) -> float:
        """应用阈值过滤"""
        # 最低评分阈值
        if score < self.config.minimum_score_threshold:
            return score * 0.5  # 降低低评分信号
        
        # 置信度阈值过滤已在组件级别处理
        
        return score

    async def _check_technical_consistency(self, fusion_input: Dict[str, Any], layer1_result: Dict[str, Any]) -> float:
        """检查技术指标一致性"""
        layer1_label = layer1_result.get('label', 0)
        technical_indicators = fusion_input.get('technical_indicators', {})
        
        consistency_score = 0.5  # 基准分
        
        if layer1_label == 1:  # 买入信号
            # 检查RSI是否支持买入
            rsi = technical_indicators.get('rsi')
            if rsi is not None and rsi < 70:
                consistency_score += 0.3
            
            # 检查布林带是否支持买入
            bb_position = technical_indicators.get('bollinger_position')
            if bb_position is not None and bb_position < 0.5:
                consistency_score += 0.2
        
        elif layer1_label == -1:  # 卖出信号
            # 检查RSI是否支持卖出
            rsi = technical_indicators.get('rsi')
            if rsi is not None and rsi > 30:
                consistency_score += 0.3
            
            # 检查布林带是否支持卖出
            bb_position = technical_indicators.get('bollinger_position')
            if bb_position is not None and bb_position > 0.5:
                consistency_score += 0.2
        
        return min(1.0, consistency_score)

    async def _check_direction_consistency(self, fusion_input: Dict[str, Any]) -> float:
        """检查方向一致性"""
        layer1_label = fusion_input.get('layer1_result', {}).get('label', 0)
        layer2_direction = fusion_input.get('layer2_result', {}).get('direction', 'hold')
        
        if layer1_label == 0 or layer2_direction == 'hold':
            return 0.5
        
        # 检查方向是否一致
        layer1_direction = 'long' if layer1_label == 1 else 'short'
        
        if layer1_direction == layer2_direction:
            return 1.0
        else:
            return 0.3  # 方向不一致给予较低分数

    async def _update_performance_history(self, fusion_input: Dict[str, Any], final_score: float) -> None:
        """更新性能历史"""
        component_scores = {}
        for component in fusion_input.get('score_components', []):
            component_scores[component.name] = component.value
        
        # 记录性能数据
        performance_record = {
            'timestamp': time.time(),
            'final_score': final_score,
            'component_scores': component_scores,
            'symbol': fusion_input.get('symbol'),
            'timeframe': fusion_input.get('timeframe')
        }
        
        self.performance_history.append(performance_record)
        
        # 保持历史记录窗口大小
        if len(self.performance_history) > self.config.performance_history_window * 2:
            self.performance_history = self.performance_history[-self.config.performance_history_window:]
        
        # 动态调整权重
        await self._adapt_weights()

    async def _adapt_weights(self) -> None:
        """动态调整权重"""
        if not self.performance_history:
            return
        
        recent_history = self.performance_history[-self.config.performance_history_window:]
        
        # 计算各组件的表现
        component_performance = {}
        for component_name in ['layer1_score', 'layer2_score', 'technical_score', 'risk_score']:
            scores = [
                record['component_scores'].get(component_name, 0.5)
                for record in recent_history
            ]
            component_performance[component_name] = np.mean(scores)
        
        # 调整权重
        if self.config.strategy == FusionStrategy.ADAPTIVE_WEIGHT:
            for component_name, performance in component_performance.items():
                if component_name == 'layer1_score':
                    self.adaptive_weights['layer1'] *= (1 + self.config.adaptation_rate * (performance - 0.5))
                elif component_name == 'layer2_score':
                    self.adaptive_weights['layer2'] *= (1 + self.config.adaptation_rate * (performance - 0.5))
                elif component_name == 'technical_score':
                    self.adaptive_weights['technical'] *= (1 + self.config.adaptation_rate * (performance - 0.5))
                elif component_name == 'risk_score':
                    self.adaptive_weights['risk'] *= (1 + self.config.adaptation_rate * (performance - 0.5))
            
            # 权重归一化
            total_weight = sum(self.adaptive_weights.values())
            for key in self.adaptive_weights:
                self.adaptive_weights[key] /= total_weight

    async def _load_performance_history(self) -> None:
        """加载性能历史（在实际实现中可能从数据库加载）"""
        # 模拟加载历史数据
        self.performance_history = []

    def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计"""
        return {
            'strategy': self.config.strategy.value,
            'adaptive_weights': self.adaptive_weights,
            'performance_history_count': len(self.performance_history),
            'config': {
                'risk_reward_threshold': self.config.risk_reward_threshold,
                'confidence_threshold': self.config.confidence_threshold,
                'minimum_score_threshold': self.config.minimum_score_threshold
            }
        }

    async def shutdown(self) -> None:
        """关闭融合器"""
        logger.info("正在关闭验证结果融合器...")
        self.is_initialized = False
        logger.info("验证结果融合器已关闭")