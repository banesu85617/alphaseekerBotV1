"""
信号处理器模块

负责处理和验证各种类型的策略信号，包括：
- 技术指标触发信号
- 机器学习预测信号
- 风险模型信号
- 回测参考信号
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .types import (
    MarketData, TechnicalIndicators, RiskMetrics, MLFeatures,
    StrategySignal, StrategyType, SignalDirection, ConfidenceLevel,
    MLPrediction, LLMAssessment, BacktestResult,
    PipelineConfig, DataInsufficientError, ModelUnavailableError
)

logger = logging.getLogger(__name__)

class SignalProcessor:
    """信号处理器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._cached_indicators = {}
        self._cached_ml_predictions = {}
        self._cached_risk_metrics = {}
        
    async def process_technical_signals(
        self, 
        market_data: MarketData,
        technical_indicators: TechnicalIndicators
    ) -> List[StrategySignal]:
        """处理技术指标信号"""
        signals = []
        
        try:
            # RSI信号
            if technical_indicators.rsi is not None:
                rsi_signal = self._generate_rsi_signal(
                    market_data, technical_indicators.rsi
                )
                if rsi_signal:
                    signals.append(rsi_signal)
            
            # MACD信号
            if technical_indicators.macd is not None and technical_indicators.macd_signal is not None:
                macd_signal = self._generate_macd_signal(
                    market_data, technical_indicators.macd, technical_indicators.macd_signal
                )
                if macd_signal:
                    signals.append(macd_signal)
            
            # 布林带信号
            if all(v is not None for v in [
                technical_indicators.bollinger_upper,
                technical_indicators.bollinger_middle,
                technical_indicators.bollinger_lower
            ]):
                bb_signal = self._generate_bollinger_signal(
                    market_data, 
                    technical_indicators.bollinger_upper,
                    technical_indicators.bollinger_middle,
                    technical_indicators.bollinger_lower
                )
                if bb_signal:
                    signals.append(bb_signal)
            
            # 趋势信号 (SMA/EMA)
            trend_signal = self._generate_trend_signal(market_data, technical_indicators)
            if trend_signal:
                signals.append(trend_signal)
                
            # ADX趋势强度信号
            if technical_indicators.adx is not None:
                adx_signal = self._generate_adx_signal(
                    market_data, technical_indicators.adx
                )
                if adx_signal:
                    signals.append(adx_signal)
            
            logger.info(f"生成了 {len(signals)} 个技术指标信号 for {market_data.symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"处理技术指标信号时出错: {e}")
            return []
    
    def _generate_rsi_signal(
        self, 
        market_data: MarketData, 
        rsi: float
    ) -> Optional[StrategySignal]:
        """生成RSI信号"""
        # RSI超卖信号 (RSI < 30)
        if rsi < 30:
            confidence = min(0.9, (30 - rsi) / 30 * 0.9 + 0.1)
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL_INDICATOR,
                direction=SignalDirection.LONG,
                confidence=confidence,
                score=confidence * 0.8,  # 技术指标权重较低
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                market_data=market_data,
                technical_indicators=TechnicalIndicators(rsi=rsi),
                metadata={"indicator": "RSI", "value": rsi, "condition": "oversold"}
            )
        
        # RSI超买信号 (RSI > 70)
        elif rsi > 70:
            confidence = min(0.9, (rsi - 70) / 30 * 0.9 + 0.1)
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL_INDICATOR,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                score=confidence * 0.8,
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                market_data=market_data,
                technical_indicators=TechnicalIndicators(rsi=rsi),
                metadata={"indicator": "RSI", "value": rsi, "condition": "overbought"}
            )
        
        return None
    
    def _generate_macd_signal(
        self,
        market_data: MarketData,
        macd: float,
        macd_signal: float
    ) -> Optional[StrategySignal]:
        """生成MACD信号"""
        # MACD金叉信号
        if macd > macd_signal and macd > 0:
            confidence = min(0.85, abs(macd - macd_signal) / abs(macd_signal) * 0.85)
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL_INDICATOR,
                direction=SignalDirection.LONG,
                confidence=confidence,
                score=confidence * 0.8,
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                market_data=market_data,
                technical_indicators=TechnicalIndicators(macd=macd, macd_signal=macd_signal),
                metadata={"indicator": "MACD", "condition": "bullish_cross"}
            )
        
        # MACD死叉信号
        elif macd < macd_signal and macd < 0:
            confidence = min(0.85, abs(macd - macd_signal) / abs(macd_signal) * 0.85)
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL_INDICATOR,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                score=confidence * 0.8,
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                market_data=market_data,
                technical_indicators=TechnicalIndicators(macd=macd, macd_signal=macd_signal),
                metadata={"indicator": "MACD", "condition": "bearish_cross"}
            )
        
        return None
    
    def _generate_bollinger_signal(
        self,
        market_data: MarketData,
        bb_upper: float,
        bb_middle: float,
        bb_lower: float
    ) -> Optional[StrategySignal]:
        """生成布林带信号"""
        current_price = market_data.price
        
        # 价格触及下轨，可能反弹
        if current_price <= bb_lower:
            distance_to_lower = (bb_lower - current_price) / bb_lower
            confidence = min(0.8, distance_to_lower * 10 + 0.2)
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL_INDICATOR,
                direction=SignalDirection.LONG,
                confidence=confidence,
                score=confidence * 0.7,  # 布林带信号权重较低
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                market_data=market_data,
                technical_indicators=TechnicalIndicators(
                    bollinger_upper=bb_upper,
                    bollinger_middle=bb_middle,
                    bollinger_lower=bb_lower
                ),
                metadata={"indicator": "Bollinger", "condition": "touch_lower"}
            )
        
        # 价格触及上轨，可能回调
        elif current_price >= bb_upper:
            distance_to_upper = (current_price - bb_upper) / bb_upper
            confidence = min(0.8, distance_to_upper * 10 + 0.2)
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL_INDICATOR,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                score=confidence * 0.7,
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                market_data=market_data,
                technical_indicators=TechnicalIndicators(
                    bollinger_upper=bb_upper,
                    bollinger_middle=bb_middle,
                    bollinger_lower=bb_lower
                ),
                metadata={"indicator": "Bollinger", "condition": "touch_upper"}
            )
        
        return None
    
    def _generate_trend_signal(
        self,
        market_data: MarketData,
        indicators: TechnicalIndicators
    ) -> Optional[StrategySignal]:
        """生成趋势信号"""
        # 检查SMA排列
        if all(v is not None for v in [indicators.sma_50, indicators.sma_200]):
            sma_50 = indicators.sma_50
            sma_200 = indicators.sma_200
            current_price = market_data.price
            
            # 多头排列 (价格 > SMA50 > SMA200)
            if current_price > sma_50 > sma_200:
                trend_strength = (sma_50 - sma_200) / sma_200
                confidence = min(0.9, trend_strength * 20 + 0.3)
                return StrategySignal(
                    strategy_type=StrategyType.TECHNICAL_INDICATOR,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    score=confidence * 0.9,  # 趋势信号权重较高
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    market_data=market_data,
                    technical_indicators=indicators,
                    metadata={"indicator": "SMA_Trend", "condition": "bullish_alignment"}
                )
            
            # 空头排列 (价格 < SMA50 < SMA200)
            elif current_price < sma_50 < sma_200:
                trend_strength = (sma_200 - sma_50) / sma_200
                confidence = min(0.9, trend_strength * 20 + 0.3)
                return StrategySignal(
                    strategy_type=StrategyType.TECHNICAL_INDICATOR,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    score=confidence * 0.9,
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    market_data=market_data,
                    technical_indicators=indicators,
                    metadata={"indicator": "SMA_Trend", "condition": "bearish_alignment"}
                )
        
        return None
    
    def _generate_adx_signal(
        self,
        market_data: MarketData,
        adx: float
    ) -> Optional[StrategySignal]:
        """生成ADX趋势强度信号"""
        # ADX > 25 表示趋势较强
        if adx > self.config.min_adx_threshold:
            strength = min(0.8, (adx - 25) / 25 * 0.8)
            return StrategySignal(
                strategy_type=StrategyType.TECHNICAL_INDICATOR,
                direction=SignalDirection.HOLD,  # ADX只表示强度，不表示方向
                confidence=strength,
                score=strength * 0.6,  # ADX信号权重较低
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                market_data=market_data,
                technical_indicators=TechnicalIndicators(adx=adx),
                metadata={"indicator": "ADX", "value": adx, "condition": "strong_trend"}
            )
        
        return None
    
    async def process_ml_signals(
        self,
        market_data: MarketData,
        ml_features: MLFeatures,
        ml_prediction: MLPrediction
    ) -> List[StrategySignal]:
        """处理机器学习预测信号"""
        signals = []
        
        try:
            # 检查概率阈值
            if ml_prediction.confidence < self.config.ml_confidence_threshold:
                logger.debug(f"ML预测置信度 {ml_prediction.confidence} 低于阈值")
                return signals
            
            # 生成ML预测信号
            if ml_prediction.label == 1:  # 买入
                confidence = ml_prediction.probability_scores.get(1, 0.0)
                signal = StrategySignal(
                    strategy_type=StrategyType.ML_PREDICTION,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    score=confidence * 0.95,  # ML信号权重很高
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    market_data=market_data,
                    ml_prediction=ml_prediction,
                    metadata={
                        "model_version": ml_prediction.model_version,
                        "prediction_type": "lightgbm_classification"
                    }
                )
                signals.append(signal)
                
            elif ml_prediction.label == -1:  # 卖出
                confidence = ml_prediction.probability_scores.get(-1, 0.0)
                signal = StrategySignal(
                    strategy_type=StrategyType.ML_PREDICTION,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    score=confidence * 0.95,
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    market_data=market_data,
                    ml_prediction=ml_prediction,
                    metadata={
                        "model_version": ml_prediction.model_version,
                        "prediction_type": "lightgbm_classification"
                    }
                )
                signals.append(signal)
            
            logger.info(f"生成了 {len(signals)} 个ML预测信号 for {market_data.symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"处理ML预测信号时出错: {e}")
            return []
    
    async def process_risk_signals(
        self,
        market_data: MarketData,
        risk_metrics: RiskMetrics
    ) -> List[StrategySignal]:
        """处理风险模型信号"""
        signals = []
        
        try:
            # 波动率风险信号
            if risk_metrics.garch_volatility is not None:
                # 如果波动率过高，倾向于保守
                vol = risk_metrics.garch_volatility
                if vol > 0.05:  # 5%以上波动率认为高风险
                    confidence = min(0.7, (vol - 0.05) * 10 + 0.3)
                    signals.append(StrategySignal(
                        strategy_type=StrategyType.RISK_MODEL,
                        direction=SignalDirection.HOLD,
                        confidence=confidence,
                        score=confidence * 0.6,
                        timestamp=datetime.now(),
                        symbol=market_data.symbol,
                        market_data=market_data,
                        risk_metrics=risk_metrics,
                        metadata={"risk_type": "volatility", "value": vol}
                    ))
            
            # VaR风险信号
            if risk_metrics.var_95 is not None:
                var_95 = risk_metrics.var_95
                if var_95 > 0.03:  # 3%以上VaR认为高风险
                    confidence = min(0.8, (var_95 - 0.03) * 20 + 0.2)
                    signals.append(StrategySignal(
                        strategy_type=StrategyType.RISK_MODEL,
                        direction=SignalDirection.HOLD,
                        confidence=confidence,
                        score=confidence * 0.7,
                        timestamp=datetime.now(),
                        symbol=market_data.symbol,
                        market_data=market_data,
                        risk_metrics=risk_metrics,
                        metadata={"risk_type": "var", "value": var_95}
                    ))
            
            # 最大回撤信号
            if risk_metrics.max_drawdown is not None:
                max_dd = abs(risk_metrics.max_drawdown)
                if max_dd > 0.15:  # 15%以上回撤认为高风险
                    confidence = min(0.9, (max_dd - 0.15) * 5 + 0.4)
                    signals.append(StrategySignal(
                        strategy_type=StrategyType.RISK_MODEL,
                        direction=SignalDirection.HOLD,
                        confidence=confidence,
                        score=confidence * 0.8,
                        timestamp=datetime.now(),
                        symbol=market_data.symbol,
                        market_data=market_data,
                        risk_metrics=risk_metrics,
                        metadata={"risk_type": "drawdown", "value": max_dd}
                    ))
            
            logger.info(f"生成了 {len(signals)} 个风险模型信号 for {market_data.symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"处理风险模型信号时出错: {e}")
            return []
    
    async def process_backtest_signals(
        self,
        market_data: MarketData,
        backtest_result: BacktestResult
    ) -> List[StrategySignal]:
        """处理回测参考信号"""
        signals = []
        
        try:
            # 检查回测分数阈值
            if backtest_result.score < self.config.min_backtest_score:
                logger.debug(f"回测分数 {backtest_result.score} 低于阈值")
                return signals
            
            # 根据回测结果生成信号
            if backtest_result.total_return > 0.1:  # 10%以上收益
                confidence = min(0.8, backtest_result.score * 0.8)
                direction = SignalDirection.LONG if backtest_result.win_rate > 0.5 else SignalDirection.HOLD
                
                signal = StrategySignal(
                    strategy_type=StrategyType.BACKTEST_REFERENCE,
                    direction=direction,
                    confidence=confidence,
                    score=confidence * 0.75,  # 回测信号权重中等
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    market_data=market_data,
                    backtest_result=backtest_result,
                    metadata={
                        "backtest_period": backtest_result.backtest_period,
                        "strategy_name": backtest_result.strategy_name,
                        "win_rate": backtest_result.win_rate
                    }
                )
                signals.append(signal)
            
            logger.info(f"生成了 {len(signals)} 个回测参考信号 for {market_data.symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"处理回测参考信号时出错: {e}")
            return []
    
    async def validate_signal(self, signal: StrategySignal) -> bool:
        """验证信号有效性"""
        try:
            # 检查必要字段
            if not all([signal.symbol, signal.direction, signal.market_data]):
                return False
            
            # 检查置信度范围
            if not (0 <= signal.confidence <= 1):
                return False
            
            # 检查数据新鲜度
            data_age = (datetime.now() - signal.market_data.timestamp).total_seconds()
            if data_age > 300:  # 5分钟以上认为数据过期
                logger.warning(f"信号数据过期: {data_age} 秒")
                return False
            
            # 根据信号类型进行特定验证
            if signal.strategy_type == StrategyType.TECHNICAL_INDICATOR:
                return self._validate_technical_signal(signal)
            elif signal.strategy_type == StrategyType.ML_PREDICTION:
                return self._validate_ml_signal(signal)
            elif signal.strategy_type == StrategyType.RISK_MODEL:
                return self._validate_risk_signal(signal)
            elif signal.strategy_type == StrategyType.BACKTEST_REFERENCE:
                return self._validate_backtest_signal(signal)
            
            return True
            
        except Exception as e:
            logger.error(f"验证信号时出错: {e}")
            return False
    
    def _validate_technical_signal(self, signal: StrategySignal) -> bool:
        """验证技术指标信号"""
        # 技术指标信号需要有效的指标数据
        if not signal.technical_indicators:
            return False
        
        # 检查具体指标的有效性
        indicators = signal.technical_indicators
        if hasattr(indicators, 'rsi') and indicators.rsi is not None:
            if not (0 <= indicators.rsi <= 100):
                return False
        
        return True
    
    def _validate_ml_signal(self, signal: StrategySignal) -> bool:
        """验证ML信号"""
        # ML信号需要有效的预测结果
        if not signal.ml_prediction:
            return False
        
        ml_pred = signal.ml_prediction
        if ml_pred.label not in [-1, 0, 1]:
            return False
        
        # 检查概率分数总和
        prob_sum = sum(ml_pred.probability_scores.values())
        if abs(prob_sum - 1.0) > 0.01:  # 允许小误差
            return False
        
        return True
    
    def _validate_risk_signal(self, signal: StrategySignal) -> bool:
        """验证风险信号"""
        # 风险信号需要有效的风险指标
        if not signal.risk_metrics:
            return False
        
        return True
    
    def _validate_backtest_signal(self, signal: StrategySignal) -> bool:
        """验证回测信号"""
        # 回测信号需要有效的回测结果
        if not signal.backtest_result:
            return False
        
        bt_result = signal.backtest_result
        if not (0 <= bt_result.score <= 1):
            return False
        
        if not (0 <= bt_result.win_rate <= 1):
            return False
        
        return True
    
    def get_cache_status(self) -> Dict[str, Dict]:
        """获取缓存状态"""
        return {
            "indicators": {"size": len(self._cached_indicators)},
            "ml_predictions": {"size": len(self._cached_ml_predictions)},
            "risk_metrics": {"size": len(self._cached_risk_metrics)}
        }
    
    def clear_cache(self):
        """清理缓存"""
        self._cached_indicators.clear()
        self._cached_ml_predictions.clear()
        self._cached_risk_metrics.clear()
        logger.info("信号处理器缓存已清理")