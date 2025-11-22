"""
指标计算工具类
提供各种评分和指标的计算功能
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoreWeights:
    """评分权重配置"""
    technical: float = 0.4
    sentiment: float = 0.2
    liquidity: float = 0.2
    momentum: float = 0.1
    risk: float = 0.1


@dataclass
class RiskMetrics:
    """风险指标"""
    volatility: float
    var_95: float  # Value at Risk 95%
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    correlation_risk: float


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, default_weights: Optional[ScoreWeights] = None):
        """
        初始化指标计算器
        
        Args:
            default_weights: 默认权重配置
        """
        self.default_weights = default_weights or ScoreWeights()
        
    def calculate_comprehensive_score(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        patterns: Optional[Dict[str, Any]] = None,
        volume_profile: Optional[Dict[str, Any]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
        weights: Optional[ScoreWeights] = None
    ) -> Dict[str, float]:
        """
        计算综合评分
        
        Args:
            market_data: 市场数据
            technical_indicators: 技术指标
            patterns: 价格模式分析
            volume_profile: 成交量分析
            sentiment: 市场情绪
            weights: 权重配置
            
        Returns:
            综合评分字典
        """
        try:
            weights = weights or self.default_weights
            
            # 计算各维度评分
            technical_score = self._calculate_technical_score(technical_indicators)
            sentiment_score = self._calculate_sentiment_score(sentiment, market_data)
            liquidity_score = self._calculate_liquidity_score(market_data)
            momentum_score = self._calculate_momentum_score(technical_indicators, market_data)
            risk_score = self._calculate_risk_score(market_data, technical_indicators)
            
            # 计算综合评分
            combined_score = (
                technical_score * weights.technical +
                sentiment_score * weights.sentiment +
                liquidity_score * weights.liquidity +
                momentum_score * weights.momentum +
                risk_score * weights.risk
            )
            
            # 计算置信度
            confidence = self._calculate_confidence([
                technical_score, sentiment_score, liquidity_score, 
                momentum_score, risk_score
            ])
            
            return {
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'liquidity_score': liquidity_score,
                'momentum_score': momentum_score,
                'risk_score': risk_score,
                'combined_score': combined_score,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive score: {e}")
            return self._get_default_scores()
    
    def _calculate_technical_score(self, indicators: Dict[str, Any]) -> float:
        """计算技术分析评分"""
        try:
            scores = []
            
            # RSI评分 (30-70为最佳)
            rsi = indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                rsi_score = 1.0 - abs(rsi - 50) / 50
            else:
                rsi_score = max(0.0, 0.5 - abs(rsi - 50) / 100)
            scores.append(rsi_score * 0.2)  # 权重20%
            
            # MACD评分
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd_signal != 0:
                macd_score = max(0.0, (macd - macd_signal) / abs(macd_signal))
                scores.append(min(1.0, macd_score) * 0.2)  # 权重20%
            
            # 布林带评分
            bollinger_position = indicators.get('bollinger_position', 0.5)
            # 中间位置最好，两端稍差
            bollinger_score = 1.0 - abs(bollinger_position - 0.5) * 2
            scores.append(max(0.0, bollinger_score) * 0.15)  # 权重15%
            
            # 趋势评分
            trend_strength = indicators.get('price_trend', 0)
            trend_score = abs(trend_strength)
            scores.append(min(1.0, trend_score) * 0.15)  # 权重15%
            
            # 移动平均线评分
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            if sma_20 > 0 and sma_50 > 0:
                ma_score = 1.0 if sma_20 > sma_50 else 0.0
                scores.append(ma_score * 0.1)  # 权重10%
            
            # ADX评分 (趋势强度)
            adx = indicators.get('adx', 25)
            if adx > 25:
                adx_score = min(1.0, (adx - 25) / 25)
                scores.append(adx_score * 0.1)  # 权重10%
            
            # 随机指标评分
            stochastic_k = indicators.get('stochastic_k', 50)
            if 20 <= stochastic_k <= 80:
                stochastic_score = 1.0 - abs(stochastic_k - 50) / 50
            else:
                stochastic_score = max(0.0, 0.5 - abs(stochastic_k - 50) / 100)
            scores.append(stochastic_score * 0.1)  # 权重10%
            
            return min(1.0, sum(scores))
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5
    
    def _calculate_sentiment_score(self, sentiment: Optional[Dict[str, Any]], market_data: Dict[str, Any]) -> float:
        """计算情绪评分"""
        try:
            scores = []
            
            # 基于市场情绪
            if sentiment:
                sentiment_score = sentiment.get('sentiment_score', 0.5)
                scores.append(sentiment_score * 0.4)  # 权重40%
            
            # 基于价格变化
            price_change = market_data.get('price_change_24h', 0)
            if price_change > 0:
                # 上涨情绪
                change_score = min(1.0, price_change / 10.0)  # 10%为基准
            else:
                # 下跌情绪惩罚较小
                change_score = max(0.0, 1.0 + price_change / 20.0)  # 20%下跌为基准
            scores.append(change_score * 0.3)  # 权重30%
            
            # 基于成交量趋势
            volume_trend = market_data.get('volume_trend', 0)
            if volume_trend > 0:
                volume_score = min(1.0, volume_trend)
            else:
                volume_score = max(0.0, 0.5 + volume_trend)
            scores.append(volume_score * 0.3)  # 权重30%
            
            return min(1.0, sum(scores))
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return 0.5
    
    def _calculate_liquidity_score(self, market_data: Dict[str, Any]) -> float:
        """计算流动性评分"""
        try:
            scores = []
            
            # 成交量评分
            volume_24h = market_data.get('volume_24h', 0)
            volume_score = min(1.0, volume_24h / 5000000)  # 500万作为基准
            scores.append(volume_score * 0.4)  # 权重40%
            
            # 买卖价差评分
            spread = market_data.get('bid_ask_spread', 0.001)
            spread_score = 1.0 - min(1.0, spread / 0.005)  # 0.5%作为基准
            scores.append(spread_score * 0.3)  # 权重30%
            
            # 订单簿深度评分
            depth = market_data.get('order_book_depth', 0)
            depth_score = min(1.0, depth / 1000000)  # 100万作为基准
            scores.append(depth_score * 0.2)  # 权重20%
            
            # 市场市值评分
            market_cap = market_data.get('market_cap', 0)
            cap_score = min(1.0, market_cap / 100000000)  # 1亿作为基准
            scores.append(cap_score * 0.1)  # 权重10%
            
            return min(1.0, sum(scores))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    def _calculate_momentum_score(self, indicators: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """计算动量评分"""
        try:
            scores = []
            
            # 动量指标
            momentum = indicators.get('momentum', 0)
            if momentum > 0:
                momentum_score = min(1.0, momentum / (market_data.get('price', 1000) * 0.05))  # 5%动量为基准
            else:
                momentum_score = max(0.0, 1.0 + momentum / (market_data.get('price', 1000) * 0.1))  # 10%下跌为基准
            scores.append(momentum_score * 0.3)  # 权重30%
            
            # ROC (变化率)
            roc = indicators.get('roc', 0)
            roc_score = min(1.0, abs(roc) / 20.0)  # 20%为基准
            scores.append(roc_score * 0.25)  # 权重25%
            
            # 价格变化强度
            price_change = market_data.get('price_change_24h', 0)
            if abs(price_change) > 0:
                change_score = min(1.0, abs(price_change) / 15.0)  # 15%为基准
            else:
                change_score = 0.5
            scores.append(change_score * 0.25)  # 权重25%
            
            # Williams %R
            williams_r = indicators.get('williams_r', -50)
            if williams_r > -20:  # 超买区域
                williams_score = (williams_r + 100) / 80  # 转换为0-1
            elif williams_r < -80:  # 超卖区域
                williams_score = (williams_r + 100) / 80
            else:
                williams_score = 0.5
            scores.append(williams_score * 0.2)  # 权重20%
            
            return min(1.0, sum(scores))
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.5
    
    def _calculate_risk_score(self, market_data: Dict[str, Any], indicators: Dict[str, Any]) -> float:
        """计算风险评分"""
        try:
            scores = []
            
            # 波动率评分 (低波动率为好)
            volatility = indicators.get('volatility', 0.1)
            if volatility < 0.02:
                vol_score = 1.0
            elif volatility > 0.2:
                vol_score = 0.2
            else:
                vol_score = 1.0 - (volatility - 0.02) / 0.18
            scores.append(vol_score * 0.4)  # 权重40%
            
            # 价格稳定性
            price_change = abs(market_data.get('price_change_24h', 0))
            if price_change < 2:
                stability_score = 1.0
            elif price_change > 10:
                stability_score = 0.2
            else:
                stability_score = 1.0 - (price_change - 2) / 8
            scores.append(stability_score * 0.3)  # 权重30%
            
            # 交易量稳定性
            volume_trend = abs(market_data.get('volume_trend', 0))
            if volume_trend < 0.2:
                volume_stability_score = 1.0
            else:
                volume_stability_score = max(0.2, 1.0 - volume_trend)
            scores.append(volume_stability_score * 0.2)  # 权重20%
            
            # 技术指标一致性
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            # 简单的指标一致性检查
            consistency_score = 0.5  # 默认分数
            
            # RSI和MACD一致性检查
            if (rsi > 50 and macd > macd_signal) or (rsi < 50 and macd < macd_signal):
                consistency_score = 0.8
            elif abs(rsi - 50) < 10 and abs(macd - macd_signal) < 0.001:
                consistency_score = 0.6  # 中性
            
            scores.append(consistency_score * 0.1)  # 权重10%
            
            return min(1.0, sum(scores))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """计算置信度"""
        try:
            if not scores:
                return 0.5
            
            # 计算分数的方差和一致性
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # 置信度 = 1 - 标准化标准差
            # 标准差越小，说明各项指标越一致，置信度越高
            confidence = 1.0 - min(1.0, std_score / 0.5)  # 0.5作为标准差基准
            
            # 如果平均分数太低，惩罚置信度
            if mean_score < 0.3:
                confidence *= 0.5
            elif mean_score < 0.5:
                confidence *= 0.8
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def calculate_risk_metrics(self, market_data: Dict[str, Any], historical_prices: Optional[List[float]] = None) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            market_data: 市场数据
            historical_prices: 历史价格列表
            
        Returns:
            风险指标
        """
        try:
            # 如果没有历史价格，使用当前数据计算
            if not historical_prices:
                # 简化的风险计算
                current_price = market_data.get('price', 1000)
                volatility = market_data.get('volatility', 0.1)
                
                risk_metrics = RiskMetrics(
                    volatility=volatility,
                    var_95=current_price * volatility * 1.65,  # 95% VaR
                    sharpe_ratio=0.5,  # 默认值
                    max_drawdown=0.1,  # 默认值
                    beta=1.0,  # 默认值
                    correlation_risk=0.5  # 默认值
                )
                
                return risk_metrics
            
            # 完整的历史风险计算
            returns = np.diff(historical_prices) / historical_prices[:-1]
            
            # 计算各种风险指标
            volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
            
            # VaR计算
            var_95 = np.percentile(returns, 5) * historical_prices[-1]
            
            # 夏普比率 (简化)
            excess_returns = returns - 0.02 / 252  # 假设2%年化无风险利率
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # 最大回撤
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = abs(np.min(drawdown))
            
            # Beta系数 (相对于市场，简化)
            beta = 1.0  # 默认值，实际需要市场数据
            
            # 相关性风险
            correlation_risk = min(1.0, np.std(returns) * 10)  # 简化计算
            
            return RiskMetrics(
                volatility=volatility,
                var_95=abs(var_95),
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                beta=beta,
                correlation_risk=correlation_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0.1, 50.0, 0.5, 0.1, 1.0, 0.5)
    
    def calculate_opportunity_score(self, score_data: Dict[str, float]) -> float:
        """
        计算机会评分
        
        Args:
            score_data: 评分数据
            
        Returns:
            机会评分
        """
        try:
            # 机会评分 = 综合评分 * 置信度 * 调整因子
            
            combined_score = score_data.get('combined_score', 0.5)
            confidence = score_data.get('confidence', 0.5)
            liquidity_score = score_data.get('liquidity_score', 0.5)
            
            # 基础机会评分
            base_score = combined_score * confidence
            
            # 流动性调整
            liquidity_adjustment = liquidity_score
            
            # 技术评分调整
            technical_score = score_data.get('technical_score', 0.5)
            
            # 综合机会评分
            opportunity_score = base_score * (0.7 + 0.3 * liquidity_score) * technical_score
            
            # 确保评分在合理范围内
            opportunity_score = max(0.0, min(1.0, opportunity_score))
            
            return opportunity_score
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return 0.5
    
    def calculate_portfolio_metrics(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算投资组合指标
        
        Args:
            positions: 持仓列表
            
        Returns:
            投资组合指标
        """
        try:
            if not positions:
                return {
                    'total_value': 0.0,
                    'total_risk': 0.0,
                    'expected_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'diversification_score': 0.0
                }
            
            # 计算总价值
            total_value = sum(pos.get('value', 0) for pos in positions)
            
            # 计算总风险
            total_risk = sum(pos.get('risk_score', 0.5) * pos.get('value', 0) for pos in positions) / total_value if total_value > 0 else 0
            
            # 计算预期收益
            expected_return = sum(pos.get('expected_return', 0) * pos.get('value', 0) for pos in positions) / total_value if total_value > 0 else 0
            
            # 计算夏普比率
            risk_adjusted_return = expected_return - 0.02  # 假设2%无风险利率
            sharpe_ratio = risk_adjusted_return / total_risk if total_risk > 0 else 0
            
            # 计算最大回撤
            max_drawdown = max(pos.get('max_drawdown', 0) for pos in positions)
            
            # 计算分散化评分
            symbols = [pos.get('symbol', '') for pos in positions]
            unique_symbols = len(set(symbols))
            diversification_score = min(1.0, unique_symbols / len(symbols)) if positions else 0
            
            return {
                'total_value': total_value,
                'total_risk': total_risk,
                'expected_return': expected_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'diversification_score': diversification_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _get_default_scores(self) -> Dict[str, float]:
        """获取默认评分"""
        return {
            'technical_score': 0.5,
            'sentiment_score': 0.5,
            'liquidity_score': 0.5,
            'momentum_score': 0.5,
            'risk_score': 0.5,
            'combined_score': 0.5,
            'confidence': 0.5
        }
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """标准化评分"""
        if max_val <= min_val:
            return 0.5
        
        normalized = (score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def calculate_score_percentile(self, score: float, score_distribution: List[float]) -> float:
        """计算评分百分位数"""
        try:
            if not score_distribution:
                return 50.0
            
            sorted_scores = sorted(score_distribution)
            position = sum(1 for s in sorted_scores if s <= score)
            percentile = (position / len(sorted_scores)) * 100
            
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating score percentile: {e}")
            return 50.0


# 示例使用
if __name__ == "__main__":
    async def test_metrics_calculator():
        calculator = MetricsCalculator()
        
        # 测试综合评分计算
        market_data = {
            'price': 50000,
            'volume_24h': 10000000,
            'price_change_24h': 2.5,
            'bid_ask_spread': 0.0002,
            'order_book_depth': 1000000,
            'market_cap': 900000000,
            'volume_trend': 0.3
        }
        
        technical_indicators = {
            'rsi': 65.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'bollinger_position': 0.7,
            'price_trend': 0.8,
            'sma_20': 49000,
            'sma_50': 48500,
            'adx': 35.0,
            'stochastic_k': 60.0,
            'momentum': 1000,
            'roc': 3.5,
            'williams_r': -30,
            'volatility': 0.15
        }
        
        patterns = {
            'is_bullish': True,
            'pattern_type': 'bullish_momentum',
            'breakout_probability': 0.7
        }
        
        volume_profile = {
            'volume_trend': 'increasing',
            'volume_strength': 0.8,
            'volume_surge': True
        }
        
        sentiment = {
            'sentiment_score': 0.75,
            'sentiment_label': 'bullish',
            'fear_greed_index': 65
        }
        
        # 计算综合评分
        scores = calculator.calculate_comprehensive_score(
            market_data, technical_indicators, patterns, volume_profile, sentiment
        )
        
        print("Comprehensive scores:")
        for key, value in scores.items():
            print(f"  {key}: {value:.3f}")
        
        # 计算风险指标
        risk_metrics = calculator.calculate_risk_metrics(market_data)
        print(f"\nRisk metrics:")
        print(f"  volatility: {risk_metrics.volatility:.3f}")
        print(f"  var_95: {risk_metrics.var_95:.2f}")
        print(f"  sharpe_ratio: {risk_metrics.sharpe_ratio:.3f}")
        print(f"  max_drawdown: {risk_metrics.max_drawdown:.3f}")
        
        # 计算机会评分
        opportunity_score = calculator.calculate_opportunity_score(scores)
        print(f"\nOpportunity score: {opportunity_score:.3f}")
    
    # 运行测试
    import asyncio
    asyncio.run(test_metrics_calculator())