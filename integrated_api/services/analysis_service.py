"""
分析服务
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from arch import arch_model
from ..core.data_fetcher import data_fetcher, data_cache
from ..core.indicators import TechnicalIndicators
from ..core.exceptions import DataError, IndicatorError, BacktestError
from ..core.models import (
    AnalysisRequest, AnalysisResponse, TechnicalIndicators as IndicatorsModel,
    RiskMetrics, AnalysisText, TradingParams, BacktestResults, TradeDirection
)
from ..services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class AnalysisService:
    """分析服务"""
    
    def __init__(self):
        self.llm_service = get_llm_service()
    
    async def analyze_symbol(self, request: AnalysisRequest) -> AnalysisResponse:
        """分析单个交易对"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting analysis for {request.symbol} - {request.timeframe}")
            
            # 1. 获取数据
            df = await self._get_data_with_cache(request.symbol, request.timeframe, request.lookback)
            if df.empty:
                raise DataError(f"No data available for {request.symbol}")
            
            # 2. 计算技术指标
            df_with_indicators = TechnicalIndicators.apply_all_indicators(df)
            latest_indicators = TechnicalIndicators.get_latest_indicators(df_with_indicators)
            
            # 3. 计算风险指标
            risk_metrics = await self._calculate_risk_metrics(df_with_indicators, request.symbol)
            
            # 4. 技术信号分析
            signal_direction = self._determine_signal_direction(latest_indicators, request.symbol)
            
            # 5. 获取当前价格
            current_price = float(df_with_indicators['close'].iloc[-1])
            
            # 6. LLM分析
            gpt_analysis = None
            gpt_params = None
            
            if self.llm_service.is_available:
                try:
                    llm_result = await self.llm_service.generate_trading_analysis(
                        symbol=request.symbol,
                        current_price=current_price,
                        technical_indicators=latest_indicators,
                        signal_direction=signal_direction,
                        market_context={
                            "timeframe": request.timeframe,
                            "garch_volatility": risk_metrics.garchVolatility,
                            "var95": risk_metrics.var95
                        }
                    )
                    
                    # 转换为模型
                    gpt_analysis = AnalysisText(
                        signal_evaluation=llm_result.get("signal_evaluation"),
                        technical_analysis=llm_result.get("technical_analysis"),
                        risk_assessment=llm_result.get("risk_assessment"),
                        market_outlook=llm_result.get("market_outlook")
                    )
                    
                    gpt_params = TradingParams(
                        optimal_entry=llm_result.get("optimal_entry"),
                        stop_loss=llm_result.get("stop_loss"),
                        take_profit=llm_result.get("take_profit"),
                        leverage=llm_result.get("leverage"),
                        position_size_usd=llm_result.get("position_size_usd"),
                        estimated_profit=llm_result.get("estimated_profit"),
                        confidence_score=llm_result.get("confidence_score", 0.0)
                    )
                    
                except Exception as e:
                    logger.warning(f"LLM analysis failed for {request.symbol}: {e}")
                    gpt_analysis = AnalysisText(signal_evaluation=f"LLM analysis failed: {str(e)}")
                    gpt_params = TradingParams(confidence_score=0.0)
            
            # 7. 回测
            backtest_results = None
            try:
                backtest_results = await self._perform_backtest(
                    df_with_indicators, signal_direction, request
                )
            except Exception as e:
                logger.warning(f"Backtest failed for {request.symbol}: {e}")
                backtest_results = BacktestResults(
                    warnings=[f"Backtest failed: {str(e)}"]
                )
            
            # 8. 构建响应
            indicators_model = IndicatorsModel(**latest_indicators)
            
            processing_time = time.time() - start_time
            
            response = AnalysisResponse(
                symbol=request.symbol,
                timeframe=request.timeframe,
                currentPrice=current_price,
                indicators=indicators_model,
                riskMetrics=risk_metrics,
                gptParams=gpt_params,
                gptAnalysis=gpt_analysis,
                backtest=backtest_results,
                processing_time=processing_time
            )
            
            logger.info(f"Analysis completed for {request.symbol} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Analysis failed for {request.symbol}: {e}")
            
            return AnalysisResponse(
                symbol=request.symbol,
                timeframe=request.timeframe,
                error=str(e),
                processing_time=processing_time
            )
    
    async def _get_data_with_cache(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback: int
    ) -> pd.DataFrame:
        """获取数据（带缓存）"""
        cache_key = f"{symbol}_{timeframe}_{lookback}"
        
        # 尝试从缓存获取
        cached_data = data_cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_data
        
        # 从数据源获取
        df = data_fetcher.get_ohlcv_data(symbol, timeframe, lookback)
        
        # 缓存结果
        if not df.empty:
            data_cache.set(cache_key, df)
            logger.debug(f"Cached data for {cache_key}")
        
        return df
    
    async def _calculate_risk_metrics(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> RiskMetrics:
        """计算风险指标"""
        try:
            risk_metrics = RiskMetrics()
            
            # 获取收益率
            returns = df['returns'].dropna()
            
            if len(returns) >= 50:
                # GARCH波动率
                risk_metrics.garchVolatility = self._fit_garch_model(returns, symbol)
            
            if len(returns) >= 20:
                # VaR
                risk_metrics.var95 = self._calculate_var(returns, 0.95)
            
            return risk_metrics
            
        except Exception as e:
            logger.warning(f"Risk metrics calculation failed for {symbol}: {e}")
            return RiskMetrics()
    
    def _fit_garch_model(self, returns: pd.Series, symbol: str) -> Optional[float]:
        """拟合GARCH模型"""
        try:
            # 放大收益率以提高数值稳定性
            scaled_returns = returns * 100
            
            if len(scaled_returns) < 50:
                return None
            
            # 拟合GARCH(1,1)模型
            model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, dist='Normal')
            
            # 抑制收敛警告
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(update_freq=0, disp='off', show_warning=False)
            
            if result.convergence_flag == 0:
                # 预测下一期波动率
                forecasts = result.forecast(horizon=1, reindex=False)
                forecast_vol = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0  # 还原尺度
                return float(forecast_vol) if np.isfinite(forecast_vol) else None
            else:
                logger.warning(f"GARCH model did not converge for {symbol}")
                return None
                
        except Exception as e:
            logger.warning(f"GARCH fitting failed for {symbol}: {e}")
            return None
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> Optional[float]:
        """计算VaR"""
        try:
            if len(returns) < 20:
                return None
            
            var_value = np.percentile(returns, (1.0 - confidence_level) * 100.0)
            return float(var_value) if np.isfinite(var_value) else None
            
        except Exception as e:
            logger.warning(f"VaR calculation failed: {e}")
            return None
    
    def _determine_signal_direction(
        self, 
        indicators: Dict[str, float], 
        symbol: str
    ) -> str:
        """确定技术信号方向"""
        try:
            # 基于RSI和趋势的基本信号
            rsi = indicators.get('RSI', 50)
            sma_50 = indicators.get('SMA_50')
            sma_200 = indicators.get('SMA_200')
            adx = indicators.get('ADX', 0)
            
            # 默认hold
            direction = "hold"
            
            # 趋势判断
            uptrend = False
            if sma_50 and sma_200:
                uptrend = sma_50 > sma_200
            
            # 趋势强度判断
            strong_trend = adx >= 25
            
            # 买入信号
            if rsi < 30:  # 超卖
                if uptrend and strong_trend:
                    direction = "long"
                elif strong_trend:
                    direction = "long"  # 即使反趋势，ADX强时也考虑
            # 卖出信号
            elif rsi > 70:  # 超买
                if not uptrend and strong_trend:
                    direction = "short"
                elif strong_trend:
                    direction = "short"  # 即使反趋势，ADX强时也考虑
            
            # 如果RSI中性但趋势很强且价格有明确位置
            elif 30 <= rsi <= 70:
                if strong_trend:
                    if uptrend and rsi < 60:
                        direction = "long"
                    elif not uptrend and rsi > 40:
                        direction = "short"
            
            logger.debug(f"{symbol}: RSI={rsi:.2f}, ADX={adx:.2f}, direction={direction}")
            return direction
            
        except Exception as e:
            logger.warning(f"Signal direction determination failed for {symbol}: {e}")
            return "hold"
    
    async def _perform_backtest(
        self,
        df: pd.DataFrame,
        signal_direction: str,
        request: AnalysisRequest
    ) -> BacktestResults:
        """执行简化回测"""
        try:
            # 简化回测策略：基于RSI
            returns = df['returns'].dropna()
            rsi = df['RSI'].dropna()
            
            if len(returns) < 50 or len(rsi) < 50:
                raise BacktestError("Insufficient data for backtest")
            
            # RSI交易策略
            trades = []
            position = None
            
            for i in range(50, len(rsi)):
                current_rsi = rsi.iloc[i]
                current_price = df['close'].iloc[i]
                
                # 买入信号
                if current_rsi < 30 and position is None:
                    position = {
                        'entry_price': current_price,
                        'entry_rsi': current_rsi,
                        'entry_index': i,
                        'direction': 'long'
                    }
                
                # 卖出信号
                elif current_rsi > 70 and position and position['direction'] == 'long':
                    position['exit_price'] = current_price
                    position['exit_rsi'] = current_rsi
                    position['exit_index'] = i
                    trades.append(position.copy())
                    position = None
                
                # 反向信号平仓
                elif current_rsi > 70 and position and position['direction'] == 'long':
                    position['exit_price'] = current_price
                    position['exit_rsi'] = current_rsi
                    position['exit_index'] = i
                    trades.append(position.copy())
                    position = None
            
            # 如果还有未平仓的仓位，强制平仓
            if position:
                position['exit_price'] = df['close'].iloc[-1]
                position['exit_rsi'] = rsi.iloc[-1]
                position['exit_index'] = len(rsi) - 1
                trades.append(position)
            
            # 计算回测结果
            if not trades:
                return BacktestResults(
                    strategy_score=0.0,
                    recommendation="No trades executed",
                    warnings=["Insufficient trading signals"]
                )
            
            # 分析交易
            trade_analysis = self._analyze_trades(trades)
            
            # 计算策略评分
            strategy_score = self._calculate_strategy_score(trade_analysis)
            
            return BacktestResults(
                strategy_score=strategy_score,
                trade_analysis=trade_analysis,
                recommendation=self._get_recommendation(strategy_score)
            )
            
        except Exception as e:
            raise BacktestError(f"Backtest execution failed: {e}")
    
    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> 'BacktestTradeAnalysis':
        """分析交易"""
        from ..core.models import BacktestTradeAnalysis
        
        total_trades = len(trades)
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        profits = []
        losses = []
        total_duration = 0
        
        for trade in trades:
            # 计算收益
            if trade['direction'] == 'long':
                pnl = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            else:
                pnl = (trade['entry_price'] - trade['exit_price']) / trade['entry_price']
            
            total_profit += pnl
            
            if pnl > 0:
                winning_trades += 1
                profits.append(pnl)
            else:
                losing_trades += 1
                losses.append(abs(pnl))
            
            # 计算持仓时长
            duration = trade['exit_index'] - trade['entry_index']
            total_duration += duration
        
        # 计算指标
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(profits) / sum(losses) if losses else float('inf')
        largest_win = max(profits) if profits else 0
        largest_loss = max(losses) if losses else 0
        avg_duration = total_duration / total_trades if total_trades > 0 else 0
        
        return BacktestTradeAnalysis(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_profit=total_profit,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration=avg_duration
        )
    
    def _calculate_strategy_score(self, trade_analysis) -> float:
        """计算策略评分"""
        score = 0.0
        
        # 胜率权重 30%
        if trade_analysis.win_rate:
            score += 0.3 * min(trade_analysis.win_rate, 1.0)
        
        # 利润因子权重 25%
        if trade_analysis.profit_factor:
            score += 0.25 * min(trade_analysis.profit_factor / 2.0, 1.0)
        
        # 总收益权重 25%
        if trade_analysis.total_profit:
            score += 0.25 * min(trade_analysis.total_profit / 0.5, 1.0)  # 假设50%为优秀
        
        # 最大回撤惩罚 - 这里用最大亏损作为近似
        if trade_analysis.largest_loss:
            penalty = min(trade_analysis.largest_loss * 0.2, 0.3)
            score = max(0, score - penalty)
        
        return min(max(score, 0.0), 1.0)
    
    def _get_recommendation(self, strategy_score: float) -> str:
        """获取建议"""
        if strategy_score >= 0.7:
            return "Strong strategy performance"
        elif strategy_score >= 0.5:
            return "Moderate strategy performance"
        elif strategy_score >= 0.3:
            return "Weak strategy performance"
        else:
            return "Poor strategy performance"


# 全局分析服务实例
analysis_service: Optional[AnalysisService] = None


def get_analysis_service() -> AnalysisService:
    """获取全局分析服务实例"""
    global analysis_service
    if analysis_service is None:
        analysis_service = AnalysisService()
    return analysis_service