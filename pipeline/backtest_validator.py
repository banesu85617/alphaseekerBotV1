"""
回测验证器模块

负责对策略信号进行历史回测验证，包括：
- 策略信号回测
- 性能指标计算
- 风险评估
- 参数优化
- 策略验证报告
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import asdict

from .types import (
    StrategySignal, FusionResult, BacktestResult, MarketData,
    PipelineConfig, SignalDirection, RiskLevel
)

logger = logging.getLogger(__name__)

class BacktestValidator:
    """回测验证器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._backtest_history = []
        self._validation_metrics = {}
        self._parameter_sets = []
        
    async def validate_strategy_signal(
        self,
        signal: StrategySignal,
        historical_data: pd.DataFrame,
        validation_period: timedelta = timedelta(days=30)
    ) -> BacktestResult:
        """验证单个策略信号"""
        try:
            logger.info(f"开始回测验证信号 for {signal.symbol}")
            
            # 1. 数据预处理
            processed_data = await self._preprocess_historical_data(
                historical_data, signal.timestamp, validation_period
            )
            
            if processed_data.empty:
                raise ValueError("历史数据不足，无法进行回测")
            
            # 2. 执行回测
            backtest_result = await self._execute_backtest(signal, processed_data)
            
            # 3. 计算性能指标
            final_metrics = await self._calculate_performance_metrics(backtest_result, processed_data)
            
            # 4. 更新回测结果
            result = BacktestResult(
                score=final_metrics["score"],
                total_return=final_metrics["total_return"],
                sharpe_ratio=final_metrics["sharpe_ratio"],
                max_drawdown=final_metrics["max_drawdown"],
                win_rate=final_metrics["win_rate"],
                profit_factor=final_metrics["profit_factor"],
                trade_count=final_metrics["trade_count"],
                backtest_period=f"{validation_period.days}天",
                strategy_name=signal.strategy_type.value,
                additional_metrics=final_metrics
            )
            
            # 5. 记录回测历史
            self._backtest_history.append({
                "timestamp": datetime.now(),
                "signal": signal,
                "result": result,
                "data_points": len(processed_data)
            })
            
            logger.info(f"回测验证完成，评分: {result.score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"回测验证信号时出错: {e}")
            # 返回默认的失败结果
            return BacktestResult(
                score=0.0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                trade_count=0,
                backtest_period="失败",
                strategy_name=signal.strategy_type.value
            )
    
    async def validate_fusion_result(
        self,
        fusion_result: FusionResult,
        signals: List[StrategySignal],
        historical_data: pd.DataFrame,
        validation_period: timedelta = timedelta(days=30)
    ) -> Dict[str, BacktestResult]:
        """验证融合结果"""
        results = {}
        
        try:
            logger.info(f"开始验证融合结果 for {fusion_result.symbol}")
            
            # 1. 对每个原始信号进行回测
            for signal in signals:
                if signal.strategy_type.value in [st.value for st in results.keys()]:
                    continue  # 避免重复验证同类型信号
                
                result = await self.validate_strategy_signal(signal, historical_data, validation_period)
                results[signal.strategy_type] = result
            
            # 2. 对融合信号进行回测
            fusion_signal = self._create_fusion_signal(fusion_result, signals)
            if fusion_signal:
                fusion_backtest = await self.validate_strategy_signal(
                    fusion_signal, historical_data, validation_period
                )
                results["fusion"] = fusion_backtest
            
            # 3. 计算融合效果
            fusion_improvement = await self._calculate_fusion_improvement(results)
            
            logger.info(f"融合验证完成，改善程度: {fusion_improvement:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"验证融合结果时出错: {e}")
            return {}
    
    async def _preprocess_historical_data(
        self,
        data: pd.DataFrame,
        current_timestamp: datetime,
        validation_period: timedelta
    ) -> pd.DataFrame:
        """预处理历史数据"""
        try:
            # 确保时间戳列为datetime类型
            if 'timestamp' not in data.columns and 'datetime' in data.columns:
                data = data.rename(columns={'datetime': 'timestamp'})
            
            # 转换时间戳
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # 过滤时间范围
            end_time = current_timestamp
            start_time = end_time - validation_period
            
            filtered_data = data[
                (data['timestamp'] >= start_time) & 
                (data['timestamp'] <= end_time)
            ].copy()
            
            # 排序并重置索引
            filtered_data = filtered_data.sort_values('timestamp').reset_index(drop=True)
            
            # 填充缺失值
            numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns
            filtered_data[numeric_columns] = filtered_data[numeric_columns].fillna(method='ffill')
            
            logger.debug(f"预处理完成，数据点: {len(filtered_data)}")
            return filtered_data
            
        except Exception as e:
            logger.error(f"预处理历史数据时出错: {e}")
            return pd.DataFrame()
    
    async def _execute_backtest(
        self,
        signal: StrategySignal,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """执行回测"""
        try:
            trades = []
            portfolio_value = 10000.0  # 初始资金
            position = 0.0  # 当前持仓
            entry_price = 0.0
            
            signal_timestamp = signal.timestamp
            current_price_idx = data[data['timestamp'] <= signal_timestamp].index
            
            if len(current_price_idx) == 0:
                return {"trades": [], "portfolio_values": [portfolio_value]}
            
            # 从信号产生点开始回测
            start_idx = max(0, len(current_price_idx) - 10)  # 信号前10个数据点
            
            for i in range(start_idx, len(data)):
                current_row = data.iloc[i]
                current_time = current_row['timestamp']
                current_price = current_row.get('close', current_row.get('price', 0))
                
                if current_price <= 0:
                    continue
                
                # 生成交易信号
                signal_generated = self._generate_trading_signal(
                    signal, current_row, current_time
                )
                
                if signal_generated and position == 0:  # 开仓
                    # 计算仓位大小
                    position_size = min(
                        portfolio_value * 0.1,  # 最大10%仓位
                        portfolio_value * signal.confidence
                    )
                    
                    if position_size > 0:
                        position = position_size / current_price
                        entry_price = current_price
                        
                        trades.append({
                            "timestamp": current_time,
                            "type": "buy",
                            "price": current_price,
                            "quantity": position,
                            "value": position_size,
                            "signal_confidence": signal.confidence
                        })
                
                elif not signal_generated and position > 0:  # 平仓
                    exit_value = position * current_price
                    profit = exit_value - (position * entry_price)
                    
                    trades.append({
                        "timestamp": current_time,
                        "type": "sell",
                        "price": current_price,
                        "quantity": position,
                        "value": exit_value,
                        "profit": profit,
                        "return": profit / (position * entry_price)
                    })
                    
                    portfolio_value += profit
                    position = 0.0
            
            # 如果最后还有持仓，按最后价格平仓
            if position > 0:
                final_price = data.iloc[-1].get('close', data.iloc[-1].get('price', 0))
                if final_price > 0:
                    exit_value = position * final_price
                    profit = exit_value - (position * entry_price)
                    
                    trades.append({
                        "timestamp": data.iloc[-1]['timestamp'],
                        "type": "sell",
                        "price": final_price,
                        "quantity": position,
                        "value": exit_value,
                        "profit": profit,
                        "return": profit / (position * entry_price)
                    })
                    
                    portfolio_value += profit
            
            # 计算组合价值曲线
            portfolio_values = self._calculate_portfolio_curve(trades, data, start_idx)
            
            return {
                "trades": trades,
                "portfolio_values": portfolio_values,
                "initial_value": 10000.0,
                "final_value": portfolio_value,
                "total_return": (portfolio_value - 10000.0) / 10000.0
            }
            
        except Exception as e:
            logger.error(f"执行回测时出错: {e}")
            return {"trades": [], "portfolio_values": [10000.0]}
    
    def _generate_trading_signal(
        self,
        signal: StrategySignal,
        market_row: pd.Series,
        current_time: datetime
    ) -> bool:
        """生成交易信号"""
        try:
            # 基于信号方向生成交易逻辑
            if signal.direction == SignalDirection.LONG:
                # 做多信号：检查是否应该开仓
                return True
            elif signal.direction == SignalDirection.SHORT:
                # 做空信号：这里简化处理，实际中需要考虑做空机制
                return True
            else:  # HOLD
                return False
                
        except Exception as e:
            logger.error(f"生成交易信号时出错: {e}")
            return False
    
    def _calculate_portfolio_curve(
        self,
        trades: List[Dict],
        data: pd.DataFrame,
        start_idx: int
    ) -> List[float]:
        """计算组合价值曲线"""
        portfolio_values = []
        current_value = 10000.0
        position = 0.0
        entry_price = 0.0
        
        trade_index = 0
        
        for i in range(start_idx, len(data)):
            current_row = data.iloc[i]
            current_price = current_row.get('close', current_row.get('price', 0))
            
            if current_price <= 0:
                portfolio_values.append(current_value)
                continue
            
            # 处理交易
            while trade_index < len(trades) and trades[trade_index]['timestamp'] <= current_row['timestamp']:
                trade = trades[trade_index]
                
                if trade['type'] == 'buy':
                    position = trade['quantity']
                    entry_price = trade['price']
                else:  # sell
                    current_value += trade.get('profit', 0)
                    position = 0.0
                
                trade_index += 1
            
            # 计算当前组合价值
            if position > 0:
                current_value = position * current_price
            else:
                # 现金价值
                pass
            
            portfolio_values.append(current_value)
        
        return portfolio_values
    
    async def _calculate_performance_metrics(
        self,
        backtest_result: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """计算性能指标"""
        trades = backtest_result.get("trades", [])
        portfolio_values = backtest_result.get("portfolio_values", [])
        
        if not portfolio_values:
            return {
                "score": 0.0,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "trade_count": 0
            }
        
        # 计算总收益率
        initial_value = backtest_result.get("initial_value", 10000.0)
        final_value = backtest_result.get("final_value", 10000.0)
        total_return = (final_value - initial_value) / initial_value
        
        # 计算夏普比率
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        risk_free_rate = 0.02 / 252  # 年化2%的无风险利率，日化
        excess_returns = returns - risk_free_rate
        
        sharpe_ratio = 0.0
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        # 计算最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 计算胜率
        winning_trades = [t for t in trades if t.get('return', 0) > 0]
        win_rate = len(winning_trades) / max(len(trades), 1)
        
        # 计算盈亏比
        gross_profit = sum(t.get('profit', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0))
        profit_factor = gross_profit / max(gross_loss, 1)
        
        # 计算综合评分
        score = self._calculate_composite_score({
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(max_drawdown),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trade_count": len(trades)
        })
        
        return {
            "score": score,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trade_count": len(trades),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_trade_return": np.mean([t.get('return', 0) for t in trades]) if trades else 0.0
        }
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        try:
            # 归一化各个指标
            total_return_score = min(1.0, max(0.0, (metrics["total_return"] + 1) / 2))
            sharpe_score = min(1.0, max(0.0, metrics["sharpe_ratio"] / 3))
            drawdown_score = min(1.0, max(0.0, 1 - metrics["max_drawdown"]))
            win_rate_score = metrics["win_rate"]
            profit_factor_score = min(1.0, metrics["profit_factor"] / 2)
            trade_count_score = min(1.0, metrics["trade_count"] / 50)  # 50次交易为满分
            
            # 加权平均
            weights = {
                "total_return": 0.25,
                "sharpe": 0.20,
                "drawdown": 0.20,
                "win_rate": 0.15,
                "profit_factor": 0.15,
                "trade_count": 0.05
            }
            
            composite_score = (
                total_return_score * weights["total_return"] +
                sharpe_score * weights["sharpe"] +
                drawdown_score * weights["drawdown"] +
                win_rate_score * weights["win_rate"] +
                profit_factor_score * weights["profit_factor"] +
                trade_count_score * weights["trade_count"]
            )
            
            return composite_score
            
        except Exception as e:
            logger.error(f"计算综合评分时出错: {e}")
            return 0.0
    
    def _create_fusion_signal(
        self,
        fusion_result: FusionResult,
        signals: List[StrategySignal]
    ) -> Optional[StrategySignal]:
        """创建融合信号"""
        if not signals:
            return None
        
        try:
            # 使用第一个信号的基础结构
            base_signal = signals[0]
            
            # 创建融合信号
            fusion_signal = StrategySignal(
                strategy_type=base_signal.strategy_type,  # 使用基础策略类型
                direction=fusion_result.final_direction,
                confidence=fusion_result.combined_confidence,
                score=fusion_result.final_score,
                timestamp=datetime.now(),
                symbol=fusion_result.symbol,
                market_data=base_signal.market_data,
                metadata={
                    "fusion_signal": True,
                    "component_scores": fusion_result.component_scores,
                    "decision_reasons": fusion_result.decision_reason
                }
            )
            
            return fusion_signal
            
        except Exception as e:
            logger.error(f"创建融合信号时出错: {e}")
            return None
    
    async def _calculate_fusion_improvement(
        self,
        results: Dict[str, BacktestResult]
    ) -> float:
        """计算融合改善程度"""
        try:
            # 找到最佳单一策略
            best_single_score = 0.0
            for strategy_type, result in results.items():
                if strategy_type != "fusion" and result.score > best_single_score:
                    best_single_score = result.score
            
            # 获取融合结果
            fusion_score = results.get("fusion", BacktestResult(0, 0, 0, 0, 0, 0, 0, "")).score
            
            # 计算改善程度
            if best_single_score > 0:
                improvement = (fusion_score - best_single_score) / best_single_score
            else:
                improvement = fusion_score
            
            return improvement
            
        except Exception as e:
            logger.error(f"计算融合改善程度时出错: {e}")
            return 0.0
    
    async def optimize_parameters(
        self,
        base_signal: StrategySignal,
        historical_data: pd.DataFrame,
        parameter_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """参数优化"""
        try:
            logger.info(f"开始参数优化 for {base_signal.symbol}")
            
            best_params = {}
            best_score = 0.0
            optimization_results = []
            
            # 生成参数组合
            param_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            for params in param_combinations:
                try:
                    # 创建带参数的信号
                    optimized_signal = self._apply_parameters(base_signal, params)
                    
                    # 执行回测
                    result = await self.validate_strategy_signal(
                        optimized_signal, historical_data
                    )
                    
                    optimization_results.append({
                        "parameters": params,
                        "score": result.score,
                        "total_return": result.total_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "win_rate": result.win_rate
                    })
                    
                    if result.score > best_score:
                        best_score = result.score
                        best_params = params
                        
                except Exception as e:
                    logger.warning(f"参数组合 {params} 验证失败: {e}")
                    continue
            
            # 排序结果
            optimization_results.sort(key=lambda x: x["score"], reverse=True)
            
            optimization_summary = {
                "best_parameters": best_params,
                "best_score": best_score,
                "total_combinations": len(param_combinations),
                "valid_combinations": len(optimization_results),
                "top_results": optimization_results[:10],  # 前10个结果
                "optimization_efficiency": len(optimization_results) / max(len(param_combinations), 1)
            }
            
            logger.info(f"参数优化完成，最佳评分: {best_score:.3f}")
            return optimization_summary
            
        except Exception as e:
            logger.error(f"参数优化时出错: {e}")
            return {}
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[Dict]:
        """生成参数组合"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        
        # 限制组合数量以避免计算时间过长
        max_combinations = 100
        if len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
        
        return combinations
    
    def _apply_parameters(self, base_signal: StrategySignal, parameters: Dict[str, Any]) -> StrategySignal:
        """应用参数到信号"""
        try:
            # 创建新的信号副本
            import copy
            optimized_signal = copy.deepcopy(base_signal)
            
            # 应用参数（这里简化处理，实际中可能需要更复杂的参数应用逻辑）
            for param_name, param_value in parameters.items():
                if param_name == "confidence_threshold":
                    optimized_signal.confidence = param_value
                elif param_name == "score_threshold":
                    optimized_signal.score = param_value
                # 可以添加更多参数应用逻辑
            
            return optimized_signal
            
        except Exception as e:
            logger.error(f"应用参数时出错: {e}")
            return base_signal
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        if not self._backtest_history:
            return {"status": "no_data"}
        
        recent_validations = self._backtest_history[-100:]  # 最近100次
        
        scores = [v["result"].score for v in recent_validations]
        returns = [v["result"].total_return for v in recent_validations]
        
        return {
            "total_validations": len(self._backtest_history),
            "recent_validations": len(recent_validations),
            "average_score": np.mean(scores) if scores else 0.0,
            "average_return": np.mean(returns) if returns else 0.0,
            "best_score": np.max(scores) if scores else 0.0,
            "worst_score": np.min(scores) if scores else 0.0,
            "validation_frequency": len(self._backtest_history) / max(1, 
                (datetime.now() - self._backtest_history[0]["timestamp"]).total_seconds() / 86400
            )
        }
    
    def export_backtest_report(
        self,
        format: str = "json",
        time_window: Optional[timedelta] = None
    ) -> str:
        """导出回测报告"""
        if not time_window:
            time_window = timedelta(days=30)
        
        current_time = datetime.now()
        window_start = current_time - time_window
        
        # 过滤时间窗口内的数据
        window_data = [
            entry for entry in self._backtest_history 
            if entry["timestamp"] >= window_start
        ]
        
        if format == "json":
            import json
            
            report_data = {
                "export_timestamp": current_time.isoformat(),
                "time_window": str(time_window),
                "validation_count": len(window_data),
                "validations": [
                    {
                        "timestamp": entry["timestamp"].isoformat(),
                        "symbol": entry["signal"].symbol,
                        "strategy_type": entry["signal"].strategy_type.value,
                        "score": entry["result"].score,
                        "total_return": entry["result"].total_return,
                        "sharpe_ratio": entry["result"].sharpe_ratio,
                        "max_drawdown": entry["result"].max_drawdown,
                        "win_rate": entry["result"].win_rate,
                        "trade_count": entry["result"].trade_count
                    }
                    for entry in window_data
                ]
            }
            
            return json.dumps(report_data, indent=2)
        
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def reset_validation_history(self):
        """重置验证历史"""
        self._backtest_history.clear()
        self._validation_metrics.clear()
        self._parameter_sets.clear()
        logger.info("回测验证历史已重置")