"""
多策略信号处理管道主模块

统一的管道实现，整合了所有组件：
- MultiStrategyPipeline: 主管道类
- 信号处理和验证
- 策略融合
- 性能监控
- 回测验证
- 批量扫描
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import pandas as pd

from .types import (
    PipelineConfig, MarketData, TechnicalIndicators, RiskMetrics, MLFeatures,
    MLPrediction, LLMAssessment, StrategySignal, FusionResult, ScanRequest, 
    ScanResult, PerformanceMetrics, BacktestResult,
    SignalDirection, StrategyType, PipelineError, ProcessingTimeoutError
)

from .signal_processor import SignalProcessor
from .strategy_fusion import StrategyFusion
from .priority_manager import PriorityManager
from .performance_monitor import PerformanceMonitor
from .backtest_validator import BacktestValidator

logger = logging.getLogger(__name__)

class MultiStrategyPipeline:
    """多策略信号处理管道"""
    
    def __init__(self, config: PipelineConfig):
        """
        初始化管道
        
        Args:
            config: 管道配置
        """
        self.config = config
        self._session_id = None
        
        # 初始化组件
        self.signal_processor = SignalProcessor(config)
        self.strategy_fusion = StrategyFusion(config)
        self.priority_manager = PriorityManager(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.backtest_validator = BacktestValidator(config)
        
        # 状态管理
        self._is_running = False
        self._active_sessions = {}
        self._global_cache = {}
        
        logger.info("多策略信号处理管道初始化完成")
    
    async def start(self) -> str:
        """启动管道"""
        if self._is_running:
            raise PipelineError("管道已经在运行中")
        
        try:
            self._is_running = True
            self._session_id = await self.performance_monitor.record_pipeline_start()
            
            logger.info(f"管道已启动，会话ID: {self._session_id}")
            return self._session_id
            
        except Exception as e:
            self._is_running = False
            logger.error(f"启动管道失败: {e}")
            raise PipelineError(f"管道启动失败: {e}")
    
    async def stop(self):
        """停止管道"""
        if not self._is_running:
            return
        
        try:
            self._is_running = False
            self._active_sessions.clear()
            
            # 生成最终性能报告
            final_report = await self.performance_monitor.generate_performance_report()
            logger.info("管道已停止，最终性能报告已生成")
            
        except Exception as e:
            logger.error(f"停止管道时出错: {e}")
    
    async def process_single_symbol(
        self,
        symbol: str,
        market_data: MarketData,
        technical_indicators: Optional[TechnicalIndicators] = None,
        risk_metrics: Optional[RiskMetrics] = None,
        ml_features: Optional[MLFeatures] = None,
        ml_prediction: Optional[MLPrediction] = None,
        llm_assessment: Optional[LLMAssessment] = None,
        backtest_result: Optional[BacktestResult] = None
    ) -> FusionResult:
        """处理单个符号的信号"""
        if not self._is_running:
            raise PipelineError("管道未启动")
        
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        try:
            logger.info(f"开始处理符号: {symbol}")
            
            # 1. 生成各种策略信号
            signals = []
            
            # 技术指标信号
            if technical_indicators:
                technical_signals = await self.signal_processor.process_technical_signals(
                    market_data, technical_indicators
                )
                signals.extend(technical_signals)
            
            # ML预测信号
            if ml_prediction and ml_features:
                ml_signals = await self.signal_processor.process_ml_signals(
                    market_data, ml_features, ml_prediction
                )
                signals.extend(ml_signals)
            
            # 风险模型信号
            if risk_metrics:
                risk_signals = await self.signal_processor.process_risk_signals(
                    market_data, risk_metrics
                )
                signals.extend(risk_signals)
            
            # 回测参考信号
            if backtest_result:
                backtest_signals = await self.signal_processor.process_backtest_signals(
                    market_data, backtest_result
                )
                signals.extend(backtest_signals)
            
            # 记录延迟
            signal_generation_time = time.time() - start_time
            await self.performance_monitor.record_latency_breakdown(
                "signal_generation", signal_generation_time, session_id
            )
            
            # 2. 信号优先级排序
            priority_start = time.time()
            prioritized_signals = await self.priority_manager.prioritize_signals(signals)
            priority_time = time.time() - priority_start
            
            await self.performance_monitor.record_latency_breakdown(
                "priority_sorting", priority_time, session_id
            )
            
            # 3. 策略融合
            fusion_start = time.time()
            
            if not prioritized_signals:
                # 没有有效信号，返回默认HOLD结果
                fusion_result = FusionResult(
                    symbol=symbol,
                    final_direction=SignalDirection.HOLD,
                    final_score=0.0,
                    combined_confidence=0.0,
                    risk_reward_ratio=0.0,
                    decision_reason=["没有有效信号"]
                )
            else:
                fusion_result = await self.strategy_fusion.fuse_signals(prioritized_signals)
            
            fusion_time = time.time() - fusion_start
            
            await self.performance_monitor.record_latency_breakdown(
                "strategy_fusion", fusion_time, session_id
            )
            
            # 4. 记录吞吐量
            total_time = time.time() - start_time
            await self.performance_monitor.record_throughput(1, total_time)
            
            # 5. 记录策略性能（如果有LLM评估）
            if llm_assessment:
                await self.performance_monitor.record_strategy_performance(
                    StrategyType.TECHNICAL_INDICATOR,  # 简化处理
                    prioritized_signals[0] if prioritized_signals else None,
                    fusion_result
                )
            
            logger.info(f"符号 {symbol} 处理完成，方向: {fusion_result.final_direction.value}")
            return fusion_result
            
        except Exception as e:
            logger.error(f"处理符号 {symbol} 时出错: {e}")
            
            # 返回错误处理的默认结果
            return FusionResult(
                symbol=symbol,
                final_direction=SignalDirection.HOLD,
                final_score=0.0,
                combined_confidence=0.0,
                risk_reward_ratio=0.0,
                decision_reason=[f"处理错误: {str(e)}"]
            )
    
    async def batch_scan(
        self,
        scan_request: ScanRequest,
        symbol_data_map: Dict[str, Dict[str, Any]]
    ) -> ScanResult:
        """批量扫描多个符号"""
        if not self._is_running:
            raise PipelineError("管道未启动")
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            logger.info(f"开始批量扫描，请求ID: {request_id}, 符号数: {len(scan_request.symbols)}")
            
            # 1. 数据预处理和过滤
            valid_symbols = await self._preprocess_scan_request(scan_request, symbol_data_map)
            
            # 2. 创建处理任务
            tasks = []
            for symbol in valid_symbols:
                symbol_data = symbol_data_map.get(symbol, {})
                
                # 构建MarketData
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=symbol_data.get('price', 0.0),
                    volume=symbol_data.get('volume', 0.0),
                    ohlcv=symbol_data.get('ohlcv'),
                    order_book=symbol_data.get('order_book'),
                    data_freshness=symbol_data.get('data_freshness', 0.0)
                )
                
                # 构建其他数据
                technical_indicators = self._extract_technical_indicators(symbol_data)
                risk_metrics = self._extract_risk_metrics(symbol_data)
                ml_features = self._extract_ml_features(symbol_data)
                ml_prediction = self._extract_ml_prediction(symbol_data)
                backtest_result = self._extract_backtest_result(symbol_data)
                
                # 创建处理任务
                task = asyncio.create_task(
                    self.process_single_symbol(
                        symbol=symbol,
                        market_data=market_data,
                        technical_indicators=technical_indicators,
                        risk_metrics=risk_metrics,
                        ml_features=ml_features,
                        ml_prediction=ml_prediction,
                        backtest_result=backtest_result
                    )
                )
                tasks.append((symbol, task))
            
            # 3. 并发处理
            results = []
            errors = []
            
            # 控制并发数量
            semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
            
            async def process_with_semaphore(symbol: str, task):
                async with semaphore:
                    try:
                        result = await asyncio.wait_for(
                            task, timeout=self.config.timeout_seconds
                        )
                        return symbol, result, None
                    except asyncio.TimeoutError:
                        error = f"处理超时: {symbol}"
                        logger.warning(error)
                        return symbol, None, {"symbol": symbol, "error": error}
                    except Exception as e:
                        error = f"处理异常: {symbol}, {str(e)}"
                        logger.error(error)
                        return symbol, None, {"symbol": symbol, "error": str(e)}
            
            # 执行所有任务
            completed_tasks = [
                process_with_semaphore(symbol, task) for symbol, task in tasks
            ]
            
            for symbol, result, error in await asyncio.gather(*completed_tasks):
                if error:
                    errors.append(error)
                elif result:
                    results.append(result)
            
            # 4. 结果排序和过滤
            filtered_results = await self._filter_and_rank_results(results, scan_request.filters)
            
            # 5. 生成扫描结果
            processing_time = time.time() - start_time
            
            scan_result = ScanResult(
                request_id=request_id,
                results=filtered_results[:scan_request.top_n or self.config.top_n_results],
                processing_time=processing_time,
                total_symbols=len(scan_request.symbols),
                filtered_symbols=len(valid_symbols),
                errors=errors,
                metadata={
                    "session_id": self._session_id,
                    "valid_symbols": len(valid_symbols),
                    "successful_processing": len(results),
                    "failed_processing": len(errors)
                }
            )
            
            # 6. 记录批量处理统计
            await self.performance_monitor.record_latency_breakdown(
                "batch_scan", processing_time, request_id
            )
            
            logger.info(f"批量扫描完成，耗时: {processing_time:.2f}s, "
                       f"成功: {len(results)}, 失败: {len(errors)}")
            
            return scan_result
            
        except Exception as e:
            logger.error(f"批量扫描失败: {e}")
            raise PipelineError(f"批量扫描失败: {e}")
    
    async def _preprocess_scan_request(
        self,
        scan_request: ScanRequest,
        symbol_data_map: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """预处理扫描请求"""
        valid_symbols = []
        
        for symbol in scan_request.symbols:
            if symbol not in symbol_data_map:
                continue
            
            symbol_data = symbol_data_map[symbol]
            
            # 检查数据完整性
            if not self._validate_symbol_data(symbol_data):
                continue
            
            # 应用过滤条件
            if not self._apply_scan_filters(symbol_data, scan_request.filters):
                continue
            
            valid_symbols.append(symbol)
            
            # 限制符号数量
            if len(valid_symbols) >= (scan_request.max_symbols or self.config.max_symbols_per_scan):
                break
        
        return valid_symbols
    
    def _validate_symbol_data(self, symbol_data: Dict[str, Any]) -> bool:
        """验证符号数据有效性"""
        # 检查基本价格数据
        if 'price' not in symbol_data or symbol_data['price'] <= 0:
            return False
        
        # 检查数据新鲜度
        if symbol_data.get('data_freshness', 0) > 300:  # 5分钟过期
            return False
        
        return True
    
    def _apply_scan_filters(self, symbol_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """应用扫描过滤条件"""
        # 这里可以实现具体的过滤逻辑
        # 例如：最小成交量、价格范围等
        
        # 示例：最小成交量过滤
        if 'min_volume' in filters:
            if symbol_data.get('volume', 0) < filters['min_volume']:
                return False
        
        # 示例：价格范围过滤
        if 'price_range' in filters:
            price = symbol_data.get('price', 0)
            min_price, max_price = filters['price_range']
            if price < min_price or price > max_price:
                return False
        
        return True
    
    async def _filter_and_rank_results(
        self,
        results: List[FusionResult],
        filters: Dict[str, Any]
    ) -> List[FusionResult]:
        """过滤和排序结果"""
        filtered_results = []
        
        for result in results:
            # 应用过滤条件
            if not self._apply_result_filters(result, filters):
                continue
            
            filtered_results.append(result)
        
        # 按综合评分排序
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return filtered_results
    
    def _apply_result_filters(self, result: FusionResult, filters: Dict[str, Any]) -> bool:
        """应用结果过滤条件"""
        # 最小置信度过滤
        if 'min_confidence' in filters:
            if result.combined_confidence < filters['min_confidence']:
                return False
        
        # 最小评分过滤
        if 'min_score' in filters:
            if result.final_score < filters['min_score']:
                return False
        
        # 最小风险回报比过滤
        if 'min_risk_reward_ratio' in filters:
            if result.risk_reward_ratio < filters['min_risk_reward_ratio']:
                return False
        
        # 方向过滤
        if 'allowed_directions' in filters:
            if result.final_direction.value not in filters['allowed_directions']:
                return False
        
        return True
    
    def _extract_technical_indicators(self, symbol_data: Dict[str, Any]) -> Optional[TechnicalIndicators]:
        """提取技术指标"""
        indicators_data = symbol_data.get('technical_indicators', {})
        if not indicators_data:
            return None
        
        return TechnicalIndicators(
            rsi=indicators_data.get('rsi'),
            macd=indicators_data.get('macd'),
            macd_signal=indicators_data.get('macd_signal'),
            bollinger_upper=indicators_data.get('bollinger_upper'),
            bollinger_middle=indicators_data.get('bollinger_middle'),
            bollinger_lower=indicators_data.get('bollinger_lower'),
            adx=indicators_data.get('adx'),
            atr=indicators_data.get('atr'),
            sma_20=indicators_data.get('sma_20'),
            sma_50=indicators_data.get('sma_50'),
            sma_200=indicators_data.get('sma_200'),
            ema_12=indicators_data.get('ema_12'),
            ema_26=indicators_data.get('ema_26')
        )
    
    def _extract_risk_metrics(self, symbol_data: Dict[str, Any]) -> Optional[RiskMetrics]:
        """提取风险指标"""
        risk_data = symbol_data.get('risk_metrics', {})
        if not risk_data:
            return None
        
        return RiskMetrics(
            garch_volatility=risk_data.get('garch_volatility'),
            var_95=risk_data.get('var_95'),
            expected_shortfall=risk_data.get('expected_shortfall'),
            max_drawdown=risk_data.get('max_drawdown'),
            sharpe_ratio=risk_data.get('sharpe_ratio'),
            sortino_ratio=risk_data.get('sortino_ratio'),
            calmar_ratio=risk_data.get('calmar_ratio')
        )
    
    def _extract_ml_features(self, symbol_data: Dict[str, Any]) -> Optional[MLFeatures]:
        """提取ML特征"""
        features_data = symbol_data.get('ml_features', {})
        if not features_data:
            return None
        
        return MLFeatures(
            spread=features_data.get('spread'),
            order_imbalance=features_data.get('order_imbalance'),
            depth_imbalance=features_data.get('depth_imbalance'),
            wap_1=features_data.get('wap_1'),
            wap_5=features_data.get('wap_5'),
            volatility_60s=features_data.get('volatility_60s'),
            mid_price=features_data.get('mid_price'),
            volume_features=features_data.get('volume_features', {})
        )
    
    def _extract_ml_prediction(self, symbol_data: Dict[str, Any]) -> Optional[MLPrediction]:
        """提取ML预测"""
        prediction_data = symbol_data.get('ml_prediction', {})
        if not prediction_data:
            return None
        
        return MLPrediction(
            label=prediction_data.get('label', 0),
            probability_scores=prediction_data.get('probability_scores', {}),
            confidence=prediction_data.get('confidence', 0.0),
            model_version=prediction_data.get('model_version', ''),
            prediction_time=datetime.fromisoformat(prediction_data.get('prediction_time', datetime.now().isoformat()))
        )
    
    def _extract_backtest_result(self, symbol_data: Dict[str, Any]) -> Optional[BacktestResult]:
        """提取回测结果"""
        backtest_data = symbol_data.get('backtest_result', {})
        if not backtest_data:
            return None
        
        return BacktestResult(
            score=backtest_data.get('score', 0.0),
            total_return=backtest_data.get('total_return', 0.0),
            sharpe_ratio=backtest_data.get('sharpe_ratio', 0.0),
            max_drawdown=backtest_data.get('max_drawdown', 0.0),
            win_rate=backtest_data.get('win_rate', 0.0),
            profit_factor=backtest_data.get('profit_factor', 0.0),
            trade_count=backtest_data.get('trade_count', 0),
            backtest_period=backtest_data.get('backtest_period', ''),
            strategy_name=backtest_data.get('strategy_name', ''),
            additional_metrics=backtest_data.get('additional_metrics', {})
        )
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        return await self.performance_monitor.generate_performance_metrics(self._session_id)
    
    async def get_performance_report(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """获取性能报告"""
        return await self.performance_monitor.generate_performance_report(time_window)
    
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """检查性能告警"""
        return await self.performance_monitor.check_performance_alerts()
    
    async def validate_signal_backtest(
        self,
        signal: StrategySignal,
        historical_data: pd.DataFrame,
        validation_period: timedelta = timedelta(days=30)
    ) -> BacktestResult:
        """验证信号回测"""
        return await self.backtest_validator.validate_strategy_signal(
            signal, historical_data, validation_period
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        return {
            "is_running": self._is_running,
            "session_id": self._session_id,
            "config": {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "timeout_seconds": self.config.timeout_seconds,
                "batch_size": self.config.batch_size,
                "max_symbols_per_scan": self.config.max_symbols_per_scan
            },
            "components": {
                "signal_processor": "active",
                "strategy_fusion": "active", 
                "priority_manager": "active",
                "performance_monitor": "active",
                "backtest_validator": "active"
            },
            "statistics": {
                "fusion_statistics": self.strategy_fusion.get_fusion_statistics(),
                "priority_statistics": self.priority_manager.get_priority_statistics(),
                "validation_statistics": self.backtest_validator.get_validation_statistics(),
                "cache_status": self.signal_processor.get_cache_status()
            }
        }
    
    async def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        try:
            # 更新配置参数
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"配置已更新: {key} = {value}")
                else:
                    logger.warning(f"未知配置项: {key}")
            
            # 更新组件配置
            self.signal_processor.config = self.config
            self.strategy_fusion.config = self.config
            self.priority_manager.config = self.config
            self.performance_monitor.config = self.config
            self.backtest_validator.config = self.config
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            raise PipelineError(f"更新配置失败: {e}")
    
    def reset_all(self):
        """重置所有组件状态"""
        try:
            # 清理缓存
            self.signal_processor.clear_cache()
            self._global_cache.clear()
            
            # 重置统计
            self.strategy_fusion.reset_performance_tracking()
            self.priority_manager.reset_statistics()
            self.performance_monitor.reset_metrics()
            self.backtest_validator.reset_validation_history()
            
            logger.info("管道状态已重置")
            
        except Exception as e:
            logger.error(f"重置管道状态失败: {e}")
            raise PipelineError(f"重置管道状态失败: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()