"""
扫描服务
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from ..core.data_fetcher import data_fetcher
from ..core.exceptions import ScanError, DataError
from ..core.models import (
    ScanRequest, ScanResponse, ScanResultItem, TradeDirection
)
from ..services.analysis_service import get_analysis_service

logger = logging.getLogger(__name__)


class ScannerService:
    """扫描服务"""
    
    def __init__(self):
        self.analysis_service = get_analysis_service()
    
    async def scan_market(self, request: ScanRequest) -> ScanResponse:
        """扫描市场"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting market scan with {request.max_concurrent_tasks} concurrent tasks")
            
            # 1. 获取交易对列表
            all_tickers = data_fetcher.get_available_tickers()
            logger.info(f"Found {len(all_tickers)} available tickers")
            
            # 2. 确定扫描范围
            start_idx = request.ticker_start_index
            end_idx = request.ticker_end_index or len(all_tickers)
            
            if start_idx >= len(all_tickers):
                raise ScanError("Start index exceeds available tickers")
            
            if end_idx > len(all_tickers):
                end_idx = len(all_tickers)
                logger.warning(f"Adjusted end index to {end_idx}")
            
            # 选择交易对
            tickers_to_scan = all_tickers[start_idx:end_idx]
            if len(tickers_to_scan) > request.max_tickers:
                tickers_to_scan = tickers_to_scan[:request.max_tickers]
                logger.info(f"Limited to {len(tickers_to_scan)} tickers as per max_tickers")
            
            # 3. 批量分析
            analysis_results = await self._scan_tickers_batch(
                tickers_to_scan, request
            )
            
            # 4. 过滤和排序
            filtered_results = self._filter_results(analysis_results, request)
            
            # 5. 取前N个结果
            top_results = filtered_results[:request.top_n]
            
            # 6. 构建响应
            processing_time = time.time() - start_time
            
            response = ScanResponse(
                scan_parameters=request,
                total_tickers_attempted=len(tickers_to_scan),
                total_tickers_succeeded=len([r for r in analysis_results.values() if r is not None]),
                ticker_start_index=start_idx,
                ticker_end_index=end_idx if request.ticker_end_index else len(all_tickers),
                total_opportunities_found=len(filtered_results),
                top_opportunities=top_results,
                processing_time=processing_time
            )
            
            logger.info(f"Market scan completed in {processing_time:.2f}s, "
                       f"found {len(filtered_results)} opportunities")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Market scan failed: {e}")
            
            return ScanResponse(
                scan_parameters=request,
                total_tickers_attempted=0,
                total_tickers_succeeded=0,
                ticker_start_index=request.ticker_start_index,
                ticker_end_index=request.ticker_end_index,
                total_opportunities_found=0,
                top_opportunities=[],
                errors={"scan_error": str(e)},
                processing_time=processing_time
            )
    
    async def _scan_tickers_batch(
        self,
        tickers: List[str],
        request: ScanRequest
    ) -> Dict[str, Any]:
        """批量扫描交易对"""
        # 创建分析请求
        analysis_request = self._create_analysis_request(request)
        
        # 限制并发数
        semaphore = asyncio.Semaphore(request.max_concurrent_tasks)
        
        async def analyze_single_ticker(ticker: str):
            async with semaphore:
                try:
                    # 设置symbol
                    analysis_request.symbol = ticker
                    
                    # 执行分析
                    result = await self.analysis_service.analyze_symbol(analysis_request)
                    
                    # 转换为扫描结果格式
                    return self._convert_to_scan_result(ticker, result, request.timeframe)
                    
                except Exception as e:
                    logger.warning(f"Analysis failed for {ticker}: {e}")
                    return None
        
        # 执行批量分析
        logger.info(f"Analyzing {len(tickers)} tickers with concurrency {request.max_concurrent_tasks}")
        
        tasks = [analyze_single_ticker(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 构建结果字典
        ticker_results = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Ticker {ticker} analysis failed with exception: {result}")
                ticker_results[ticker] = None
            else:
                ticker_results[ticker] = result
        
        return ticker_results
    
    def _create_analysis_request(self, request: ScanRequest):
        """从扫描请求创建分析请求"""
        from ..core.models import AnalysisRequest, TimeFrame
        
        return AnalysisRequest(
            symbol="",  # 将在具体分析时设置
            timeframe=request.timeframe,
            lookback=request.lookback,
            accountBalance=request.accountBalance,
            maxLeverage=request.maxLeverage
        )
    
    def _convert_to_scan_result(
        self,
        ticker: str,
        analysis_result,
        timeframe: str
    ) -> Optional[ScanResultItem]:
        """转换为扫描结果格式"""
        try:
            if analysis_result.error:
                return None
            
            # 检查是否有交易参数
            if not analysis_result.gptParams:
                return None
            
            gpt_params = analysis_result.gptParams
            
            # 如果没有交易方向或置信度太低，跳过
            if (not gpt_params.confidence_score or 
                gpt_params.confidence_score < 0.3 or
                gpt_params.trade_direction == 'hold'):
                return None
            
            # 计算综合评分
            backtest_score = analysis_result.backtest.strategy_score if analysis_result.backtest else 0.0
            combined_score = self._calculate_combined_score(
                gpt_params.confidence_score,
                backtest_score
            )
            
            # 构建分析摘要
            analysis_summary = self._build_analysis_summary(analysis_result)
            
            return ScanResultItem(
                rank=0,  # 将在排序后设置
                symbol=ticker,
                timeframe=timeframe,
                currentPrice=analysis_result.currentPrice,
                gptConfidence=gpt_params.confidence_score,
                backtestScore=backtest_score,
                combinedScore=combined_score,
                tradeDirection=gpt_params.trade_direction,
                optimalEntry=gpt_params.optimal_entry,
                stopLoss=gpt_params.stop_loss,
                takeProfit=gpt_params.take_profit,
                gptAnalysisSummary=analysis_summary
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert analysis result for {ticker}: {e}")
            return None
    
    def _calculate_combined_score(
        self,
        gpt_confidence: float,
        backtest_score: float
    ) -> float:
        """计算综合评分"""
        # 权重分配：GPT 60%，回测 40%
        return 0.6 * gpt_confidence + 0.4 * (backtest_score or 0)
    
    def _build_analysis_summary(self, analysis_result) -> str:
        """构建分析摘要"""
        summary_parts = []
        
        if analysis_result.gptAnalysis:
            if analysis_result.gptAnalysis.signal_evaluation:
                summary_parts.append(f"信号评估: {analysis_result.gptAnalysis.signal_evaluation[:100]}...")
            
            if analysis_result.gptAnalysis.technical_analysis:
                summary_parts.append(f"技术分析: {analysis_result.gptAnalysis.technical_analysis[:100]}...")
        
        # 添加风险信息
        if analysis_result.riskMetrics:
            risk_info = []
            if analysis_result.riskMetrics.garchVolatility:
                risk_info.append(f"GARCH波动率: {analysis_result.riskMetrics.garchVolatility:.2%}")
            if analysis_result.riskMetrics.var95:
                risk_info.append(f"VaR(95%): {analysis_result.riskMetrics.var95:.2%}")
            
            if risk_info:
                summary_parts.append(f"风险指标: {', '.join(risk_info)}")
        
        return " | ".join(summary_parts) if summary_parts else "分析完成"
    
    def _filter_results(
        self,
        results: Dict[str, Any],
        request: ScanRequest
    ) -> List[ScanResultItem]:
        """过滤结果"""
        filtered_items = []
        
        for ticker, result in results.items():
            if result is None:
                continue
            
            # 各种过滤条件
            if not self._passes_filters(result, request):
                continue
            
            filtered_items.append(result)
        
        # 按综合评分排序
        filtered_items.sort(key=lambda x: x.combinedScore or 0, reverse=True)
        
        # 设置排名
        for i, item in enumerate(filtered_items):
            item.rank = i + 1
        
        logger.info(f"Filtered from {len(results)} to {len(filtered_items)} results")
        
        return filtered_items
    
    def _passes_filters(self, item: ScanResultItem, request: ScanRequest) -> bool:
        """检查是否通过过滤条件"""
        
        # GPT置信度过滤
        if item.gptConfidence and item.gptConfidence < request.min_gpt_confidence:
            return False
        
        # 回测评分过滤
        if item.backtestScore and item.backtestScore < request.min_backtest_score:
            return False
        
        # 交易方向过滤
        if (request.trade_direction and 
            item.tradeDirection != request.trade_direction):
            return False
        
        # 回测交易数过滤
        if request.min_backtest_trades > 0:
            # 这里需要从分析结果中获取交易数
            # 由于ScanResultItem没有这个字段，暂时跳过
            pass
        
        # 胜率过滤
        if request.min_backtest_win_rate > 0:
            # 同上，需要从分析结果获取
            pass
        
        # 利润因子过滤
        if request.min_backtest_profit_factor > 0:
            # 同上
            pass
        
        # 风险回报比过滤
        if (request.min_risk_reward_ratio > 0 and
            item.optimalEntry and item.stopLoss and item.takeProfit):
            if item.tradeDirection == 'long':
                risk = abs(item.optimalEntry - item.stopLoss)
                reward = abs(item.takeProfit - item.optimalEntry)
            else:
                risk = abs(item.stopLoss - item.optimalEntry)
                reward = abs(item.optimalEntry - item.takeProfit)
            
            if risk > 0 and reward / risk < request.min_risk_reward_ratio:
                return False
        
        # ADX过滤
        if request.min_adx > 0:
            # 需要从分析结果获取ADX值
            # 暂时跳过
            pass
        
        # SMA对齐过滤
        if request.require_sma_alignment:
            # 需要检查SMA排列
            # 暂时跳过
            pass
        
        return True
    
    async def get_available_symbols(self) -> List[str]:
        """获取可扫描的交易对"""
        try:
            return data_fetcher.get_available_tickers()
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            raise ScanError(f"Symbol retrieval failed: {e}")
    
    def get_scanner_statistics(self) -> Dict[str, Any]:
        """获取扫描器统计信息"""
        try:
            all_tickers = data_fetcher.get_available_tickers()
            
            return {
                "total_available_symbols": len(all_tickers),
                "supported_timeframes": [
                    "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"
                ],
                "default_max_tickers": 100,
                "default_max_concurrent_tasks": 16,
                "supported_filters": [
                    "min_gpt_confidence",
                    "min_backtest_score", 
                    "trade_direction",
                    "min_adx",
                    "require_sma_alignment"
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get scanner statistics: {e}")
            return {}


# 全局扫描服务实例
scanner_service: Optional[ScannerService] = None


def get_scanner_service() -> ScannerService:
    """获取全局扫描服务实例"""
    global scanner_service
    if scanner_service is None:
        scanner_service = ScannerService()
    return scanner_service