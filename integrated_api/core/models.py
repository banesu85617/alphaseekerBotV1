"""
数据模型定义
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class TradeDirection(str, Enum):
    """交易方向"""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class TimeFrame(str, Enum):
    """时间周期"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


class TechnicalIndicators(BaseModel):
    """技术指标数据"""
    # 趋势指标
    SMA_50: Optional[float] = Field(default=None, description="50期简单移动平均")
    SMA_200: Optional[float] = Field(default=None, description="200期简单移动平均")
    EMA_12: Optional[float] = Field(default=None, description="12期指数移动平均")
    EMA_26: Optional[float] = Field(default=None, description="26期指数移动平均")
    MACD: Optional[float] = Field(default=None, description="MACD指标")
    Signal_Line: Optional[float] = Field(default=None, description="MACD信号线")
    
    # 动量指标
    RSI: Optional[float] = Field(default=None, description="相对强弱指标")
    Stochastic_K: Optional[float] = Field(default=None, description="随机指标%K")
    Stochastic_D: Optional[float] = Field(default=None, description="随机指标%D")
    Williams_R: Optional[float] = Field(default=None, description="威廉指标")
    Momentum: Optional[float] = Field(default=None, description="动量指标")
    
    # 波动性指标
    ATR: Optional[float] = Field(default=None, description="真实波动幅度均值")
    Bollinger_Upper: Optional[float] = Field(default=None, description="布林带上轨")
    Bollinger_Middle: Optional[float] = Field(default=None, description="布林带中轨")
    Bollinger_Lower: Optional[float] = Field(default=None, description="布林带下轨")
    
    # 趋势强度指标
    ADX: Optional[float] = Field(default=None, description="平均趋向指标")
    CCI: Optional[float] = Field(default=None, description="商品通道指数")
    
    # 成交量指标
    OBV: Optional[float] = Field(default=None, description="能量潮指标")
    
    # 收益率
    returns: Optional[float] = Field(default=None, description="当前收益率")
    
    model_config = ConfigDict(extra='allow', populate_by_name=True)


class RiskMetrics(BaseModel):
    """风险指标数据"""
    garchVolatility: Optional[float] = Field(default=None, description="GARCH波动率预测")
    var95: Optional[float] = Field(default=None, description="95%置信度VaR")
    maxDrawdown: Optional[float] = Field(default=None, description="最大回撤")
    sharpeRatio: Optional[float] = Field(default=None, description="夏普比率")


class MarketData(BaseModel):
    """市场数据"""
    symbol: str = Field(..., description="交易对")
    timeframe: TimeFrame = Field(..., description="时间周期")
    timestamp: datetime = Field(..., description="时间戳")
    open: float = Field(..., description="开盘价")
    high: float = Field(..., description="最高价")
    low: float = Field(..., description="最低价")
    close: float = Field(..., description="收盘价")
    volume: float = Field(..., description="成交量")


class AnalysisText(BaseModel):
    """分析文本"""
    signal_evaluation: Optional[str] = Field(default=None, description="信号评估")
    technical_analysis: Optional[str] = Field(default=None, description="技术分析")
    risk_assessment: Optional[str] = Field(default=None, description="风险评估")
    market_outlook: Optional[str] = Field(default=None, description="市场展望")
    raw_text: Optional[str] = Field(default=None, description="原始文本")


class TradingParams(BaseModel):
    """交易参数"""
    optimal_entry: Optional[float] = Field(default=None, description="最佳入场价")
    stop_loss: Optional[float] = Field(default=None, description="止损价")
    take_profit: Optional[float] = Field(default=None, description="止盈价")
    leverage: Optional[int] = Field(default=None, description="杠杆倍数", ge=1)
    position_size_usd: Optional[float] = Field(default=None, description="仓位大小(USD)", ge=0)
    estimated_profit: Optional[float] = Field(default=None, description="预估利润")
    confidence_score: Optional[float] = Field(default=None, description="置信度", ge=0.0, le=1.0)


class BacktestTradeAnalysis(BaseModel):
    """回测交易分析"""
    total_trades: int = Field(default=0, description="总交易数")
    winning_trades: int = Field(default=0, description="盈利交易数")
    losing_trades: int = Field(default=0, description="亏损交易数")
    win_rate: Optional[float] = Field(default=None, description="胜率", ge=0.0, le=1.0)
    avg_profit: Optional[float] = Field(default=None, description="平均盈利")
    avg_loss: Optional[float] = Field(default=None, description="平均亏损")
    profit_factor: Optional[float] = Field(default=None, description="利润因子")
    total_profit: Optional[float] = Field(default=None, description="总利润")
    largest_win: Optional[float] = Field(default=None, description="最大单笔盈利")
    largest_loss: Optional[float] = Field(default=None, description="最大单笔亏损")
    average_trade_duration: Optional[float] = Field(default=None, description="平均持仓时长")


class BacktestResults(BaseModel):
    """回测结果"""
    strategy_score: Optional[float] = Field(default=None, description="策略评分", ge=0.0, le=1.0)
    trade_analysis: Optional[BacktestTradeAnalysis] = Field(default=None, description="交易分析")
    recommendation: Optional[str] = Field(default=None, description="建议")
    warnings: List[str] = Field(default_factory=list, description="警告")


class AnalysisRequest(BaseModel):
    """分析请求"""
    symbol: str = Field(..., description="交易对", example="BTC/USDT:USDT")
    timeframe: TimeFrame = Field(default=TimeFrame.ONE_HOUR, description="时间周期")
    lookback: int = Field(default=1000, description="回看数据量", ge=250)
    accountBalance: float = Field(default=1000.0, description="账户余额", ge=0)
    maxLeverage: float = Field(default=10.0, description="最大杠杆", ge=1)


class AnalysisResponse(BaseModel):
    """分析响应"""
    symbol: str = Field(..., description="交易对")
    timeframe: TimeFrame = Field(..., description="时间周期")
    currentPrice: Optional[float] = Field(default=None, description="当前价格")
    indicators: Optional[TechnicalIndicators] = Field(default=None, description="技术指标")
    riskMetrics: Optional[RiskMetrics] = Field(default=None, description="风险指标")
    gptParams: Optional[TradingParams] = Field(default=None, description="交易参数")
    gptAnalysis: Optional[AnalysisText] = Field(default=None, description="分析文本")
    backtest: Optional[BacktestResults] = Field(default=None, description="回测结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")


class ScanRequest(BaseModel):
    """扫描请求"""
    # 核心扫描参数
    ticker_start_index: int = Field(default=0, description="开始索引", ge=0)
    ticker_end_index: Optional[int] = Field(default=None, description="结束索引(不含)", ge=0)
    timeframe: TimeFrame = Field(default=TimeFrame.ONE_MINUTE, description="时间周期")
    max_tickers: int = Field(default=100, description="最大扫描数量", ge=1)
    top_n: int = Field(default=10, description="返回前N个结果", ge=1)
    
    # 过滤条件
    min_gpt_confidence: float = Field(default=0.65, description="最小GPT置信度", ge=0.0, le=1.0)
    min_backtest_score: float = Field(default=0.60, description="最小回测评分", ge=0.0, le=1.0)
    trade_direction: Optional[TradeDirection] = Field(default=None, description="交易方向")
    
    # BTC趋势过滤
    filter_by_btc_trend: bool = Field(default=True, description="启用BTC趋势过滤")
    
    # 回测过滤
    min_backtest_trades: int = Field(default=15, description="最小回测交易数", ge=0)
    min_backtest_win_rate: float = Field(default=0.52, description="最小胜率", ge=0.0, le=1.0)
    min_backtest_profit_factor: float = Field(default=1.5, description="最小利润因子", ge=0.0)
    
    # 风险过滤
    min_risk_reward_ratio: float = Field(default=1.8, description="最小风险回报比", ge=0.0)
    min_adx: float = Field(default=25.0, description="最小ADX", ge=0.0)
    require_sma_alignment: bool = Field(default=True, description="要求SMA对齐")
    
    # 分析配置
    lookback: int = Field(default=2000, description="回看数据量", ge=250)
    accountBalance: float = Field(default=5000.0, description="账户余额", ge=0)
    maxLeverage: float = Field(default=20.0, description="最大杠杆", ge=1)
    max_concurrent_tasks: int = Field(default=16, description="最大并发任务", ge=1)


class ScanResultItem(BaseModel):
    """扫描结果项"""
    rank: int = Field(..., description="排名")
    symbol: str = Field(..., description="交易对")
    timeframe: TimeFrame = Field(..., description="时间周期")
    currentPrice: Optional[float] = Field(default=None, description="当前价格")
    gptConfidence: Optional[float] = Field(default=None, description="GPT置信度")
    backtestScore: Optional[float] = Field(default=None, description="回测评分")
    combinedScore: Optional[float] = Field(default=None, description="综合评分")
    tradeDirection: Optional[TradeDirection] = Field(default=None, description="交易方向")
    optimalEntry: Optional[float] = Field(default=None, description="最佳入场")
    stopLoss: Optional[float] = Field(default=None, description="止损价")
    takeProfit: Optional[float] = Field(default=None, description="止盈价")
    gptAnalysisSummary: Optional[str] = Field(default=None, description="GPT分析摘要")


class ScanResponse(BaseModel):
    """扫描响应"""
    scan_parameters: ScanRequest = Field(..., description="扫描参数")
    total_tickers_attempted: int = Field(..., description="尝试扫描的交易对数量")
    total_tickers_succeeded: int = Field(..., description="成功扫描的交易对数量")
    ticker_start_index: int = Field(..., description="实际开始索引")
    ticker_end_index: Optional[int] = Field(default=None, description="实际结束索引")
    total_opportunities_found: int = Field(..., description="发现的机会数量")
    top_opportunities: List[ScanResultItem] = Field(..., description="排名前列的机会")
    errors: Dict[str, str] = Field(default_factory=dict, description="错误信息")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")


class TickerRequest(BaseModel):
    """交易对请求"""
    pass


class TickersResponse(BaseModel):
    """交易对响应"""
    tickers: List[str] = Field(..., description="交易对列表")


class LLMHealthCheck(BaseModel):
    """LLM健康检查"""
    status: str = Field(..., description="状态")
    provider: str = Field(..., description="提供商")
    model: str = Field(..., description="模型")
    base_url: str = Field(..., description="基础URL")
    response_time: Optional[float] = Field(default=None, description="响应时间")
    error: Optional[str] = Field(default=None, description="错误信息")


class SystemStatus(BaseModel):
    """系统状态"""
    status: str = Field(..., description="系统状态")
    version: str = Field(..., description="版本")
    uptime: float = Field(..., description="运行时间(秒)")
    llm_status: LLMHealthCheck = Field(..., description="LLM状态")
    memory_usage: Optional[float] = Field(default=None, description="内存使用(MB)")
    active_connections: Optional[int] = Field(default=None, description="活跃连接数")