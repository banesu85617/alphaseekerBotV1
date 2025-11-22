"""
AlphaSeeker-API - 重构版本
集成本地LLM的加密货币技术分析与市场扫描API
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import settings
from .config.llm_config import LLMProvider

# 服务
from .services.llm_service import get_llm_service
from .services.analysis_service import get_analysis_service
from .services.scanner_service import get_scanner_service

# 核心模块
from .core.models import (
    TickerRequest, TickersResponse,
    AnalysisRequest, AnalysisResponse,
    ScanRequest, ScanResponse,
    SystemStatus
)

# 工具
from .utils.validation import ValidationUtils
from .utils.performance import PerformanceOptimizer

# 配置日志
def setup_logging():
    """配置日志"""
    log_level = getattr(logging, settings.api.log_level.upper(), logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(settings.api.log_format)
    
    # 创建处理器
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


# 全局变量
start_time = time.time()
llm_health = {"status": "unknown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger = logging.getLogger(__name__)
    logger.info("🚀 AlphaSeeker-API starting up...")
    
    try:
        # 初始化服务
        llm_service = get_llm_service()
        if llm_service.is_available:
            logger.info(f"✅ LLM service initialized with {llm_service.provider}")
            
            # 健康检查
            health = await llm_service.health_check()
            global llm_health
            llm_health = health
            logger.info(f"LLM health: {health}")
        else:
            logger.warning("❌ LLM service not available")
        
        # 优化垃圾回收
        PerformanceOptimizer.optimize_gc()
        
        logger.info("✅ AlphaSeeker-API startup completed")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # 关闭
    logger.info("🔄 AlphaSeeker-API shutting down...")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    description="AI驱动的加密货币技术分析与市场扫描引擎，支持本地LLM集成",
    version=settings.app_version,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "uptime": time.time() - start_time,
        "docs": "/docs"
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """健康检查"""
    try:
        memory_stats = PerformanceOptimizer.get_memory_stats()
        
        # 获取LLM状态
        llm_service = get_llm_service()
        llm_health_check = await llm_service.health_check()
        
        return SystemStatus(
            status="healthy",
            version=settings.app_version,
            uptime=time.time() - start_time,
            llm_status=llm_health_check,
            memory_usage=memory_stats.get("rss_mb"),
            active_connections=None  # 这里可以添加连接数统计
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/api/crypto/tickers", response_model=TickersResponse)
async def get_tickers(ticker_request: TickerRequest =Body(...)):
    """获取可用的交易对列表"""
    try:
        scanner_service = get_scanner_service()
        tickers = await scanner_service.get_available_symbols()
        
        return TickersResponse(tickers=tickers)
        
    except Exception as e:
        logger.error(f"Failed to get tickers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tickers: {str(e)}")


@app.post("/api/crypto/analyze", response_model=AnalysisResponse)
@PerformanceOptimizer.time_execution
@PerformanceOptimizer.memory_monitor
async def analyze_symbol(request: AnalysisRequest):
    """分析单个交易对"""
    try:
        # 验证输入
        request.symbol = ValidationUtils.validate_ticker(request.symbol)
        request.timeframe = ValidationUtils.validate_timeframe(request.timeframe.value)
        
        logger.info(f"Analyzing {request.symbol} on {request.timeframe}")
        
        # 执行分析
        analysis_service = get_analysis_service()
        result = await analysis_service.analyze_symbol(request)
        
        # 检查是否有错误
        if result.error:
            logger.warning(f"Analysis completed with error for {request.symbol}: {result.error}")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/crypto/scan", response_model=ScanResponse)
@PerformanceOptimizer.time_execution
@PerformanceOptimizer.memory_monitor
async def scan_market(request: ScanRequest):
    """扫描市场"""
    try:
        # 验证输入
        if request.timeframe:
            request.timeframe = ValidationUtils.validate_timeframe(request.timeframe.value)
        
        if request.trade_direction:
            request.trade_direction = ValidationUtils.validate_trade_direction(request.trade_direction.value)
        
        # 限制扫描参数
        request.max_tickers = ValidationUtils.validate_api_request_limit(
            request.max_tickers, default_limit=100
        )
        request.max_concurrent_tasks = ValidationUtils.validate_numeric_range(
            request.max_concurrent_tasks, 1, 32
        )
        
        logger.info(f"Starting market scan with {request.max_concurrent_tasks} concurrent tasks")
        
        # 执行扫描
        scanner_service = get_scanner_service()
        result = await scanner_service.scan_market(request)
        
        # 检查扫描结果
        if result.errors:
            logger.warning(f"Scan completed with errors: {result.errors}")
        
        logger.info(f"Scan completed: {result.total_opportunities_found} opportunities found")
        
        return result
        
    except Exception as e:
        logger.error(f"Market scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market scan failed: {str(e)}")


@app.get("/api/system/status")
async def get_system_status():
    """获取系统状态"""
    try:
        performance_summary = PerformanceOptimizer.get_performance_summary()
        
        scanner_service = get_scanner_service()
        scanner_stats = scanner_service.get_scanner_statistics()
        
        return {
            "system": performance_summary,
            "scanner": scanner_stats,
            "config": {
                "llm_provider": settings.llm.provider.value,
                "llm_model": settings.llm.model_name,
                "llm_base_url": settings.llm.base_url,
                "max_concurrent_tasks": settings.performance.max_concurrent_tasks,
                "batch_processing": settings.performance.batch_processing
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@app.get("/api/system/performance")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        summary = PerformanceOptimizer.get_performance_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@app.get("/api/llm/health")
async def get_llm_health():
    """获取LLM健康状态"""
    try:
        llm_service = get_llm_service()
        health = await llm_service.health_check()
        return health
        
    except Exception as e:
        logger.error(f"Failed to get LLM health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM health: {str(e)}")


# 异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.api.log_level.lower()
    )
