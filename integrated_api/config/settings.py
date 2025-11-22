"""
应用配置管理
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from .llm_config import LLMConfig, LLMProvider


class APISettings(BaseModel):
    """API服务器配置"""
    host: str = Field(default="0.0.0.0", description="服务器主机")
    port: int = Field(default=8000, description="服务器端口", ge=1, le=65535)
    reload: bool = Field(default=False, description="热重载")
    workers: int = Field(default=1, description="工作进程数", ge=1, le=16)
    
    # 安全配置
    cors_origins: list = Field(
        default=["*"],
        description="CORS允许的源"
    )
    
    # 日志配置
    log_level: str = Field(
        default="INFO",
        description="日志级别",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )


class DataSettings(BaseModel):
    """数据源配置"""
    # CCXT配置
    exchange_name: str = Field(default="binanceusdm", description="交易所名称")
    enable_rate_limit: bool = Field(default=True, description="启用速率限制")
    rate_limit: int = Field(default=200, description="速率限制（毫秒）", ge=100, le=5000)
    timeout: int = Field(default=30000, description="请求超时（毫秒）", ge=5000, le=120000)
    
    # 数据缓存配置
    cache_ttl: int = Field(default=300, description="缓存生存时间（秒）", ge=60, le=3600)
    max_cache_size: int = Field(default=1000, description="最大缓存条目数", ge=100, le=10000)


class PerformanceSettings(BaseModel):
    """性能优化配置"""
    # 并发控制
    max_concurrent_tasks: int = Field(default=16, description="最大并发任务数", ge=1, le=64)
    task_timeout: int = Field(default=10, description="任务超时（秒）", ge=5, le=60)
    
    # 批处理
    batch_processing: bool = Field(default=True, description="启用批处理")
    batch_size: int = Field(default=10, description="批处理大小", ge=5, le=50)
    
    # 缓存策略
    enable_cache: bool = Field(default=True, description="启用缓存")
    cache_strategy: str = Field(default="memory", description="缓存策略")
    
    # 内存优化
    gc_threshold: int = Field(default=700, description="垃圾回收阈值", ge=100, le=1000)


class Settings(BaseModel):
    """应用主配置"""
    
    # 子配置
    api: APISettings = Field(default_factory=APISettings)
    data: DataSettings = Field(default_factory=DataSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # 版本信息
    app_name: str = Field(default="AlphaSeeker-API", description="应用名称")
    app_version: str = Field(default="2.0.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    
    # 环境变量
    environment: str = Field(default="development", description="运行环境")
    
    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载配置"""
        settings = cls()
        
        # API配置
        if host := os.getenv("API_HOST"):
            settings.api.host = host
        if port := os.getenv("API_PORT"):
            settings.api.port = int(port)
        if reload := os.getenv("API_RELOAD"):
            settings.api.reload = reload.lower() == "true"
        
        # LLM配置
        if provider := os.getenv("LLM_PROVIDER"):
            settings.llm.provider = LLMProvider(provider)
        if base_url := os.getenv("LLM_BASE_URL"):
            settings.llm.base_url = base_url
        if model_name := os.getenv("LLM_MODEL_NAME"):
            settings.llm.model_name = model_name
        if max_tokens := os.getenv("LLM_MAX_TOKENS"):
            settings.llm.max_tokens = int(max_tokens)
        if temperature := os.getenv("LLM_TEMPERATURE"):
            settings.llm.temperature = float(temperature)
        if timeout := os.getenv("LLM_TIMEOUT"):
            settings.llm.timeout = float(timeout)
        if api_key := os.getenv("LLM_API_KEY"):
            settings.llm.api_key = api_key
        
        # 性能配置
        if max_concurrent := os.getenv("MAX_CONCURRENT_TASKS"):
            settings.performance.max_concurrent_tasks = int(max_concurrent)
        if batch_processing := os.getenv("BATCH_PROCESSING"):
            settings.performance.batch_processing = batch_processing.lower() == "true"
        if batch_size := os.getenv("BATCH_SIZE"):
            settings.performance.batch_size = int(batch_size)
        
        # 其他配置
        if debug := os.getenv("DEBUG"):
            settings.debug = debug.lower() == "true"
        if environment := os.getenv("ENVIRONMENT"):
            settings.environment = environment
        
        return settings
    
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment == "production"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 全局配置实例
settings = Settings.from_env()