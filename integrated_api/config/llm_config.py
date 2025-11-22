"""
LLM配置管理
支持LM Studio、Ollama、AnythingLLM等本地模型服务器
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """支持的LLM提供商"""
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    ANYTHING_LLM = "anything_llm"
    OPENAI = "openai"  # 保留兼容


class LLMConfig(BaseModel):
    """LLM统一配置"""
    provider: LLMProvider = Field(
        default=LLMProvider.LM_STUDIO,
        description="LLM提供商"
    )
    
    # 通用配置
    base_url: Optional[str] = Field(
        default="http://localhost:1234",
        description="LLM服务器基础URL"
    )
    model_name: str = Field(
        default="llama-3-8b-instruct",
        description="模型名称"
    )
    max_tokens: int = Field(
        default=1800,
        description="最大令牌数",
        ge=100,
        le=8000
    )
    temperature: float = Field(
        default=0.3,
        description="温度参数",
        ge=0.0,
        le=2.0
    )
    timeout: float = Field(
        default=30.0,
        description="请求超时（秒）",
        ge=5.0,
        le=300.0
    )
    
    # OpenAI兼容性配置
    api_key: Optional[str] = Field(
        default=None,
        description="API密钥（如果需要）"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="自定义请求头"
    )
    
    # 性能优化配置
    enable_streaming: bool = Field(
        default=False,
        description="启用流式响应"
    )
    batch_size: int = Field(
        default=16,
        description="批处理大小",
        ge=1,
        le=64
    )
    
    # 错误处理配置
    max_retries: int = Field(
        default=3,
        description="最大重试次数",
        ge=0,
        le=10
    )
    retry_delay: float = Field(
        default=1.0,
        description="重试延迟（秒）",
        ge=0.1,
        le=10.0
    )
    
    @property
    def client_config(self) -> Dict[str, Any]:
        """获取客户端配置"""
        config = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "headers": self.headers
        }
        
        # 根据不同提供商调整配置
        if self.provider == LLMProvider.OPENAI:
            if self.api_key:
                config["api_key"] = self.api_key
        elif self.provider in [LLMProvider.LM_STUDIO, LLMProvider.ANYTHING_LLM]:
            # 这些服务通常不需要API密钥
            pass
        elif self.provider == LLMProvider.OLLAMA:
            # Ollama通常使用本地服务器
            if not self.base_url:
                config["base_url"] = "http://localhost:11434"
        
        return config
    
    def validate_config(self) -> bool:
        """验证配置"""
        if not self.base_url:
            return False
        
        if self.provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.ANYTHING_LLM]:
            # 本地服务验证
            return self.base_url.startswith(("http://", "https://"))
        
        return True
    
    class Config:
        env_prefix = "LLM_"
        case_sensitive = False