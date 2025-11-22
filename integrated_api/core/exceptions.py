"""
异常处理模块
"""


class AlphaSeekerError(Exception):
    """AlphaSeeker基础异常"""
    pass


class DataError(AlphaSeekerError):
    """数据相关异常"""
    pass


class LLMServerError(AlphaSeekerError):
    """LLM服务异常"""
    pass


class IndicatorError(AlphaSeekerError):
    """技术指标计算异常"""
    pass


class BacktestError(AlphaSeekerError):
    """回测异常"""
    pass


class ScanError(AlphaSeekerError):
    """扫描异常"""
    pass


class ConfigurationError(AlphaSeekerError):
    """配置异常"""
    pass


class RateLimitError(AlphaSeekerError):
    """速率限制异常"""
    pass


class NetworkError(AlphaSeekerError):
    """网络异常"""
    pass


class ValidationError(AlphaSeekerError):
    """验证异常"""
    pass