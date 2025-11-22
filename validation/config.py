"""
双重验证机制配置管理
提供配置化的验证流程参数
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json

from .lgbm_filter import LightGBMConfig
from .llm_evaluator import LLMConfig, LLMProvider
from .fusion_algorithm import FusionConfig, FusionStrategy


class LoggingLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TimeoutConfig:
    """超时配置"""
    layer1_timeout: float = 2.0  # 第一层超时时间
    layer2_timeout: float = 5.0  # 第二层超时时间
    total_timeout: float = 10.0  # 总超时时间
    per_request_timeout: float = 8.0  # 单个请求超时时间


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    retry_on_timeout: bool = True
    retry_on_error: bool = True


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_performance_monitoring: bool = True
    enable_error_tracking: bool = True
    metrics_retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'avg_processing_time': 5.0,
        'error_rate': 0.1,
        'timeout_rate': 0.05
    })


@dataclass
class ValidationConfig:
    """双重验证配置主类"""
    
    # 基础配置
    max_concurrent_tasks: int = 16
    max_queue_size: int = 1000
    batch_size: int = 50
    
    # 子组件配置
    lgbm_config: Optional[LightGBMConfig] = None
    llm_config: Optional[LLMConfig] = None
    fusion_config: Optional[FusionConfig] = None
    
    # 系统配置
    timeout_config: Optional[TimeoutConfig] = None
    retry_config: Optional[RetryConfig] = None
    monitoring_config: Optional[MonitoringConfig] = None
    
    # 验证阈值
    min_combined_score: float = 0.5
    min_confidence_level: float = 0.65
    risk_reward_threshold: float = 1.0
    
    # 性能配置
    cache_ttl_seconds: int = 300  # 5分钟
    enable_caching: bool = True
    cache_size_limit: int = 10000
    
    # 异步配置
    worker_pool_size: int = 4
    enable_priority_queue: bool = True
    priority_queue_size: int = 100
    
    # 安全配置
    rate_limit_per_minute: int = 100
    max_requests_per_hour: int = 1000
    enable_authentication: bool = False
    
    # 日志配置
    logging_level: LoggingLevel = LoggingLevel.INFO
    log_to_file: bool = False
    log_file_path: str = "logs/validation.log"
    
    # 环境配置
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        # 设置默认配置
        if self.lgbm_config is None:
            self.lgbm_config = LightGBMConfig()
        if self.llm_config is None:
            self.llm_config = LLMConfig()
        if self.fusion_config is None:
            self.fusion_config = FusionConfig()
        if self.timeout_config is None:
            self.timeout_config = TimeoutConfig()
        if self.retry_config is None:
            self.retry_config = RetryConfig()
        if self.monitoring_config is None:
            self.monitoring_config = MonitoringConfig()
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        errors = []
        
        # 验证超时配置
        if self.timeout_config.layer1_timeout <= 0:
            errors.append("layer1_timeout必须大于0")
        
        if self.timeout_config.layer2_timeout <= 0:
            errors.append("layer2_timeout必须大于0")
        
        if self.timeout_config.total_timeout <= 0:
            errors.append("total_timeout必须大于0")
        
        if self.timeout_config.layer1_timeout + self.timeout_config.layer2_timeout > self.timeout_config.total_timeout:
            errors.append("分层超时时间之和不能超过总超时时间")
        
        # 验证阈值
        if not (0 <= self.min_combined_score <= 1):
            errors.append("min_combined_score必须在0-1之间")
        
        if not (0 <= self.min_confidence_level <= 1):
            errors.append("min_confidence_level必须在0-1之间")
        
        if self.risk_reward_threshold < 0:
            errors.append("risk_reward_threshold必须大于等于0")
        
        # 验证并发配置
        if self.max_concurrent_tasks <= 0:
            errors.append("max_concurrent_tasks必须大于0")
        
        if self.max_queue_size <= 0:
            errors.append("max_queue_size必须大于0")
        
        if errors:
            raise ValueError(f"配置验证失败: {', '.join(errors)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ValidationConfig':
        """从字典创建配置"""
        # 提取子配置
        lgbm_config = LightGBMConfig(**config_dict.get('lgbm_config', {}))
        llm_config = LLMConfig(**config_dict.get('llm_config', {}))
        fusion_config = FusionConfig(**config_dict.get('fusion_config', {}))
        timeout_config = TimeoutConfig(**config_dict.get('timeout_config', {}))
        retry_config = RetryConfig(**config_dict.get('retry_config', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring_config', {}))
        
        # 创建主配置
        return cls(
            **config_dict.get('base_config', {}),
            lgbm_config=lgbm_config,
            llm_config=llm_config,
            fusion_config=fusion_config,
            timeout_config=timeout_config,
            retry_config=retry_config,
            monitoring_config=monitoring_config
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ValidationConfig':
        """从YAML文件加载配置"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ValueError(f"加载YAML配置失败: {str(e)}")

    @classmethod
    def from_json(cls, json_path: str) -> 'ValidationConfig':
        """从JSON文件加载配置"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ValueError(f"加载JSON配置失败: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'base_config': {
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'max_queue_size': self.max_queue_size,
                'batch_size': self.batch_size,
                'min_combined_score': self.min_combined_score,
                'min_confidence_level': self.min_confidence_level,
                'risk_reward_threshold': self.risk_reward_threshold,
                'cache_ttl_seconds': self.cache_ttl_seconds,
                'enable_caching': self.enable_caching,
                'cache_size_limit': self.cache_size_limit,
                'worker_pool_size': self.worker_pool_size,
                'enable_priority_queue': self.enable_priority_queue,
                'priority_queue_size': self.priority_queue_size,
                'rate_limit_per_minute': self.rate_limit_per_minute,
                'max_requests_per_hour': self.max_requests_per_hour,
                'enable_authentication': self.enable_authentication,
                'logging_level': self.logging_level.value,
                'log_to_file': self.log_to_file,
                'log_file_path': self.log_file_path,
                'environment': self.environment,
                'debug_mode': self.debug_mode
            },
            'lgbm_config': self.lgbm_config.__dict__ if hasattr(self.lgbm_config, '__dict__') else {},
            'llm_config': self.llm_config.__dict__ if hasattr(self.llm_config, '__dict__') else {},
            'fusion_config': self.fusion_config.__dict__ if hasattr(self.fusion_config, '__dict__') else {},
            'timeout_config': self.timeout_config.__dict__ if hasattr(self.timeout_config, '__dict__') else {},
            'retry_config': self.retry_config.__dict__ if hasattr(self.retry_config, '__dict__') else {},
            'monitoring_config': self.monitoring_config.__dict__ if hasattr(self.monitoring_config, '__dict__') else {}
        }

    def save_to_yaml(self, yaml_path: str) -> None:
        """保存到YAML文件"""
        config_dict = self.to_dict()
        
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            raise ValueError(f"保存YAML配置失败: {str(e)}")

    def save_to_json(self, json_path: str) -> None:
        """保存到JSON文件"""
        config_dict = self.to_dict()
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"保存JSON配置失败: {str(e)}")

    def create_development_config() -> 'ValidationConfig':
        """创建开发环境配置"""
        return cls(
            max_concurrent_tasks=4,
            max_queue_size=100,
            batch_size=10,
            lgbm_config=LightGBMConfig(
                model_path="models/dev_lgbm_model.txt",
                probability_threshold=0.6,
                batch_size=10
            ),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                base_url="http://localhost:11434",
                model_name="llama2:7b",
                timeout=15.0
            ),
            fusion_config=FusionConfig(
                strategy=FusionStrategy.EQUAL_WEIGHT,
                risk_reward_threshold=0.8
            ),
            timeout_config=TimeoutConfig(
                layer1_timeout=1.0,
                layer2_timeout=3.0,
                total_timeout=5.0
            ),
            environment="development",
            debug_mode=True,
            logging_level=LoggingLevel.DEBUG,
            enable_caching=False  # 开发环境禁用缓存以获得最新结果
        )

    def create_production_config() -> 'ValidationConfig':
        """创建生产环境配置"""
        return cls(
            max_concurrent_tasks=32,
            max_queue_size=5000,
            batch_size=100,
            lgbm_config=LightGBMConfig(
                model_path="models/prod_lgbm_model.txt",
                probability_threshold=0.7,
                batch_size=100
            ),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                base_url="http://ollama:11434",
                model_name="llama2:13b",
                timeout=10.0,
                max_retries=2
            ),
            fusion_config=FusionConfig(
                strategy=FusionStrategy.ADAPTIVE_WEIGHT,
                risk_reward_threshold=1.2,
                performance_history_window=500
            ),
            timeout_config=TimeoutConfig(
                layer1_timeout=2.0,
                layer2_timeout=8.0,
                total_timeout=12.0
            ),
            monitoring_config=MonitoringConfig(
                enable_performance_monitoring=True,
                enable_error_tracking=True,
                metrics_retention_hours=168,  # 7天
                alert_thresholds={
                    'avg_processing_time': 3.0,
                    'error_rate': 0.05,
                    'timeout_rate': 0.02
                }
            ),
            environment="production",
            debug_mode=False,
            logging_level=LoggingLevel.INFO,
            log_to_file=True,
            enable_caching=True,
            cache_ttl_seconds=180,  # 3分钟
            rate_limit_per_minute=1000,
            max_requests_per_hour=10000
        )

    def create_test_config() -> 'ValidationConfig':
        """创建测试环境配置"""
        return cls(
            max_concurrent_tasks=2,
            max_queue_size=50,
            batch_size=5,
            lgbm_config=LightGBMConfig(
                model_path="models/test_lgbm_model.txt",
                probability_threshold=0.5,
                batch_size=5
            ),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                base_url="http://localhost:11434",
                model_name="llama2:3b",
                timeout=5.0
            ),
            fusion_config=FusionConfig(
                strategy=FusionStrategy.EQUAL_WEIGHT,
                risk_reward_threshold=0.5
            ),
            timeout_config=TimeoutConfig(
                layer1_timeout=0.5,
                layer2_timeout=2.0,
                total_timeout=3.0
            ),
            environment="test",
            debug_mode=True,
            logging_level=LoggingLevel.DEBUG,
            enable_caching=False,
            cache_ttl_seconds=60  # 测试环境使用较短缓存
        )

    def validate_environment(self) -> bool:
        """验证当前环境的配置适用性"""
        if self.environment not in ["development", "staging", "production", "test"]:
            return False
        
        if self.environment == "production":
            # 生产环境必须的配置检查
            if not self.log_to_file:
                return False
            if not self.enable_caching:
                return False
            if self.max_concurrent_tasks < 8:
                return False
            if not self.monitoring_config.enable_performance_monitoring:
                return False
        
        return True


# 预定义配置模板
VALIDATION_CONFIG_TEMPLATES = {
    "development": ValidationConfig.create_development_config,
    "production": ValidationConfig.create_production_config,
    "test": ValidationConfig.create_test_config
}


def get_config_template(environment: str) -> ValidationConfig:
    """获取指定环境的配置模板"""
    if environment not in VALIDATION_CONFIG_TEMPLATES:
        raise ValueError(f"不支持的环境: {environment}")
    
    return VALIDATION_CONFIG_TEMPLATES[environment]()


def load_config(
    config_path: Optional[str] = None,
    environment: str = "development"
) -> ValidationConfig:
    """加载配置"""
    if config_path:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return ValidationConfig.from_yaml(config_path)
        elif config_path.endswith('.json'):
            return ValidationConfig.from_json(config_path)
        else:
            raise ValueError("配置文件必须是.yaml/.yml或.json格式")
    else:
        # 使用环境模板
        return get_config_template(environment)