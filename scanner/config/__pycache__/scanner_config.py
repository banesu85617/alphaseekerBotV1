"""
配置管理模块
提供可配置的扫描策略和系统参数
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """扫描器配置"""
    # 基础配置
    name: str = "default_scanner"
    version: str = "1.0.0"
    debug: bool = False
    
    # 并行处理配置
    max_workers: int = 10
    batch_size: int = 20
    timeout: float = 30.0
    max_concurrent_scans: int = 5
    
    # 过滤配置
    min_volume: float = 1000000.0
    min_market_cap: float = 10000000.0
    max_volatility: float = 0.8
    max_spread: float = 0.001
    
    # 深度分析配置
    enable_deep_analysis: bool = True
    deep_analysis_threshold: float = 0.7
    max_deep_analysis_pairs: int = 5
    deep_analysis_timeout: float = 60.0
    
    # 缓存配置
    cache_ttl: int = 60
    enable_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    max_memory_cache_size: int = 10000
    
    # 监控配置
    enable_monitoring: bool = True
    monitoring_interval: float = 10.0
    alert_threshold: float = 0.8
    max_alerts: int = 1000
    
    # 策略配置
    priority_strategy: str = "volume"
    filter_strategy: str = "balanced"
    scan_frequency: int = 300  # 秒
    
    # API配置
    exchange_api_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 资源限制
    max_memory_usage_mb: int = 1024
    max_cpu_usage_percent: float = 80.0
    
    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """策略配置"""
    # 优先级策略
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        'volume': 0.3,
        'volatility': 0.2,
        'trend': 0.2,
        'liquidity': 0.15,
        'quality': 0.15
    })
    
    # 过滤阈值
    strict_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_quality_score': 0.8,
        'max_spread': 0.0005,
        'min_volume': 5000000.0,
        'max_volatility': 0.3
    })
    
    balanced_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_quality_score': 0.6,
        'max_spread': 0.001,
        'min_volume': 1000000.0,
        'max_volatility': 0.6
    })
    
    permissive_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_quality_score': 0.4,
        'max_spread': 0.002,
        'min_volume': 500000.0,
        'max_volatility': 0.8
    })
    
    # 评分权重
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        'technical': 0.4,
        'sentiment': 0.2,
        'liquidity': 0.2,
        'momentum': 0.1,
        'risk': 0.1
    })


@dataclass
class AlertConfig:
    """警报配置"""
    # 启用/禁用警报
    enable_alerts: bool = True
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = True
    
    # 邮件配置
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = ""
    to_emails: List[str] = field(default_factory=list)
    
    # Webhook配置
    webhook_url: str = ""
    webhook_timeout: float = 10.0
    
    # 警报阈值
    high_opportunity_threshold: float = 0.9
    performance_alert_threshold: float = 30.0
    error_rate_threshold: float = 0.1
    memory_usage_threshold: float = 1024.0
    
    # 冷却时间
    alert_cooldown: int = 300  # 秒
    
    # 警报规则
    alert_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    """数据库配置"""
    # SQLite配置
    sqlite_path: str = "scanner.db"
    enable_sqlite: bool = True
    
    # Redis配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    # 连接池配置
    max_connections: int = 20
    connection_timeout: float = 10.0
    pool_timeout: float = 30.0
    
    # 备份配置
    enable_backup: bool = True
    backup_interval: int = 3600  # 秒
    backup_path: str = "backups/"


@dataclass
class MonitoringConfig:
    """监控配置"""
    # 启用监控
    enable_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_resource_monitoring: bool = True
    
    # 监控间隔
    metrics_interval: float = 10.0
    system_check_interval: float = 30.0
    cleanup_interval: int = 3600  # 秒
    
    # 性能阈值
    max_scan_duration: float = 30.0
    min_throughput: float = 10.0
    max_memory_usage_mb: float = 1024.0
    max_cpu_usage_percent: float = 80.0
    max_error_rate: float = 0.1
    max_latency_p95: float = 5.0
    
    # 历史记录
    max_metrics_history: int = 1000
    max_performance_history: int = 500
    
    # 导出配置
    enable_metrics_export: bool = True
    export_format: str = "json"
    export_path: str = "metrics/"


@dataclass
class SystemConfig:
    """系统配置"""
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 全局设置
    timezone: str = "UTC"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 特性开关
    enable_async_processing: bool = True
    enable_caching: bool = True
    enable_validation: bool = True
    enable_metrics: bool = True


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        
        # 默认配置
        self.default_config = SystemConfig()
        
        logger.info("ConfigManager initialized")
    
    def load_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            系统配置
        """
        try:
            path = config_path or self.config_path
            
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                self.config = self._dict_to_config(config_data)
                logger.info(f"Loaded configuration from {path}")
            else:
                # 使用默认配置
                self.config = self.default_config
                logger.info("Using default configuration")
            
            # 验证配置
            self.validate_config(self.config)
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # 使用默认配置
            self.config = self.default_config
            return self.config
    
    def save_config(self, config: SystemConfig, config_path: Optional[str] = None):
        """
        保存配置
        
        Args:
            config: 系统配置
            config_path: 保存路径
        """
        try:
            path = config_path or self.config_path
            
            if not path:
                raise ValueError("No configuration path provided")
            
            # 创建目录
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 转换为字典并保存
            config_dict = self._config_to_dict(config)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate_config(self, config: SystemConfig) -> bool:
        """
        验证配置
        
        Args:
            config: 系统配置
            
        Returns:
            True if valid
        """
        try:
            # 验证扫描器配置
            scanner = config.scanner
            if scanner.max_workers <= 0:
                raise ValueError("max_workers must be positive")
            
            if scanner.timeout <= 0:
                raise ValueError("timeout must be positive")
            
            if scanner.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            
            # 验证策略配置
            strategy = config.strategy
            weights_sum = sum(strategy.score_weights.values())
            if abs(weights_sum - 1.0) > 0.01:
                logger.warning(f"Score weights sum {weights_sum:.3f} != 1.0")
            
            # 验证数据库配置
            database = config.database
            if database.max_connections <= 0:
                raise ValueError("max_connections must be positive")
            
            # 验证监控配置
            monitoring = config.monitoring
            if monitoring.metrics_interval <= 0:
                raise ValueError("metrics_interval must be positive")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_config(self) -> SystemConfig:
        """获取当前配置"""
        if self.config is None:
            self.config = self.default_config
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 配置更新字典
        """
        if self.config is None:
            self.config = self.default_config
        
        # 更新配置
        self._update_config_dict(self.config, updates)
        logger.info("Configuration updated")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """字典转换为配置对象"""
        # 转换嵌套配置
        scanner_dict = config_dict.get('scanner', {})
        strategy_dict = config_dict.get('strategy', {})
        alert_dict = config_dict.get('alert', {})
        database_dict = config_dict.get('database', {})
        monitoring_dict = config_dict.get('monitoring', {})
        
        scanner = ScannerConfig(**scanner_dict)
        strategy = StrategyConfig(**strategy_dict)
        alert = AlertConfig(**alert_dict)
        database = DatabaseConfig(**database_dict)
        monitoring = MonitoringConfig(**monitoring_dict)
        
        # 处理特殊字段
        if alert_dict.get('to_emails'):
            alert.to_emails = alert_dict['to_emails']
        
        if strategy_dict.get('alert_rules'):
            strategy.alert_rules = strategy_dict['alert_rules']
        
        return SystemConfig(
            scanner=scanner,
            strategy=strategy,
            alert=alert,
            database=database,
            monitoring=monitoring
        )
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """配置对象转换为字典"""
        return {
            'scanner': asdict(config.scanner),
            'strategy': asdict(config.strategy),
            'alert': asdict(config.alert),
            'database': asdict(config.database),
            'monitoring': asdict(config.monitoring)
        }
    
    def _update_config_dict(self, config: Any, updates: Dict[str, Any]):
        """更新配置字典"""
        for key, value in updates.items():
            if hasattr(config, key):
                if isinstance(value, dict) and isinstance(getattr(config, key), dict):
                    # 嵌套字典更新
                    nested_config = getattr(config, key)
                    self._update_config_dict(nested_config, value)
                else:
                    # 直接属性更新
                    setattr(config, key, value)
            else:
                # 存储到custom_params
                if not hasattr(config, 'custom_params'):
                    config.custom_params = {}
                config.custom_params[key] = value
    
    def create_template_config(self, template_path: str):
        """
        创建配置模板
        
        Args:
            template_path: 模板保存路径
        """
        try:
            template_config = self.default_config
            
            # 保存模板配置
            self.save_config(template_config, template_path)
            
            logger.info(f"Configuration template created at {template_path}")
            
        except Exception as e:
            logger.error(f"Error creating configuration template: {e}")
            raise
    
    def merge_configs(self, base_config: SystemConfig, override_config: SystemConfig) -> SystemConfig:
        """
        合并配置
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        try:
            # 使用override_config的值覆盖base_config
            merged = SystemConfig()
            
            # 合并各个配置部分
            for attr_name in ['scanner', 'strategy', 'alert', 'database', 'monitoring']:
                base_attr = getattr(base_config, attr_name)
                override_attr = getattr(override_config, attr_name)
                
                if override_attr:
                    merged_attr = self._merge_config_objects(base_attr, override_attr)
                    setattr(merged, attr_name, merged_attr)
                else:
                    setattr(merged, attr_name, base_attr)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging configurations: {e}")
            return base_config
    
    def _merge_config_objects(self, base_obj: Any, override_obj: Any) -> Any:
        """合并配置对象"""
        merged_dict = asdict(base_obj)
        override_dict = asdict(override_obj)
        
        # 深度合并字典
        for key, value in override_dict.items():
            if key in merged_dict and isinstance(value, dict) and isinstance(merged_dict[key], dict):
                merged_dict[key].update(value)
            else:
                merged_dict[key] = value
        
        # 重新创建对象
        return type(base_obj)(**merged_dict)


# 预设配置
class PresetConfigs:
    """预设配置"""
    
    @staticmethod
    def high_frequency_config() -> SystemConfig:
        """高频扫描配置"""
        config = SystemConfig()
        config.scanner.max_workers = 20
        config.scanner.batch_size = 50
        config.scanner.scan_frequency = 60  # 1分钟
        config.scanner.timeout = 15.0
        config.scanner.enable_deep_analysis = False  # 关闭深度分析以提高速度
        
        config.strategy.filter_strategy = "strict"
        config.strategy.score_weights = {
            'technical': 0.5,
            'sentiment': 0.1,
            'liquidity': 0.3,
            'momentum': 0.1,
            'risk': 0.0  # 关闭风险因子以提高速度
        }
        
        config.monitoring.max_scan_duration = 15.0
        
        return config
    
    @staticmethod
    def quality_focused_config() -> SystemConfig:
        """质量优先配置"""
        config = SystemConfig()
        config.scanner.max_workers = 5
        config.scanner.batch_size = 10
        config.scanner.enable_deep_analysis = True
        config.scanner.deep_analysis_threshold = 0.6
        
        config.strategy.filter_strategy = "strict"
        config.strategy.score_weights = {
            'technical': 0.3,
            'sentiment': 0.2,
            'liquidity': 0.2,
            'momentum': 0.15,
            'risk': 0.15
        }
        
        config.monitoring.max_scan_duration = 60.0
        
        return config
    
    @staticmethod
    def balanced_config() -> SystemConfig:
        """平衡配置"""
        return SystemConfig()  # 使用默认配置


# 示例使用
if __name__ == "__main__":
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 加载配置
    config = config_manager.load_config()
    
    # 更新配置
    updates = {
        'scanner': {
            'max_workers': 15,
            'batch_size': 25
        },
        'strategy': {
            'filter_strategy': 'strict'
        }
    }
    config_manager.update_config(updates)
    
    # 保存配置
    config_manager.save_config(config, 'scanner_config.json')
    
    # 创建配置模板
    config_manager.create_template_config('config_template.json')
    
    # 测试预设配置
    hf_config = PresetConfigs.high_frequency_config()
    quality_config = PresetConfigs.quality_focused_config()
    
    print("High frequency config:", hf_config.scanner.max_workers)
    print("Quality config:", quality_config.scanner.enable_deep_analysis)