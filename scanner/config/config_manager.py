"""
扫描器配置管理模块 (Scanner Configuration Manager)

该模块提供AlphaSeeker系统的配置管理功能，包括：
- 市场扫描配置
- 数据源配置
- 扫描参数管理
- 配置验证和加载
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

class ScanMode(Enum):
    """扫描模式枚举"""
    FAST = "fast"
    DEEP = "deep"
    BALANCED = "balanced"
    REAL_TIME = "real_time"

class DataSource(Enum):
    """数据源枚举"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    LOCAL = "local"

@dataclass
class ScanConfig:
    """扫描配置"""
    name: str
    mode: ScanMode = ScanMode.BALANCED
    data_source: DataSource = DataSource.YAHOO_FINANCE
    symbols: List[str] = field(default_factory=list)
    max_symbols: int = 50
    update_interval: int = 60  # 秒
    enable_indicators: bool = True
    confidence_threshold: float = 0.7
    enable_alerts: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        if not self.symbols:
            # 默认股票列表
            self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    api_key: Optional[str] = None
    base_url: str = "https://query1.finance.yahoo.com"
    rate_limit: int = 100  # 每分钟请求数
    timeout: int = 30  # 请求超时秒数
    retry_attempts: int = 3
    enable_cache: bool = True
    cache_duration: int = 300  # 缓存时间（秒）
    
    def is_configured(self) -> bool:
        """检查数据源是否已正确配置"""
        if self.api_key is not None:
            return len(self.api_key.strip()) > 0
        return True  # 对于免费数据源，API密钥可能不需要

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config/scanner_config.yaml")
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.scan_config = ScanConfig("default")
        self.data_source_configs = {
            DataSource.YAHOO_FINANCE: DataSourceConfig("Yahoo Finance"),
            DataSource.ALPHA_VANTAGE: DataSourceConfig("Alpha Vantage", base_url="https://www.alphavantage.co/query"),
            DataSource.FINNHUB: DataSourceConfig("Finnhub", base_url="https://finnhub.io/api/v1"),
            DataSource.POLYGON: DataSourceConfig("Polygon", base_url="https://api.polygon.io"),
            DataSource.LOCAL: DataSourceConfig("Local", base_url="file://")
        }
        
        # 加载配置
        self.load_config()
    
    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> bool:
        """加载配置文件"""
        config_file = config_file or self.config_path
        
        try:
            if not config_file.exists():
                self.logger.warning(f"配置文件不存在: {config_file}")
                self._create_default_config(config_file)
                return True
            
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)
            
            # 加载扫描配置
            if 'scan' in config_data:
                scan_data = config_data['scan']
                self.scan_config = ScanConfig(
                    name=scan_data.get('name', 'default'),
                    mode=ScanMode(scan_data.get('mode', 'balanced')),
                    data_source=DataSource(scan_data.get('data_source', 'yahoo_finance')),
                    symbols=scan_data.get('symbols', []),
                    max_symbols=scan_data.get('max_symbols', 50),
                    update_interval=scan_data.get('update_interval', 60),
                    enable_indicators=scan_data.get('enable_indicators', True),
                    confidence_threshold=scan_data.get('confidence_threshold', 0.7),
                    enable_alerts=scan_data.get('enable_alerts', False),
                    log_level=scan_data.get('log_level', 'INFO')
                )
            
            # 加载数据源配置
            if 'data_sources' in config_data:
                for source_name, source_data in config_data['data_sources'].items():
                    try:
                        source_enum = DataSource(source_name)
                        self.data_source_configs[source_enum] = DataSourceConfig(
                            name=source_data.get('name', source_name),
                            api_key=source_data.get('api_key'),
                            base_url=source_data.get('base_url', ''),
                            rate_limit=source_data.get('rate_limit', 100),
                            timeout=source_data.get('timeout', 30),
                            retry_attempts=source_data.get('retry_attempts', 3),
                            enable_cache=source_data.get('enable_cache', True),
                            cache_duration=source_data.get('cache_duration', 300)
                        )
                    except ValueError:
                        self.logger.warning(f"未知的数据源类型: {source_name}")
            
            self.logger.info(f"配置文件加载成功: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return False
    
    def save_config(self, config_file: Optional[Union[str, Path]] = None) -> bool:
        """保存配置文件"""
        config_file = config_file or self.config_path
        
        try:
            # 确保目录存在
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'scan': {
                    'name': self.scan_config.name,
                    'mode': self.scan_config.mode.value,
                    'data_source': self.scan_config.data_source.value,
                    'symbols': self.scan_config.symbols,
                    'max_symbols': self.scan_config.max_symbols,
                    'update_interval': self.scan_config.update_interval,
                    'enable_indicators': self.scan_config.enable_indicators,
                    'confidence_threshold': self.scan_config.confidence_threshold,
                    'enable_alerts': self.scan_config.enable_alerts,
                    'log_level': self.scan_config.log_level
                },
                'data_sources': {}
            }
            
            for source, config in self.data_source_configs.items():
                config_data['data_sources'][source.value] = {
                    'name': config.name,
                    'api_key': config.api_key,
                    'base_url': config.base_url,
                    'rate_limit': config.rate_limit,
                    'timeout': config.timeout,
                    'retry_attempts': config.retry_attempts,
                    'enable_cache': config.enable_cache,
                    'cache_duration': config.cache_duration
                }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(config_data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            
            self.logger.info(f"配置文件保存成功: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            return False
    
    def _create_default_config(self, config_file: Path) -> None:
        """创建默认配置文件"""
        try:
            config_data = {
                'scan': {
                    'name': 'alphaseeker_default',
                    'mode': 'balanced',
                    'data_source': 'yahoo_finance',
                    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                    'max_symbols': 50,
                    'update_interval': 60,
                    'enable_indicators': True,
                    'confidence_threshold': 0.7,
                    'enable_alerts': False,
                    'log_level': 'INFO'
                },
                'data_sources': {
                    'yahoo_finance': {
                        'name': 'Yahoo Finance',
                        'base_url': 'https://query1.finance.yahoo.com',
                        'rate_limit': 100,
                        'timeout': 30,
                        'retry_attempts': 3,
                        'enable_cache': True,
                        'cache_duration': 300
                    },
                    'alpha_vantage': {
                        'name': 'Alpha Vantage',
                        'api_key': '',  # 需要用户配置
                        'base_url': 'https://www.alphavantage.co/query',
                        'rate_limit': 5,
                        'timeout': 30,
                        'retry_attempts': 3,
                        'enable_cache': True,
                        'cache_duration': 300
                    }
                }
            }
            
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"创建默认配置文件: {config_file}")
            
        except Exception as e:
            self.logger.error(f"创建默认配置文件失败: {e}")
    
    def get_scan_config(self) -> ScanConfig:
        """获取扫描配置"""
        return self.scan_config
    
    def get_data_source_config(self, source: DataSource) -> Optional[DataSourceConfig]:
        """获取指定数据源配置"""
        return self.data_source_configs.get(source)
    
    def update_scan_config(self, **kwargs) -> bool:
        """更新扫描配置"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.scan_config, key):
                    setattr(self.scan_config, key, value)
            return True
        except Exception as e:
            self.logger.error(f"更新扫描配置失败: {e}")
            return False
    
    def validate_config(self) -> bool:
        """验证配置"""
        try:
            # 验证扫描配置
            assert self.scan_config.name, "扫描配置名称不能为空"
            assert 0 < self.scan_config.max_symbols <= 1000, "最大股票数量必须在1-1000之间"
            assert 0 < self.scan_config.update_interval <= 3600, "更新间隔必须在1-3600秒之间"
            assert 0 <= self.scan_config.confidence_threshold <= 1, "置信度阈值必须在0-1之间"
            
            # 验证数据源配置
            for source, config in self.data_source_configs.items():
                if not config.is_configured():
                    self.logger.warning(f"数据源 {source.value} 未正确配置")
            
            return True
        except AssertionError as e:
            self.logger.error(f"配置验证失败: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'scan_config': {
                'name': self.scan_config.name,
                'mode': self.scan_config.mode.value,
                'data_source': self.scan_config.data_source.value,
                'symbols_count': len(self.scan_config.symbols),
                'max_symbols': self.scan_config.max_symbols,
                'update_interval': self.scan_config.update_interval,
                'confidence_threshold': self.scan_config.confidence_threshold
            },
            'data_sources': {
                source.value: {
                    'name': config.name,
                    'configured': config.is_configured(),
                    'base_url': config.base_url,
                    'rate_limit': config.rate_limit
                } for source, config in self.data_source_configs.items()
            }
        }

# 全局配置管理器实例
_global_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager

def load_config(config_path: Optional[str] = None) -> bool:
    """加载配置"""
    return get_config_manager().load_config(config_path)

def get_scan_config() -> ScanConfig:
    """获取扫描配置"""
    return get_config_manager().get_scan_config()

__version__ = "1.0.0"
__author__ = "AlphaSeeker Team"

__all__ = [
    "ConfigManager",
    "ScanConfig", 
    "DataSourceConfig",
    "ScanMode",
    "DataSource",
    "get_config_manager",
    "load_config",
    "get_scan_config"
]
