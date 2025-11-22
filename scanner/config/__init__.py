"""
扫描器配置模块
"""

from .config_manager import ConfigManager, ScanConfig, DataSourceConfig, ScanMode, DataSource

__version__ = "1.0.0"
__author__ = "AlphaSeeker Team"

__all__ = [
    "ConfigManager",
    "ScanConfig",
    "DataSourceConfig", 
    "ScanMode",
    "DataSource"
]
