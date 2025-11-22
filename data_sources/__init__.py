"""
数据源模块
提供统一的真实市场数据获取接口
"""

from .multi_source_manager import (
    MultiSourceManager,
    DataSource,
    MarketData,
    data_source_manager
)

__all__ = [
    'MultiSourceManager',
    'DataSource', 
    'MarketData',
    'data_source_manager'
]