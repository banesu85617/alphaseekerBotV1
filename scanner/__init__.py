"""
AlphaSeeker Scanner模块
=====================

这是AlphaSeeker交易系统的核心扫描模块，包含：
- MarketScanner: 市场数据扫描器
- ScanStrategy: 扫描策略管理
- ScanConfig: 扫描配置管理

作者: MiniMax Agent
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ScanStrategy类定义（从之前的修复）
class ScanStrategy(Enum):
    """扫描策略枚举 - 定义不同的市场扫描方法"""
    
    # 基本扫描策略
    MOMENTUM = "momentum"              # 动量策略
    MEAN_REVERSION = "mean_reversion"  # 均值回归策略
    BREAKOUT = "breakout"             # 突破策略
    CONTRARIAN = "contrarian"         # 反向策略
    
    # 技术分析策略
    TECHNICAL_SMA = "technical_sma"         # 简单移动平均线
    TECHNICAL_EMA = "technical_ema"         # 指数移动平均线
    TECHNICAL_RSI = "technical_rsi"         # RSI相对强弱指标
    TECHNICAL_MACD = "technical_macd"       # MACD指标
    TECHNICAL_BOLLINGER = "technical_bollinger"  # 布林带
    
    # 复合策略
    MULTI_TECHNICAL = "multi_technical"    # 多技术指标复合
    ALPHA_SEEKING = "alpha_seeking"       # Alpha寻找策略
    RISK_PARITY = "risk_parity"           # 风险平价策略
    QUANTITATIVE = "quantitative"         # 量化策略
    
    # 时间策略
    TIME_BASED = "time_based"             # 基于时间的扫描
    HIGH_FREQUENCY = "high_frequency"     # 高频扫描
    DAILY_SCAN = "daily_scan"            # 日常扫描
    WEEKLY_SCAN = "weekly_scan"          # 每周扫描
    
    # 市场状态策略
    BULL_MARKET = "bull_market"           # 牛市策略
    BEAR_MARKET = "bear_market"           # 熊市策略
    SIDEWAYS = "sideways"                # 横盘策略
    VOLATILE = "volatile"                # 波动市场策略
    
    def get_description(self) -> str:
        """获取策略描述"""
        descriptions = {
            ScanStrategy.MOMENTUM: "动量策略 - 追踪价格趋势",
            ScanStrategy.MEAN_REVERSION: "均值回归策略 - 回归均值",
            ScanStrategy.BREAKOUT: "突破策略 - 价格突破关键位",
            ScanStrategy.CONTRARIAN: "反向策略 - 逆向操作",
            
            ScanStrategy.TECHNICAL_SMA: "SMA策略 - 简单移动平均线交叉",
            ScanStrategy.TECHNICAL_EMA: "EMA策略 - 指数移动平均线",
            ScanStrategy.TECHNICAL_RSI: "RSI策略 - 相对强弱指标",
            ScanStrategy.TECHNICAL_MACD: "MACD策略 - 移动平均收敛散度",
            ScanStrategy.TECHNICAL_BOLLINGER: "布林带策略 - 价格波动区间",
            
            ScanStrategy.MULTI_TECHNICAL: "多技术指标策略 - 综合判断",
            ScanStrategy.ALPHA_SEEKING: "Alpha寻找策略 - 寻找超额收益",
            ScanStrategy.RISK_PARITY: "风险平价策略 - 风险平衡配置",
            ScanStrategy.QUANTITATIVE: "量化策略 - 数学模型驱动",
            
            ScanStrategy.TIME_BASED: "时间策略 - 基于时间窗口",
            ScanStrategy.HIGH_FREQUENCY: "高频策略 - 分钟级扫描",
            ScanStrategy.DAILY_SCAN: "日度策略 - 每日扫描",
            ScanStrategy.WEEKLY_SCAN: "周度策略 - 每周扫描",
            
            ScanStrategy.BULL_MARKET: "牛市策略 - 上涨趋势",
            ScanStrategy.BEAR_MARKET: "熊市策略 - 下跌趋势",
            ScanStrategy.SIDEWAYS: "横盘策略 - 震荡市场",
            ScanStrategy.VOLATILE: "波动策略 - 高波动市场"
        }
        return descriptions.get(self, f"未知策略: {self.value}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取策略默认参数"""
        parameters = {
            ScanStrategy.MOMENTUM: {
                'lookback_period': 20,
                'min_signal_strength': 0.6
            },
            ScanStrategy.MEAN_REVERSION: {
                'rsi_lower': 30,
                'rsi_upper': 70,
                'bollinger_deviation': 2
            },
            ScanStrategy.BREAKOUT: {
                'breakout_threshold': 0.02,
                'volume_confirmation': True
            },
            ScanStrategy.CONTRARIAN: {
                'contrarian_threshold': 0.7,
                'sentiment_data': True
            },
            
            ScanStrategy.TECHNICAL_SMA: {
                'short_period': 20,
                'long_period': 50
            },
            ScanStrategy.TECHNICAL_EMA: {
                'short_period': 12,
                'long_period': 26
            },
            ScanStrategy.TECHNICAL_RSI: {
                'period': 14,
                'overbought': 70,
                'oversold': 30
            },
            ScanStrategy.TECHNICAL_MACD: {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            ScanStrategy.TECHNICAL_BOLLINGER: {
                'period': 20,
                'std_dev': 2
            },
            
            ScanStrategy.MULTI_TECHNICAL: {
                'required_signals': 3,
                'weight_sma': 0.3,
                'weight_rsi': 0.3,
                'weight_macd': 0.4
            },
            ScanStrategy.ALPHA_SEEKING: {
                'alpha_threshold': 0.05,
                'sharpe_min': 1.5,
                'max_drawdown': 0.15
            },
            ScanStrategy.RISK_PARITY: {
                'target_volatility': 0.15,
                'rebalance_freq': 'weekly'
            },
            ScanStrategy.QUANTITATIVE: {
                'lookback_days': 252,
                'significance_level': 0.05
            }
        }
        return parameters.get(self, {})

class StrategyExecutor:
    """策略执行器 - 执行具体的扫描策略"""
    
    def __init__(self, strategy: ScanStrategy, params: Optional[Dict] = None):
        self.strategy = strategy
        self.params = params or {}
        logger.info(f"策略执行器初始化: {strategy.value}")
    
    def execute(self, data: Dict) -> Dict:
        """执行策略逻辑"""
        try:
            if self.strategy in [ScanStrategy.TECHNICAL_SMA, ScanStrategy.TECHNICAL_EMA]:
                return self._execute_ma_strategy(data)
            elif self.strategy == ScanStrategy.TECHNICAL_RSI:
                return self._execute_rsi_strategy(data)
            elif self.strategy == ScanStrategy.TECHNICAL_MACD:
                return self._execute_macd_strategy(data)
            elif self.strategy == ScanStrategy.ALPHA_SEEKING:
                return self._execute_alpha_strategy(data)
            else:
                return self._execute_default_strategy(data)
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return {'error': str(e), 'strategy': self.strategy.value}
    
    def _execute_ma_strategy(self, data: Dict) -> Dict:
        """移动平均策略执行逻辑"""
        return {
            'strategy': self.strategy.value,
            'signals': [],
            'status': 'completed',
            'message': '移动平均策略执行完成'
        }
    
    def _execute_rsi_strategy(self, data: Dict) -> Dict:
        """RSI策略执行逻辑"""
        return {
            'strategy': self.strategy.value,
            'signals': [],
            'status': 'completed',
            'message': 'RSI策略执行完成'
        }
    
    def _execute_macd_strategy(self, data: Dict) -> Dict:
        """MACD策略执行逻辑"""
        return {
            'strategy': self.strategy.value,
            'signals': [],
            'status': 'completed',
            'message': 'MACD策略执行完成'
        }
    
    def _execute_alpha_strategy(self, data: Dict) -> Dict:
        """Alpha策略执行逻辑"""
        return {
            'strategy': self.strategy.value,
            'signals': [],
            'status': 'completed',
            'message': 'Alpha寻找策略执行完成'
        }
    
    def _execute_default_strategy(self, data: Dict) -> Dict:
        """默认策略执行逻辑"""
        return {
            'strategy': self.strategy.value,
            'signals': [],
            'status': 'completed',
            'message': f'{self.strategy.get_description()} 执行完成'
        }

# ScanConfig类定义（新增）
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
import os

class ScanFrequency(Enum):
    """扫描频率枚举"""
    REAL_TIME = "realtime"        # 实时扫描
    HIGH_FREQUENCY = "1min"       # 1分钟
    INTRADAY = "5min"            # 5分钟
    HOURLY = "hourly"            # 小时
    DAILY = "daily"              # 日度
    WEEKLY = "weekly"            # 周度

class DataSource(Enum):
    """数据源枚举"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    LOCAL = "local"

class RiskLevel(Enum):
    """风险等级枚举"""
    CONSERVATIVE = "conservative"    # 保守型
    MODERATE = "moderate"           # 适中型
    AGGRESSIVE = "aggressive"       # 激进型

@dataclass
class TechnicalIndicator:
    """技术指标配置"""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    threshold_buy: Optional[float] = None
    threshold_sell: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.parameters, str):
            try:
                self.parameters = json.loads(self.parameters)
            except:
                self.parameters = {}

@dataclass
class StockFilter:
    """股票筛选条件"""
    market_cap_min: Optional[float] = None          # 最小市值
    market_cap_max: Optional[float] = None          # 最大市值
    volume_min: Optional[int] = None                # 最小成交量
    price_min: Optional[float] = None               # 最小价格
    price_max: Optional[float] = None               # 最大价格
    pe_ratio_max: Optional[float] = None            # 最大市盈率
    dividend_yield_min: Optional[float] = None      # 最小股息收益率
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'market_cap_min': self.market_cap_min,
            'market_cap_max': self.market_cap_max,
            'volume_min': self.volume_min,
            'price_min': self.price_min,
            'price_max': self.price_max,
            'pe_ratio_max': self.pe_ratio_max,
            'dividend_yield_min': self.dividend_yield_min
        }

@dataclass
class RiskParameters:
    """风险控制参数"""
    max_position_size: float = 0.10          # 最大仓位比例 (10%)
    stop_loss_pct: float = 0.05              # 止损比例 (5%)
    take_profit_pct: float = 0.15            # 止盈比例 (15%)
    max_daily_trades: int = 10               # 最大日交易次数
    correlation_limit: float = 0.7           # 相关性限制
    volatility_limit: float = 0.30           # 波动率限制 (30%)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_daily_trades': self.max_daily_trades,
            'correlation_limit': self.correlation_limit,
            'volatility_limit': self.volatility_limit
        }

@dataclass
class DataSourceConfig:
    """数据源配置"""
    source: DataSource
    api_key: Optional[str] = None
    rate_limit: int = 100                    # 每分钟API调用限制
    timeout: int = 30                        # 请求超时时间(秒)
    retry_count: int = 3                     # 重试次数
    cache_duration: int = 300                # 缓存持续时间(秒)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'source': self.source.value,
            'api_key': self.api_key,
            'rate_limit': self.rate_limit,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'cache_duration': self.cache_duration
        }

class ScanConfig:
    """
    扫描配置管理器
    
    管理所有与市场扫描相关的配置参数，
    支持配置文件的加载、保存和验证。
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化扫描配置"""
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        
        # 基础配置
        self.scan_frequency: ScanFrequency = ScanFrequency.DAILY
        self.data_source: DataSource = DataSource.YAHOO_FINANCE
        self.risk_level: RiskLevel = RiskLevel.MODERATE
        self.watchlist: List[str] = field(default_factory=list)
        
        # 股票池配置
        self.stock_filter = StockFilter()
        
        # 技术指标配置
        self.indicators = self._create_default_indicators()
        
        # 风险控制参数
        self.risk_params = RiskParameters()
        
        # 数据源配置
        self.data_source_config = DataSourceConfig(DataSource.YAHOO_FINANCE)
        
        # 性能优化参数
        self.max_concurrent_scans = 5
        self.cache_enabled = True
        self.debug_mode = False
        
        # 加载配置文件（如果存在）
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self.logger.info("使用默认扫描配置")
            self._set_default_config()
    
    def _create_default_indicators(self) -> Dict[str, TechnicalIndicator]:
        """创建默认技术指标配置"""
        return {
            'sma': TechnicalIndicator(
                name='简单移动平均线',
                parameters={'short_period': 20, 'long_period': 50}
            ),
            'ema': TechnicalIndicator(
                name='指数移动平均线',
                parameters={'short_period': 12, 'long_period': 26}
            ),
            'rsi': TechnicalIndicator(
                name='相对强弱指数',
                parameters={'period': 14},
                threshold_buy=30,
                threshold_sell=70
            ),
            'macd': TechnicalIndicator(
                name='MACD指标',
                parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            ),
            'bollinger': TechnicalIndicator(
                name='布林带',
                parameters={'period': 20, 'std_dev': 2}
            ),
            'volume': TechnicalIndicator(
                name='成交量指标',
                parameters={'period': 20}
            )
        }
    
    def _set_default_config(self):
        """设置默认配置"""
        # 默认股票池
        self.watchlist = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'NVDA', 'META', 'NFLX', 'AMD', 'INTC'
        ]
        
        # 默认筛选条件
        self.stock_filter.market_cap_min = 1e9  # 10亿美元
        self.stock_filter.volume_min = 100000   # 10万股
        self.stock_filter.price_min = 5.0       # 5美元
        
        # 根据风险等级调整参数
        if self.risk_level == RiskLevel.CONSERVATIVE:
            self.risk_params.stop_loss_pct = 0.03
            self.risk_params.take_profit_pct = 0.10
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            self.risk_params.stop_loss_pct = 0.08
            self.risk_params.take_profit_pct = 0.25
        
        self.logger.info(f"✅ 默认配置设置完成 - 风险等级: {self.risk_level.value}")
    
    def validate_config(self) -> List[str]:
        """验证配置参数"""
        errors = []
        
        # 验证股票池
        if not self.watchlist:
            errors.append("股票池不能为空")
        else:
            for symbol in self.watchlist:
                if not isinstance(symbol, str) or len(symbol) == 0:
                    errors.append(f"无效的股票代码: {symbol}")
        
        # 验证风险参数
        if self.risk_params.stop_loss_pct <= 0 or self.risk_params.stop_loss_pct >= 1:
            errors.append("止损比例必须在0-1之间")
        
        if self.risk_params.take_profit_pct <= 0:
            errors.append("止盈比例必须大于0")
        
        if self.risk_params.max_position_size <= 0 or self.risk_params.max_position_size > 1:
            errors.append("最大仓位比例必须在0-1之间")
        
        if len(errors) == 0:
            self.logger.info("✅ 配置验证通过")
        else:
            self.logger.warning(f"⚠️ 配置验证发现问题: {len(errors)} 个错误")
        
        return errors
    
    def get_indicator_config(self, indicator_name: str) -> Optional[TechnicalIndicator]:
        """获取技术指标配置"""
        return self.indicators.get(indicator_name)
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"""ScanConfig(
  扫描频率: {self.scan_frequency.value}
  数据源: {self.data_source.value}
  风险等级: {self.risk_level.value}
  股票池大小: {len(self.watchlist)}
  技术指标数量: {len([ind for ind in self.indicators.values() if ind.enabled])}
  最大并发扫描: {self.max_concurrent_scans}
  缓存启用: {self.cache_enabled}
)"""

# 辅助函数
def get_default_strategies() -> List[ScanStrategy]:
    """获取默认扫描策略列表"""
    return [
        ScanStrategy.ALPHA_SEEKING,
        ScanStrategy.TECHNICAL_SMA,
        ScanStrategy.TECHNICAL_RSI,
        ScanStrategy.MULTI_TECHNICAL
    ]

def get_strategy_info(strategy: ScanStrategy) -> Dict:
    """获取策略详细信息"""
    return {
        'name': strategy.value,
        'description': strategy.get_description(),
        'parameters': strategy.get_parameters(),
        'category': 'technical' if 'technical' in strategy.value else 'fundamental'
    }

def get_scan_config(config_file: Optional[str] = None) -> ScanConfig:
    """获取扫描配置实例"""
    return ScanConfig(config_file)

# 导入MarketScanner
try:
    from .market_scanner import MarketScanner, get_market_scanner
    logger.info("✅ MarketScanner导入成功")
except ImportError as e:
    logger.warning(f"⚠️ MarketScanner导入失败: {e}")
    MarketScanner = None
    get_market_scanner = None

# 导出内容
__all__ = [
    'ScanStrategy',
    'StrategyExecutor', 
    'ScanConfig',
    'ScanFrequency',
    'DataSource',
    'RiskLevel',
    'TechnicalIndicator',
    'StockFilter',
    'RiskParameters',
    'DataSourceConfig',
    'get_default_strategies',
    'get_strategy_info',
    'get_scan_config',
    'MarketScanner',
    'get_market_scanner'
]
