"""
扫描策略模块
实现智能交易对筛选和优先级排序策略
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """策略类型"""
    PRIORITY = "priority"
    FILTER = "filter"
    COMBINED = "combined"


class FilterLevel(Enum):
    """过滤级别"""
    STRICT = "strict"      # 严格过滤，只保留最优质
    BALANCED = "balanced"  # 平衡过滤
    PERMISSIVE = "permissive"  # 宽松过滤


class PriorityMethod(Enum):
    """优先级方法"""
    VOLUME = "volume"      # 成交量优先
    VOLATILITY = "volatility"  # 波动率优先
    TREND = "trend"        # 趋势优先
    LIQUIDITY = "liquidity"  # 流动性优先
    CUSTOM = "custom"      # 自定义


@dataclass
class FilterConfig:
    """过滤配置"""
    # 基础阈值
    min_volume: float = 1000000.0
    min_market_cap: float = 10000000.0
    min_volume_24h: float = 500000.0
    
    # 技术指标阈值
    min_liquidity_score: float = 0.5
    max_volatility: float = 0.8
    min_price_change_24h: float = -5.0  # 最小24h涨跌幅
    
    # 质量过滤
    min_data_quality: float = 0.7
    max_spread: float = 0.001  # 最大买卖价差
    min_market_hours: int = 6  # 最小交易时间
    
    # 禁止列表
    excluded_symbols: List[str] = None
    excluded_exchanges: List[str] = None
    excluded_categories: List[str] = None
    
    def __post_init__(self):
        if self.excluded_symbols is None:
            self.excluded_symbols = []
        if self.excluded_exchanges is None:
            self.excluded_exchanges = []
        if self.excluded_categories is None:
            self.excluded_categories = []


@dataclass
class PriorityConfig:
    """优先级配置"""
    method: PriorityMethod = PriorityMethod.VOLUME
    weights: Dict[str, float] = None
    custom_scoring_function: Optional[callable] = None
    normalization_method: str = "min_max"  # min_max, z_score, robust
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'volume': 0.3,
                'volatility': 0.2,
                'trend': 0.2,
                'liquidity': 0.15,
                'quality': 0.15
            }


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, config: Any):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用策略"""
        pass
    
    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """验证数据质量"""
        if not data:
            return False
        
        required_fields = ['symbol', 'volume', 'price']
        for item in data:
            if not all(field in item for field in required_fields):
                logger.warning(f"Missing required fields in data item: {item}")
                return False
        
        return True


class PriorityStrategy(BaseStrategy):
    """优先级排序策略"""
    
    def __init__(self, method: PriorityMethod, config: Optional[PriorityConfig] = None):
        super().__init__(config or PriorityConfig(method=method))
        self.method = method
        self.config: PriorityConfig = self.config
    
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用优先级策略"""
        if not self.validate_data(data):
            logger.warning("Invalid data for priority strategy")
            return data
        
        if not data:
            return data
        
        # 计算优先级分数
        scored_data = []
        for item in data:
            try:
                score = self._calculate_priority_score(item)
                item_copy = item.copy()
                item_copy['priority_score'] = score
                item_copy['priority_method'] = self.method.value
                scored_data.append(item_copy)
            except Exception as e:
                logger.error(f"Error calculating priority score for {item.get('symbol', 'unknown')}: {e}")
                continue
        
        # 按优先级分数排序
        scored_data.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"Applied {self.method.value} priority strategy to {len(scored_data)} items")
        return scored_data
    
    def _calculate_priority_score(self, item: Dict[str, Any]) -> float:
        """计算优先级分数"""
        if self.method == PriorityMethod.VOLUME:
            return self._volume_priority(item)
        elif self.method == PriorityMethod.VOLATILITY:
            return self._volatility_priority(item)
        elif self.method == PriorityMethod.TREND:
            return self._trend_priority(item)
        elif self.method == PriorityMethod.LIQUIDITY:
            return self._liquidity_priority(item)
        elif self.method == PriorityMethod.CUSTOM and self.config.custom_scoring_function:
            return self.config.custom_scoring_function(item)
        else:
            return self._default_priority(item)
    
    def _volume_priority(self, item: Dict[str, Any]) -> float:
        """基于成交量的优先级计算"""
        volume = item.get('volume_24h', 0)
        market_cap = item.get('market_cap', 0)
        
        # 归一化成交量
        if volume <= 0:
            return 0.0
        
        normalized_volume = np.log1p(volume) / 20  # 对数缩放
        
        # 考虑市场市值
        market_cap_factor = min(1.0, np.log1p(market_cap) / 25) if market_cap > 0 else 0.5
        
        # 综合分数
        score = normalized_volume * 0.7 + market_cap_factor * 0.3
        
        return min(1.0, score)
    
    def _volatility_priority(self, item: Dict[str, Any]) -> float:
        """基于波动率的优先级计算"""
        volatility = item.get('volatility', 0)
        price_change_24h = abs(item.get('price_change_24h', 0))
        
        # 理想波动率范围：0.02 - 0.15
        if volatility <= 0:
            return 0.0
        
        # 波动率适中的分数更高
        if 0.02 <= volatility <= 0.15:
            volatility_score = 1.0
        elif volatility < 0.02:
            volatility_score = volatility / 0.02 * 0.5  # 低波动率惩罚
        else:  # volatility > 0.15
            volatility_score = max(0.3, 1.0 - (volatility - 0.15) / 0.35)  # 高波动率惩罚
        
        # 考虑价格变化的合理性
        change_score = min(1.0, price_change_24h / 20.0)  # 20%作为基准
        
        # 综合分数
        score = volatility_score * 0.7 + change_score * 0.3
        
        return min(1.0, score)
    
    def _trend_priority(self, item: Dict[str, Any]) -> float:
        """基于趋势的优先级计算"""
        price_change_24h = item.get('price_change_24h', 0)
        volume_trend = item.get('volume_trend', 0)
        trend_strength = item.get('trend_strength', 0.5)
        
        # 价格趋势分数
        if price_change_24h > 0:
            price_score = min(1.0, price_change_24h / 10.0)  # 10%作为基准
        else:
            price_score = max(0.0, 1.0 + price_change_24h / 20.0)  # 下跌的惩罚较小
        
        # 成交量趋势分数
        volume_score = (volume_trend + 1) / 2  # 从[-1,1]归一化到[0,1]
        
        # 趋势强度分数
        strength_score = trend_strength
        
        # 综合分数
        score = price_score * 0.5 + volume_score * 0.3 + strength_score * 0.2
        
        return min(1.0, score)
    
    def _liquidity_priority(self, item: Dict[str, Any]) -> float:
        """基于流动性的优先级计算"""
        bid_ask_spread = item.get('bid_ask_spread', 0)
        volume_24h = item.get('volume_24h', 0)
        order_book_depth = item.get('order_book_depth', 0)
        
        # 买卖价差分数（越小越好）
        if bid_ask_spread <= 0:
            spread_score = 0.5
        else:
            spread_score = max(0.0, 1.0 - bid_ask_spread / 0.01)  # 1%作为基准
        
        # 成交量分数
        volume_score = min(1.0, np.log1p(volume_24h) / 18)
        
        # 订单簿深度分数
        depth_score = min(1.0, order_book_depth / 1000000) if order_book_depth > 0 else 0.5
        
        # 综合分数
        score = spread_score * 0.5 + volume_score * 0.3 + depth_score * 0.2
        
        return min(1.0, score)
    
    def _default_priority(self, item: Dict[str, Any]) -> float:
        """默认优先级计算"""
        volume_score = self._volume_priority(item)
        volatility_score = self._volatility_priority(item)
        trend_score = self._trend_priority(item)
        liquidity_score = self._liquidity_priority(item)
        
        # 使用配置的权重
        weights = self.config.weights
        score = (
            volume_score * weights.get('volume', 0.25) +
            volatility_score * weights.get('volatility', 0.25) +
            trend_score * weights.get('trend', 0.25) +
            liquidity_score * weights.get('liquidity', 0.25)
        )
        
        return min(1.0, score)


class FilterStrategy(BaseStrategy):
    """过滤策略"""
    
    def __init__(self, level: FilterLevel, config: Optional[FilterConfig] = None):
        super().__init__(config or FilterConfig())
        self.level = level
        self.config: FilterConfig = self.config
    
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用过滤策略"""
        if not data:
            return data
        
        logger.info(f"Applying {self.level.value} filter to {len(data)} items")
        
        # 逐级过滤
        filtered_data = data.copy()
        
        # 1. 基础质量过滤
        filtered_data = self._basic_quality_filter(filtered_data)
        
        # 2. 阈值过滤
        filtered_data = self._threshold_filter(filtered_data)
        
        # 3. 禁止列表过滤
        filtered_data = self._blacklist_filter(filtered_data)
        
        # 4. 高级过滤
        filtered_data = self._advanced_filter(filtered_data)
        
        logger.info(f"Filter reduced {len(data)} items to {len(filtered_data)} items")
        return filtered_data
    
    def filter_symbols(
        self,
        symbols: List[str],
        allowed: Optional[List[str]] = None,
        excluded: Optional[List[str]] = None
    ) -> List[str]:
        """过滤交易对列表"""
        result = symbols.copy()
        
        # 应用允许列表
        if allowed:
            result = [s for s in result if s in allowed]
        
        # 应用排除列表
        if excluded:
            result = [s for s in result if s not in excluded]
        
        return result
    
    def _basic_quality_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基础质量过滤"""
        filtered = []
        
        for item in data:
            symbol = item.get('symbol', '')
            
            # 检查必需字段
            if not self._has_required_fields(item):
                continue
            
            # 检查数据有效性
            if not self._is_valid_data(item):
                continue
            
            # 添加质量评分
            item['quality_score'] = self._calculate_quality_score(item)
            filtered.append(item)
        
        # 按质量排序，只保留质量好的
        threshold = self._get_quality_threshold()
        filtered = [item for item in filtered if item['quality_score'] >= threshold]
        
        return filtered
    
    def _threshold_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """阈值过滤"""
        filtered = []
        
        for item in data:
            # 成交量检查
            volume = item.get('volume_24h', 0)
            if volume < self.config.min_volume_24h:
                continue
            
            # 市值检查
            market_cap = item.get('market_cap', 0)
            if market_cap < self.config.min_market_cap:
                continue
            
            # 波动率检查
            volatility = item.get('volatility', 0)
            if volatility > self.config.max_volatility:
                continue
            
            # 价格变化检查
            price_change = item.get('price_change_24h', 0)
            if price_change < self.config.min_price_change_24h:
                continue
            
            # 买卖价差检查
            spread = item.get('bid_ask_spread', 0)
            if spread > self.config.max_spread:
                continue
            
            filtered.append(item)
        
        return filtered
    
    def _blacklist_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """禁止列表过滤"""
        excluded_symbols = set(self.config.excluded_symbols)
        excluded_exchanges = set(self.config.excluded_exchanges)
        excluded_categories = set(self.config.excluded_categories)
        
        filtered = []
        
        for item in data:
            symbol = item.get('symbol', '')
            exchange = item.get('exchange', '')
            category = item.get('category', '')
            
            # 排除匹配的交易对
            if symbol in excluded_symbols:
                continue
            
            # 排除匹配的交易所
            if exchange in excluded_exchanges:
                continue
            
            # 排除匹配的分类
            if category in excluded_categories:
                continue
            
            filtered.append(item)
        
        return filtered
    
    def _advanced_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """高级过滤"""
        # 根据过滤级别应用不同的标准
        if self.level == FilterLevel.STRICT:
            return self._strict_filter(data)
        elif self.level == FilterLevel.BALANCED:
            return self._balanced_filter(data)
        else:  # PERMISSIVE
            return self._permissive_filter(data)
    
    def _strict_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """严格过滤"""
        filtered = []
        
        for item in data:
            # 严格的质量标准
            if item.get('quality_score', 0) < 0.8:
                continue
            
            # 严格的流动性标准
            if item.get('bid_ask_spread', 1) > 0.0005:  # 0.05%
                continue
            
            # 严格的数据完整性
            if not self._has_complete_data(item):
                continue
            
            filtered.append(item)
        
        return filtered
    
    def _balanced_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """平衡过滤"""
        filtered = []
        
        for item in data:
            # 平衡的质量标准
            if item.get('quality_score', 0) < 0.6:
                continue
            
            # 平衡的流动性标准
            if item.get('bid_ask_spread', 1) > 0.001:  # 0.1%
                continue
            
            filtered.append(item)
        
        return filtered
    
    def _permissive_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """宽松过滤"""
        filtered = []
        
        for item in data:
            # 宽松的质量标准
            if item.get('quality_score', 0) < 0.4:
                continue
            
            # 宽松的流动性标准
            if item.get('bid_ask_spread', 1) > 0.002:  # 0.2%
                continue
            
            filtered.append(item)
        
        return filtered
    
    def _has_required_fields(self, item: Dict[str, Any]) -> bool:
        """检查必需字段"""
        required = ['symbol', 'volume_24h', 'price']
        return all(field in item for field in required)
    
    def _is_valid_data(self, item: Dict[str, Any]) -> bool:
        """检查数据有效性"""
        # 检查数值有效性
        volume = item.get('volume_24h', 0)
        price = item.get('price', 0)
        
        if volume <= 0 or price <= 0:
            return False
        
        # 检查极端值
        if price < 0.0001 or price > 1000000:
            return False
        
        return True
    
    def _has_complete_data(self, item: Dict[str, Any]) -> bool:
        """检查数据完整性"""
        optional_fields = [
            'volatility', 'bid_ask_spread', 'market_cap',
            'volume_trend', 'trend_strength', 'order_book_depth'
        ]
        missing_count = sum(1 for field in optional_fields if field not in item)
        return missing_count <= 2  # 允许最多缺失2个可选字段
    
    def _calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """计算质量评分"""
        score = 0.0
        
        # 数据完整性 (30%)
        required_fields = ['symbol', 'volume_24h', 'price']
        completeness = sum(1 for field in required_fields if field in item) / len(required_fields)
        score += completeness * 0.3
        
        # 数据合理性 (40%)
        rationality_factor = 1.0
        
        # 成交量合理性
        volume = item.get('volume_24h', 0)
        if volume > 1000000:  # 100万以上
            rationality_factor += 0.2
        
        # 价格合理性
        price = item.get('price', 0)
        if 0.01 <= price <= 100000:  # 价格在合理范围
            rationality_factor += 0.2
        
        score += min(1.0, rationality_factor) * 0.4
        
        # 实时性 (30%)
        timestamp = item.get('timestamp')
        if timestamp:
            # 检查数据的新鲜度
            score += 0.3
        
        return min(1.0, score)
    
    def _get_quality_threshold(self) -> float:
        """获取质量阈值"""
        thresholds = {
            FilterLevel.STRICT: 0.7,
            FilterLevel.BALANCED: 0.5,
            FilterLevel.PERMISSIVE: 0.3
        }
        return thresholds.get(self.level, 0.5)


class CombinedStrategy(BaseStrategy):
    """组合策略"""
    
    def __init__(
        self,
        priority_strategy: PriorityStrategy,
        filter_strategy: FilterStrategy,
        apply_sequence: str = "filter_then_prioritize"  # filter_then_prioritize, prioritize_then_filter
    ):
        super().__init__(None)
        self.priority_strategy = priority_strategy
        self.filter_strategy = filter_strategy
        self.apply_sequence = apply_sequence
    
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用组合策略"""
        if not data:
            return data
        
        logger.info(f"Applying combined strategy with sequence: {self.apply_sequence}")
        
        if self.apply_sequence == "filter_then_prioritize":
            # 先过滤再排序
            filtered_data = self.filter_strategy.apply(data)
            prioritized_data = self.priority_strategy.apply(filtered_data)
        else:  # prioritize_then_filter
            # 先排序再过滤
            prioritized_data = self.priority_strategy.apply(data)
            filtered_data = self.filter_strategy.apply(prioritized_data)
        
        return filtered_data


# 策略工厂
class StrategyFactory:
    """策略工厂"""
    
    @staticmethod
    def create_priority_strategy(
        method: str,
        config: Optional[PriorityConfig] = None
    ) -> PriorityStrategy:
        """创建优先级策略"""
        try:
            priority_method = PriorityMethod(method)
            return PriorityStrategy(priority_method, config)
        except ValueError:
            logger.warning(f"Unknown priority method: {method}, using default VOLUME")
            return PriorityStrategy(PriorityMethod.VOLUME, config)
    
    @staticmethod
    def create_filter_strategy(
        level: str,
        config: Optional[FilterConfig] = None
    ) -> FilterStrategy:
        """创建过滤策略"""
        try:
            filter_level = FilterLevel(level)
            return FilterStrategy(filter_level, config)
        except ValueError:
            logger.warning(f"Unknown filter level: {level}, using default BALANCED")
            return FilterStrategy(FilterLevel.BALANCED, config)
    
    @staticmethod
    def create_combined_strategy(
        priority_method: str,
        filter_level: str,
        sequence: str = "filter_then_prioritize"
    ) -> CombinedStrategy:
        """创建组合策略"""
        priority_strategy = StrategyFactory.create_priority_strategy(priority_method)
        filter_strategy = StrategyFactory.create_filter_strategy(filter_level)
        
        return CombinedStrategy(
            priority_strategy=priority_strategy,
            filter_strategy=filter_strategy,
            apply_sequence=sequence
        )


# 示例使用
if __name__ == "__main__":
    # 测试数据
    test_data = [
        {
            'symbol': 'BTCUSDT',
            'volume_24h': 10000000,
            'price': 50000,
            'volatility': 0.05,
            'price_change_24h': 2.5,
            'market_cap': 900000000,
            'bid_ask_spread': 0.0002
        },
        {
            'symbol': 'ETHUSDT',
            'volume_24h': 5000000,
            'price': 3000,
            'volatility': 0.08,
            'price_change_24h': 3.2,
            'market_cap': 350000000,
            'bid_ask_spread': 0.0003
        }
    ]
    
    # 创建策略
    priority_strategy = StrategyFactory.create_priority_strategy("volume")
    filter_strategy = StrategyFactory.create_filter_strategy("balanced")
    combined_strategy = StrategyFactory.create_combined_strategy("volume", "balanced")
    
    # 应用策略
    print("Original data:", len(test_data))
    
    prioritized = priority_strategy.apply(test_data)
    print("After priority:", len(prioritized))
    
    filtered = filter_strategy.apply(test_data)
    print("After filter:", len(filtered))
    
    combined = combined_strategy.apply(test_data)
    print("After combined:", len(combined))