"""
验证工具
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ValidationUtils:
    """验证工具类"""
    
    # 交易对格式验证
    TICKER_PATTERN = re.compile(r'^[A-Z]+/[A-Z]+:[A-Z]+$')
    
    # 时间周期验证
    TIMEFRAME_PATTERN = re.compile(r'^(1m|5m|15m|30m|1h|4h|1d|1w)$')
    
    # 数字范围验证
    @staticmethod
    def validate_numeric_range(
        value: Union[int, float],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        field_name: str = "value"
    ) -> Union[int, float]:
        """验证数值范围"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be numeric, got {type(value)}")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"{field_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{field_name} must be <= {max_val}, got {value}")
        
        return value
    
    @staticmethod
    def validate_ticker(symbol: str) -> str:
        """验证交易对格式"""
        if not isinstance(symbol, str):
            raise ValidationError("Symbol must be a string")
        
        symbol = symbol.strip().upper()
        
        if not symbol:
            raise ValidationError("Symbol cannot be empty")
        
        if not ValidationUtils.TICKER_PATTERN.match(symbol):
            raise ValidationError(f"Invalid symbol format: {symbol}. Expected format: BASE/QUOTE:SETTLEMENT")
        
        return symbol
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """验证时间周期"""
        if not isinstance(timeframe, str):
            raise ValidationError("Timeframe must be a string")
        
        timeframe = timeframe.strip().lower()
        
        if not timeframe:
            raise ValidationError("Timeframe cannot be empty")
        
        if not ValidationUtils.TIMEFRAME_PATTERN.match(timeframe):
            raise ValidationError(f"Invalid timeframe: {timeframe}. Supported: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w")
        
        return timeframe
    
    @staticmethod
    def validate_trade_direction(direction: str) -> str:
        """验证交易方向"""
        if not isinstance(direction, str):
            raise ValidationError("Trade direction must be a string")
        
        direction = direction.strip().lower()
        
        valid_directions = ["long", "short", "hold"]
        if direction not in valid_directions:
            raise ValidationError(f"Invalid trade direction: {direction}. Must be one of: {valid_directions}")
        
        return direction
    
    @staticmethod
    def validate_confidence_score(score: Union[int, float, None]) -> Optional[float]:
        """验证置信度分数"""
        if score is None:
            return None
        
        try:
            score = float(score)
        except (ValueError, TypeError):
            raise ValidationError("Confidence score must be numeric")
        
        if score < 0 or score > 1:
            raise ValidationError("Confidence score must be between 0 and 1")
        
        return score
    
    @staticmethod
    def validate_percentage(value: Union[int, float]) -> float:
        """验证百分比值"""
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValidationError("Percentage value must be numeric")
        
        if value < 0 or value > 100:
            raise ValidationError("Percentage must be between 0 and 100")
        
        return value
    
    @staticmethod
    def validate_dict_values(
        data: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """验证字典值"""
        if not isinstance(data, dict):
            raise ValidationError("Data must be a dictionary")
        
        validated_data = {}
        
        # 验证必需字段
        if required_keys:
            for key in required_keys:
                if key not in data:
                    raise ValidationError(f"Missing required field: {key}")
                validated_data[key] = data[key]
        
        # 验证可选字段
        if optional_keys:
            for key in optional_keys:
                if key in data:
                    validated_data[key] = data[key]
        
        # 添加其他字段
        for key, value in data.items():
            if key not in (required_keys or []) + (optional_keys or []):
                validated_data[key] = value
        
        return validated_data
    
    @staticmethod
    def validate_list_length(
        data: List[Any],
        min_length: int = 0,
        max_length: Optional[int] = None,
        field_name: str = "list"
    ) -> List[Any]:
        """验证列表长度"""
        if not isinstance(data, list):
            raise ValidationError(f"{field_name} must be a list")
        
        if len(data) < min_length:
            raise ValidationError(f"{field_name} must have at least {min_length} items, got {len(data)}")
        
        if max_length is not None and len(data) > max_length:
            raise ValidationError(f"{field_name} must have at most {max_length} items, got {len(data)}")
        
        return data
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全除法"""
        if denominator == 0:
            logger.warning(f"Division by zero: {numerator} / {denominator}")
            return default
        
        try:
            return numerator / denominator
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Division error: {e}")
            return default
    
    @staticmethod
    def validate_price_value(price: Union[int, float, None]) -> Optional[float]:
        """验证价格值"""
        if price is None:
            return None
        
        try:
            price = float(price)
        except (ValueError, TypeError):
            raise ValidationError("Price must be numeric")
        
        if price <= 0:
            raise ValidationError("Price must be positive")
        
        if not float('-inf') < price < float('inf'):
            raise ValidationError("Price must be a valid number")
        
        return price
    
    @staticmethod
    def validate_leverage(leverage: Union[int, None]) -> Optional[int]:
        """验证杠杆倍数"""
        if leverage is None:
            return None
        
        try:
            leverage = int(leverage)
        except (ValueError, TypeError):
            raise ValidationError("Leverage must be an integer")
        
        if leverage < 1:
            raise ValidationError("Leverage must be at least 1")
        
        if leverage > 100:  # 合理的杠杆上限
            raise ValidationError("Leverage seems too high")
        
        return leverage
    
    @staticmethod
    def validate_risk_reward_ratio(
        entry: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        min_ratio: float = 1.0
    ) -> bool:
        """验证风险回报比"""
        if not all([entry, stop_loss, take_profit]):
            return False
        
        try:
            entry = float(entry)
            stop_loss = float(stop_loss)
            take_profit = float(take_profit)
        except (ValueError, TypeError):
            return False
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return False
        
        return (reward / risk) >= min_ratio
    
    @staticmethod
    def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
        """清理字符串"""
        if not isinstance(text, str):
            return str(text)
        
        # 移除前后空白
        text = text.strip()
        
        # 限制长度
        if max_length and len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def validate_api_request_limit(limit: int, default_limit: int = 1000) -> int:
        """验证API请求限制"""
        if limit <= 0:
            logger.warning(f"Invalid limit {limit}, using default {default_limit}")
            return default_limit
        
        if limit > default_limit:
            logger.warning(f"Limit {limit} exceeds maximum {default_limit}")
            return default_limit
        
        return limit