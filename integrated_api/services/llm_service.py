"""
LLM服务
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional
from ..core.llm_interface import LLMInterface
from ..config.llm_config import LLMConfig
from ..core.exceptions import LLMServerError, ValidationError

logger = logging.getLogger(__name__)


class LLMService:
    """LLM服务"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm_interface: Optional[LLMInterface] = None
        self._init_interface()
    
    def _init_interface(self):
        """初始化LLM接口"""
        try:
            self.llm_interface = LLMInterface(self.config)
            logger.info(f"Initialized LLM service with {self.config.provider.value}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM interface: {e}")
            raise LLMServerError(f"LLM initialization failed: {e}")
    
    async def generate_trading_analysis(
        self,
        symbol: str,
        current_price: float,
        technical_indicators: Dict[str, Any],
        signal_direction: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """生成交易分析"""
        try:
            # 验证输入
            self._validate_analysis_input(
                symbol, current_price, technical_indicators, signal_direction
            )
            
            # 构建市场数据
            market_data = {
                "symbol": symbol,
                "current_price": current_price,
                "signal_direction": signal_direction
            }
            
            if market_context:
                market_data.update(market_context)
            
            # 调用LLM生成分析
            result = await self.llm_interface.generate_trading_analysis(
                market_data=market_data,
                technical_indicators=technical_indicators,
                signal_direction=signal_direction
            )
            
            # 后处理结果
            processed_result = self._process_analysis_result(result)
            
            logger.info(f"Generated trading analysis for {symbol}")
            return processed_result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate trading analysis for {symbol}: {e}")
            raise LLMServerError(f"Trading analysis generation failed: {e}")
    
    def _validate_analysis_input(
        self,
        symbol: str,
        current_price: float,
        technical_indicators: Dict[str, Any],
        signal_direction: str
    ):
        """验证分析输入"""
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValidationError("Current price must be a positive number")
        
        if not technical_indicators:
            raise ValidationError("Technical indicators cannot be empty")
        
        if signal_direction not in ['long', 'short', 'hold']:
            raise ValidationError("Signal direction must be 'long', 'short', or 'hold'")
    
    def _process_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """处理分析结果"""
        try:
            # 默认值
            default_result = {
                "signal_evaluation": None,
                "technical_analysis": None,
                "risk_assessment": None,
                "market_outlook": None,
                "trade_direction": "hold",
                "optimal_entry": None,
                "stop_loss": None,
                "take_profit": None,
                "leverage": None,
                "position_size_usd": None,
                "estimated_profit": None,
                "confidence_score": 0.0
            }
            
            # 合并结果
            processed = {**default_result, **result}
            
            # 验证和清理数值字段
            numeric_fields = [
                "optimal_entry", "stop_loss", "take_profit", 
                "position_size_usd", "estimated_profit", "confidence_score"
            ]
            
            for field in numeric_fields:
                if field in processed:
                    try:
                        value = processed[field]
                        if value is not None:
                            processed[field] = float(value)
                    except (ValueError, TypeError):
                        processed[field] = None
            
            # 验证置信度
            if processed["confidence_score"] is not None:
                processed["confidence_score"] = max(0.0, min(1.0, processed["confidence_score"]))
            
            # 验证交易方向
            if processed["trade_direction"] not in ["long", "short", "hold"]:
                processed["trade_direction"] = "hold"
            
            # 如果是hold，清空交易参数
            if processed["trade_direction"] == "hold":
                for field in ["optimal_entry", "stop_loss", "take_profit", "leverage", "position_size_usd", "estimated_profit"]:
                    processed[field] = None
            
            # 验证R/R比例（如果提供了完整参数）
            if all(processed[field] is not None for field in ["optimal_entry", "stop_loss", "take_profit"]):
                risk = abs(processed["optimal_entry"] - processed["stop_loss"])
                reward = abs(processed["take_profit"] - processed["optimal_entry"])
                
                if risk > 0 and reward / risk < 1.0:
                    logger.warning(f"Low risk/reward ratio: {reward/risk:.2f}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process analysis result: {e}")
            return {
                "trade_direction": "hold",
                "confidence_score": 0.0,
                "signal_evaluation": f"Error processing analysis: {e}"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self.llm_interface:
                return {
                    "status": "unavailable",
                    "error": "LLM interface not initialized"
                }
            
            health_result = await self.llm_interface.health_check()
            
            return {
                "status": health_result.get("status", "unknown"),
                "provider": self.config.provider.value,
                "model": self.config.model_name,
                "base_url": self.config.base_url,
                "response_time": health_result.get("response_time"),
                "error": health_result.get("error")
            }
            
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return {
                "status": "error",
                "provider": self.config.provider.value,
                "error": str(e)
            }
    
    def update_config(self, new_config: LLMConfig):
        """更新配置"""
        try:
            self.config = new_config
            self._init_interface()
            logger.info("LLM service configuration updated")
        except Exception as e:
            logger.error(f"Failed to update LLM config: {e}")
            raise LLMServerError(f"Config update failed: {e}")
    
    @property
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.llm_interface is not None and self.llm_interface.is_available
    
    @property
    def provider(self) -> str:
        """获取当前提供商"""
        return self.config.provider.value
    
    @property
    def model(self) -> str:
        """获取当前模型"""
        return self.config.model_name


# 全局LLM服务实例
llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取全局LLM服务实例"""
    global llm_service
    if llm_service is None:
        from ..config.settings import settings
        llm_service = LLMService(settings.llm)
    return llm_service