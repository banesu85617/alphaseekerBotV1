"""
AlphaSeeker-Bot风险管理模块
实现止损止盈、仓位管理和风险控制机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass
from enum import Enum

from ..config.settings import RISK_CONFIG


class PositionSide(Enum):
    """仓位方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class RiskMetrics:
    """风险指标"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    var_95: float = 0.0  # 95% VaR
    position_size: float = 0.0
    leverage: float = 1.0


@dataclass
class Position:
    """仓位信息"""
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    entry_time: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0.0
    unrealized_pnl: float = 0.0


class RiskManager:
    """
    风险管理器
    
    基于分析文档实现：
    - 固定止盈止损(0.4%/0.4%)
    - 动态风控(基于波动率)
    - 仓位管理
    - 风险指标监控
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化风险管理器
        
        Args:
            config: 风险管理配置
        """
        self.config = config or RISK_CONFIG
        
        # 风控参数
        self.take_profit_pct = self.config.get("TAKE_PROFIT_PCT", 0.004)
        self.stop_loss_pct = self.config.get("STOP_LOSS_PCT", 0.004)
        self.transaction_cost_pct = self.config.get("TRANSACTION_COST_PCT", 0.0005)
        
        # 仓位限制
        self.max_position_size = self.config.get("MAX_POSITION_SIZE", 1.0)
        self.max_daily_loss = self.config.get("MAX_DAILY_LOSS", 0.02)
        self.max_drawdown = self.config.get("MAX_DRAWDOWN", 0.1)
        
        # 动态风控
        self.volatility_based_sl = self.config.get("VOLATILITY_BASED_SL", True)
        self.volatility_multiplier = self.config.get("VOLATILITY_MULTIPLIER", 2.0)
        
        # 状态跟踪
        self.current_position = None
        self.position_history = []
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.equity_curve = []
        
        # 风险指标
        self.risk_metrics = RiskMetrics()
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, signal_strength: float, 
                              account_balance: float,
                              price: float,
                              volatility: float = None) -> float:
        """
        计算仓位大小
        
        Args:
            signal_strength: 信号强度 [0, 1]
            account_balance: 账户余额
            price: 当前价格
            volatility: 当前波动率
            
        Returns:
            建议仓位大小
        """
        # 基础仓位大小
        base_size = min(signal_strength, 1.0) * self.max_position_size
        
        # 基于波动率调整
        if self.volatility_based_sl and volatility is not None:
            # 波动率越高，仓位越小
            vol_adjustment = 1.0 / (1.0 + self.volatility_multiplier * volatility)
            base_size *= vol_adjustment
            
        # 风险预算调整
        if self.risk_metrics.current_drawdown > 0:
            drawdown_factor = max(0.1, 1.0 - self.risk_metrics.current_drawdown * 2)
            base_size *= drawdown_factor
            
        # 确保不超过最大仓位
        final_size = min(base_size, self.max_position_size)
        
        self.logger.debug(f"仓位计算: 信号强度={signal_strength:.3f}, 建议仓位={final_size:.3f}")
        
        return final_size
    
    def calculate_stop_loss(self, entry_price: float, side: PositionSide, 
                          volatility: float = None) -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            side: 仓位方向
            volatility: 当前波动率
            
        Returns:
            止损价格
        """
        if self.volatility_based_sl and volatility is not None:
            # 动态止损基于波动率
            vol_stop_pct = self.volatility_multiplier * volatility
            stop_pct = min(self.stop_loss_pct * 2, vol_stop_pct)  # 不超过固定止损的2倍
        else:
            # 固定止损
            stop_pct = self.stop_loss_pct
            
        if side == PositionSide.LONG:
            stop_price = entry_price * (1 - stop_pct)
        else:  # SHORT
            stop_price = entry_price * (1 + stop_pct)
            
        return stop_price
    
    def calculate_take_profit(self, entry_price: float, side: PositionSide,
                            stop_loss: float = None) -> float:
        """
        计算止盈价格
        
        Args:
            entry_price: 入场价格
            side: 仓位方向
            stop_loss: 止损价格（用于计算风险回报比）
            
        Returns:
            止盈价格
        """
        if stop_loss is None:
            stop_loss = self.calculate_stop_loss(entry_price, side)
            
        # 计算风险距离
        if side == PositionSide.LONG:
            risk_distance = entry_price - stop_loss
            reward_distance = risk_distance * 2  # 2:1的风险回报比
            take_profit_price = entry_price + reward_distance
        else:  # SHORT
            risk_distance = stop_loss - entry_price
            reward_distance = risk_distance * 2
            take_profit_price = entry_price - reward_distance
            
        return take_profit_price
    
    def can_open_position(self, signal: int, current_price: float,
                         account_balance: float, position_size: float) -> Dict:
        """
        检查是否可以开仓
        
        Args:
            signal: 交易信号 (1:买入, -1:卖出, 0:持有)
            current_price: 当前价格
            account_balance: 账户余额
            position_size: 建议仓位大小
            
        Returns:
            检查结果字典
        """
        result = {
            "can_open": True,
            "reason": "",
            "risk_level": "LOW"
        }
        
        # 检查是否已有仓位
        if self.current_position is not None:
            # 如果当前有相同方向的仓位，不再开仓
            if ((signal == 1 and self.current_position.side == PositionSide.LONG) or
                (signal == -1 and self.current_position.side == PositionSide.SHORT)):
                result["can_open"] = False
                result["reason"] = "already_has_position"
                return result
                
        # 检查风险限制
        if self.risk_metrics.current_drawdown >= self.max_drawdown:
            result["can_open"] = False
            result["reason"] = "max_drawdown_exceeded"
            result["risk_level"] = "HIGH"
            return result
            
        if self.daily_pnl <= -account_balance * self.max_daily_loss:
            result["can_open"] = False
            result["reason"] = "daily_loss_limit"
            result["risk_level"] = "HIGH"
            return result
            
        # 检查仓位大小
        if position_size > self.max_position_size:
            result["can_open"] = False
            result["reason"] = "position_size_exceeded"
            result["risk_level"] = "MEDIUM"
            return result
            
        # 检查信号强度
        if abs(signal) < 0.5:  # 信号不够强
            result["can_open"] = False
            result["reason"] = "weak_signal"
            result["risk_level"] = "LOW"
            return result
            
        return result
    
    def open_position(self, signal: int, current_price: float,
                     position_size: float, account_balance: float,
                     current_time: float, volatility: float = None) -> Dict:
        """
        开仓
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            position_size: 仓位大小
            account_balance: 账户余额
            current_time: 当前时间
            volatility: 当前波动率
            
        Returns:
            开仓结果字典
        """
        # 检查是否可以开仓
        check_result = self.can_open_position(signal, current_price, 
                                            account_balance, position_size)
        
        if not check_result["can_open"]:
            return {
                "success": False,
                "reason": check_result["reason"],
                "risk_level": check_result["risk_level"]
            }
            
        # 确定仓位方向
        if signal == 1:
            side = PositionSide.LONG
        elif signal == -1:
            side = PositionSide.SHORT
        else:
            return {"success": False, "reason": "invalid_signal"}
            
        # 计算止损止盈
        stop_loss = self.calculate_stop_loss(current_price, side, volatility)
        take_profit = self.calculate_take_profit(current_price, side, stop_loss)
        
        # 创建仓位
        position = Position(
            side=side,
            size=position_size,
            entry_price=current_price,
            current_price=current_price,
            entry_time=current_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=0.0,
            unrealized_pnl=0.0
        )
        
        self.current_position = position
        
        self.logger.info(f"开仓: {side.value}, 仓位: {position_size:.3f}, "
                        f"价格: {current_price:.6f}, 止损: {stop_loss:.6f}, "
                        f"止盈: {take_profit:.6f}")
        
        return {
            "success": True,
            "position": position,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_level": check_result["risk_level"]
        }
    
    def update_position(self, current_price: float, current_time: float,
                       market_data: Dict = None) -> Dict:
        """
        更新仓位状态
        
        Args:
            current_price: 当前价格
            current_time: 当前时间
            market_data: 市场数据（用于计算波动率等）
            
        Returns:
            更新结果字典
        """
        if self.current_position is None:
            return {"status": "no_position"}
            
        position = self.current_position
        position.current_price = current_price
        
        # 计算未实现盈亏
        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
        else:  # SHORT
            position.unrealized_pnl = (position.entry_price - current_price) * position.size
            
        # 考虑交易成本
        position.unrealized_pnl -= self.transaction_cost_pct * position.entry_price * position.size
        
        # 检查止损止盈条件
        trigger_result = self._check_exit_conditions(position, market_data)
        
        return {
            "status": "updated",
            "position": position,
            "exit_triggered": trigger_result["exit_triggered"],
            "exit_reason": trigger_result["exit_reason"],
            "unrealized_pnl": position.unrealized_pnl
        }
    
    def _check_exit_conditions(self, position: Position, 
                             market_data: Dict = None) -> Dict:
        """
        检查出场条件
        
        Args:
            position: 仓位对象
            market_data: 市场数据
            
        Returns:
            出场检查结果
        """
        current_price = position.current_price
        
        # 检查止损
        if position.stop_loss is not None:
            if ((position.side == PositionSide.LONG and current_price <= position.stop_loss) or
                (position.side == PositionSide.SHORT and current_price >= position.stop_loss)):
                return {
                    "exit_triggered": True,
                    "exit_reason": "stop_loss",
                    "exit_price": current_price
                }
                
        # 检查止盈
        if position.take_profit is not None:
            if ((position.side == PositionSide.LONG and current_price >= position.take_profit) or
                (position.side == PositionSide.SHORT and current_price <= position.take_profit)):
                return {
                    "exit_triggered": True,
                    "exit_reason": "take_profit",
                    "exit_price": current_price
                }
                
        # 时间止损（可选）
        if market_data and "time_in_position" in market_data:
            max_holding_time = 3600  # 1小时
            if market_data["time_in_position"] > max_holding_time:
                return {
                    "exit_triggered": True,
                    "exit_reason": "time_stop",
                    "exit_price": current_price
                }
                
        return {"exit_triggered": False, "exit_reason": None}
    
    def close_position(self, exit_price: float, current_time: float,
                      exit_reason: str = "manual") -> Dict:
        """
        平仓
        
        Args:
            exit_price: 出场价格
            current_time: 当前时间
            exit_reason: 出场原因
            
        Returns:
            平仓结果字典
        """
        if self.current_position is None:
            return {"success": False, "reason": "no_position_to_close"}
            
        position = self.current_position
        
        # 计算最终盈亏
        if position.side == PositionSide.LONG:
            final_pnl = (exit_price - position.entry_price) * position.size
        else:  # SHORT
            final_pnl = (position.entry_price - exit_price) * position.size
            
        # 扣除交易成本
        final_pnl -= self.transaction_cost_pct * position.entry_price * position.size
        
        position.pnl = final_pnl
        
        # 更新累计统计
        self.daily_pnl += final_pnl
        self.position_history.append(position)
        
        # 保存历史记录的副本（防止引用问题）
        self.position_history = self.position_history[-1000:]  # 保留最近1000条记录
        
        # 清空当前仓位
        self.current_position = None
        
        self.logger.info(f"平仓: {exit_reason}, PnL: {final_pnl:.6f}, "
                        f"累计日PnL: {self.daily_pnl:.6f}")
        
        return {
            "success": True,
            "position": position,
            "final_pnl": final_pnl,
            "exit_reason": exit_reason,
            "daily_pnl": self.daily_pnl
        }
    
    def update_risk_metrics(self, current_equity: float):
        """
        更新风险指标
        
        Args:
            current_equity: 当前权益
        """
        # 更新权益曲线
        self.equity_curve.append(current_equity)
        if len(self.equity_curve) > 10000:
            self.equity_curve = self.equity_curve[-10000:]
            
        # 更新峰值权益
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # 计算当前回撤
        if self.peak_equity > 0:
            self.risk_metrics.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
        # 更新最大回撤
        self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, 
                                           self.risk_metrics.current_drawdown)
        
        # 更新PnL指标
        self.risk_metrics.daily_pnl = self.daily_pnl
        self.risk_metrics.total_pnl = sum([p.pnl for p in self.position_history])
        
        # 计算胜率
        if self.position_history:
            winning_trades = sum(1 for p in self.position_history if p.pnl > 0)
            self.risk_metrics.win_rate = winning_trades / len(self.position_history)
            
        # 更新仓位信息
        if self.current_position:
            self.risk_metrics.position_size = self.current_position.size
            self.risk_metrics.unrealized_pnl = self.current_position.unrealized_pnl
        else:
            self.risk_metrics.position_size = 0.0
            self.risk_metrics.unrealized_pnl = 0.0
    
    def get_risk_status(self) -> Dict:
        """
        获取风险状态
        
        Returns:
            风险状态字典
        """
        # 计算Sharpe比率（如果有足够数据）
        if len(self.equity_curve) > 10:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if returns.std() > 0:
                self.risk_metrics.sharpe_ratio = returns.mean() / returns.std()
                
        # 计算VaR
        if len(self.equity_curve) > 30:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            self.risk_metrics.var_95 = np.percentile(returns, 5)
            
        # 总体风险等级
        risk_level = "LOW"
        if self.risk_metrics.current_drawdown > 0.05:
            risk_level = "HIGH"
        elif self.risk_metrics.current_drawdown > 0.02:
            risk_level = "MEDIUM"
            
        return {
            "risk_level": risk_level,
            "metrics": {
                "current_drawdown": self.risk_metrics.current_drawdown,
                "max_drawdown": self.risk_metrics.max_drawdown,
                "daily_pnl": self.risk_metrics.daily_pnl,
                "total_pnl": self.risk_metrics.total_pnl,
                "win_rate": self.risk_metrics.win_rate,
                "sharpe_ratio": self.risk_metrics.sharpe_ratio,
                "var_95": self.risk_metrics.var_95,
                "position_size": self.risk_metrics.position_size
            },
            "position": {
                "has_position": self.current_position is not None,
                "side": self.current_position.side.value if self.current_position else None,
                "size": self.current_position.size if self.current_position else 0,
                "unrealized_pnl": self.current_position.unrealized_pnl if self.current_position else 0
            },
            "limits": {
                "max_drawdown_limit": self.max_drawdown,
                "daily_loss_limit": self.max_daily_loss,
                "max_position_size": self.max_position_size
            }
        }
    
    def reset_daily_stats(self):
        """重置日统计"""
        self.daily_pnl = 0.0
        self.logger.info("日统计已重置")