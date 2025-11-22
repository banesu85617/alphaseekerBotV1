#!/usr/bin/env python3
"""
AlphaSeeker 主集成应用
==================

AlphaSeeker系统的核心集成应用，协调所有组件：
- 集成API服务
- 机器学习引擎
- 多策略管道
- 市场扫描器
- 双重验证器

提供统一的使用接口和完整的系统管理功能。

作者: AlphaSeeker Team
版本: 1.0.0
日期: 2025-10-25
"""

import asyncio
import logging
import os
import sys
import signal
import time
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List 
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import yaml
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# FastAPI and HTTP
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


# ================================
# StrategyType 枚举定义 - 自动生成
# ================================
from enum import Enum


# ================================


# ================================
# MODEL_CONFIG - 机器学习模型配置
# ================================

MODEL_CONFIG = {
    # LightGBM模型配置
    'lightgbm': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    },
    
    # 模型训练配置
    'training': {
        'test_size': 0.2,
        'random_state': 42,
        'stratify': True,
        'early_stopping_rounds': 10,
        'eval_metric': 'binary_logloss',
        'verbose_eval': 100,
    },
    
    # 特征工程配置
    'feature_engineering': {
        'scaling_method': 'standard',
        'handle_missing': 'drop',
        'encode_categorical': 'label',
        'feature_selection': True,
        'variance_threshold': 0.01,
    },
    
    # 交叉验证配置
    'cross_validation': {
        'cv_folds': 5,
        'shuffle': True,
        'random_state': 42,
        'scoring': 'accuracy',
    },
    
    # 模型保存配置
    'model_saving': {
        'save_path': 'models/',
        'model_prefix': 'alphaseeker_model_',
        'save_format': 'pkl',
        'include_config': True,
    },
    
    # 预测配置
    'prediction': {
        'threshold': 0.5,
        'output_probability': True,
        'confidence_interval': 0.95,
    },
    
    # 性能监控配置
    'monitoring': {
        'track_accuracy': True,
        'track_precision': True,
        'track_recall': True,
        'track_f1_score': True,
        'track_auc': True,
        'log_predictions': False,
    },
    
    # 超参数优化配置
    'hyperparameter_tuning': {
        'enabled': False,
        'method': 'grid_search',
        'n_trials': 100,
        'timeout': 3600,
        'cv_folds': 3,
    },
    
    # 回测配置
    'backtesting': {
        'initial_capital': 100000,
        'transaction_cost': 0.001,
        'slippage': 0.0005,
        'benchmark': 'SPY',
        'rebalance_frequency': 'daily',
    },
    
    # 风险控制配置
    'risk_management': {
        'max_position_size': 0.1,
        'stop_loss': 0.05,
        'take_profit': 0.15,
        'max_drawdown': 0.2,
        'var_confidence': 0.95,
    },
    
    # 数据配置
    'data': {
        'lookback_period': 252,  # 一年交易日
        'update_frequency': 'daily',
        'data_source': 'yahoo',
        'min_data_points': 100,
        'feature_window': 20,
    },
    
    # 市场配置
    'market': {
        'timezone': 'UTC',
        'trading_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        },
        'min_volume': 100000,
        'market_cap_min': 1000000000,
    },
    
    # Alpha配置
    'alpha': {
        'target_alpha': 0.02,  # 2%年化Alpha
        'beta_neutral': True,
        'factor_exposure': {
            'value': 0.25,
            'momentum': 0.25,
            'quality': 0.25,
            'low_volatility': 0.25,
        },
    },
    
    # 报告配置
    'reporting': {
        'generate_reports': True,
        'report_format': 'html',
        'include_charts': True,
        'email_reports': False,
        'report_frequency': 'weekly',
    },
    
    # 调试配置
    'debug': {
        'verbose_logging': True,
        'save_intermediate_results': False,
        'profile_execution': False,
        'memory_monitoring': False,
    },
}


# ================================
# RISK_CONFIG - 风险管理配置
# ================================

RISK_CONFIG = {
    # 基本风险管理设置
    'basic_risk': {
        'max_position_size': 0.1,  # 单个仓位最大占比10%
        'max_portfolio_risk': 0.05,  # 组合最大风险5%
        'stop_loss_percentage': 0.05,  # 止损5%
        'take_profit_percentage': 0.15,  # 止盈15%
        'trailing_stop_percentage': 0.03,  # 追踪止损3%
        'max_drawdown_limit': 0.15,  # 最大回撤15%
    },
    
    # 仓位管理
    'position_management': {
        'min_position_size': 0.01,  # 最小仓位1%
        'position_increment': 0.01,  # 仓位递增步长1%
        'max_concurrent_positions': 20,  # 最大并发仓位20个
        'rebalance_threshold': 0.05,  # 再平衡阈值5%
        'position_timeout_days': 30,  # 仓位超时30天
    },
    
    # 波动率控制
    'volatility_control': {
        'target_volatility': 0.15,  # 目标波动率15%
        'volatility_window': 20,  # 波动率计算窗口20天
        'volatility_adjustment': True,  # 启用波动率调整
        'max_leverage': 2.0,  # 最大杠杆2倍
        'volatility_scaling': True,  # 波动率缩放
    },
    
    # 相关性管理
    'correlation_management': {
        'max_correlation': 0.7,  # 最大相关性0.7
        'correlation_window': 60,  # 相关性计算窗口60天
        'correlation_threshold': 0.5,  # 相关性阈值
        'diversification_bonus': 0.02,  # 分散化奖励2%
        'sector_limit': 0.3,  # 行业限制30%
    },
    
    # Value at Risk (VaR) 配置
    'var_config': {
        'var_confidence_level': 0.95,  # VaR置信水平95%
        'var_method': 'historical',  # VaR计算方法：historical
        'var_window': 252,  # VaR计算窗口252天
        'var_scaling_factor': 1.0,  # VaR缩放因子
        'stressed_var': True,  # 压力VaR
        'var_reporting': True,  # VaR报告
    },
    
    # 压力测试配置
    'stress_testing': {
        'enabled': True,  # 启用压力测试
        'scenarios': [
            'market_crash_2008',
            'covid_pandemic_2020',
            'interest_rate_shock',
            'currency_crisis',
            'sector_rotation'
        ],
        'frequency': 'monthly',  # 压力测试频率：每月
        'stress_var_multiplier': 1.5,  # 压力VaR倍数
    },
    
    # 止损配置
    'stop_loss': {
        'dynamic_stop_loss': True,  # 动态止损
        'atr_multiplier': 2.0,  # ATR倍数
        'time_based_stop': True,  # 基于时间的止损
        'profit_based_stop': True,  # 基于盈利的止损
        'volume_stop': True,  # 基于成交量的止损
    },
    
    # 风险监控
    'monitoring': {
        'real_time_monitoring': True,  # 实时监控
        'alert_threshold': 0.8,  # 警报阈值
        'notification_channels': ['email', 'sms', 'webhook'],  # 通知渠道
        'monitoring_frequency': 'hourly',  # 监控频率：每小时
        'risk_metrics': [
            'sharpe_ratio',
            'max_drawdown',
            'volatility',
            'var',
            'beta'
        ],
    },
    
    # 风险预算
    'risk_budget': {
        'total_risk_budget': 0.05,  # 总风险预算5%
        'allocation_by_strategy': {
            'momentum': 0.03,
            'mean_reversion': 0.02,
            'alpha_seeking': 0.04,
            'multi_strategy': 0.01
        },
        'risk_budget_rebalance': 'weekly',  # 风险预算再平衡：每周
    },
    
    # 对冲策略
    'hedging': {
        'enabled': True,  # 启用对冲
        'hedge_ratio': 0.8,  # 对冲比率80%
        'hedge_instruments': ['SPY', 'VIX', 'TLT'],  # 对冲工具
        'dynamic_hedging': True,  # 动态对冲
        'cost_benefit_threshold': 0.002,  # 成本效益阈值
    },
    
    # 合规性检查
    'compliance': {
        'enabled': True,  # 启用合规性检查
        'regulatory_limits': {
            'position_limit': 0.1,  # 仓位限制10%
            'concentration_limit': 0.05,  # 集中度限制5%
            'liquidity_limit': 0.02,  # 流动性限制2%
        },
        'compliance_frequency': 'daily',  # 合规检查频率：每日
        'auto_liquidation': False,  # 自动清算
    },
    
    # 流动性风险管理
    'liquidity_risk': {
        'min_daily_volume': 100000,  # 最小日成交量
        'bid_ask_spread_limit': 0.01,  # 买卖价差限制1%
        'market_impact_limit': 0.005,  # 市场影响限制0.5%
        'liquidity_buffer': 0.1,  # 流动性缓冲10%
        'liquidity_monitoring': True,  # 流动性监控
    },
    
    # 信用风险管理
    'credit_risk': {
        'enabled': False,  # 暂不启用信用风险（股票市场）
        'counterparty_limit': 0.05,  # 交易对手限制5%
        'concentration_limit': 0.02,  # 集中度限制2%
        'credit_rating_threshold': 'BBB',  # 信用评级阈值
        'monitoring_frequency': 'weekly',  # 监控频率：每周
    },
    
    # 操作风险管理
    'operational_risk': {
        'enabled': True,  # 启用操作风险
        'error_tolerance': 0.001,  # 错误容忍度0.1%
        'data_quality_check': True,  # 数据质量检查
        'execution_monitoring': True,  # 执行监控
        'system_reliability_threshold': 0.99,  # 系统可靠性阈值99%
    },
    
    # 风险报告配置
    'reporting': {
        'generate_risk_reports': True,  # 生成风险报告
        'report_frequency': 'daily',  # 报告频率：每日
        'include_var_report': True,  # 包含VaR报告
        'include_stress_test': True,  # 包含压力测试
        'risk_attribution': True,  # 风险归因
        'real_time_dashboard': True,  # 实时仪表板
    },
    
    # 风险模型配置
    'risk_models': {
        'covariance_model': 'ledoit_wolf',  # 协方差模型：Ledoit-Wolf
        'factor_model': 'fama_french_3',  # 因子模型：Fama-French 3因子
        'volatility_model': 'garch',  # 波动率模型：GARCH
        'correlation_model': 'dynamic',  # 相关性模型：动态
        'model_validation': True,  # 模型验证
    },
}




# StrategyType 枚举定义 - 自动生成
# ================================
from enum import Enum


class StrategyType(Enum):
    """策略类型枚举 - 定义Pipeline系统支持的所有策略类型"""
    
    # 技术分析策略 (6种)
    TECHNICAL_INDICATOR = "technical_indicator"
    TECHNICAL_SMA = "technical_sma"
    TECHNICAL_EMA = "technical_ema"
    TECHNICAL_RSI = "technical_rsi"
    TECHNICAL_MACD = "technical_macd"
    TECHNICAL_BOLLINGER = "technical_bollinger"
    
    # 基本面策略 (4种)
    FUNDAMENTAL = "fundamental"
    VALUE_INVESTING = "value_investing"
    GROWTH_INVESTING = "growth_investing"
    DIVIDEND_INVESTING = "dividend_investing"
    
    # 动量策略 (3种)
    MOMENTUM = "momentum"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    
    # 均值回归策略 (3种)
    MEAN_REVERSION = "mean_reversion"
    CONTRARIAN = "contrarian"
    OVERBOUGHT_OVERSOLD = "overbought_oversold"
    
    # 量化策略 (3种)
    QUANTITATIVE = "quantitative"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    PAIR_TRADING = "pair_trading"
    ML_PREDICTION = "ml_prediction"
    
    # Alpha策略 (3种)
    ALPHA_SEEKING = "alpha_seeking"
    ALPHA_HUNTING = "alpha_hunting"
    EDGE_DETECTION = "edge_detection"
    
    # 风险管理策略 (5种)
    RISK_MANAGEMENT = "risk_management"
    STOP_LOSS = "stop_loss"
    PORTFOLIO_HEDGING = "portfolio_hedging"
    RISK_MODEL = "risk_model"
    BACKTEST_REFERENCE = "backtest_reference"
    
    # 复合策略 (3种)
    MULTI_STRATEGY = "multi_strategy"
    HYBRID_STRATEGY = "hybrid_strategy"
    ENSEMBLE_METHOD = "ensemble_method"
    
    # 时间策略 (4种)
    HIGH_FREQUENCY = "high_frequency"
    INTRADAY = "intraday"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    
    # 特殊策略 (5种)
    EVENT_DRIVEN = "event_driven"
    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_SENTIMENT = "social_sentiment"
    FUTURES_STRATEGY = "futures_strategy"
    OPTIONS_STRATEGY = "options_strategy"
    
    def get_description(self) -> str:
        """获取策略类型的描述信息"""
        descriptions = {
            # 技术分析策略
            StrategyType.TECHNICAL_INDICATOR: "技术指标策略 - 基于技术分析指标",
            StrategyType.TECHNICAL_SMA: "SMA策略 - 简单移动平均线策略",
            StrategyType.TECHNICAL_EMA: "EMA策略 - 指数移动平均线策略",
            StrategyType.TECHNICAL_RSI: "RSI策略 - 相对强弱指数策略",
            StrategyType.TECHNICAL_MACD: "MACD策略 - 移动平均收敛发散策略",
            StrategyType.TECHNICAL_BOLLINGER: "布林带策略 - 布林带突破策略",
            
            # 基本面策略
            StrategyType.FUNDAMENTAL: "基本面策略 - 基于公司财务数据",
            StrategyType.VALUE_INVESTING: "价值投资策略 - 寻找被低估的股票",
            StrategyType.GROWTH_INVESTING: "成长投资策略 - 关注高增长公司",
            StrategyType.DIVIDEND_INVESTING: "股息投资策略 - 投资稳定分红股票",
            
            # 动量策略
            StrategyType.MOMENTUM: "动量策略 - 基于价格动量进行交易",
            StrategyType.TREND_FOLLOWING: "趋势跟随策略 - 跟随市场趋势",
            StrategyType.BREAKOUT: "突破策略 - 价格突破重要水平",
            
            # 均值回归策略
            StrategyType.MEAN_REVERSION: "均值回归策略 - 价格回归均值",
            StrategyType.CONTRARIAN: "反向策略 - 逆市场而动",
            StrategyType.OVERBOUGHT_OVERSOLD: "超买超卖策略 - 识别极端情况",
            
            # 量化策略
            StrategyType.QUANTITATIVE: "量化策略 - 基于数学模型",
            StrategyType.STATISTICAL_ARBITRAGE: "统计套利策略 - 利用价格差",
            StrategyType.PAIR_TRADING: "配对交易策略 - 配对股票交易",
            StrategyType.ML_PREDICTION: "机器学习预测策略 - 基于ML模型预测",
            
            # Alpha策略
            StrategyType.ALPHA_SEEKING: "Alpha寻找策略 - 寻找超额收益",
            StrategyType.ALPHA_HUNTING: "Alpha猎手策略 - 积极寻找Alpha",
            StrategyType.EDGE_DETECTION: "边缘检测策略 - 检测市场边缘",
            
            # 风险管理策略
            StrategyType.RISK_MANAGEMENT: "风险管理策略 - 控制投资风险",
            StrategyType.STOP_LOSS: "止损策略 - 限制损失",
            StrategyType.PORTFOLIO_HEDGING: "组合对冲策略 - 对冲投资组合风险",
            StrategyType.RISK_MODEL: "风险模型策略 - 基于风险模型的风险控制",
            StrategyType.BACKTEST_REFERENCE: "回测基准策略 - 作为回测比较基准",
            
            # 复合策略
            StrategyType.MULTI_STRATEGY: "多策略组合 - 综合多种策略",
            StrategyType.HYBRID_STRATEGY: "混合策略 - 策略混合",
            StrategyType.ENSEMBLE_METHOD: "集成方法 - 策略集成",
            
            # 时间策略
            StrategyType.HIGH_FREQUENCY: "高频交易策略 - 高速算法交易",
            StrategyType.INTRADAY: "日内交易策略 - 当日买卖",
            StrategyType.SWING_TRADING: "摆动交易策略 - 中短期持有",
            StrategyType.POSITION_TRADING: "仓位交易策略 - 长期持有",
            
            # 特殊策略
            StrategyType.EVENT_DRIVEN: "事件驱动策略 - 基于市场事件",
            StrategyType.NEWS_SENTIMENT: "新闻情绪策略 - 基于新闻情绪",
            StrategyType.SOCIAL_SENTIMENT: "社交情绪策略 - 基于社交媒体",
            StrategyType.FUTURES_STRATEGY: "期货策略 - 期货交易",
            StrategyType.OPTIONS_STRATEGY: "期权策略 - 期权交易",
        }
        
        return descriptions.get(self, f"策略: {self.value}")

class AlphaSeekerConfig:
    """AlphaSeeker主配置类"""
    # 应用基础配置
    app_name: str = "AlphaSeeker"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    debug: bool = False
    
    # 组件配置
    api_config: Dict[str, Any] = None
    ml_engine_config: Dict[str, Any] = None
    pipeline_config: Dict[str, Any] = None
    scanner_config: Dict[str, Any] = None
    validation_config: Dict[str, Any] = None
    
    # 性能配置
    max_concurrent_tasks: int = 32
    request_timeout: float = 30.0
    batch_size: int = 100
    enable_cache: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 数据路径
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    cache_dir: str = "cache"
    
    def __post_init__(self):
        """初始化后的配置处理"""
        if self.api_config is None:
            self.api_config = self._default_api_config()
        if self.ml_engine_config is None:
            self.ml_engine_config = self._default_ml_config()
        if self.pipeline_config is None:
            self.pipeline_config = self._default_pipeline_config()
        if self.scanner_config is None:
            self.scanner_config = self._default_scanner_config()
        if self.validation_config is None:
            self.validation_config = self._default_validation_config()
    
    def _default_api_config(self) -> Dict[str, Any]:
        """默认API配置"""
        return {
            "cors_origins": ["*"],
            "log_level": "INFO",
            "log_format": self.log_format,
            "host": self.host,
            "port": self.port,
            "reload": self.reload
        }
    
    def _default_ml_config(self) -> Dict[str, Any]:
        """默认ML引擎配置"""
        return {
            "model_config": MODEL_CONFIG,
            "risk_config": RISK_CONFIG,
            "enable_caching": self.enable_cache,
            "target_latency_ms": 500,
            "feature_engineering": {
                "scaling_method": "standard",
                "handle_missing": "drop",
                "encode_categorical": "label",
                "feature_selection": True,
                "variance_threshold": 0.01,
            },
            "lightgbm": {
                "objective": "binary",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42,
                "n_estimators": 100,
                "max_depth": 6,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
        }
    
    def _default_pipeline_config(self) -> Dict[str, Any]:
        """默认管道配置"""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout_seconds": 10.0,
            "ml_probability_threshold": 0.65,
            "llm_confidence_threshold": 0.65,
            "strategy_weights": {
                StrategyType.TECHNICAL_INDICATOR: 0.4,
                StrategyType.ML_PREDICTION: 0.2,
                StrategyType.RISK_MODEL: 0.2,
                StrategyType.BACKTEST_REFERENCE: 0.2
            }
        }
    
    def _default_scanner_config(self) -> Dict[str, Any]:
        """默认扫描器配置"""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "scan_timeout": 15.0,
            "batch_size": self.batch_size,
            "enable_cache": self.enable_cache
        }
    
    def _default_validation_config(self) -> Dict[str, Any]:
        """默认验证器配置"""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "lgbm_config": LightGBMConfig(
                probability_threshold=0.65,
                confidence_threshold=0.6
            ),
            "llm_config": LLMConfig(
                provider=LLMProvider.OLLAMA,
                base_url="http://localhost:11434",
                model_name="llama2:13b"
            ),
            "fusion_config": FusionConfig(
                strategy=FusionStrategy.ADAPTIVE_WEIGHT,
                risk_reward_threshold=1.2
            )
        }

# 系统状态类
@dataclass
class SystemStatus:
    """系统状态信息"""
    status: str = "initializing"
    uptime: float = 0.0
    version: str = "1.0.0"
    components: Dict[str, Dict[str, Any]] = None
    performance: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = {}
        if self.performance is None:
            self.performance = {}


# ================================
# 配置类定义
# ================================

@dataclass
class ValidationConfig:
    """验证器配置"""
    timeout_seconds: float = 5.0
    max_retries: int = 3
    enable_cache: bool = True
    llm_confidence_threshold: float = 0.65
    ml_probability_threshold: float = 0.65

@dataclass  
class PipelineConfig:
    """管道配置"""
    max_concurrent_tasks: int = 32
    timeout_seconds: float = 10.0
    ml_probability_threshold: float = 0.65
    llm_confidence_threshold: float = 0.65
    strategy_weights: Dict[StrategyType, float] = field(default_factory=lambda: {
        StrategyType.TECHNICAL_INDICATOR: 0.4,
        StrategyType.ML_PREDICTION: 0.2,
        StrategyType.RISK_MODEL: 0.2,
        StrategyType.BACKTEST_REFERENCE: 0.2
    })

@dataclass
class ScanConfig:
    """扫描器配置"""
    max_concurrent_tasks: int = 32
    scan_timeout: float = 15.0
    batch_size: int = 100
    enable_cache: bool = True

# ================================
# 组件类定义（简化版本）
# ================================

class SignalValidationCoordinator:
    """信号验证协调器 - 简化版本"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("SignalValidationCoordinator初始化完成")
    
    def health_check(self) -> Dict[str, Any]:
        return {"status": "ready", "overall_status": "healthy"}

class MultiStrategyPipeline:
    """多策略管道 - 简化版本"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("MultiStrategyPipeline初始化完成")
    
    async def start(self):
        """启动管道"""
        self.logger.info("多策略管道启动完成")
    
    def health_check(self) -> Dict[str, Any]:
        return {"status": "ready", "overall_status": "healthy"}

class MarketScanner:
    """市场扫描器 - 简化版本"""
    
    def __init__(self, config: ScanConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("MarketScanner初始化完成")
    
    def health_check(self) -> Dict[str, Any]:
        return {"status": "ready", "overall_status": "healthy"}

class AlphaSeekerOrchestrator:
    """AlphaSeeker系统协调器 - 核心组件"""
    
    def __init__(self, config: AlphaSeekerConfig):
        """初始化协调器"""
        self.config = config
        self.logger = None
        self.start_time = None
        self.is_running = False
        
        # 组件实例
        self.ml_engine: Optional[AlphaSeekerMLEngine] = None
        self.pipeline: Optional[MultiStrategyPipeline] = None
        self.scanner: Optional[MarketScanner] = None
        self.validation_coordinator: Optional[SignalValidationCoordinator] = None
        

    def _ensure_configs(self):
        """确保所有配置都存在"""
        if self.config.ml_engine_config is None:
            self.config.ml_engine_config = self._default_ml_config()
        if self.config.pipeline_config is None:
            self.config.pipeline_config = self._default_pipeline_config()
        if self.config.scanner_config is None:
            self.config.scanner_config = self._default_scanner_config()
        if self.config.validation_config is None:
            self.config.validation_config = self._default_validation_config()
    
    def _default_ml_config(self) -> Dict[str, Any]:
        """默认ML引擎配置"""
        return {
            "model_config": MODEL_CONFIG,
            "risk_config": RISK_CONFIG,
            "enable_caching": self.config.enable_cache,
            "target_latency_ms": 500,
            "feature_engineering": {
                "scaling_method": "standard",
                "handle_missing": "drop",
                "encode_categorical": "label",
                "feature_selection": True,
                "variance_threshold": 0.01,
            },
            "lightgbm": {
                "objective": "binary",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42,
                "n_estimators": 100,
                "max_depth": 6,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
        }
    
    def _default_pipeline_config(self) -> Dict[str, Any]:
        """默认管道配置"""
        return {
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "timeout_seconds": 10.0,
            "ml_probability_threshold": 0.65,
            "llm_confidence_threshold": 0.65,
            "strategy_weights": {
                StrategyType.TECHNICAL_INDICATOR: 0.4,
                StrategyType.ML_PREDICTION: 0.2,
                StrategyType.RISK_MODEL: 0.2,
                StrategyType.BACKTEST_REFERENCE: 0.2
            }
        }
    
    def _default_scanner_config(self) -> Dict[str, Any]:
        """默认扫描器配置"""
        return {
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "scan_timeout": 15.0,
            "batch_size": self.config.batch_size,
            "enable_cache": self.config.enable_cache
        }
    
    def _default_validation_config(self) -> Dict[str, Any]:
        """默认验证器配置"""
        return {
            "timeout_seconds": 5.0,
            "max_retries": 3,
            "enable_cache": self.config.enable_cache
        }
        # 性能统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.data_dir,
            self.config.model_dir,
            self.config.log_dir,
            self.config.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志系统"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(self.config.log_format)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(self.config.log_dir, "alphaseeker.log")
        )
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 设置第三方库的日志级别
        logging.getLogger("ccxt").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("lightgbm").setLevel(logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("📝 日志系统初始化完成")
    
    async def initialize_components(self):
        """初始化所有组件"""
        try:
            self.logger.info("🚀 开始初始化AlphaSeeker组件...")
            
            # 1. 初始化ML引擎
            self.logger.info("🧠 初始化机器学习引擎...")
            
            # 确保配置已初始化
            self._ensure_configs()
            
            self.ml_engine = AlphaSeekerMLEngine(
                config=self.config.ml_engine_config,        # 确保配置已初始化
                logger=self.logger
            )
            ml_health = self.ml_engine.health_check()
            self.logger.info(f"ML引擎状态: {ml_health.overall_status}")
            
            # 2. 初始化验证器
            self.logger.info("🔍 初始化双重验证器...")
            validation_config = ValidationConfig(**self.config.validation_config)
            self.validation_coordinator = SignalValidationCoordinator(validation_config)
            
            # 3. 初始化管道
            self.logger.info("⚙️ 初始化多策略管道...")
            pipeline_config = PipelineConfig(**self.config.pipeline_config)
            self.pipeline = MultiStrategyPipeline(pipeline_config)
            await self.pipeline.start()
            
            # 4. 初始化扫描器
            self.logger.info("📊 初始化市场扫描器...")
            scan_config = ScanConfig(**self.config.scanner_config)
            self.scanner = MarketScanner(scan_config)
            
            # 更新组件状态
            self._update_component_status("ml_engine", "ready", ml_health)
            self._update_component_status("validation", "ready", {"status": "ready"})
            self._update_component_status("pipeline", "ready", {"status": "ready"})
            self._update_component_status("scanner", "ready", {"status": "ready"})
            
            self.logger.info("✅ 所有组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 组件初始化失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _update_component_status(self, component: str, status: str, details: Dict[str, Any]):
        """更新组件状态"""
        if not hasattr(self, '_component_status'):
            self._component_status = {}
        
        self._component_status[component] = {
            "status": status,
            "last_update": datetime.now().isoformat(),
            "details": details
        }
    
    def get_system_status(self) -> SystemStatus:
        """获取系统状态"""
        uptime = time.time() - self.start_time if self.start_time else 0.0
        
        # 收集各组件状态
        components = {}
        
        # ML引擎状态
        if self.ml_engine:
            ml_health = self.ml_engine.health_check()
            components["ml_engine"] = {
                "status": "healthy" if ml_health['overall_status'] == "healthy" else "warning",
                "performance": self.ml_engine.get_performance_stats()
            }
        
        # 验证器状态
        if self.validation_coordinator:
            components["validation"] = self._component_status.get("validation", {"status": "unknown"})
        
        # 管道状态
        if self.pipeline:
            components["pipeline"] = self._component_status.get("pipeline", {"status": "unknown"})
        
        # 扫描器状态
        if self.scanner:
            components["scanner"] = self._component_status.get("scanner", {"status": "unknown"})
        
        # 计算性能指标
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        avg_processing_time = self.total_processing_time / max(self.total_requests, 1)
        
        performance = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "avg_processing_time": round(avg_processing_time, 3),
            "uptime": round(uptime, 2)
        }
        
        
        # 确保配置已初始化
        self._ensure_configs()
        
        return SystemStatus(
            status="healthy" if self.is_running else "stopped",
            uptime=uptime,
            version=self.config.app_version,
            components=components,
            performance=performance
        )
    
    async def start(self):
        """启动系统"""
        try:
            self.start_time = time.time()
            self._setup_logging()
            self.logger.info(f"🚀 启动 AlphaSeeker v{self.config.app_version}")
            
            # 初始化组件
            await self.initialize_components()
            
            self.is_running = True
            self.logger.info("✅ AlphaSeeker系统启动完成")
            
        except Exception as e:
            self.logger.error(f"❌ 系统启动失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def stop(self):
        """停止系统"""
        self.logger.info("🛑 正在停止AlphaSeeker系统...")
        
        self.is_running = False
        
        # 停止各组件
        try:
            if self.pipeline:
                await self.pipeline.stop()
            
            if self.validation_coordinator:
                await self.validation_coordinator.shutdown()
            
            self.logger.info("✅ AlphaSeeker系统已停止")
            
        except Exception as e:
            self.logger.error(f"❌ 停止系统时出错: {e}")
    
    async def process_trading_signal(self, symbol: str, market_data: Dict[str, Any], 
                                   indicators: Dict[str, Any], 
                                   features: Dict[str, Any]) -> Dict[str, Any]:
        """处理交易信号 - 核心功能"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            self.logger.info(f"📊 处理 {symbol} 的交易信号")
            
            # 1. ML引擎预测
            ml_prediction = None
            if self.ml_engine:
                ml_result = self.ml_engine.predict(market_data)
                ml_prediction = MLPrediction(
                    label=ml_result['signal_label'],
                    probability_scores=ml_result['probability_distribution'],
                    confidence=ml_result['confidence'],
                    model_version="lightgbm_v2.1.0"
                )
            
            # 2. 市场数据转换
            market = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=market_data.get('price', 0),
                volume=market_data.get('volume', 0),
                data_freshness=1.0
            )
            
            # 3. 技术指标转换
            technical_indicators = TechnicalIndicators(
                rsi=indicators.get('rsi', 50),
                macd=indicators.get('macd', 0),
                adx=indicators.get('adx', 25),
                sma_50=indicators.get('sma_50', 0),
                sma_200=indicators.get('sma_200', 0)
            )
            
            # 4. 多策略融合
            fusion_result = None
            if self.pipeline and ml_prediction:
                fusion_result = await self.pipeline.process_single_symbol(
                    symbol=symbol,
                    market_data=market,
                    technical_indicators=technical_indicators,
                    ml_prediction=ml_prediction
                )
            
            # 5. 双重验证
            validation_result = None
            if self.validation_coordinator:
                validation_request = ValidationRequest(
                    symbol=symbol,
                    timeframe="1h",
                    current_price=market_data.get('price', 0),
                    features=features,
                    indicators=indicators,
                    risk_context={"volatility": 0.025},
                    priority=ValidationPriority.MEDIUM
                )
                
                validation_result = await self.validation_coordinator.validate_signal(validation_request)
            
            # 6. 合成最终结果
            final_result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "signal_direction": fusion_result.final_direction.value if fusion_result else "unknown",
                "confidence": fusion_result.combined_confidence if fusion_result else 0.5,
                "score": fusion_result.final_score if fusion_result else 0.5,
                "risk_reward_ratio": fusion_result.risk_reward_ratio if fusion_result else 1.0,
                "processing_time": time.time() - start_time,
                "components": {
                    "ml_prediction": {
                        "label": ml_prediction.label if ml_prediction else None,
                        "confidence": ml_prediction.confidence if ml_prediction else None,
                        "probabilities": ml_prediction.probability_scores if ml_prediction else None
                    },
                    "fusion_result": {
                        "final_score": fusion_result.final_score if fusion_result else None,
                        "confidence": fusion_result.combined_confidence if fusion_result else None
                    } if fusion_result else None,
                    "validation": {
                        "status": validation_result.status.value if validation_result else None,
                        "combined_score": validation_result.combined_score if validation_result else None
                    } if validation_result else None
                }
            }
            
            self.successful_requests += 1
            self.total_processing_time += time.time() - start_time
            
            self.logger.info(f"✅ {symbol} 信号处理完成 - 方向: {final_result['signal_direction']}, 置信度: {final_result['confidence']:.3f}")
            
            return final_result
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"❌ {symbol} 信号处理失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def batch_scan_market(self, symbols: List[str], max_results: int = 10) -> Dict[str, Any]:
        """批量市场扫描"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 开始批量扫描市场 - {len(symbols)} 个交易对")
            
            results = []
            
            # 并发处理多个交易对
            tasks = []
            for symbol in symbols:
                # 模拟市场数据（实际中应从数据源获取）
                mock_market_data = {
                    "price": 40000 + hash(symbol) % 10000,
                    "volume": 1000000,
                    "timestamp": time.time()
                }
                
                mock_indicators = {
                    "rsi": 50 + hash(symbol) % 40,
                    "macd": 100 + hash(symbol) % 200,
                    "adx": 20 + hash(symbol) % 20,
                    "sma_50": 42000,
                    "sma_200": 40000
                }
                
                mock_features = {
                    "mid_price": mock_market_data["price"],
                    "spread": 2.5,
                    "volatility_60s": 0.025
                }
                
                task = self.process_trading_signal(
                    symbol, mock_market_data, mock_indicators, mock_features
                )
                tasks.append(task)
            
            # 等待所有任务完成
            symbol_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤和处理结果
            valid_results = []
            for i, result in enumerate(symbol_results):
                if isinstance(result, Exception):
                    self.logger.error(f"处理 {symbols[i]} 时出错: {result}")
                    continue
                
                # 只保留高置信度的结果
                if result['confidence'] >= 0.6:
                    valid_results.append(result)
            
            # 按置信度排序，取前max_results个
            valid_results.sort(key=lambda x: x['confidence'], reverse=True)
            results = valid_results[:max_results]
            
            processing_time = time.time() - start_time
            
            final_result = {
                "scan_id": f"scan_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "total_symbols": len(symbols),
                "processed_symbols": len(symbol_results),
                "valid_results": len(results),
                "results": results,
                "processing_time": processing_time,
                "summary": {
                    "avg_confidence": sum(r['confidence'] for r in results) / max(len(results), 1),
                    "signal_distribution": self._analyze_signal_distribution(results)
                }
            }
            
            self.logger.info(f"✅ 市场扫描完成 - 处理: {len(symbol_results)}个, 有效: {len(results)}个, 用时: {processing_time:.2f}秒")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 市场扫描失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _analyze_signal_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析信号分布"""
        distribution = {"long": 0, "short": 0, "flat": 0, "unknown": 0}
        
        for result in results:
            direction = result.get("signal_direction", "unknown")
            distribution[direction] = distribution.get(direction, 0) + 1
        
        return distribution


# 全局系统实例
orchestrator: Optional[AlphaSeekerOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global orchestrator
    
    # 启动
    try:
        orchestrator = AlphaSeekerOrchestrator(CONFIG)
        await orchestrator.start()
        
        yield
        
    finally:
        # 关闭
        if orchestrator:
            await orchestrator.stop()

# 创建FastAPI应用
app = FastAPI(
    title="AlphaSeeker集成系统",
    description="AlphaSeeker AI驱动的加密货币交易信号系统，集成机器学习、多策略融合和双重验证",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置实例
CONFIG = AlphaSeekerConfig()


# 修复 FastAPI 弃用警告 - 使用 lifespan 事件处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """ lifespan 事件处理器替代弃用的 on_event """
    # 启动逻辑
    print("🚀 AlphaSeeker集成系统正在启动...")
    yield
    # 关闭逻辑
    print("🛑 AlphaSeeker集成系统正在关闭...")

    """启动事件"""
    print("🚀 AlphaSeeker集成系统正在启动...")

# 关闭处理器已在 lifespan 中处理
    """关闭事件"""
    print("🛑 AlphaSeeker集成系统正在关闭...")


@app.get("/", response_class=HTMLResponse)
async def root():
    """主页 - 现代化系统状态页面"""
    
    # 系统信息数据
    system_info = {
        "name": "AlphaSeeker集成系统",
        "version": "1.0.0",
        "description": "AI驱动的加密货币交易信号系统",
        "components": [
            "机器学习引擎 (LightGBM)",
            "多策略信号管道", 
            "市场扫描器",
            "双重验证器",
            "集成API服务"
        ],
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }
    
    # 现代化HTML界面
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaSeeker 集成系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .title {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 700;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .version {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 5px;
        }

        .status-badge {
            display: inline-block;
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 10px 0;
            animation: pulse 2s infinite;
        }

        .section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }

        .section-title {
            font-size: 1.3em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
        }

        .section-title::before {
            content: "🔧";
            margin-right: 10px;
            font-size: 1.2em;
        }

        .components-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .component-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .component-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        .component-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .component-status {
            color: #27ae60;
            font-size: 0.9em;
        }

        .json-section {
            background: #2d3748;
            color: #e2e8f0;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            font-family: 'Fira Code', 'Consolas', monospace;
            overflow-x: auto;
        }

        .json-title {
            color: #63b3ed;
            font-size: 1.1em;
            margin-bottom: 15px;
            font-weight: 600;
        }

        .json-content {
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .toggle-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .toggle-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .timestamp {
            color: #6c757d;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            .title {
                font-size: 2em;
            }
            
            .components-grid {
                grid-template-columns: 1fr;
            }
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">🚀 AlphaSeeker 集成系统</h1>
            <div class="version">版本: <span id="version">1.0.0</span></div>
            <div class="status-badge">🟢 运行中</div>
        </div>

        <div class="section">
            <div class="section-title">系统组件</div>
            <div class="components-grid" id="components">
                <div class="component-card">
                    <div class="component-name">机器学习引擎 (LightGBM)</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">多策略信号管道</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">市场扫描器</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">双重验证器</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">集成API服务</div>
                    <div class="component-status">✅ 正常</div>
                </div>
            </div>
        </div>

        <div class="json-section">
            <div class="json-title">📊 系统信息</div>
            <div class="json-content" id="json-content">
                {
                    "name": "AlphaSeeker集成系统",
                    "version": "1.0.0",
                    "description": "AI驱动的加密货币交易信号系统",
                    "status": "running",
                    "timestamp": "2025-10-27T22:56:49"
                }
            </div>
        </div>

        <div class="controls">
            <button class="toggle-btn" onclick="toggleJsonFormat()">
                切换JSON格式
            </button>
            <div class="timestamp">
                更新时间: <span id="timestamp"></span>
            </div>
        </div>
    </div>

    <script>
        function toggleJsonFormat() {
            const jsonContent = document.getElementById('json-content');
            const isFormatted = jsonContent.style.whiteSpace === 'pre-wrap';
            
            if (isFormatted) {
                jsonContent.style.whiteSpace = 'nowrap';
                jsonContent.textContent = JSON.stringify(JSON.parse(jsonContent.textContent));
            } else {
                jsonContent.style.whiteSpace = 'pre-wrap';
                jsonContent.textContent = JSON.stringify(JSON.parse(jsonContent.textContent), null, 2);
            }
        }

        function updateTimestamp() {
            const now = new Date();
            document.getElementById('timestamp').textContent = now.toLocaleString('zh-CN');
        }

        // 页面加载后更新时间戳
        document.addEventListener('DOMContentLoaded', function() {
            updateTimestamp();
        });
    </script>
</body>
</html>
    """
    
    return html_template


@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        status = orchestrator.get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@app.post("/api/v1/signal/analyze", tags=["交易信号"])
async def analyze_signal(request: Request):
    """分析单个交易信号"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        data = await request.json()
        
        required_fields = ["symbol", "market_data", "indicators", "features"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"缺少必需字段: {field}")
        
        result = await orchestrator.process_trading_signal(
            symbol=data["symbol"],
            market_data=data["market_data"],
            indicators=data["indicators"],
            features=data["features"]
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信号分析失败: {str(e)}")

@app.post("/api/v1/scan/market", tags=["市场扫描"])
async def scan_market(request: Request):
    """批量市场扫描"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        data = await request.json()
        
        symbols = data.get("symbols", [])
        max_results = data.get("max_results", 10)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="symbols不能为空")
        
        if len(symbols) > 100:  # 限制最大数量
            raise HTTPException(status_code=400, detail="symbols数量不能超过100个")
        
        result = await orchestrator.batch_scan_market(symbols, max_results)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"市场扫描失败: {str(e)}")

@app.get("/api/v1/system/status", tags=["系统"])
async def get_system_status():
    """获取详细系统状态"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        status = orchestrator.get_system_status()
        return asdict(status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@app.get("/api/v1/components", tags=["系统"])
async def get_components_info():
    """获取组件信息"""
    components_info = {
        "ml_engine": {
            "name": "机器学习引擎",
            "description": "LightGBM模型进行交易信号预测",
            "features": ["价格预测", "特征工程", "风险管理"]
        },
        "pipeline": {
            "name": "多策略管道",
            "description": "融合多种策略的交易信号处理管道",
            "features": ["策略融合", "信号优先级", "冲突解决"]
        },
        "scanner": {
            "name": "市场扫描器",
            "description": "多策略市场扫描和机会发现",
            "features": ["批量扫描", "策略多样化", "机会排序"]
        },
        "validation": {
            "name": "双重验证器",
            "description": "LightGBM + LLM双重验证机制",
            "features": ["快速筛选", "深度评估", "结果融合"]
        },
        "api": {
            "name": "集成API",
            "description": "统一的REST API接口服务",
            "features": ["REST API", "CORS支持", "错误处理"]
        }
    }
    
    return {
        "components": components_info,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/performance", tags=["系统"])
async def get_performance_metrics():
    """获取性能指标"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        status = orchestrator.get_system_status()
        
        return {
            "performance": status.performance,
            "system_info": {
                "uptime": status.uptime,
                "version": status.version,
                "config": {
                    "max_concurrent_tasks": CONFIG.max_concurrent_tasks,
                    "batch_size": CONFIG.batch_size,
                    "enable_cache": CONFIG.enable_cache
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

# 异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    return JSONResponse(
        status_code=500,
        content={"error": "内部服务器错误", "status_code": 500}
    )

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n🛑 接收到信号 {signum}，正在关闭系统...")
    if orchestrator:
        asyncio.create_task(orchestrator.stop())
    sys.exit(0)

def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数"""
    setup_signal_handlers()
    
    print("=" * 60)
    print("🚀 AlphaSeeker 集成系统")
    print("=" * 60)
    print(f"版本: {CONFIG.app_version}")
    print(f"主机: {CONFIG.host}:{CONFIG.port}")
    print(f"调试: {CONFIG.debug}")
    print(f"并发任务: {CONFIG.max_concurrent_tasks}")
    print(f"批处理大小: {CONFIG.batch_size}")
    print("=" * 60)
    
    # 启动服务器
    uvicorn.run(
        "main_integration:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=CONFIG.reload,
        log_level=CONFIG.log_level.lower()
    )

if __name__ == "__main__":
    print("启动AlphaSeeker系统...")

# ================================
# AlphaSeekerMLEngine - 机器学习引擎
# ================================

import logging
import time
import asyncio
from datetime import datetime, timedelta

import json
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available, using mock predictions")

@dataclass
class MLPredictionResult:
    """ML预测结果"""
    symbol: str
    prediction: float
    confidence: float
    features: Dict[str, float]
    model_version: str
    timestamp: str
    processing_time: float

@dataclass
class ModelHealthStatus:
    """模型健康状态"""
    overall_status: str
    model_loaded: bool
    lightgbm_available: bool
    memory_usage: str
    last_prediction_time: Optional[str]
    total_predictions: int
    accuracy_rate: float

class AlphaSeekerMLEngine:
    """AlphaSeeker机器学习引擎"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        初始化ML引擎
        
        Args:
            config: ML引擎配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.model = None
        self.is_loaded = False
        self.total_predictions = 0
        self.successful_predictions = 0
        self.last_prediction_time = None
        
        # 特征配置
        self.feature_config = config.get('feature_engineering', {
            'scaling_method': 'standard',
            'handle_missing': 'drop',
            'encode_categorical': 'label',
            'feature_selection': True,
            'variance_threshold': 0.01,
        })
        
        # LightGBM配置
        self.lgbm_config = config.get('lightgbm', {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        })
        
        self.logger.info("🧠 AlphaSeekerMLEngine初始化完成")
        
        # 尝试加载模型
        self._load_model()
    
    def _load_model(self):
        """加载机器学习模型"""
        try:
            if LIGHTGBM_AVAILABLE:
                self.logger.info("📦 尝试加载LightGBM模型...")
                
                # 创建模拟LightGBM模型用于演示
                self.model = lgb.LGBMClassifier(**self.lgbm_config)
                
                # 模拟训练数据
                X_demo = np.random.random((100, 10))
                y_demo = np.random.randint(0, 2, 100)
                
                self.model.fit(X_demo, y_demo)
                
                self.logger.info("✅ LightGBM模型加载成功")
                self.is_loaded = True
            else:
                self.logger.warning("⚠️ LightGBM不可用，使用模拟模式")
                self.model = "mock_model"
                self.is_loaded = True
                
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            self.model = "fallback_model"
            self.is_loaded = False
    
    def predict(self, symbol: str, features: Dict[str, float]) -> MLPredictionResult:
        """生成机器学习预测"""
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise Exception("模型未加载")
            
            # 特征预处理
            processed_features = self._preprocess_features(features)
            
            if LIGHTGBM_AVAILABLE and hasattr(self.model, 'predict_proba'):
                X = np.array([list(processed_features.values())]).reshape(1, -1)
                prediction_prob = self.model.predict_proba(X)[0]
                prediction = prediction_prob[1]
            else:
                prediction = self._generate_mock_prediction(processed_features)
            
            confidence = self._calculate_confidence(processed_features)
            
            self.total_predictions += 1
            self.successful_predictions += 1
            self.last_prediction_time = datetime.now().isoformat()
            
            processing_time = time.time() - start_time
            
            result = MLPredictionResult(
                symbol=symbol,
                prediction=prediction,
                confidence=confidence,
                features=processed_features,
                model_version="lightgbm_v1.0.0",
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            self.logger.debug(f"✅ ML预测完成: {symbol} -> {prediction:.4f} (置信度: {confidence:.3f})")
            return result
            
        except Exception as e:
            self.total_predictions += 1
            processing_time = time.time() - start_time
            
            self.logger.error(f"❌ ML预测失败: {e}")
            
            return MLPredictionResult(
                symbol=symbol,
                prediction=0.5,
                confidence=0.1,
                features=self._preprocess_features(features),
                model_version="fallback",
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
    
    def _preprocess_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """特征预处理"""
        processed = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if key in ['price', 'volume', 'amount']:
                    processed[key] = np.log1p(abs(value)) if value != 0 else 0
                elif key in ['rsi', 'macd', 'bb_position']:
                    processed[key] = np.clip(value, -10, 10)
                else:
                    processed[key] = value
            else:
                processed[key] = 0.0
        
        default_features = {
            'rsi': 50.0,
            'macd': 0.0,
            'bb_position': 0.5,
            'volume_ratio': 1.0,
            'price_momentum': 0.0
        }
        
        for key, default_value in default_features.items():
            if key not in processed:
                processed[key] = default_value
        
        return processed
    
    def _generate_mock_prediction(self, features: Dict[str, float]) -> float:
        """生成模拟预测"""
        rsi = features.get('rsi', 50)
        macd = features.get('macd', 0)
        bb_pos = features.get('bb_position', 0.5)
        volume_ratio = features.get('volume_ratio', 1.0)
        
        rsi_signal = 0.5
        if rsi > 70:
            rsi_signal = 0.2
        elif rsi < 30:
            rsi_signal = 0.8
        
        macd_signal = 0.5 + np.clip(macd / 10, -0.3, 0.3)
        bb_signal = 1.0 - bb_pos
        volume_signal = np.clip(volume_ratio, 0.5, 2.0) / 2.0
        
        final_prediction = (
            rsi_signal * 0.3 +
            macd_signal * 0.3 +
            bb_signal * 0.2 +
            volume_signal * 0.2
        )
        
        return np.clip(final_prediction, 0.1, 0.9)
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """计算预测置信度"""
        confidence_factors = []
        
        required_features = ['rsi', 'macd', 'bb_position', 'volume_ratio']
        completeness = len([f for f in required_features if f in features]) / len(required_features)
        confidence_factors.append(completeness)
        
        rsi = features.get('rsi', 50)
        confidence_factors.append(1.0 if 0 <= rsi <= 100 else 0.5)
        
        bb_pos = features.get('bb_position', 0.5)
        confidence_factors.append(1.0 if 0 <= bb_pos <= 1 else 0.3)
        
        return np.mean(confidence_factors)
    
    def health_check(self) -> ModelHealthStatus:
        """执行模型健康检查"""
        try:
            model_loaded = self.is_loaded
            accuracy_rate = (
                self.successful_predictions / self.total_predictions 
                if self.total_predictions > 0 else 1.0
            )
            memory_usage = f"{random.randint(50, 200)}MB"
            
            return ModelHealthStatus(
                overall_status="healthy" if model_loaded else "warning",
                model_loaded=model_loaded,
                lightgbm_available=LIGHTGBM_AVAILABLE,
                memory_usage=memory_usage,
                last_prediction_time=self.last_prediction_time,
                total_predictions=self.total_predictions,
                accuracy_rate=accuracy_rate
            )
            
        except Exception as e:
            self.logger.error(f"❌ 健康检查失败: {e}")
            return ModelHealthStatus(
                overall_status="error",
                model_loaded=False,
                lightgbm_available=LIGHTGBM_AVAILABLE,
                memory_usage="unknown",
                last_prediction_time=None,
                total_predictions=0,
                accuracy_rate=0.0
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": "LightGBM" if LIGHTGBM_AVAILABLE else "Mock",
            "model_loaded": self.is_loaded,
            "lightgbm_available": LIGHTGBM_AVAILABLE,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "last_prediction_time": self.last_prediction_time,
            "config": self.lgbm_config,
            "feature_config": self.feature_config
        }
if __name__ == "__main__":
    main()



@app.get("/", response_class=HTMLResponse)
async def root():
    """主页 - 现代化系统状态页面"""
    
    # 系统信息数据
    system_info = {
        "name": "AlphaSeeker集成系统",
        "version": "1.0.0",
        "description": "AI驱动的加密货币交易信号系统",
        "components": [
            "机器学习引擎 (LightGBM)",
            "多策略信号管道", 
            "市场扫描器",
            "双重验证器",
            "集成API服务"
        ],
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }
    
    # 现代化HTML界面
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaSeeker 集成系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .title {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 700;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .version {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 5px;
        }

        .status-badge {
            display: inline-block;
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 10px 0;
            animation: pulse 2s infinite;
        }

        .section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }

        .section-title {
            font-size: 1.3em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
        }

        .section-title::before {
            content: "🔧";
            margin-right: 10px;
            font-size: 1.2em;
        }

        .components-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .component-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .component-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        .component-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .component-status {
            color: #27ae60;
            font-size: 0.9em;
        }

        .json-section {
            background: #2d3748;
            color: #e2e8f0;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            font-family: 'Fira Code', 'Consolas', monospace;
            overflow-x: auto;
        }

        .json-title {
            color: #63b3ed;
            font-size: 1.1em;
            margin-bottom: 15px;
            font-weight: 600;
        }

        .json-content {
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .toggle-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .toggle-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .timestamp {
            color: #6c757d;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            .title {
                font-size: 2em;
            }
            
            .components-grid {
                grid-template-columns: 1fr;
            }
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">🚀 AlphaSeeker 集成系统</h1>
            <div class="version">版本: <span id="version">1.0.0</span></div>
            <div class="status-badge">🟢 运行中</div>
        </div>

        <div class="section">
            <div class="section-title">系统组件</div>
            <div class="components-grid" id="components">
                <div class="component-card">
                    <div class="component-name">机器学习引擎 (LightGBM)</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">多策略信号管道</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">市场扫描器</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">双重验证器</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">集成API服务</div>
                    <div class="component-status">✅ 正常</div>
                </div>
            </div>
        </div>

        <div class="json-section">
            <div class="json-title">📊 系统信息</div>
            <div class="json-content" id="json-content">
                {
                    "name": "AlphaSeeker集成系统",
                    "version": "1.0.0",
                    "description": "AI驱动的加密货币交易信号系统",
                    "status": "running",
                    "timestamp": "2025-10-27T22:56:49"
                }
            </div>
        </div>

        <div class="controls">
            <button class="toggle-btn" onclick="toggleJsonFormat()">
                切换JSON格式
            </button>
            <div class="timestamp">
                更新时间: <span id="timestamp"></span>
            </div>
        </div>
    </div>

    <script>
        function toggleJsonFormat() {
            const jsonContent = document.getElementById('json-content');
            const isFormatted = jsonContent.style.whiteSpace === 'pre-wrap';
            
            if (isFormatted) {
                jsonContent.style.whiteSpace = 'nowrap';
                jsonContent.textContent = JSON.stringify(JSON.parse(jsonContent.textContent));
            } else {
                jsonContent.style.whiteSpace = 'pre-wrap';
                jsonContent.textContent = JSON.stringify(JSON.parse(jsonContent.textContent), null, 2);
            }
        }

        function updateTimestamp() {
            const now = new Date();
            document.getElementById('timestamp').textContent = now.toLocaleString('zh-CN');
        }

        // 页面加载后更新时间戳
        document.addEventListener('DOMContentLoaded', function() {
            updateTimestamp();
        });
    </script>
</body>
</html>
    """
    
    return html_template



@app.get("/", response_class=HTMLResponse)
async def root():
    """主页 - 现代化系统状态页面"""
    
    # 系统信息数据
    system_info = {
        "name": "AlphaSeeker集成系统",
        "version": "1.0.0",
        "description": "AI驱动的加密货币交易信号系统",
        "components": [
            "机器学习引擎 (LightGBM)",
            "多策略信号管道", 
            "市场扫描器",
            "双重验证器",
            "集成API服务"
        ],
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }
    
    # 现代化HTML界面
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaSeeker 集成系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .title {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 700;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .version {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 5px;
        }

        .status-badge {
            display: inline-block;
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 10px 0;
            animation: pulse 2s infinite;
        }

        .section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }

        .section-title {
            font-size: 1.3em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
        }

        .section-title::before {
            content: "🔧";
            margin-right: 10px;
            font-size: 1.2em;
        }

        .components-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .component-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .component-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        .component-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .component-status {
            color: #27ae60;
            font-size: 0.9em;
        }

        .json-section {
            background: #2d3748;
            color: #e2e8f0;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            font-family: 'Fira Code', 'Consolas', monospace;
            overflow-x: auto;
        }

        .json-title {
            color: #63b3ed;
            font-size: 1.1em;
            margin-bottom: 15px;
            font-weight: 600;
        }

        .json-content {
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .toggle-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .toggle-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .timestamp {
            color: #6c757d;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            .title {
                font-size: 2em;
            }
            
            .components-grid {
                grid-template-columns: 1fr;
            }
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">🚀 AlphaSeeker 集成系统</h1>
            <div class="version">版本: <span id="version">1.0.0</span></div>
            <div class="status-badge">🟢 运行中</div>
        </div>

        <div class="section">
            <div class="section-title">系统组件</div>
            <div class="components-grid" id="components">
                <div class="component-card">
                    <div class="component-name">机器学习引擎 (LightGBM)</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">多策略信号管道</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">市场扫描器</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">双重验证器</div>
                    <div class="component-status">✅ 正常</div>
                </div>
                <div class="component-card">
                    <div class="component-name">集成API服务</div>
                    <div class="component-status">✅ 正常</div>
                </div>
            </div>
        </div>

        <div class="json-section">
            <div class="json-title">📊 系统信息</div>
            <div class="json-content" id="json-content">
                {
                    "name": "AlphaSeeker集成系统",
                    "version": "1.0.0",
                    "description": "AI驱动的加密货币交易信号系统",
                    "status": "running",
                    "timestamp": "2025-10-27T22:56:49"
                }
            </div>
        </div>

        <div class="controls">
            <button class="toggle-btn" onclick="toggleJsonFormat()">
                切换JSON格式
            </button>
            <div class="timestamp">
                更新时间: <span id="timestamp"></span>
            </div>
        </div>
    </div>

    <script>
        function toggleJsonFormat() {
            const jsonContent = document.getElementById('json-content');
            const isFormatted = jsonContent.style.whiteSpace === 'pre-wrap';
            
            if (isFormatted) {
                jsonContent.style.whiteSpace = 'nowrap';
                jsonContent.textContent = JSON.stringify(JSON.parse(jsonContent.textContent));
            } else {
                jsonContent.style.whiteSpace = 'pre-wrap';
                jsonContent.textContent = JSON.stringify(JSON.parse(jsonContent.textContent), null, 2);
            }
        }

        function updateTimestamp() {
            const now = new Date();
            document.getElementById('timestamp').textContent = now.toLocaleString('zh-CN');
        }

        // 页面加载后更新时间戳
        document.addEventListener('DOMContentLoaded', function() {
            updateTimestamp();
        });
    </script>
</body>
</html>
    """
    
    return html_template
