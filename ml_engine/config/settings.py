"""
AlphaSeeker-Bot机器学习引擎配置模块
基于分析文档中的系统参数和架构设计
"""

# =============================================================================
# LightGBM模型配置
# =============================================================================
MODEL_CONFIG = {
    "objective": "multiclass",
    "num_class": 3,
    "random_state": 42,
    "class_weight": "balanced",  # 类别平衡处理
    "verbose": -1,
    "n_jobs": -1,
    # 可调整的超参数（在实际训练中会被优化）
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "max_depth": 6,
}

# =============================================================================
# 目标变量和阈值配置
# =============================================================================
TARGET_CONFIG = {
    "HORIZON_SECONDS": 300,  # 未来收益窗口(5分钟)
    "PROFIT_THRESHOLD": 0.0002,  # 未来收益阈值(万分之二)
    "LABELS": {
        1: "BUY",    # 买入
        0: "HOLD",   # 持有  
        -1: "SELL"   # 卖出
    }
}

# =============================================================================
# 特征工程配置
# =============================================================================
FEATURE_CONFIG = {
    # 核心微结构特征（基于分析文档中的60余项特征）
    "CORE_FEATURES": [
        # 价格相关
        "mid_price", "spread", "wap_1", "wap_5", 
        "wap_1_diff", "mid_price_diff", "spread_diff",
        
        # 订单簿不平衡
        "order_imbalance_1", "depth_imbalance",
        "bid_volume", "ask_volume", "total_volume",
        
        # 波动率和统计特征
        "volatility_60s", "price_volatility", "return_std",
        
        # 成交量特征
        "volume_trend", "volume_change", "volume_ratio",
        
        # 微结构特征
        "trade_intensity", "order_flow_imbalance", 
        "price_impact", "effective_spread"
    ],
    
    # 特征差分和滞后配置
    "DIFF_FEATURES": True,  # 对价格相关特征进行差分
    "LAG_FEATURES": [1, 2, 3],  # 滞后特征窗口
    
    # 滚动统计窗口
    "ROLLING_WINDOWS": {
        "short": 60,    # 1分钟
        "medium": 300,  # 5分钟  
        "long": 900     # 15分钟
    }
}

# =============================================================================
# 因子评估配置
# =============================================================================
FACTOR_CONFIG = {
    # 评分权重（基于分析文档中的加权方案）
    "SCORING_WEIGHTS": {
        "IC_score": 4,
        "p_value_score": 3, 
        "Sharpe_score": 2,
        "Sortino_score": 1
    },
    
    # 分级阈值
    "GRADE_THRESHOLDS": {
        "AAA": {"IC": 0.05, "p_value": 0.01},
        "AA": {"IC": 0.03, "p_value": 0.05},
        "A": {"IC": 0.02, "p_value": 0.1},
        "B": {"IC": 0.01, "p_value": 0.2},
        "C": {"IC": 0.005, "p_value": 0.3},
        "D": {"IC": 0.001, "p_value": 0.5},
        "E": {"IC": 0.0, "p_value": 1.0}
    },
    
    # Bonferroni校正
    "MULTIPLE_TEST_CORRECTION": "bonferroni",
    "SIGNIFICANCE_LEVEL": 0.05
}

# =============================================================================
# 风险管理配置
# =============================================================================
RISK_CONFIG = {
    # 止盈止损（基于分析文档中的0.4%/0.4%）
    "TAKE_PROFIT_PCT": 0.004,  # 0.4%
    "STOP_LOSS_PCT": 0.004,    # 0.4%
    
    # 交易成本（基于分析文档中的0.05%）
    "TRANSACTION_COST_PCT": 0.0005,  # 0.05%
    
    # 风险控制
    "MAX_POSITION_SIZE": 1.0,  # 最大仓位
    "MAX_DAILY_LOSS": 0.02,    # 最大日亏损2%
    "MAX_DRAWDOWN": 0.1,       # 最大回撤10%
    
    # 动态风控
    "VOLATILITY_BASED_SL": True,  # 基于波动的动态止损
    "VOLATILITY_MULTIPLIER": 2.0,
}

# =============================================================================
# 推理性能配置
# =============================================================================
INFERENCE_CONFIG = {
    "TARGET_LATENCY_MS": 500,  # 目标推理延迟0.5秒
    "ENABLE_CACHING": True,    # 启用特征缓存
    "BATCH_SIZE": 32,          # 推理批次大小
    "PRECISION": "float32",    # 推理精度
    "NUM_THREADS": -1,         # 推理线程数
}

# =============================================================================
# 数据存储配置
# =============================================================================
DATA_CONFIG = {
    # SQLite数据库配置
    "DB_PATH": "market_features.db",
    "TABLE_NAME": "market_features",
    "TIMESTAMP_COLUMN": "timestamp_utc",
    
    # 时间序列切分
    "TRAIN_TEST_SPLIT": 0.8,
    "VAL_SPLIT": 0.2,
    "SHUFFLE": False,  # 时间序列不shuffle
    
    # 数据质量
    "MIN_DATA_POINTS": 1000,
    "MAX_MISSING_RATIO": 0.1,  # 最大缺失比例10%
    "OUTLIER_THRESHOLD": 3.0,  # 异常值标准差倍数
}

# =============================================================================
# 监控和评估配置
# =============================================================================
MONITORING_CONFIG = {
    # 模型重训触发条件
    "RETRAIN_TRIGGERS": {
        "days": 7,                    # 7天重训
        "ic_threshold": 0.01,         # IC低于阈值
        "drawdown_threshold": 0.05,   # 回撤超过5%
        "performance_decline": 0.1    # 性能下降10%
    },
    
    # 性能指标监控
    "METRICS": [
        "IC", "Return_Spread", "Sharpe", 
        "Sortino", "Calmar", "p_value"
    ],
    
    # 日志配置
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# =============================================================================
# 导出所有配置
# =============================================================================
__all__ = [
    "MODEL_CONFIG",
    "TARGET_CONFIG", 
    "FEATURE_CONFIG",
    "FACTOR_CONFIG",
    "RISK_CONFIG",
    "INFERENCE_CONFIG",
    "DATA_CONFIG", 
    "MONITORING_CONFIG"
]