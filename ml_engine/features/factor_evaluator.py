"""
AlphaSeeker-Bot Alpha因子评估与分级模块
基于分析文档中的IC、分位数收益价差、Sharpe、Sortino、Calmar和p值评估体系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import FACTOR_CONFIG


class AlphaFactorEvaluator:
    """
    Alpha因子评估器
    
    基于分析文档实现：
    - IC(信息系数)计算
    - 分位数收益价差分析
    - 风险调整收益指标(Sharpe, Sortino, Calmar)
    - 统计显著性检验(p值)
    - 因子分级(AAA-E)
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or FACTOR_CONFIG
        self.scoring_weights = self.config["SCORING_WEIGHTS"]
        self.grade_thresholds = self.config["GRADE_THRESHOLDS"]
        self.logger = logging.getLogger(__name__)
        
    def calculate_ic(self, factor: pd.Series, returns: pd.Series, 
                    lag: int = 1) -> Dict:
        """
        计算信息系数(IC)
        
        Args:
            factor: 因子值序列
            returns: 未来收益序列
            lag: 滞后阶数，避免前瞻偏差
            
        Returns:
            IC计算结果字典
        """
        # 滞后因子值，避免前瞻偏差
        factor_lagged = factor.shift(lag)
        
        # 移除缺失值
        valid_idx = ~(factor_lagged.isna() | returns.isna())
        factor_clean = factor_lagged[valid_idx]
        returns_clean = returns[valid_idx]
        
        if len(factor_clean) == 0:
            return {"ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "n_observations": 0}
            
        # 计算IC
        ic = np.corrcoef(factor_clean, returns_clean)[0, 1]
        
        # 滚动IC（可选）
        rolling_ic = []
        window = min(60, len(factor_clean) // 4)  # 使用数据长度的1/4或60的较小值
        
        for i in range(window, len(factor_clean)):
            window_factor = factor_clean.iloc[i-window:i]
            window_returns = returns_clean.iloc[i-window:i]
            
            if len(window_factor) > 10:  # 至少需要10个观测值
                window_ic = np.corrcoef(window_factor, window_returns)[0, 1]
                if not np.isnan(window_ic):
                    rolling_ic.append(window_ic)
                    
        # 计算IC统计量
        if rolling_ic:
            ic_std = np.std(rolling_ic)
            ic_ir = ic / (ic_std + 1e-8) if ic_std > 0 else 0
        else:
            ic_std = 0.0
            ic_ir = 0.0
            
        return {
            "ic": ic,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "rolling_ic": rolling_ic,
            "n_observations": len(factor_clean)
        }
    
    def calculate_return_spread(self, factor: pd.Series, returns: pd.Series,
                              n_quantiles: int = 5, lag: int = 1) -> Dict:
        """
        计算分位数收益价差
        
        Args:
            factor: 因子值序列
            returns: 未来收益序列
            n_quantiles: 分位数数量
            lag: 滞后阶数
            
        Returns:
            分位数收益价差结果字典
        """
        # 滞后因子值
        factor_lagged = factor.shift(lag)
        
        # 合并数据并移除缺失值
        data = pd.DataFrame({"factor": factor_lagged, "returns": returns})
        data = data.dropna()
        
        if len(data) == 0:
            return {"spread": 0.0, "quantile_returns": [], "top_return": 0.0, "bottom_return": 0.0}
            
        # 分位数分组
        data["quantile"] = pd.qcut(data["factor"], q=n_quantiles, labels=False, duplicates='drop')
        
        # 计算各分位数组的平均收益
        quantile_returns = data.groupby("quantile")["returns"].mean()
        
        # 价差(顶底分组)
        if len(quantile_returns) >= 2:
            top_return = quantile_returns.iloc[-1]  # 最高分位数组
            bottom_return = quantile_returns.iloc[0]  # 最低分位数组
            spread = top_return - bottom_return
        else:
            top_return = bottom_return = spread = 0.0
            
        return {
            "spread": spread,
            "quantile_returns": quantile_returns.to_dict(),
            "top_return": top_return,
            "bottom_return": bottom_return,
            "n_observations": len(data)
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益序列
            risk_free_rate: 无风险利率
            
        Returns:
            夏普比率
        """
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / (excess_returns.std() + 1e-8)
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.0) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益序列
            risk_free_rate: 无风险利率
            
        Returns:
            索提诺比率
        """
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        return excess_returns.mean() / (downside_std + 1e-8)
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        计算卡尔玛比率(收益/最大回撤)
        
        Args:
            returns: 收益序列
            
        Returns:
            卡尔玛比率
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown == 0:
            return 0.0
            
        return returns.mean() / max_drawdown
    
    def calculate_p_value(self, factor: pd.Series, returns: pd.Series, 
                         lag: int = 1) -> Dict:
        """
        计算统计显著性(p值)
        
        Args:
            factor: 因子值序列
            returns: 未来收益序列
            lag: 滞后阶数
            
        Returns:
            p值计算结果字典
        """
        # 滞后因子值
        factor_lagged = factor.shift(lag)
        
        # 合并数据并移除缺失值
        data = pd.DataFrame({"factor": factor_lagged, "returns": returns})
        data = data.dropna()
        
        if len(data) < 10:  # 数据太少
            return {"p_value": 1.0, "t_stat": 0.0, "significant": False}
            
        # 相关性检验
        correlation, p_value = stats.pearsonr(data["factor"], data["returns"])
        t_stat = correlation * np.sqrt((len(data) - 2) / (1 - correlation**2 + 1e-8))
        
        return {
            "p_value": p_value,
            "t_stat": t_stat,
            "significant": p_value < self.config["SIGNIFICANCE_LEVEL"]
        }
    
    def evaluate_factor(self, factor: pd.Series, returns: pd.Series) -> Dict:
        """
        综合评估单个因子
        
        Args:
            factor: 因子值序列
            returns: 未来收益序列
            
        Returns:
            因子评估结果字典
        """
        self.logger.info(f"开始评估因子，数据长度: 因子={len(factor)}, 收益={len(returns)}")
        
        # 计算各项指标
        ic_result = self.calculate_ic(factor, returns)
        spread_result = self.calculate_return_spread(factor, returns)
        p_value_result = self.calculate_p_value(factor, returns)
        
        # 风险调整收益（基于因子分位数组的收益）
        factor_lagged = factor.shift(1)
        data = pd.DataFrame({"factor": factor_lagged, "returns": returns}).dropna()
        
        if len(data) > 10:
            data["quantile"] = pd.qcut(data["factor"], q=5, labels=False, duplicates='drop')
            quantile_returns = data.groupby("quantile")["returns"].apply(lambda x: x.values)
            
            # 选择极端分位数组的收益序列
            extreme_returns = pd.concat([
                quantile_returns.iloc[0],  # 最低分位数组
                quantile_returns.iloc[-1]  # 最高分位数组
            ])
            
            sharpe = self.calculate_sharpe_ratio(extreme_returns)
            sortino = self.calculate_sortino_ratio(extreme_returns)
            calmar = self.calculate_calmar_ratio(extreme_returns)
        else:
            sharpe = sortino = calmar = 0.0
            
        # 组装结果
        result = {
            "ic": ic_result["ic"],
            "ic_std": ic_result["ic_std"],
            "ic_ir": ic_result["ic_ir"],
            "return_spread": spread_result["spread"],
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "p_value": p_value_result["p_value"],
            "significant": p_value_result["significant"],
            "n_observations": min(ic_result["n_observations"], spread_result["n_observations"])
        }
        
        self.logger.info(f"因子评估完成: IC={result['ic']:.4f}, 价差={result['return_spread']:.6f}")
        
        return result
    
    def score_factor(self, factor_metrics: Dict) -> Dict:
        """
        对因子进行打分
        
        Args:
            factor_metrics: 因子指标字典
            
        Returns:
            因子评分字典
        """
        # IC评分 (0-4分)
        ic_score = min(4, max(0, abs(factor_metrics["ic"]) * 80))
        
        # p值评分 (0-4分) - p值越小分数越高
        if factor_metrics["p_value"] <= 0.01:
            p_value_score = 4
        elif factor_metrics["p_value"] <= 0.05:
            p_value_score = 3
        elif factor_metrics["p_value"] <= 0.1:
            p_value_score = 2
        elif factor_metrics["p_value"] <= 0.2:
            p_value_score = 1
        else:
            p_value_score = 0
            
        # Sharpe评分 (0-4分)
        sharpe_score = min(4, max(0, factor_metrics["sharpe"]))
        
        # Sortino评分 (0-4分)  
        sortino_score = min(4, max(0, factor_metrics["sortino"]))
        
        # 总分计算
        total_score = (
            ic_score * self.scoring_weights["IC_score"] +
            p_value_score * self.scoring_weights["p_value_score"] +
            sharpe_score * self.scoring_weights["Sharpe_score"] +
            sortino_score * self.scoring_weights["Sortino_score"]
        )
        
        return {
            "ic_score": ic_score,
            "p_value_score": p_value_score,
            "sharpe_score": sharpe_score,
            "sortino_score": sortino_score,
            "total_score": total_score
        }
    
    def grade_factor(self, factor_metrics: Dict, factor_scores: Dict) -> str:
        """
        对因子进行分级(AAA-E)
        
        Args:
            factor_metrics: 因子指标字典
            factor_scores: 因子评分字典
            
        Returns:
            因子等级
        """
        total_score = factor_scores["total_score"]
        
        if total_score >= 15:
            return "AAA"
        elif total_score >= 12:
            return "AA"
        elif total_score >= 9:
            return "A"
        elif total_score >= 6:
            return "B"
        elif total_score >= 3:
            return "C"
        elif total_score >= 1:
            return "D"
        else:
            return "E"
    
    def evaluate_multiple_factors(self, factors_df: pd.DataFrame, 
                                returns: pd.Series) -> pd.DataFrame:
        """
        评估多个因子
        
        Args:
            factors_df: 因子DataFrame (列为因子名，行为时间)
            returns: 收益序列
            
        Returns:
            包含所有评估结果的DataFrame
        """
        results = []
        
        for factor_name in factors_df.columns:
            try:
                self.logger.info(f"评估因子: {factor_name}")
                
                # 评估因子
                metrics = self.evaluate_factor(factors_df[factor_name], returns)
                
                # 因子打分
                scores = self.score_factor(metrics)
                
                # 因子分级
                grade = self.grade_factor(metrics, scores)
                
                # 组装结果
                result = {
                    "factor_name": factor_name,
                    "grade": grade,
                    "total_score": scores["total_score"],
                    **metrics,
                    **scores
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"评估因子 {factor_name} 时出错: {str(e)}")
                continue
                
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 按总分排序
        results_df = results_df.sort_values("total_score", ascending=False)
        
        self.logger.info(f"完成 {len(results)} 个因子的评估")
        
        return results_df
    
    def select_top_factors(self, factor_evaluation_df: pd.DataFrame,
                          top_n: int = 20, 
                          min_grade: str = "C") -> List[str]:
        """
        选择顶级因子
        
        Args:
            factor_evaluation_df: 因子评估结果DataFrame
            top_n: 选择因子数量
            min_grade: 最小等级
            
        Returns:
            选中因子名称列表
        """
        # 等级过滤
        grade_order = ["AAA", "AA", "A", "B", "C", "D", "E"]
        min_grade_idx = grade_order.index(min_grade)
        
        filtered_df = factor_evaluation_df[
            factor_evaluation_df["grade"].apply(
                lambda x: grade_order.index(x) <= min_grade_idx
            )
        ]
        
        # 选择顶级因子
        selected_factors = filtered_df.head(top_n)["factor_name"].tolist()
        
        self.logger.info(f"选择 {len(selected_factors)} 个顶级因子")
        
        return selected_factors
    
    def generate_factor_report(self, factor_evaluation_df: pd.DataFrame) -> str:
        """
        生成因子评估报告
        
        Args:
            factor_evaluation_df: 因子评估结果DataFrame
            
        Returns:
            报告字符串
        """
        report = []
        report.append("# Alpha因子评估报告\n")
        
        # 总体统计
        report.append("## 总体统计")
        report.append(f"- 评估因子数量: {len(factor_evaluation_df)}")
        report.append(f"- 平均IC: {factor_evaluation_df['ic'].mean():.4f}")
        report.append(f"- 平均价差: {factor_evaluation_df['return_spread'].mean():.6f}")
        
        # 等级分布
        grade_dist = factor_evaluation_df["grade"].value_counts().sort_index()
        report.append("\n## 等级分布")
        for grade, count in grade_dist.items():
            report.append(f"- {grade}: {count} 个因子")
            
        # 顶级因子
        report.append("\n## 顶级因子(前10)")
        top_factors = factor_evaluation_df.head(10)
        for _, row in top_factors.iterrows():
            report.append(f"- {row['factor_name']}: {row['grade']} (IC={row['ic']:.4f}, 价差={row['return_spread']:.6f})")
            
        return "\n".join(report)