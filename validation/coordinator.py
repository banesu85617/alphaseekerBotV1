"""
双重验证机制核心协调器
实现LightGBM快速筛选 + 本地LLM深度评估的两层验证流程
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd

from .lgbm_filter import LightGBMFilter
from .llm_evaluator import LLMEvaluator
from .fusion_algorithm import ValidationFusion
from .config import ValidationConfig
from .monitoring import ValidationMonitor
from .utils import TimeoutManager, RetryManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """验证状态枚举"""
    PENDING = "pending"
    LAYER1_PASSED = "layer1_passed"
    LAYER2_PASSED = "layer2_passed"
    LAYER1_FAILED = "layer1_failed"
    LAYER2_FAILED = "layer2_failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    HOLD = "hold"


class ValidationPriority(Enum):
    """验证优先级"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class ValidationRequest:
    """验证请求数据类"""
    symbol: str
    timeframe: str
    current_price: float
    features: Dict[str, Any]
    indicators: Dict[str, float]
    risk_context: Dict[str, float]
    priority: ValidationPriority = ValidationPriority.MEDIUM
    request_id: str = None
    timestamp: float = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = f"{self.symbol}_{self.timeframe}_{int(time.time() * 1000)}"
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Layer1Result:
    """第一层验证结果"""
    label: int  # -1: 卖出, 0: 持有, 1: 买入
    probability: float
    confidence: float
    processing_time: float
    passed: bool
    reason: str = ""


@dataclass
class Layer2Result:
    """第二层验证结果"""
    direction: str  # "long", "short", "hold"
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    risk_assessment: str
    analysis_summary: str
    processing_time: float
    passed: bool
    reason: str = ""


@dataclass
class ValidationResult:
    """最终验证结果"""
    request_id: str
    symbol: str
    timeframe: str
    status: ValidationStatus
    layer1_result: Optional[Layer1Result]
    layer2_result: Optional[Layer2Result]
    combined_score: Optional[float]
    risk_reward_ratio: Optional[float]
    total_processing_time: float
    timestamp: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        # 处理Enum类型
        result['status'] = self.status.value
        return result


class SignalValidationCoordinator:
    """
    双重验证机制核心协调器
    
    负责协调两层验证流程：
    1. 第一层：LightGBM快速筛选
    2. 第二层：本地LLM深度评估
    3. 结果融合与综合评分
    """

    def __init__(self, config: ValidationConfig):
        """
        初始化协调器
        
        Args:
            config: 验证配置对象
        """
        self.config = config
        self.lgbm_filter = LightGBMFilter(config.lgbm_config)
        self.llm_evaluator = LLMEvaluator(config.llm_config)
        self.fusion = ValidationFusion(config.fusion_config)
        self.monitor = ValidationMonitor(config.monitoring_config)
        self.timeout_manager = TimeoutManager(config.timeout_config)
        self.retry_manager = RetryManager(config.retry_config)
        
        # 验证队列
        self.validation_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.processing_tasks = []
        self.shutdown_event = asyncio.Event()
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'layer1_passed': 0,
            'layer2_passed': 0,
            'timeout_errors': 0,
            'validation_errors': 0,
            'hold_results': 0
        }

    async def initialize(self) -> None:
        """初始化验证器组件"""
        logger.info("正在初始化双重验证协调器...")
        
        try:
            # 初始化LightGBM过滤器
            await self.lgbm_filter.initialize()
            logger.info("LightGBM过滤器初始化完成")
            
            # 初始化LLM评估器
            await self.llm_evaluator.initialize()
            logger.info("LLM评估器初始化完成")
            
            # 初始化融合器
            await self.fusion.initialize()
            logger.info("融合算法初始化完成")
            
            # 启动监控
            await self.monitor.initialize()
            logger.info("监控系统初始化完成")
            
            # 启动工作进程
            await self._start_workers()
            logger.info("工作进程启动完成")
            
        except Exception as e:
            logger.error(f"协调器初始化失败: {str(e)}")
            raise

    async def validate_signal(self, request: ValidationRequest) -> ValidationResult:
        """
        执行信号验证
        
        Args:
            request: 验证请求
            
        Returns:
            验证结果
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        logger.info(f"开始验证信号: {request.symbol} {request.timeframe}")
        
        try:
            # 第一层验证
            layer1_result = await self._execute_layer1_validation(request)
            
            if not layer1_result.passed:
                return ValidationResult(
                    request_id=request.request_id,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    status=ValidationStatus.LAYER1_FAILED,
                    layer1_result=layer1_result,
                    layer2_result=None,
                    combined_score=None,
                    risk_reward_ratio=None,
                    total_processing_time=time.time() - start_time,
                    timestamp=time.time(),
                    metadata={'reason': layer1_result.reason}
                )

            # 第二层验证
            layer2_result = await self._execute_layer2_validation(request, layer1_result)
            
            # 结果融合
            fusion_result = await self._fuse_results(
                request, layer1_result, layer2_result
            )
            
            # 更新统计
            self._update_stats(layer1_result, layer2_result, fusion_result)
            
            # 记录性能指标
            processing_time = time.time() - start_time
            await self.monitor.record_validation_performance(
                request.request_id, processing_time, fusion_result
            )
            
            logger.info(f"信号验证完成: {request.symbol}, "
                       f"状态: {fusion_result.status.value}, "
                       f"耗时: {processing_time:.3f}s")
            
            return fusion_result
            
        except asyncio.TimeoutError:
            self.stats['timeout_errors'] += 1
            logger.warning(f"验证超时: {request.request_id}")
            return ValidationResult(
                request_id=request.request_id,
                symbol=request.symbol,
                timeframe=request.timeframe,
                status=ValidationStatus.TIMEOUT,
                layer1_result=None,
                layer2_result=None,
                combined_score=None,
                risk_reward_ratio=None,
                total_processing_time=time.time() - start_time,
                timestamp=time.time(),
                metadata={'error': 'timeout'}
            )
            
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.error(f"验证异常: {str(e)}")
            return ValidationResult(
                request_id=request.request_id,
                symbol=request.symbol,
                timeframe=request.timeframe,
                status=ValidationStatus.ERROR,
                layer1_result=None,
                layer2_result=None,
                combined_score=None,
                risk_reward_ratio=None,
                total_processing_time=time.time() - start_time,
                timestamp=time.time(),
                metadata={'error': str(e)}
            )

    async def _execute_layer1_validation(self, request: ValidationRequest) -> Layer1Result:
        """执行第一层验证（LightGBM快速筛选）"""
        start_time = time.time()
        
        try:
            # 设置超时
            async with self.timeout_manager.timeout_context(
                self.config.timeout_config.layer1_timeout
            ) as timeout_task:
                # 执行LightGBM筛选
                result = await self.lgbm_filter.validate(request.features)
                
                processing_time = time.time() - start_time
                
                # 检查是否通过第一层
                passed = self._check_layer1_gate(result)
                if not passed:
                    self.stats['layer1_passed'] += 1
                else:
                    self.stats['layer1_passed'] += 1
                
                return Layer1Result(
                    label=result['label'],
                    probability=result['probability'],
                    confidence=result['confidence'],
                    processing_time=processing_time,
                    passed=passed,
                    reason="Gate condition not met" if not passed else ""
                )
                
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(f"第一层验证超时: {request.request_id}")
            return Layer1Result(
                label=0,
                probability=0.0,
                confidence=0.0,
                processing_time=processing_time,
                passed=False,
                reason="Layer1 timeout"
            )

    async def _execute_layer2_validation(
        self, 
        request: ValidationRequest, 
        layer1_result: Layer1Result
    ) -> Layer2Result:
        """执行第二层验证（LLM深度评估）"""
        start_time = time.time()
        
        try:
            # 设置超时
            async with self.timeout_manager.timeout_context(
                self.config.timeout_config.layer2_timeout
            ) as timeout_task:
                # 构建LLM评估输入
                evaluation_input = {
                    'symbol': request.symbol,
                    'timeframe': request.timeframe,
                    'current_price': request.current_price,
                    'technical_indicators': request.indicators,
                    'risk_context': request.risk_context,
                    'layer1_prediction': {
                        'label': layer1_result.label,
                        'probability': layer1_result.probability,
                        'confidence': layer1_result.confidence
                    },
                    'constraints': self.config.llm_config.output_constraints
                }
                
                # 执行LLM评估
                result = await self.llm_evaluator.evaluate(evaluation_input)
                
                processing_time = time.time() - start_time
                
                # 验证结果并检查R/R比
                validated_result = self._validate_layer2_result(result)
                validated_result.processing_time = processing_time
                
                self.stats['layer2_passed'] += 1 if validated_result.passed else 0
                
                return validated_result
                
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(f"第二层验证超时: {request.request_id}")
            return Layer2Result(
                direction="hold",
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                confidence=0.0,
                risk_assessment="Timeout",
                analysis_summary="Layer2 validation timeout",
                processing_time=processing_time,
                passed=False,
                reason="Layer2 timeout"
            )

    def _check_layer1_gate(self, lgbm_result: Dict[str, Any]) -> bool:
        """检查第一层门控条件"""
        # 检查概率阈值
        if lgbm_result['probability'] < self.config.lgbm_config.probability_threshold:
            return False
        
        # 检查标签
        if self.config.lgbm_config.require_direction and lgbm_result['label'] == 0:
            return False
        
        # 检查置信度
        if lgbm_result['confidence'] < self.config.lgbm_config.confidence_threshold:
            return False
        
        return True

    def _validate_layer2_result(self, result: Dict[str, Any]) -> Layer2Result:
        """验证第二层结果并计算R/R比"""
        try:
            # 验证必需字段
            required_fields = ['direction', 'entry', 'stop_loss', 'take_profit', 'confidence']
            for field in required_fields:
                if field not in result:
                    result[field] = None if field != 'direction' else 'hold'
            
            # 验证方向枚举
            if result['direction'] not in ['long', 'short', 'hold']:
                result['direction'] = 'hold'
            
            # 计算风险回报比
            risk_reward_ratio = None
            if (result['entry'] is not None and 
                result['stop_loss'] is not None and 
                result['take_profit'] is not None):
                
                if result['direction'] == 'long':
                    risk = result['entry'] - result['stop_loss']
                    reward = result['take_profit'] - result['entry']
                elif result['direction'] == 'short':
                    risk = result['stop_loss'] - result['entry']
                    reward = result['entry'] - result['take_profit']
                else:
                    risk = reward = 0
                
                if risk > 0:
                    risk_reward_ratio = reward / risk
            
            # 检查R/R比阈值
            passed = True
            reason = ""
            
            if result['direction'] == 'hold':
                passed = False
                reason = "Direction set to hold"
            elif risk_reward_ratio is None:
                passed = False
                reason = "Unable to calculate risk/reward ratio"
            elif risk_reward_ratio < self.config.fusion_config.risk_reward_threshold:
                passed = False
                reason = f"Risk/reward ratio {risk_reward_ratio:.2f} below threshold"
            elif result['confidence'] < self.config.llm_config.min_confidence:
                passed = False
                reason = f"Confidence {result['confidence']:.2f} below minimum"
            
            return Layer2Result(
                direction=result['direction'],
                entry_price=result['entry'],
                stop_loss=result['stop_loss'],
                take_profit=result['take_profit'],
                confidence=result['confidence'],
                risk_assessment=result.get('risk_assessment', ''),
                analysis_summary=result.get('analysis_summary', ''),
                processing_time=0.0,
                passed=passed,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"第二层结果验证失败: {str(e)}")
            return Layer2Result(
                direction="hold",
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                confidence=0.0,
                risk_assessment="Validation failed",
                analysis_summary="Result validation failed",
                processing_time=0.0,
                passed=False,
                reason=str(e)
            )

    async def _fuse_results(
        self, 
        request: ValidationRequest, 
        layer1_result: Layer1Result, 
        layer2_result: Layer2Result
    ) -> ValidationResult:
        """融合两层验证结果"""
        
        # 确定最终状态
        if layer2_result.passed:
            status = ValidationStatus.LAYER2_PASSED
        elif layer1_result.passed and not layer2_result.passed:
            status = ValidationStatus.HOLD
        else:
            status = ValidationStatus.LAYER1_FAILED
        
        # 计算综合评分
        fusion_input = {
            'layer1_result': asdict(layer1_result),
            'layer2_result': asdict(layer2_result),
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'current_price': request.current_price,
            'technical_indicators': request.indicators,
            'risk_context': request.risk_context
        }
        
        combined_score = await self.fusion.calculate_combined_score(fusion_input)
        risk_reward_ratio = self._calculate_risk_reward_ratio(layer2_result)
        
        return ValidationResult(
            request_id=request.request_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            status=status,
            layer1_result=layer1_result,
            layer2_result=layer2_result,
            combined_score=combined_score,
            risk_reward_ratio=risk_reward_ratio,
            total_processing_time=0.0,  # 将在调用方计算
            timestamp=time.time(),
            metadata={
                'layer1_processing_time': layer1_result.processing_time,
                'layer2_processing_time': layer2_result.processing_time
            }
        )

    def _calculate_risk_reward_ratio(self, layer2_result: Layer2Result) -> Optional[float]:
        """计算风险回报比"""
        if (layer2_result.entry_price is None or 
            layer2_result.stop_loss is None or 
            layer2_result.take_profit is None):
            return None
        
        if layer2_result.direction == 'long':
            risk = layer2_result.entry_price - layer2_result.stop_loss
            reward = layer2_result.take_profit - layer2_result.entry_price
        elif layer2_result.direction == 'short':
            risk = layer2_result.stop_loss - layer2_result.entry_price
            reward = layer2_result.entry_price - layer2_result.take_profit
        else:
            return None
        
        return reward / risk if risk > 0 else None

    def _update_stats(
        self, 
        layer1_result: Layer1Result, 
        layer2_result: Layer2Result, 
        final_result: ValidationResult
    ) -> None:
        """更新性能统计"""
        if final_result.status == ValidationStatus.HOLD:
            self.stats['hold_results'] += 1

    async def _start_workers(self) -> None:
        """启动工作进程"""
        for i in range(self.config.max_concurrent_tasks):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.processing_tasks.append(task)

    async def _worker(self, worker_id: str) -> None:
        """工作进程"""
        logger.info(f"工作进程 {worker_id} 已启动")
        
        while not self.shutdown_event.is_set():
            try:
                # 从队列获取验证请求
                request = await asyncio.wait_for(
                    self.validation_queue.get(), 
                    timeout=1.0
                )
                
                # 执行验证
                await self.validate_signal(request)
                
            except asyncio.TimeoutError:
                # 继续循环
                continue
            except Exception as e:
                logger.error(f"工作进程 {worker_id} 异常: {str(e)}")
                await asyncio.sleep(1)

    async def batch_validate(self, requests: List[ValidationRequest]) -> List[ValidationResult]:
        """
        批量验证信号
        
        Args:
            requests: 验证请求列表
            
        Returns:
            验证结果列表
        """
        logger.info(f"开始批量验证 {len(requests)} 个信号")
        
        # 创建验证任务
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.validate_signal(request))
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量验证任务 {i} 异常: {str(result)}")
                processed_results.append(
                    ValidationResult(
                        request_id=requests[i].request_id,
                        symbol=requests[i].symbol,
                        timeframe=requests[i].timeframe,
                        status=ValidationStatus.ERROR,
                        layer1_result=None,
                        layer2_result=None,
                        combined_score=None,
                        risk_reward_ratio=None,
                        total_processing_time=0.0,
                        timestamp=time.time(),
                        metadata={'error': str(result)}
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'total_requests': self.stats['total_requests'],
            'layer1_passed': self.stats['layer1_passed'],
            'layer2_passed': self.stats['layer2_passed'],
            'timeout_errors': self.stats['timeout_errors'],
            'validation_errors': self.stats['validation_errors'],
            'hold_results': self.stats['hold_results'],
            'success_rate': (
                (self.stats['total_requests'] - self.stats['validation_errors']) / 
                max(self.stats['total_requests'], 1)
            )
        }

    async def shutdown(self) -> None:
        """关闭协调器"""
        logger.info("正在关闭双重验证协调器...")
        
        # 设置关闭事件
        self.shutdown_event.set()
        
        # 等待工作进程完成
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # 关闭组件
        await self.lgbm_filter.shutdown()
        await self.llm_evaluator.shutdown()
        await self.fusion.shutdown()
        await self.monitor.shutdown()
        
        logger.info("双重验证协调器已关闭")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.shutdown()