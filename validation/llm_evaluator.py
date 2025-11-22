"""
本地LLM深度评估器
实现第二层的LLM参数建议与解释性评估
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
import httpx

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """LLM提供商枚举"""
    LM_STUDIO = "lm_studio"
    ANYTHING_LLM = "anything_llm"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider = LLMProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    model_name: str = "llama2"
    api_key: Optional[str] = None
    
    # 请求配置
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 生成配置
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 0.9
    
    # 验证配置
    min_confidence: float = 0.65
    output_constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.output_constraints is None:
            self.output_constraints = {
                'required_fields': ['direction', 'entry', 'stop_loss', 'take_profit', 'confidence'],
                'valid_directions': ['long', 'short', 'hold'],
                'min_confidence': self.min_confidence
            }


@dataclass
class LLMRequest:
    """LLM请求结构"""
    prompt: str
    system_prompt: str = ""
    temperature: float = 0.3
    max_tokens: int = 1024


class LLMEvaluator:
    """
    本地LLM深度评估器
    
    负责第二层的参数建议与解释性评估
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化LLM评估器
        
        Args:
            config: LLM配置对象
        """
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        self.provider_handlers = {
            LLMProvider.LM_STUDIO: self._handle_lm_studio,
            LLMProvider.ANYTHING_LLM: self._handle_anything_llm,
            LLMProvider.OLLAMA: self._handle_ollama
        }
        self.is_initialized = False

    async def initialize(self) -> None:
        """初始化LLM评估器"""
        logger.info(f"初始化LLM评估器，提供商: {self.config.provider.value}")
        
        try:
            # 创建HTTP客户端
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            # 健康检查
            await self._health_check()
            
            self.is_initialized = True
            logger.info("LLM评估器初始化完成")
            
        except Exception as e:
            logger.error(f"LLM评估器初始化失败: {str(e)}")
            raise

    async def _health_check(self) -> None:
        """健康检查"""
        try:
            if self.config.provider == LLMProvider.OLLAMA:
                # Ollama健康检查
                async with self.client.get(f"{self.config.base_url}/api/tags") as response:
                    if response.status_code != 200:
                        raise Exception(f"健康检查失败: {response.status_code}")
                    
                    data = response.json()
                    model_names = [model['name'] for model in data.get('models', [])]
                    
                    if self.config.model_name not in model_names:
                        logger.warning(f"模型 {self.config.model_name} 未找到，可用模型: {model_names}")
            
            elif self.config.provider == LLMProvider.LM_STUDIO:
                # LM Studio健康检查
                async with self.client.get(f"{self.config.base_url}/v1/models") as response:
                    if response.status_code != 200:
                        raise Exception(f"健康检查失败: {response.status_code}")
            
            logger.info("LLM健康检查通过")
            
        except Exception as e:
            logger.error(f"LLM健康检查失败: {str(e)}")
            raise

    async def evaluate(self, evaluation_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行LLM评估
        
        Args:
            evaluation_input: 评估输入数据
            
        Returns:
            评估结果字典
        """
        if not self.is_initialized:
            raise RuntimeError("LLM评估器未初始化")
        
        start_time = time.time()
        
        try:
            # 构建提示词
            prompt = self._build_evaluation_prompt(evaluation_input)
            
            # 执行LLM请求
            response = await self._call_llm(prompt)
            
            # 解析响应
            result = self._parse_llm_response(response)
            
            # 验证结果
            validated_result = self._validate_result(result)
            
            processing_time = time.time() - start_time
            validated_result['processing_time'] = processing_time
            
            logger.debug(f"LLM评估完成，耗时: {processing_time:.3f}s")
            
            return validated_result
            
        except Exception as e:
            logger.error(f"LLM评估失败: {str(e)}")
            return self._create_error_result(str(e))

    def _build_evaluation_prompt(self, evaluation_input: Dict[str, Any]) -> str:
        """构建评估提示词"""
        symbol = evaluation_input.get('symbol', 'UNKNOWN')
        timeframe = evaluation_input.get('timeframe', '1h')
        current_price = evaluation_input.get('current_price', 0.0)
        
        # 技术指标
        indicators = evaluation_input.get('technical_indicators', {})
        risk_context = evaluation_input.get('risk_context', {})
        
        # 第一层结果
        layer1 = evaluation_input.get('layer1_prediction', {})
        
        # 约束条件
        constraints = evaluation_input.get('constraints', {})
        
        prompt = f"""
你是一个专业的量化交易分析师。请基于以下信息对 {symbol} {timeframe} 进行深度技术分析，并提供交易建议。

当前价格: {current_price}

技术指标:
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- 布林带: {indicators.get('bollinger', 'N/A')}
- ADX: {indicators.get('adx', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}

风险上下文:
- 波动率: {risk_context.get('volatility', 'N/A')}
- VaR(95%): {risk_context.get('var_95', 'N/A')}

第一层预测:
- 方向: {layer1.get('label', 'N/A')}
- 概率: {layer1.get('probability', 'N/A')}
- 置信度: {layer1.get('confidence', 'N/A')}

请提供JSON格式的分析结果，包含以下字段:
{{
    "direction": "long|short|hold",
    "entry": float (建议入场价格),
    "stop_loss": float (止损价格),
    "take_profit": float (止盈价格),
    "confidence": float (0-1置信度),
    "risk_assessment": "风险评估",
    "analysis_summary": "四段式分析摘要"
}}

约束条件:
- 只在风险回报比 >= 1.0 时建议交易
- 置信度必须 >= {constraints.get('min_confidence', 0.65)}
- 考虑当前市场趋势和风险水平
- 如果信息不足或风险过高，请设置direction为"hold"

请基于技术分析给出专业的交易建议。
"""
        
        return prompt.strip()

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        if self.config.provider not in self.provider_handlers:
            raise ValueError(f"不支持的LLM提供商: {self.config.provider}")
        
        return await self.provider_handlers[self.config.provider](prompt)

    async def _handle_ollama(self, prompt: str) -> str:
        """处理Ollama请求"""
        url = f"{self.config.base_url}/api/generate"
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p
            }
        }
        
        async with self.client.post(url, json=payload) as response:
            if response.status_code != 200:
                raise Exception(f"Ollama API调用失败: {response.status_code}")
            
            data = response.json()
            return data.get('response', '')

    async def _handle_lm_studio(self, prompt: str) -> str:
        """处理LM Studio请求"""
        url = f"{self.config.base_url}/v1/chat/completions"
        
        headers = {}
        if self.config.api_key:
            headers['Authorization'] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p
        }
        
        async with self.client.post(url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                raise Exception(f"LM Studio API调用失败: {response.status_code}")
            
            data = response.json()
            return data['choices'][0]['message']['content']

    async def _handle_anything_llm(self, prompt: str) -> str:
        """处理AnythingLLM请求"""
        # AnythingLLM的API实现（需要根据实际API调整）
        url = f"{self.config.base_url}/api/chat"
        
        payload = {
            "message": prompt,
            "mode": "chat",
            "history": []
        }
        
        headers = {}
        if self.config.api_key:
            headers['Authorization'] = f"Bearer {self.config.api_key}"
        
        async with self.client.post(url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                raise Exception(f"AnythingLLM API调用失败: {response.status_code}")
            
            data = response.json()
            return data.get('response', '')

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试直接解析JSON
            # 清理响应文本，提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                return parsed
            else:
                # 如果没有找到JSON，尝试使用正则表达式
                import re
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, response)
                
                if matches:
                    try:
                        return json.loads(matches[0])
                    except:
                        pass
                
                logger.warning(f"无法解析LLM响应为JSON: {response[:200]}")
                return self._parse_text_response(response)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {str(e)}")
            return self._parse_text_response(response)

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """解析文本响应"""
        # 简化的文本解析逻辑
        result = {
            'direction': 'hold',
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0.5,
            'risk_assessment': 'Unable to parse LLM response',
            'analysis_summary': response[:500]  # 截取前500字符作为摘要
        }
        
        # 简单的关键词匹配
        response_lower = response.lower()
        
        if 'long' in response_lower and 'buy' in response_lower:
            result['direction'] = 'long'
        elif 'short' in response_lower and 'sell' in response_lower:
            result['direction'] = 'short'
        
        # 提取置信度
        confidence_match = response_lower.search(r'confidence[:\s]+([0-9.]+)', response)
        if confidence_match:
            try:
                result['confidence'] = min(1.0, float(confidence_match.group(1)))
            except:
                pass
        
        return result

    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证结果"""
        validated = result.copy()
        
        # 验证方向
        if validated.get('direction') not in ['long', 'short', 'hold']:
            logger.warning(f"无效的方向: {validated.get('direction')}，设置为hold")
            validated['direction'] = 'hold'
        
        # 验证数值字段
        numeric_fields = ['entry', 'stop_loss', 'take_profit', 'confidence']
        for field in numeric_fields:
            if field in validated:
                try:
                    value = float(validated[field])
                    if field == 'confidence' and not (0 <= value <= 1):
                        validated[field] = 0.5
                    else:
                        validated[field] = value
                except (ValueError, TypeError):
                    if field == 'confidence':
                        validated[field] = 0.5
                    else:
                        validated[field] = None
        
        # 确保必需字段存在
        required_fields = ['direction', 'entry', 'stop_loss', 'take_profit', 'confidence']
        for field in required_fields:
            if field not in validated:
                if field == 'confidence':
                    validated[field] = 0.5
                else:
                    validated[field] = None
        
        # 添加默认值
        if 'risk_assessment' not in validated:
            validated['risk_assessment'] = 'No risk assessment provided'
        
        if 'analysis_summary' not in validated:
            validated['analysis_summary'] = 'No analysis summary provided'
        
        return validated

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'direction': 'hold',
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0.0,
            'risk_assessment': f'LLM evaluation failed: {error_message}',
            'analysis_summary': 'LLM evaluation failed',
            'error': error_message,
            'processing_time': 0.0
        }

    async def batch_evaluate(self, evaluation_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量评估"""
        logger.info(f"开始批量LLM评估 {len(evaluation_inputs)} 个样本")
        
        results = []
        
        # 限制并发数量
        semaphore = asyncio.Semaphore(3)
        
        async def evaluate_with_semaphore(evaluation_input):
            async with semaphore:
                return await self.evaluate(evaluation_input)
        
        # 并发执行
        tasks = [evaluate_with_semaphore(input_data) for input_data in evaluation_inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量评估任务 {i} 异常: {str(result)}")
                processed_results.append(self._create_error_result(str(result)))
            else:
                processed_results.append(result)
        
        logger.info(f"批量LLM评估完成，处理 {len(processed_results)} 个结果")
        return processed_results

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'healthy',
            'provider': self.config.provider.value,
            'model': self.config.model_name,
            'base_url': self.config.base_url,
            'timeout': self.config.timeout
        }

    async def shutdown(self) -> None:
        """关闭LLM评估器"""
        logger.info("正在关闭LLM评估器...")
        
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.is_initialized = False
        logger.info("LLM评估器已关闭")