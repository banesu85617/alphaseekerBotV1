"""
统一LLM接口
支持LM Studio、Ollama、AnythingLLM等本地模型
"""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, Union
import aiohttp
import logging
from ..config.llm_config import LLMProvider, LLMConfig

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: list,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """聊天完成接口"""
        pass
    
    @abstractmethod
    def get_api_endpoint(self) -> str:
        """获取API端点"""
        pass
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close_session()
    
    async def start_session(self):
        """启动HTTP会话"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.headers
            )
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
            self.session = None


class LMStudioClient(BaseLLMClient):
    """LM Studio客户端"""
    
    def get_api_endpoint(self) -> str:
        return f"{self.config.base_url}/v1/chat/completions"
    
    async def chat_completion(
        self,
        messages: list,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """LM Studio聊天完成"""
        await self.start_session()
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream or self.config.enable_streaming,
            **kwargs
        }
        
        # LM Studio兼容OpenAI格式
        url = self.get_api_endpoint()
        
        try:
            async with self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"LM Studio API error: {response.status} - {error_text}")
                
                if stream or self.config.enable_streaming:
                    return self._handle_streaming_response(response)
                else:
                    result = await response.json()
                    return result
        
        except Exception as e:
            logger.error(f"LM Studio request failed: {e}")
            raise
    
    async def _handle_streaming_response(self, response) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                data = line[6:]  # 移除 'data: ' 前缀
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            yield delta['content']
                except json.JSONDecodeError:
                    continue


class OllamaClient(BaseLLMClient):
    """Ollama客户端"""
    
    def get_api_endpoint(self) -> str:
        return f"{self.config.base_url}/api/chat"
    
    async def chat_completion(
        self,
        messages: list,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Ollama聊天完成"""
        await self.start_session()
        
        # 转换消息格式为Ollama格式
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        payload = {
            "model": self.config.model_name,
            "messages": ollama_messages,
            "stream": stream or self.config.enable_streaming,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        url = self.get_api_endpoint()
        
        try:
            async with self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                
                if stream or self.config.enable_streaming:
                    return self._handle_ollama_streaming(response)
                else:
                    result = await response.json()
                    # 转换为OpenAI兼容格式
                    return self._convert_ollama_to_openai(result)
        
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise
    
    async def _handle_ollama_streaming(self, response) -> AsyncGenerator[str, None]:
        """处理Ollama流式响应"""
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line:
                try:
                    chunk = json.loads(line)
                    if 'message' in chunk and 'content' in chunk['message']:
                        yield chunk['message']['content']
                    if chunk.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
    
    def _convert_ollama_to_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """将Ollama响应转换为OpenAI格式"""
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result.get('message', {}).get('content', '')
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "total_tokens": result.get('eval_count', 0)
            }
        }


class AnythingLLMClient(BaseLLMClient):
    """AnythingLLM客户端"""
    
    def get_api_endpoint(self) -> str:
        return f"{self.config.base_url}/api/v1/chat"
    
    async def chat_completion(
        self,
        messages: list,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """AnythingLLM聊天完成"""
        await self.start_session()
        
        # AnythingLLM可能有特定的聊天接口格式
        payload = {
            "message": messages[-1]["content"] if messages else "",
            "mode": "chat",  # 或其他模式
            "workspace": "default",
            "history": messages[:-1] if len(messages) > 1 else [],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        url = self.get_api_endpoint()
        
        try:
            async with self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"AnythingLLM API error: {response.status} - {error_text}")
                
                result = await response.json()
                return self._convert_anythingllm_to_openai(result)
        
        except Exception as e:
            logger.error(f"AnythingLLM request failed: {e}")
            raise
    
    def _convert_anythingllm_to_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """将AnythingLLM响应转换为OpenAI格式"""
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result.get('response', '')
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "total_tokens": result.get('tokensUsed', 0)
            }
        }


class LLMInterface:
    """统一LLM接口"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client: Optional[BaseLLMClient] = None
        self._init_client()
    
    def _init_client(self):
        """根据提供商初始化客户端"""
        if self.config.provider == LLMProvider.LM_STUDIO:
            self.client = LMStudioClient(self.config)
        elif self.config.provider == LLMProvider.OLLAMA:
            self.client = OllamaClient(self.config)
        elif self.config.provider == LLMProvider.ANYTHING_LLM:
            self.client = AnythingLLMClient(self.config)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def generate_trading_analysis(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        signal_direction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """生成交易分析（兼容原API）"""
        
        # 构建提示词
        prompt = self._build_trading_prompt(market_data, technical_indicators, signal_direction)
        
        messages = [
            {
                "role": "system",
                "content": "你是一个加密货币交易分析师，评估技术信号并提供可执行的交易参数。请以JSON格式回答。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 设置响应格式
        response_format = {"type": "json_object"}
        
        # 重试机制
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                async with self.client:
                    result = await self.client.chat_completion(
                        messages=messages,
                        response_format=response_format,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        **kwargs
                    )
                
                # 处理流式响应
                if hasattr(result, '__aiter__'):
                    content = ""
                    async for chunk in result:
                        content += chunk
                    result = {"choices": [{"message": {"content": content}}]}
                
                # 解析响应
                if isinstance(result, dict) and 'choices' in result:
                    content = result['choices'][0]['message']['content']
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON response: {content}")
                        return {"error": "Invalid JSON response", "content": content}
                
                return result
            
            except Exception as e:
                last_error = e
                logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        # 所有重试都失败
        logger.error(f"All LLM requests failed. Last error: {last_error}")
        return {
            "error": f"LLM service unavailable: {str(last_error)}",
            "trade_direction": "hold",
            "confidence_score": 0.0
        }
    
    def _build_trading_prompt(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        signal_direction: str
    ) -> str:
        """构建交易分析提示词"""
        
        # 格式化技术指标
        indicators_str = json.dumps(technical_indicators, indent=2, ensure_ascii=False)
        
        # 格式化市场数据
        market_str = json.dumps(market_data, indent=2, ensure_ascii=False)
        
        prompt = f"""
你是专业的加密货币交易分析师。请评估以下技术信号并提供交易参数。

技术信号方向: {signal_direction}

市场数据:
{market_str}

技术指标:
{indicators_str}

请提供以下分析（以JSON格式）:

{{
    "signal_evaluation": "对{signal_direction}信号的评估",
    "technical_analysis": "技术分析",
    "risk_assessment": "风险评估", 
    "market_outlook": "市场展望",
    "trade_direction": "long/short/hold",
    "optimal_entry": 入场价格,
    "stop_loss": 止损价格,
    "take_profit": 止盈价格,
    "leverage": 建议杠杆,
    "position_size_usd": 仓位大小(USD),
    "confidence_score": 0.0-1.0之间的置信度,
    "estimated_profit": 预估利润(USD)
}}

要求:
1. 严格输出有效JSON格式
2. confidence_score必须在0-1之间
3. 如果有重大矛盾信号，使用"hold"
4. 止损止盈比例建议至少1:1.5
5. 基于当前价格和技术指标给出合理参数
"""
        
        return prompt
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 简单的测试请求
            test_messages = [
                {"role": "user", "content": "测试连接"}
            ]
            
            async with self.client:
                result = await self.client.chat_completion(
                    messages=test_messages,
                    max_tokens=10
                )
            
            return {
                "status": "healthy",
                "provider": self.config.provider.value,
                "model": self.config.model_name,
                "base_url": self.config.base_url
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.config.provider.value,
                "model": self.config.model_name,
                "base_url": self.config.base_url,
                "error": str(e)
            }
    
    @property
    def is_available(self) -> bool:
        """检查LLM服务是否可用"""
        return self.client is not None