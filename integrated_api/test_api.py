#!/usr/bin/env python3
"""
AlphaSeeker-API 测试脚本
测试API的主要功能
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# API配置
API_BASE_URL = "http://localhost:8000"
API_PREFIX = "/api"


class APITester:
    """API测试器"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health(self) -> bool:
        """测试健康检查"""
        try:
            print("🔍 测试健康检查...")
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 健康检查通过: {data['status']}")
                    return True
                else:
                    print(f"❌ 健康检查失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 健康检查异常: {e}")
            return False
    
    async def test_tickers(self) -> bool:
        """测试获取交易对"""
        try:
            print("🔍 测试获取交易对...")
            async with self.session.get(f"{self.base_url}{API_PREFIX}/crypto/tickers") as response:
                if response.status == 200:
                    data = await response.json()
                    tickers = data['tickers']
                    print(f"✅ 获取交易对成功: {len(tickers)} 个交易对")
                    if tickers:
                        print(f"   示例交易对: {tickers[:5]}")
                    return True
                else:
                    print(f"❌ 获取交易对失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 获取交易对异常: {e}")
            return False
    
    async def test_single_analysis(self) -> bool:
        """测试单个分析"""
        try:
            print("🔍 测试单个分析...")
            
            # 使用BTC/USDT进行分析
            request_data = {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "lookback": 500,
                "accountBalance": 1000.0,
                "maxLeverage": 10.0
            }
            
            async with self.session.post(
                f"{self.base_url}{API_PREFIX}/crypto/analyze",
                json=request_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 单个分析成功")
                    print(f"   交易对: {data['symbol']}")
                    print(f"   当前价格: {data.get('currentPrice', 'N/A')}")
                    
                    if data.get('gptParams'):
                        params = data['gptParams']
                        print(f"   交易方向: {params.get('tradeDirection', 'N/A')}")
                        print(f"   置信度: {params.get('confidenceScore', 'N/A')}")
                    
                    if data.get('indicators'):
                        indicators = data['indicators']
                        print(f"   RSI: {indicators.get('RSI', 'N/A')}")
                        print(f"   ADX: {indicators.get('ADX', 'N/A')}")
                    
                    return True
                else:
                    error_data = await response.json()
                    print(f"❌ 单个分析失败: {response.status} - {error_data}")
                    return False
        except Exception as e:
            print(f"❌ 单个分析异常: {e}")
            return False
    
    async def test_market_scan(self) -> bool:
        """测试市场扫描"""
        try:
            print("🔍 测试市场扫描...")
            
            request_data = {
                "timeframe": "1h",
                "max_tickers": 20,
                "top_n": 5,
                "min_gpt_confidence": 0.5,
                "min_backtest_score": 0.3,
                "max_concurrent_tasks": 4  # 减少并发以加快测试
            }
            
            async with self.session.post(
                f"{self.base_url}{API_PREFIX}/crypto/scan",
                json=request_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 市场扫描成功")
                    print(f"   尝试扫描: {data['total_tickers_attempted']} 个交易对")
                    print(f"   成功扫描: {data['total_tickers_succeeded']} 个")
                    print(f"   发现机会: {data['total_opportunities_found']} 个")
                    print(f"   返回结果: {len(data['top_opportunities'])} 个")
                    
                    if data['top_opportunities']:
                        top_result = data['top_opportunities'][0]
                        print(f"   最佳机会: {top_result['symbol']}")
                        print(f"   交易方向: {top_result.get('tradeDirection', 'N/A')}")
                        print(f"   综合评分: {top_result.get('combinedScore', 'N/A')}")
                    
                    return True
                else:
                    error_data = await response.json()
                    print(f"❌ 市场扫描失败: {response.status} - {error_data}")
                    return False
        except Exception as e:
            print(f"❌ 市场扫描异常: {e}")
            return False
    
    async def test_llm_health(self) -> bool:
        """测试LLM健康状态"""
        try:
            print("🔍 测试LLM健康状态...")
            async with self.session.get(f"{self.base_url}{API_PREFIX}/llm/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ LLM健康状态: {data['status']}")
                    print(f"   提供商: {data.get('provider', 'N/A')}")
                    print(f"   模型: {data.get('model', 'N/A')}")
                    print(f"   基础URL: {data.get('base_url', 'N/A')}")
                    
                    if data.get('error'):
                        print(f"   错误: {data['error']}")
                    
                    return True
                else:
                    print(f"❌ LLM健康检查失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ LLM健康检查异常: {e}")
            return False
    
    async def test_system_status(self) -> bool:
        """测试系统状态"""
        try:
            print("🔍 测试系统状态...")
            async with self.session.get(f"{self.base_url}{API_PREFIX}/system/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 系统状态获取成功")
                    
                    config = data.get('config', {})
                    print(f"   LLM提供商: {config.get('llm_provider', 'N/A')}")
                    print(f"   LLM模型: {config.get('llm_model', 'N/A')}")
                    print(f"   最大并发: {config.get('max_concurrent_tasks', 'N/A')}")
                    
                    return True
                else:
                    print(f"❌ 系统状态获取失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 系统状态获取异常: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        print("🧪 开始API测试...\n")
        
        tests = [
            ("健康检查", self.test_health),
            ("获取交易对", self.test_tickers),
            ("LLM健康状态", self.test_llm_health),
            ("单个分析", self.test_single_analysis),
            ("市场扫描", self.test_market_scan),
            ("系统状态", self.test_system_status)
        ]
        
        results = {}
        start_time = time.time()
        
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            try:
                success = await test_func()
                results[test_name] = success
            except Exception as e:
                print(f"❌ {test_name} 测试异常: {e}")
                results[test_name] = False
        
        # 测试总结
        total_time = time.time() - start_time
        passed = sum(results.values())
        total = len(results)
        
        print(f"\n📊 测试总结:")
        print(f"   总测试数: {total}")
        print(f"   通过: {passed}")
        print(f"   失败: {total - passed}")
        print(f"   成功率: {passed/total*100:.1f}%")
        print(f"   总耗时: {total_time:.2f}秒")
        
        if passed == total:
            print("🎉 所有测试通过!")
        else:
            print("⚠️  部分测试失败，请检查API配置")
        
        return results


async def main():
    """主函数"""
    print("🚀 AlphaSeeker-API 测试工具")
    print("=" * 50)
    
    # 检查API是否运行
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/") as response:
                if response.status != 200:
                    print(f"❌ API未运行或无法访问: {response.status}")
                    return
    except Exception as e:
        print(f"❌ 无法连接到API服务器: {e}")
        print(f"   请确保API服务在 {API_BASE_URL} 上运行")
        return
    
    # 运行测试
    async with APITester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())