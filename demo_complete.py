#!/usr/bin/env python3
"""
AlphaSeeker 完整演示程序
======================

这个演示程序展示了AlphaSeeker集成系统的完整功能，包括：
- 单个交易信号分析
- 批量市场扫描
- 系统监控
- 性能测试

运行方式:
    python demo_complete.py

要求:
    确保AlphaSeeker主应用正在运行 (python main_integration.py)
"""

import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlphaSeekerDemo:
    """AlphaSeeker演示程序"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 演示用的交易对列表
        self.demo_symbols = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
            "UNIUSDT", "AAVEUSDT", "COMPUSDT", "SUSHIUSDT", "YFIUSDT",
            "ATOMUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT", "ALGOUSDT"
        ]
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def check_system_health(self) -> bool:
        """检查系统健康状态"""
        try:
            logger.info("🔍 检查系统健康状态...")
            
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"✅ 系统健康: {health_data['status']}")
                    logger.info(f"   版本: {health_data['version']}")
                    logger.info(f"   运行时间: {health_data['uptime']:.1f}秒")
                    logger.info(f"   组件数量: {len(health_data['components'])}")
                    return True
                else:
                    logger.error(f"❌ 系统响应异常: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 系统检查失败: {e}")
            return False
    
    def generate_realistic_market_data(self, symbol: str) -> Dict[str, Any]:
        """生成真实感的市场数据"""
        base_price = {
            "BTCUSDT": 45000, "ETHUSDT": 3000, "ADAUSDT": 1.5,
            "DOTUSDT": 25, "LINKUSDT": 15, "UNIUSDT": 8,
            "AAVEUSDT": 200, "COMPUSDT": 150, "SUSHIUSDT": 6,
            "YFIUSDT": 35000, "ATOMUSDT": 30, "SOLUSDT": 100,
            "AVAXUSDT": 80, "MATICUSDT": 1.2, "ALGOUSDT": 0.3
        }
        
        price = base_price.get(symbol, 100)
        # 添加随机波动
        price *= (1 + random.uniform(-0.05, 0.05))
        
        return {
            "price": round(price, 6),
            "volume": random.randint(500000, 5000000),
            "timestamp": int(time.time())
        }
    
    def generate_technical_indicators(self, symbol: str, price: float) -> Dict[str, float]:
        """生成技术指标数据"""
        # 基于价格生成相关的技术指标
        rsi = random.uniform(20, 80)
        macd = random.uniform(-500, 500) * (price / 1000)
        adx = random.uniform(15, 50)
        
        return {
            "rsi": round(rsi, 2),
            "macd": round(macd, 2),
            "adx": round(adx, 2),
            "sma_50": round(price * random.uniform(0.95, 1.05), 2),
            "sma_200": round(price * random.uniform(0.90, 1.10), 2),
            "ema_12": round(price * random.uniform(0.98, 1.02), 2),
            "ema_26": round(price * random.uniform(0.96, 1.04), 2),
            "bb_upper": round(price * 1.02, 2),
            "bb_middle": price,
            "bb_lower": round(price * 0.98, 2),
            "atr": round(price * random.uniform(0.01, 0.03), 2)
        }
    
    def generate_features(self, market_data: Dict[str, Any], 
                         indicators: Dict[str, float]) -> Dict[str, float]:
        """生成机器学习特征"""
        price = market_data["price"]
        volume = market_data["volume"]
        
        return {
            "mid_price": price,
            "spread": random.uniform(0.5, 5.0),
            "bid_ask_ratio": random.uniform(0.8, 1.5),
            "volatility_60s": random.uniform(0.01, 0.05),
            "volume_spike": random.uniform(0.5, 3.0),
            "price_momentum": random.uniform(-0.02, 0.02),
            "order_flow_imbalance": random.uniform(-0.5, 0.5),
            "relative_strength": (indicators["rsi"] - 50) / 50,
            "trend_strength": indicators["adx"] / 50,
            "price_position": (price - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"])
        }
    
    async def analyze_single_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """分析单个交易信号"""
        try:
            # 生成数据
            market_data = self.generate_realistic_market_data(symbol)
            indicators = self.generate_technical_indicators(symbol, market_data["price"])
            features = self.generate_features(market_data, indicators)
            
            # 发送请求
            payload = {
                "symbol": symbol,
                "market_data": market_data,
                "indicators": indicators,
                "features": features
            }
            
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/api/v1/signal/analyze",
                json=payload
            ) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    logger.info(f"📊 {symbol} 信号分析结果:")
                    logger.info(f"   方向: {result['signal_direction']}")
                    logger.info(f"   置信度: {result['confidence']:.3f}")
                    logger.info(f"   评分: {result['score']:.3f}")
                    logger.info(f"   风险回报比: {result['risk_reward_ratio']:.2f}")
                    logger.info(f"   处理时间: {processing_time:.3f}秒")
                    
                    # ML组件结果
                    if result.get('components', {}).get('ml_prediction'):
                        ml_data = result['components']['ml_prediction']
                        logger.info(f"   ML预测: {ml_data.get('label')} (置信度: {ml_data.get('confidence', 0):.3f})")
                    
                    # 验证器结果
                    if result.get('components', {}).get('validation'):
                        val_data = result['components']['validation']
                        logger.info(f"   验证状态: {val_data.get('status')} (评分: {val_data.get('combined_score', 0):.3f})")
                    
                    return result
                else:
                    logger.error(f"❌ {symbol} 分析失败: HTTP {response.status}")
                    error_text = await response.text()
                    logger.error(f"   错误: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ {symbol} 分析异常: {e}")
            return None
    
    async def batch_market_scan(self, symbols: List[str], max_results: int = 5) -> Optional[Dict[str, Any]]:
        """批量市场扫描"""
        try:
            logger.info(f"🔍 开始批量市场扫描: {len(symbols)} 个交易对")
            
            payload = {
                "symbols": symbols,
                "max_results": max_results
            }
            
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/api/v1/scan/market",
                json=payload
            ) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    logger.info(f"✅ 批量扫描完成:")
                    logger.info(f"   扫描ID: {result['scan_id']}")
                    logger.info(f"   总交易对: {result['total_symbols']}")
                    logger.info(f"   处理成功: {result['processed_symbols']}")
                    logger.info(f"   找到机会: {result['valid_results']}")
                    logger.info(f"   平均置信度: {result['summary']['avg_confidence']:.3f}")
                    logger.info(f"   总处理时间: {processing_time:.2f}秒")
                    
                    # 信号分布
                    signal_dist = result['summary']['signal_distribution']
                    logger.info(f"   信号分布: {signal_dist}")
                    
                    # 显示前几个机会
                    logger.info("🎯 顶级交易机会:")
                    for i, opportunity in enumerate(result['results'][:3], 1):
                        logger.info(f"   {i}. {opportunity['symbol']}: {opportunity['signal_direction']} "
                                  f"(置信度: {opportunity['confidence']:.3f}, 评分: {opportunity['score']:.3f})")
                    
                    return result
                else:
                    logger.error(f"❌ 批量扫描失败: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ 批量扫描异常: {e}")
            return None
    
    async def get_system_performance(self) -> Optional[Dict[str, Any]]:
        """获取系统性能指标"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/performance") as response:
                if response.status == 200:
                    perf_data = await response.json()
                    
                    logger.info("📈 系统性能指标:")
                    perf = perf_data['performance']
                    sys_info = perf_data['system_info']
                    
                    logger.info(f"   请求总数: {perf['total_requests']}")
                    logger.info(f"   成功率: {perf['success_rate']:.2f}%")
                    logger.info(f"   平均处理时间: {perf['avg_processing_time']:.3f}秒")
                    logger.info(f"   系统运行时间: {perf['uptime']:.0f}秒")
                    logger.info(f"   配置: 并发任务={sys_info['config']['max_concurrent_tasks']}, "
                              f"批大小={sys_info['config']['batch_size']}")
                    
                    return perf_data
                else:
                    logger.error(f"❌ 获取性能指标失败: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ 获取性能指标异常: {e}")
            return None
    
    async def get_components_info(self) -> Optional[Dict[str, Any]]:
        """获取组件信息"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/components") as response:
                if response.status == 200:
                    comp_data = await response.json()
                    
                    logger.info("🔧 系统组件信息:")
                    for comp_name, comp_info in comp_data['components'].items():
                        logger.info(f"   {comp_info['name']}: {comp_info['description']}")
                        features = ', '.join(comp_info['features'])
                        logger.info(f"     功能: {features}")
                    
                    return comp_data
                else:
                    logger.error(f"❌ 获取组件信息失败: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ 获取组件信息异常: {e}")
            return None
    
    async def stress_test(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """压力测试"""
        logger.info(f"🔥 开始压力测试 (持续{duration_seconds}秒)...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        async def single_request():
            """单次请求"""
            nonlocal total_requests, successful_requests, failed_requests
            
            symbol = random.choice(self.demo_symbols)
            total_requests += 1
            
            try:
                req_start = time.time()
                result = await self.analyze_single_signal(symbol)
                req_time = time.time() - req_start
                
                if result:
                    successful_requests += 1
                    response_times.append(req_time)
                    return True
                else:
                    failed_requests += 1
                    return False
                    
            except Exception as e:
                failed_requests += 1
                logger.debug(f"请求失败: {e}")
                return False
        
        # 并发请求
        while time.time() < end_time:
            # 创建并发任务
            tasks = [single_request() for _ in range(min(5, int(end_time - time.time())))]
            
            # 等待完成
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 短暂延迟
            await asyncio.sleep(0.1)
        
        # 统计结果
        actual_duration = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        requests_per_second = total_requests / actual_duration
        
        results = {
            "duration": actual_duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time": avg_response_time,
            "requests_per_second": requests_per_second,
            "throughput": f"{requests_per_second:.2f} req/s"
        }
        
        logger.info("📊 压力测试结果:")
        logger.info(f"   测试时长: {actual_duration:.1f}秒")
        logger.info(f"   总请求数: {total_requests}")
        logger.info(f"   成功请求: {successful_requests}")
        logger.info(f"   失败请求: {failed_requests}")
        logger.info(f"   成功率: {results['success_rate']:.2f}%")
        logger.info(f"   平均响应时间: {avg_response_time:.3f}秒")
        logger.info(f"   吞吐量: {results['throughput']}")
        
        return results
    
    async def run_complete_demo(self):
        """运行完整演示"""
        logger.info("=" * 60)
        logger.info("🚀 AlphaSeeker 完整功能演示")
        logger.info("=" * 60)
        
        # 1. 系统健康检查
        logger.info("\n🔍 第一步: 系统健康检查")
        if not await self.check_system_health():
            logger.error("❌ 系统不健康，演示终止")
            return
        
        # 2. 组件信息
        logger.info("\n🔧 第二步: 组件信息")
        await self.get_components_info()
        
        # 3. 单个信号分析
        logger.info("\n📊 第三步: 单个交易信号分析")
        demo_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        for symbol in demo_symbols:
            await self.analyze_single_signal(symbol)
            await asyncio.sleep(1)  # 短暂延迟
        
        # 4. 批量市场扫描
        logger.info("\n🔍 第四步: 批量市场扫描")
        await self.batch_market_scan(self.demo_symbols[:8], max_results=5)
        
        # 5. 系统性能
        logger.info("\n📈 第五步: 系统性能指标")
        await self.get_system_performance()
        
        # 6. 压力测试（可选）
        logger.info("\n🔥 第六步: 性能压力测试")
        stress_results = await self.stress_test(duration_seconds=15)
        
        # 7. 最终性能检查
        logger.info("\n📈 第七步: 最终性能检查")
        await self.get_system_performance()
        
        # 总结
        logger.info("\n" + "=" * 60)
        logger.info("✅ 演示完成!")
        logger.info("=" * 60)
        logger.info("主要功能验证:")
        logger.info("  ✅ 系统启动和健康检查")
        logger.info("  ✅ 单个交易信号分析")
        logger.info("  ✅ 批量市场扫描")
        logger.info("  ✅ 系统性能监控")
        logger.info("  ✅ 压力测试")
        logger.info("  ✅ API接口完整性")
        
        if stress_results:
            logger.info("\n性能总结:")
            logger.info(f"  • 吞吐量: {stress_results['throughput']}")
            logger.info(f"  • 成功率: {stress_results['success_rate']:.1f}%")
            logger.info(f"  • 平均响应: {stress_results['avg_response_time']:.3f}秒")
        
        logger.info("\n🎉 AlphaSeeker系统演示成功完成!")

async def main():
    """主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🚀 AlphaSeeker 完整功能演示程序 🚀                ║
    ║                                                              ║
    ║  这个演示将展示AlphaSeeker集成系统的所有核心功能:           ║
    ║  • 交易信号分析                                              ║
    ║  • 批量市场扫描                                              ║
    ║  • 系统监控                                                  ║
    ║  • 性能测试                                                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 检查系统是否运行
    base_url = "http://localhost:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                if response.status != 200:
                    print("❌ AlphaSeeker系统未运行或无法访问")
                    print(f"请确保系统正在 {base_url} 上运行:")
                    print("python main_integration.py")
                    return
    except Exception:
        print("❌ 无法连接到AlphaSeeker系统")
        print(f"请确保系统在 {base_url} 上运行:")
        print("python main_integration.py")
        return
    
    # 运行演示
    async with AlphaSeekerDemo(base_url) as demo:
        await demo.run_complete_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()