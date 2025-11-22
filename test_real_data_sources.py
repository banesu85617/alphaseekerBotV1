#!/usr/bin/env python3
"""
AlphaSeeker 2.0 真实数据源测试脚本
测试多数据源切换和真实数据获取功能
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import Dict, Any, List


class DataSourceTester:
    """数据源测试器"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> Dict[str, Any]:
        """测试健康检查"""
        print("🔍 测试健康检查...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                
                print(f"   状态: {data.get('status', 'unknown')}")
                print(f"   版本: {data.get('version', 'unknown')}")
                print(f"   运行时长: {data.get('uptime', 'unknown')}")
                
                components = data.get('components', {})
                for name, status in components.items():
                    status_icon = "✅" if status == "healthy" else "❌"
                    print(f"   {status_icon} {name}: {status}")
                
                return {
                    "test": "health_check",
                    "passed": response.status == 200,
                    "response": data
                }
        except Exception as e:
            print(f"   ❌ 健康检查失败: {e}")
            return {"test": "health_check", "passed": False, "error": str(e)}
    
    async def test_signal_analysis(self) -> Dict[str, Any]:
        """测试信号分析API"""
        print("🔍 测试信号分析API...")
        
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
        results = []
        
        for symbol in test_symbols:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.base_url}/api/v1/signal/analyze", 
                                          params={"symbol": symbol}) as response:
                    data = await response.json()
                    response_time = (time.time() - start_time) * 1000
                    
                    # 检查数据质量
                    has_real_data = data.get("is_real_data", False)
                    data_source = data.get("data_source", "unknown")
                    exchange = data.get("exchange", "unknown")
                    
                    print(f"   📊 {symbol}:")
                    print(f"      价格: ${data.get('price', 0):.4f}")
                    print(f"      信号: {data.get('signal', 'N/A')}")
                    print(f"      数据源: {data_source} ({exchange})")
                    print(f"      实时数据: {'✅' if has_real_data else '⚠️'}")
                    print(f"      响应时间: {response_time:.1f}ms")
                    
                    results.append({
                        "symbol": symbol,
                        "price": data.get("price"),
                        "signal": data.get("signal"),
                        "data_source": data_source,
                        "exchange": exchange,
                        "is_real_data": has_real_data,
                        "response_time_ms": response_time,
                        "passed": response.status == 200
                    })
                    
            except Exception as e:
                print(f"   ❌ {symbol} 分析失败: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e),
                    "passed": False
                })
        
        return {
            "test": "signal_analysis",
            "results": results,
            "passed": all(r.get("passed", False) for r in results)
        }
    
    async def test_market_scan(self) -> Dict[str, Any]:
        """测试市场扫描API"""
        print("🔍 测试市场扫描API...")
        
        try:
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/api/v1/scan/market") as response:
                data = await response.json()
                response_time = (time.time() - start_time) * 1000
                
                print(f"   扫描币种: {data.get('total_symbols', 0)}个")
                
                summary = data.get("summary", {})
                print(f"   买入信号: {summary.get('buy_signals', 0)}")
                print(f"   卖出信号: {summary.get('sell_signals', 0)}")
                print(f"   观望信号: {summary.get('hold_signals', 0)}")
                print(f"   系统状态: {summary.get('system_status', 'unknown')}")
                
                results = data.get("results", [])
                real_data_count = sum(1 for r in results if r.get("is_real_data", False))
                print(f"   实时数据: {real_data_count}/{len(results)} 个币种")
                print(f"   响应时间: {response_time:.1f}ms")
                
                # 检查数据源分布
                data_sources = {}
                for result in results:
                    source = result.get("data_source", "unknown")
                    data_sources[source] = data_sources.get(source, 0) + 1
                
                print(f"   数据源分布:")
                for source, count in data_sources.items():
                    print(f"      {source}: {count} 个币种")
                
                return {
                    "test": "market_scan",
                    "total_symbols": data.get("total_symbols"),
                    "real_data_count": real_data_count,
                    "data_sources": data_sources,
                    "response_time_ms": response_time,
                    "passed": response.status == 200
                }
                
        except Exception as e:
            print(f"   ❌ 市场扫描失败: {e}")
            return {"test": "market_scan", "passed": False, "error": str(e)}
    
    async def test_pages(self) -> Dict[str, Any]:
        """测试页面访问"""
        print("🔍 测试页面访问...")
        
        pages = [
            ("/", "主页"),
            ("/scan", "市场扫描"),
            ("/analyze/BTCUSDT", "分析详情"),
            ("/performance", "性能统计")
        ]
        
        results = []
        
        for path, name in pages:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.base_url}{path}") as response:
                    response_time = (time.time() - start_time) * 1000
                    content = await response.text()
                    
                    # 检查是否包含预期的内容
                    has_expected_content = ("AlphaSeeker" in content or 
                                          "crypto" in content.lower() or 
                                          name in content)
                    
                    print(f"   📄 {name} ({path}): {'✅' if response.status == 200 else '❌'} "
                          f"{response_time:.1f}ms")
                    
                    results.append({
                        "page": name,
                        "path": path,
                        "status": response.status,
                        "response_time_ms": response_time,
                        "has_expected_content": has_expected_content,
                        "passed": response.status == 200 and has_expected_content
                    })
                    
            except Exception as e:
                print(f"   ❌ {name} 访问失败: {e}")
                results.append({
                    "page": name,
                    "path": path,
                    "error": str(e),
                    "passed": False
                })
        
        return {
            "test": "pages",
            "results": results,
            "passed": all(r.get("passed", False) for r in results)
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🧪 开始AlphaSeeker 2.0真实数据源全面测试")
        print("=" * 60)
        
        test_results = []
        
        # 测试1: 健康检查
        result = await self.test_health_check()
        test_results.append(result)
        print()
        
        # 测试2: 信号分析
        result = await self.test_signal_analysis()
        test_results.append(result)
        print()
        
        # 测试3: 市场扫描
        result = await self.test_market_scan()
        test_results.append(result)
        print()
        
        # 测试4: 页面访问
        result = await self.test_pages()
        test_results.append(result)
        print()
        
        # 生成测试报告
        passed_tests = sum(1 for r in test_results if r.get("passed", False))
        total_tests = len(test_results)
        
        print("📊 测试结果汇总")
        print("=" * 60)
        
        for result in test_results:
            test_name = result.get("test", "unknown")
            status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
            print(f"{status} {test_name}")
            
            # 显示详细信息
            if test_name == "signal_analysis":
                real_data_count = sum(1 for r in result.get("results", []) 
                                    if r.get("is_real_data", False))
                avg_response_time = sum(r.get("response_time_ms", 0) for r in result.get("results", [])) / max(len(result.get("results", [])), 1)
                print(f"    实时数据: {real_data_count} 个币种")
                print(f"    平均响应时间: {avg_response_time:.1f}ms")
            
            elif test_name == "market_scan":
                scan_result = result
                print(f"    扫描币种: {scan_result.get('total_symbols', 0)} 个")
                print(f"    实时数据: {scan_result.get('real_data_count', 0)} 个")
                print(f"    响应时间: {scan_result.get('response_time_ms', 0):.1f}ms")
            
            elif test_name == "pages":
                pages_result = result
                working_pages = sum(1 for r in pages_result.get("results", []) if r.get("passed", False))
                print(f"    正常页面: {working_pages}/{len(pages_result.get('results', []))} 个")
        
        print()
        print("🏆 总体测试结果:")
        print(f"   通过: {passed_tests}/{total_tests} 个测试")
        print(f"   成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("   🎉 所有测试通过！系统运行正常！")
            print("   ✨ 真实数据源集成成功！")
        else:
            print("   ⚠️ 部分测试失败，请检查系统配置")
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": (passed_tests/total_tests)*100,
                "timestamp": datetime.now().isoformat()
            },
            "details": test_results
        }


async def main():
    """主函数"""
    print("🚀 AlphaSeeker 2.0 真实数据源测试")
    print("=" * 60)
    
    # 检查服务是否运行
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ 检测到服务正在运行")
    except:
        print("❌ 错误：AlphaSeeker服务未运行")
        print("💡 请先启动服务：python3 main_integration.py")
        return
    
    print()
    
    # 运行测试
    async with DataSourceTester() as tester:
        results = await tester.run_all_tests()
    
    # 保存测试报告
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细测试报告已保存到: {report_file}")
    
    # 退出码
    success_rate = results["summary"]["success_rate"]
    if success_rate >= 75:
        exit(0)  # 成功
    else:
        exit(1)  # 失败


if __name__ == "__main__":
    asyncio.run(main())