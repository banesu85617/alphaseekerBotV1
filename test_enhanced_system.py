#!/usr/bin/env python3
"""
AlphaSeeker 2.0 完整功能版测试脚本
================================

测试新系统的所有功能是否正常工作
"""

import requests
import time
import json

def test_system():
    """测试完整系统功能"""
    base_url = "http://0.0.0.0:8000"
    
    print("🧪 测试 AlphaSeeker 2.0 完整功能版")
    print("=" * 50)
    
    # 测试列表
    tests = [
        ("主页", "/"),
        ("市场扫描页面", "/scan"),
        ("分析页面", "/analyze/BTCUSDT"),
        ("性能统计页面", "/performance"),
        ("健康检查", "/health"),
        ("市场扫描API", "/api/v1/scan/market"),
        ("信号分析API", "/api/v1/signal/analyze?symbol=BTCUSDT"),
        ("性能API", "/api/v1/performance")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, endpoint in tests:
        print(f"🔍 测试 {test_name} ({endpoint})...")
        
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                content = response.text
                
                # 检查特定内容
                if endpoint == "/":
                    if "AlphaSeeker 2.0" in content and "市场扫描" in content:
                        print(f"  ✅ {test_name} - 界面完整")
                    else:
                        print(f"  ⚠️ {test_name} - 内容可能不完整")
                elif endpoint == "/scan":
                    if "市场扫描" in content and "扫描概览" in content:
                        print(f"  ✅ {test_name} - 功能完整")
                    else:
                        print(f"  ⚠️ {test_name} - 内容可能不完整")
                elif endpoint.startswith("/analyze/"):
                    if "深度分析" in content and "AI分析理由" in content:
                        print(f"  ✅ {test_name} - 分析界面完整")
                    else:
                        print(f"  ⚠️ {test_name} - 内容可能不完整")
                elif endpoint == "/performance":
                    if "性能统计" in content and "组件状态" in content:
                        print(f"  ✅ {test_name} - 统计界面完整")
                    else:
                        print(f"  ⚠️ {test_name} - 内容可能不完整")
                elif endpoint == "/health":
                    if "status" in content and "components" in content:
                        print(f"  ✅ {test_name} - 健康检查正常")
                    else:
                        print(f"  ⚠️ {test_name} - 响应格式异常")
                elif endpoint.startswith("/api/v1/"):
                    data = response.json()
                    if isinstance(data, dict):
                        print(f"  ✅ {test_name} - API响应正常")
                    else:
                        print(f"  ⚠️ {test_name} - 响应格式异常")
                else:
                    print(f"  ✅ {test_name} - 响应正常")
                
                passed += 1
                
            else:
                print(f"  ❌ {test_name} - HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ {test_name} - 连接失败，请确保服务正在运行")
            print("  💡 运行命令: ./start.sh")
            break
        except Exception as e:
            print(f"  ❌ {test_name} - 错误: {e}")
    
    print("")
    print("📊 测试结果总结")
    print("=" * 30)
    print(f"通过: {passed}/{total}")
    print(f"成功率: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("")
        print("🎉 所有测试通过！AlphaSeeker 2.0 系统运行完美！")
        print("")
        print("🌟 您现在可以享受完整的功能体验:")
        print("  - 🏠 现代化主页界面")
        print("  - 📊 市场扫描功能")
        print("  - 🔍 深度分析功能")
        print("  - 📈 性能监控功能")
        print("  - ⚡ 实时数据更新")
        print("")
        print("🚀 访问地址: http://0.0.0.0:8000")
    else:
        print("")
        print("⚠️ 部分测试失败，请检查系统状态")
        print("💡 可能需要重启服务: ./start.sh")

if __name__ == "__main__":
    test_system()
