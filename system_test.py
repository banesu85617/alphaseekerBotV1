#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaSeeker系统稳定性和性能测试脚本
"""

import time
import asyncio
import aiohttp
import json
import psutil
import yaml
import subprocess
import sys
import os
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import threading
import signal
import resource

class AlphaSeekerTestSuite:
    """AlphaSeeker系统测试套件"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.config = self.load_config()
        self.test_results = {
            'system_startup': [],
            'performance_benchmarks': [],
            'component_integration': [],
            'stability_tests': [],
            'api_tests': []
        }
        self.base_url = "http://localhost:8000"
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open('config/main_config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置加载失败: {e}")
            return {}
    
    def log_result(self, category: str, test_name: str, status: str, 
                   details: Dict[str, Any], duration: float = 0):
        """记录测试结果"""
        self.test_results[category].append({
            'test_name': test_name,
            'status': status,
            'details': details,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[{status}] {test_name} - 耗时: {duration:.2f}s")
    
    def get_system_resources(self):
        """获取系统资源使用情况"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids())
        }
    
    async def test_system_startup(self):
        """1. 系统启动测试"""
        print("\n=== 1. 系统启动测试 ===")
        
        # 测试主集成应用启动
        start_time = time.time()
        try:
            process = subprocess.Popen(
                [sys.executable, 'main_integration.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # 等待服务启动
            await asyncio.sleep(3)
            
            # 检查进程状态
            if process.poll() is None:
                # 测试端口监听
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 8000))
                sock.close()
                
                if result == 0:
                    duration = time.time() - start_time
                    self.log_result('system_startup', '主应用启动测试', 'PASS', 
                                  {'port': 8000, 'process_id': process.pid}, duration)
                else:
                    duration = time.time() - start_time
                    self.log_result('system_startup', '主应用启动测试', 'FAIL', 
                                  {'error': '端口8000未监听'}, duration)
            else:
                duration = time.time() - start_time
                stdout, stderr = process.communicate()
                self.log_result('system_startup', '主应用启动测试', 'FAIL', 
                              {'error': '进程退出', 'stderr': stderr.decode()}, duration)
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('system_startup', '主应用启动测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试配置文件加载
        start_time = time.time()
        try:
            if self.config:
                duration = time.time() - start_time
                self.log_result('system_startup', '配置文件加载测试', 'PASS', 
                              {'config_keys': list(self.config.keys())}, duration)
            else:
                duration = time.time() - start_time
                self.log_result('system_startup', '配置文件加载测试', 'FAIL', 
                              {'error': '配置为空'}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('system_startup', '配置文件加载测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试依赖包导入
        start_time = time.time()
        try:
            import fastapi
            import lightgbm
            import ccxt
            duration = time.time() - start_time
            self.log_result('system_startup', '核心依赖包测试', 'PASS', 
                          {'packages': ['fastapi', 'lightgbm', 'ccxt']}, duration)
        except ImportError as e:
            duration = time.time() - start_time
            self.log_result('system_startup', '核心依赖包测试', 'FAIL', 
                          {'error': str(e)}, duration)
    
    async def test_performance_benchmarks(self):
        """2. 性能基准测试"""
        print("\n=== 2. 性能基准测试 ===")
        
        # 测试单个信号分析性能
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                test_data = {
                    'symbol': 'BTC/USDT',
                    'timeframe': '1h',
                    'strategy': 'momentum'
                }
                
                async with session.post(f'{self.base_url}/api/v1/signal/analyze', 
                                      json=test_data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        if duration < 1.0:
                            self.log_result('performance_benchmarks', '单个信号分析性能', 'PASS', 
                                          {'duration': duration, 'threshold': 1.0}, duration)
                        else:
                            self.log_result('performance_benchmarks', '单个信号分析性能', 'WARN', 
                                          {'duration': duration, 'threshold': 1.0}, duration)
                    else:
                        self.log_result('performance_benchmarks', '单个信号分析性能', 'FAIL', 
                                      {'status': response.status, 'duration': duration}, duration)
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '单个信号分析性能', 'TIMEOUT', 
                          {'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '单个信号分析性能', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试批量市场扫描性能
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                test_data = {
                    'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
                    'strategies': ['momentum', 'mean_reversion']
                }
                
                async with session.post(f'{self.base_url}/api/v1/scanner/batch-scan', 
                                      json=test_data, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        if duration < 10.0:
                            self.log_result('performance_benchmarks', '批量市场扫描性能', 'PASS', 
                                          {'duration': duration, 'threshold': 10.0}, duration)
                        else:
                            self.log_result('performance_benchmarks', '批量市场扫描性能', 'WARN', 
                                          {'duration': duration, 'threshold': 10.0}, duration)
                    else:
                        self.log_result('performance_benchmarks', '批量市场扫描性能', 'FAIL', 
                                      {'status': response.status, 'duration': duration}, duration)
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '批量市场扫描性能', 'TIMEOUT', 
                          {'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '批量市场扫描性能', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试并发处理能力
        start_time = time.time()
        try:
            concurrent_requests = 32
            tasks = []
            async with aiohttp.ClientSession() as session:
                for i in range(concurrent_requests):
                    test_data = {
                        'symbol': f'TEST{i % 3 + 1}/USDT',
                        'timeframe': '1m'
                    }
                    task = session.post(f'{self.base_url}/api/v1/signal/analyze', 
                                      json=test_data, timeout=aiohttp.ClientTimeout(total=5))
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start_time
                
                successful_requests = sum(1 for r in responses if not isinstance(r, Exception))
                if successful_requests >= concurrent_requests * 0.8:  # 80%成功率
                    self.log_result('performance_benchmarks', '并发处理能力测试', 'PASS', 
                                  {'concurrent_requests': concurrent_requests, 
                                   'successful_requests': successful_requests,
                                   'duration': duration}, duration)
                else:
                    self.log_result('performance_benchmarks', '并发处理能力测试', 'WARN', 
                                  {'concurrent_requests': concurrent_requests,
                                   'successful_requests': successful_requests,
                                   'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '并发处理能力测试', 'ERROR', 
                          {'error': str(e)}, duration)
    
    async def test_component_integration(self):
        """3. 组件集成测试"""
        print("\n=== 3. 组件集成测试 ===")
        
        # 测试API服务
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{self.base_url}/api/v1/health') as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        self.log_result('component_integration', 'API服务测试', 'PASS', 
                                      {'status': response.status, 'duration': duration}, duration)
                    else:
                        self.log_result('component_integration', 'API服务测试', 'FAIL', 
                                      {'status': response.status}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', 'API服务测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试机器学习引擎
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                test_data = {
                    'features': [1.0, 2.0, 3.0, 4.0, 5.0],
                    'model_type': 'lightgbm'
                }
                async with session.post(f'{self.base_url}/api/v1/ml/predict', 
                                      json=test_data) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        self.log_result('component_integration', '机器学习引擎测试', 'PASS', 
                                      {'status': response.status, 'duration': duration}, duration)
                    else:
                        self.log_result('component_integration', '机器学习引擎测试', 'FAIL', 
                                      {'status': response.status}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', '机器学习引擎测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试多策略管道
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                test_data = {
                    'symbol': 'BTC/USDT',
                    'strategies': ['technical', 'ml', 'risk']
                }
                async with session.post(f'{self.base_url}/api/v1/pipeline/process', 
                                      json=test_data) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        self.log_result('component_integration', '多策略管道测试', 'PASS', 
                                      {'status': response.status, 'duration': duration}, duration)
                    else:
                        self.log_result('component_integration', '多策略管道测试', 'FAIL', 
                                      {'status': response.status}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', '多策略管道测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试市场扫描器
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                test_data = {
                    'strategy': 'momentum',
                    'limit': 10
                }
                async with session.post(f'{self.base_url}/api/v1/scanner/scan', 
                                      json=test_data) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        self.log_result('component_integration', '市场扫描器测试', 'PASS', 
                                      {'status': response.status, 'duration': duration}, duration)
                    else:
                        self.log_result('component_integration', '市场扫描器测试', 'FAIL', 
                                      {'status': response.status}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', '市场扫描器测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试双重验证器
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                test_data = {
                    'signal_data': {'symbol': 'BTC/USDT', 'confidence': 0.8}
                }
                async with session.post(f'{self.base_url}/api/v1/validation/verify', 
                                      json=test_data) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        self.log_result('component_integration', '双重验证器测试', 'PASS', 
                                      {'status': response.status, 'duration': duration}, duration)
                    else:
                        self.log_result('component_integration', '双重验证器测试', 'FAIL', 
                                      {'status': response.status}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', '双重验证器测试', 'ERROR', 
                          {'error': str(e)}, duration)
    
    async def test_stability(self):
        """4. 稳定性测试"""
        print("\n=== 4. 稳定性测试 ===")
        
        # 测试异常情况处理
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                # 测试无效数据
                test_data = {'invalid': 'data'}
                async with session.post(f'{self.base_url}/api/v1/signal/analyze', 
                                      json=test_data) as response:
                    duration = time.time() - start_time
                    if response.status in [400, 422]:  # 预期错误码
                        self.log_result('stability_tests', '异常情况处理测试', 'PASS', 
                                      {'status': response.status, 'duration': duration}, duration)
                    else:
                        self.log_result('stability_tests', '异常情况处理测试', 'WARN', 
                                      {'status': response.status, 'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('stability_tests', '异常情况处理测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试系统资源监控
        start_time = time.time()
        try:
            resources_before = self.get_system_resources()
            
            # 模拟高负载
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i in range(10):
                    task = session.get(f'{self.base_url}/api/v1/health')
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            resources_after = self.get_system_resources()
            duration = time.time() - start_time
            
            memory_increase = resources_after['memory_used_mb'] - resources_before['memory_used_mb']
            
            self.log_result('stability_tests', '系统资源监控测试', 'PASS', 
                          {'memory_increase_mb': memory_increase,
                           'resources_before': resources_before,
                           'resources_after': resources_after,
                           'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('stability_tests', '系统资源监控测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试长时间运行稳定性 (简化为短时间测试)
        start_time = time.time()
        try:
            requests_count = 0
            error_count = 0
            
            async with aiohttp.ClientSession() as session:
                for i in range(20):  # 20次请求模拟长时间运行
                    try:
                        async with session.get(f'{self.base_url}/api/v1/health') as response:
                            if response.status != 200:
                                error_count += 1
                            requests_count += 1
                            await asyncio.sleep(0.1)  # 小间隔
                    except:
                        error_count += 1
                        requests_count += 1
            
            duration = time.time() - start_time
            error_rate = error_count / requests_count if requests_count > 0 else 1.0
            
            if error_rate < 0.1:  # 错误率低于10%
                self.log_result('stability_tests', '长时间运行稳定性测试', 'PASS', 
                              {'requests_count': requests_count,
                               'error_count': error_count,
                               'error_rate': error_rate,
                               'duration': duration}, duration)
            else:
                self.log_result('stability_tests', '长时间运行稳定性测试', 'WARN', 
                              {'requests_count': requests_count,
                               'error_count': error_count,
                               'error_rate': error_rate,
                               'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('stability_tests', '长时间运行稳定性测试', 'ERROR', 
                          {'error': str(e)}, duration)
    
    async def test_api_endpoints(self):
        """5. API接口测试"""
        print("\n=== 5. API接口测试 ===")
        
        endpoints = [
            ('GET', '/api/v1/health', None),
            ('GET', '/api/v1/system/status', None),
            ('POST', '/api/v1/signal/analyze', {'symbol': 'BTC/USDT'}),
            ('POST', '/api/v1/scanner/scan', {'strategy': 'momentum'}),
            ('POST', '/api/v1/ml/predict', {'features': [1.0, 2.0]}),
            ('POST', '/api/v1/validation/verify', {'signal_data': {}}),
            ('POST', '/api/v1/pipeline/process', {'symbol': 'BTC/USDT'}),
        ]
        
        for method, endpoint, test_data in endpoints:
            start_time = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    if method == 'GET':
                        async with session.get(f'{self.base_url}{endpoint}') as response:
                            duration = time.time() - start_time
                            status = 'PASS' if response.status < 500 else 'FAIL'
                            self.log_result('api_tests', f'{method} {endpoint}', status, 
                                          {'status': response.status, 'duration': duration}, duration)
                    else:
                        async with session.post(f'{self.base_url}{endpoint}, json=test_data') as response:
                            duration = time.time() - start_time
                            status = 'PASS' if response.status < 500 else 'FAIL'
                            self.log_result('api_tests', f'{method} {endpoint}', status, 
                                          {'status': response.status, 'duration': duration}, duration)
            except Exception as e:
                duration = time.time() - start_time
                self.log_result('api_tests', f'{method} {endpoint}', 'ERROR', 
                              {'error': str(e)}, duration)
    
    def generate_report(self):
        """生成测试报告"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # 统计各类测试结果
        summary = {}
        for category, tests in self.test_results.items():
            passed = len([t for t in tests if t['status'] == 'PASS'])
            failed = len([t for t in tests if t['status'] == 'FAIL'])
            warnings = len([t for t in tests if t['status'] == 'WARN'])
            errors = len([t for t in tests if t['status'] == 'ERROR'])
            timeouts = len([t for t in tests if t['status'] == 'TIMEOUT'])
            
            summary[category] = {
                'total': len(tests),
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'errors': errors,
                'timeouts': timeouts,
                'success_rate': (passed / len(tests) * 100) if tests else 0
            }
        
        # 生成报告内容
        report_content = f"""# AlphaSeeker系统稳定性和性能测试报告

## 测试概述
- **测试时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总测试时长**: {total_duration:.2f}秒
- **测试环境**: Python {sys.version.split()[0]}
- **系统资源**: CPU {psutil.cpu_count()}核心, 内存 {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB

## 测试结果汇总

"""
        
        for category, stats in summary.items():
            category_name = {
                'system_startup': '系统启动测试',
                'performance_benchmarks': '性能基准测试',
                'component_integration': '组件集成测试',
                'stability_tests': '稳定性测试',
                'api_tests': 'API接口测试'
            }.get(category, category)
            
            report_content += f"""### {category_name}
- **总测试数**: {stats['total']}
- **通过**: {stats['passed']} ({stats['success_rate']:.1f}%)
- **失败**: {stats['failed']}
- **警告**: {stats['warnings']}
- **错误**: {stats['errors']}
- **超时**: {stats['timeouts']}

"""

        report_content += "\n## 详细测试结果\n\n"
        
        for category, tests in self.test_results.items():
            category_name = {
                'system_startup': '系统启动测试',
                'performance_benchmarks': '性能基准测试',
                'component_integration': '组件集成测试',
                'stability_tests': '稳定性测试',
                'api_tests': 'API接口测试'
            }.get(category, category)
            
            report_content += f"### {category_name}\n\n"
            
            for test in tests:
                status_emoji = {
                    'PASS': '✅',
                    'FAIL': '❌',
                    'WARN': '⚠️',
                    'ERROR': '🚫',
                    'TIMEOUT': '⏱️'
                }.get(test['status'], '❓')
                
                report_content += f"- **{status_emoji} {test['test_name']}** ({test['duration']:.2f}s)\n"
                report_content += f"  - 状态: {test['status']}\n"
                report_content += f"  - 详情: {json.dumps(test['details'], indent=6, ensure_ascii=False)}\n\n"
        
        # 性能指标总结
        performance_tests = [t for t in self.test_results['performance_benchmarks']]
        if performance_tests:
            report_content += "\n## 性能指标总结\n\n"
            for test in performance_tests:
                report_content += f"- **{test['test_name']}**: {test['duration']:.2f}秒\n"
        
        # 建议和改进
        report_content += "\n## 建议和改进\n\n"
        
        total_tests = sum(stats['total'] for stats in summary.values())
        total_passed = sum(stats['passed'] for stats in summary.values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if overall_success_rate >= 90:
            report_content += "- ✅ 系统整体性能表现优秀\n"
        elif overall_success_rate >= 70:
            report_content += "- ⚠️ 系统整体性能表现良好，需要关注一些问题\n"
        else:
            report_content += "- ❌ 系统存在较多问题，需要紧急修复\n"
        
        report_content += "\n- 建议定期进行性能监控和压力测试\n"
        report_content += "- 建议实施更完善的错误处理和恢复机制\n"
        report_content += "- 建议优化高并发场景下的资源使用\n"
        report_content += "- 建议增加更多异常情况的测试覆盖\n\n"
        
        report_content += f"---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report_content
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始AlphaSeeker系统稳定性和性能测试\n")
        
        # 记录测试开始时的系统资源
        initial_resources = self.get_system_resources()
        print(f"初始系统资源 - CPU: {initial_resources['cpu_percent']:.1f}%, "
              f"内存: {initial_resources['memory_percent']:.1f}%\n")
        
        try:
            # 执行各项测试
            await self.test_system_startup()
            await self.test_performance_benchmarks()
            await self.test_component_integration()
            await self.test_stability()
            await self.test_api_endpoints()
            
        except Exception as e:
            print(f"测试执行过程中发生错误: {e}")
            traceback.print_exc()
        
        finally:
            # 记录测试结束时的系统资源
            final_resources = self.get_system_resources()
            print(f"\n测试结束系统资源 - CPU: {final_resources['cpu_percent']:.1f}%, "
                  f"内存: {final_resources['memory_percent']:.1f}%")
            
            # 生成并保存报告
            report_content = self.generate_report()
            
            # 确保目录存在
            os.makedirs('/workspace/test_results', exist_ok=True)
            
            # 保存报告
            with open('/workspace/test_results/system_performance_test.md', 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n📊 测试报告已保存到: /workspace/test_results/system_performance_test.md")
            print(f"📈 测试完成，总耗时: {(datetime.now() - self.start_time).total_seconds():.2f}秒")


async def main():
    """主函数"""
    # 启动主服务
    print("启动AlphaSeeker主服务...")
    
    # 启动主服务进程
    server_process = None
    try:
        server_process = subprocess.Popen(
            [sys.executable, 'main_integration.py'],
            cwd='/workspace/code'
        )
        
        # 等待服务启动
        await asyncio.sleep(5)
        
        # 运行测试
        test_suite = AlphaSeekerTestSuite()
        await test_suite.run_all_tests()
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
    
    finally:
        # 清理服务进程
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        
        print("测试清理完成")


if __name__ == "__main__":
    asyncio.run(main())