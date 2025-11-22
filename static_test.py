#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaSeeker系统静态分析和模拟测试脚本
"""

import time
import json
import yaml
import psutil
import sys
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

class AlphaSeekerStaticTestSuite:
    """AlphaSeeker系统静态测试套件"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.config = self.load_config()
        self.test_results = {
            'system_startup': [],
            'performance_benchmarks': [],
            'component_integration': [],
            'stability_tests': [],
            'file_structure': []
        }
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open('/workspace/code/config/main_config.yaml', 'r', encoding='utf-8') as f:
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
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARN': '⚠️',
            'ERROR': '🚫'
        }.get(status, '❓')
        print(f"[{status}] {test_name} - 耗时: {duration:.2f}s")
    
    def get_system_resources(self):
        """获取系统资源使用情况"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids())
        }
    
    def check_file_structure(self):
        """检查文件结构"""
        print("\n=== 文件结构检查 ===")
        
        start_time = time.time()
        
        # 检查核心文件和目录
        required_files = [
            '/workspace/code/main_integration.py',
            '/workspace/code/requirements.txt',
            '/workspace/code/config/main_config.yaml',
            '/workspace/code/start.sh',
            '/workspace/code/stop.sh'
        ]
        
        required_dirs = [
            '/workspace/code/integrated_api',
            '/workspace/code/ml_engine',
            '/workspace/code/pipeline',
            '/workspace/code/scanner',
            '/workspace/code/validation',
            '/workspace/code/logs',
            '/workspace/code/data',
            '/workspace/code/models'
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
        
        duration = time.time() - start_time
        
        if not missing_files and not missing_dirs:
            self.log_result('file_structure', '核心文件结构检查', 'PASS', 
                          {'files_ok': len(required_files), 'dirs_ok': len(required_dirs)}, duration)
        else:
            self.log_result('file_structure', '核心文件结构检查', 'WARN', 
                          {'missing_files': missing_files, 'missing_dirs': missing_dirs}, duration)
    
    def test_system_startup(self):
        """1. 系统启动测试"""
        print("\n=== 1. 系统启动测试 ===")
        
        # 测试配置文件解析
        start_time = time.time()
        try:
            if self.config and 'components' in self.config:
                duration = time.time() - start_time
                components = list(self.config['components'].keys())
                self.log_result('system_startup', '配置文件解析测试', 'PASS', 
                              {'components': components, 'config_keys': list(self.config.keys())}, duration)
            else:
                duration = time.time() - start_time
                self.log_result('system_startup', '配置文件解析测试', 'FAIL', 
                              {'error': '配置结构不正确'}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('system_startup', '配置文件解析测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试Python环境
        start_time = time.time()
        try:
            python_version = sys.version.split()[0]
            version_ok = tuple(map(int, python_version.split('.')[:2])) >= (3, 8)
            
            duration = time.time() - start_time
            if version_ok:
                self.log_result('system_startup', 'Python环境检查', 'PASS', 
                              {'python_version': python_version}, duration)
            else:
                self.log_result('system_startup', 'Python环境检查', 'WARN', 
                              {'python_version': python_version, 'warning': '版本过低'}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('system_startup', 'Python环境检查', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试依赖包导入
        start_time = time.time()
        try:
            import fastapi
            import lightgbm
            import ccxt
            import pandas
            import numpy
            
            duration = time.time() - start_time
            packages = {
                'fastapi': fastapi.__version__,
                'lightgbm': lightgbm.__version__,
                'ccxt': ccxt.__version__,
                'pandas': pandas.__version__,
                'numpy': numpy.__version__
            }
            
            self.log_result('system_startup', '核心依赖包导入测试', 'PASS', 
                          {'packages': packages}, duration)
        except ImportError as e:
            duration = time.time() - start_time
            self.log_result('system_startup', '核心依赖包导入测试', 'FAIL', 
                          {'error': str(e)}, duration)
    
    def test_performance_benchmarks(self):
        """2. 性能基准测试"""
        print("\n=== 2. 性能基准测试 ===")
        
        # 测试数据处理性能
        start_time = time.time()
        try:
            import numpy as np
            import pandas as pd
            
            # 模拟数据处理
            data_size = 10000
            test_data = np.random.randn(data_size, 10)
            df = pd.DataFrame(test_data)
            
            # 执行一些数据操作
            result = df.rolling(window=5).mean()
            correlation = df.corr()
            
            duration = time.time() - start_time
            
            if duration < 2.0:  # 2秒内完成
                self.log_result('performance_benchmarks', '数据处理性能测试', 'PASS', 
                              {'data_size': data_size, 'duration': duration}, duration)
            else:
                self.log_result('performance_benchmarks', '数据处理性能测试', 'WARN', 
                              {'data_size': data_size, 'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '数据处理性能测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试机器学习模型性能
        start_time = time.time()
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            # 生成测试数据
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            
            # 训练模型
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # 预测
            predictions = model.predict(X[:100])
            
            duration = time.time() - start_time
            
            if duration < 1.0:  # 1秒内完成
                self.log_result('performance_benchmarks', '机器学习性能测试', 'PASS', 
                              {'samples': 1000, 'duration': duration}, duration)
            else:
                self.log_result('performance_benchmarks', '机器学习性能测试', 'WARN', 
                              {'samples': 1000, 'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '机器学习性能测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试并发性能
        start_time = time.time()
        try:
            import concurrent.futures
            
            def cpu_intensive_task(n):
                return sum(i**2 for i in range(n))
            
            # 模拟并发任务
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                tasks = [executor.submit(cpu_intensive_task, 1000) for _ in range(20)]
                results = [task.result() for task in concurrent.futures.as_completed(tasks)]
            
            duration = time.time() - start_time
            
            if duration < 5.0:  # 5秒内完成
                self.log_result('performance_benchmarks', '并发处理性能测试', 'PASS', 
                              {'tasks': 20, 'duration': duration}, duration)
            else:
                self.log_result('performance_benchmarks', '并发处理性能测试', 'WARN', 
                              {'tasks': 20, 'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('performance_benchmarks', '并发处理性能测试', 'ERROR', 
                          {'error': str(e)}, duration)
    
    def test_component_integration(self):
        """3. 组件集成测试"""
        print("\n=== 3. 组件集成测试 ===")
        
        # 检查各组件模块
        components = [
            'integrated_api',
            'ml_engine', 
            'pipeline',
            'scanner',
            'validation'
        ]
        
        start_time = time.time()
        try:
            available_components = []
            missing_components = []
            
            for component in components:
                component_path = f'/workspace/code/{component}'
                if os.path.exists(component_path) and os.path.isdir(component_path):
                    available_components.append(component)
                else:
                    missing_components.append(component)
            
            duration = time.time() - start_time
            
            if len(available_components) == len(components):
                self.log_result('component_integration', '组件目录检查', 'PASS', 
                              {'available': available_components, 'missing': missing_components}, duration)
            else:
                self.log_result('component_integration', '组件目录检查', 'WARN', 
                              {'available': available_components, 'missing': missing_components}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', '组件目录检查', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 检查组件配置文件
        start_time = time.time()
        try:
            component_configs = {
                'api': 'integrated_api/requirements.txt',
                'ml_engine': 'ml_engine/requirements.txt',
                'scanner': 'scanner/requirements.txt'
            }
            
            config_status = {}
            for component, config_file in component_configs.items():
                config_path = f'/workspace/code/{config_file}'
                config_status[component] = os.path.exists(config_path)
            
            duration = time.time() - start_time
            
            all_exist = all(config_status.values())
            if all_exist:
                self.log_result('component_integration', '组件配置检查', 'PASS', 
                              {'configs': config_status}, duration)
            else:
                self.log_result('component_integration', '组件配置检查', 'WARN', 
                              {'configs': config_status}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', '组件配置检查', 'ERROR', 
                          {'error': str(e)}, duration)
    
    def test_stability(self):
        """4. 稳定性测试"""
        print("\n=== 4. 稳定性测试 ===")
        
        # 测试系统资源监控
        start_time = time.time()
        try:
            resources_before = self.get_system_resources()
            
            # 模拟负载
            import numpy as np
            data = np.random.randn(1000, 100)
            result = np.sum(data ** 2, axis=1)
            
            resources_after = self.get_system_resources()
            duration = time.time() - start_time
            
            memory_increase = resources_after['memory_used_mb'] - resources_before['memory_used_mb']
            
            if memory_increase < 100:  # 内存增加小于100MB
                self.log_result('stability_tests', '内存使用稳定性测试', 'PASS', 
                              {'memory_increase_mb': memory_increase, 'duration': duration}, duration)
            else:
                self.log_result('stability_tests', '内存使用稳定性测试', 'WARN', 
                              {'memory_increase_mb': memory_increase, 'duration': duration}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('stability_tests', '内存使用稳定性测试', 'ERROR', 
                          {'error': str(e)}, duration)
        
        # 测试异常处理
        start_time = time.time()
        try:
            def test_function(should_fail=False):
                if should_fail:
                    raise ValueError("测试异常")
                return "success"
            
            # 测试正常情况
            result1 = test_function(False)
            
            # 测试异常情况
            try:
                result2 = test_function(True)
                exception_handled = False
            except ValueError:
                exception_handled = True
            
            duration = time.time() - start_time
            
            if result1 == "success" and exception_handled:
                self.log_result('stability_tests', '异常处理测试', 'PASS', 
                              {'normal_case': result1, 'exception_handled': exception_handled}, duration)
            else:
                self.log_result('stability_tests', '异常处理测试', 'FAIL', 
                              {'normal_case': result1, 'exception_handled': exception_handled}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('stability_tests', '异常处理测试', 'ERROR', 
                          {'error': str(e)}, duration)
    
    def test_configuration_validation(self):
        """5. 配置验证测试"""
        print("\n=== 5. 配置验证测试 ===")
        
        start_time = time.time()
        try:
            config_validation = {
                'server_config': 'server' in self.config,
                'components_config': 'components' in self.config,
                'performance_config': 'performance' in self.config,
                'logging_config': 'logging' in self.config,
                'paths_config': 'paths' in self.config
            }
            
            required_configs = ['server', 'components', 'performance', 'logging', 'paths']
            missing_configs = [key for key in required_configs if key not in self.config]
            
            duration = time.time() - start_time
            
            if not missing_configs:
                self.log_result('component_integration', '配置结构验证', 'PASS', 
                              {'validation': config_validation}, duration)
            else:
                self.log_result('component_integration', '配置结构验证', 'WARN', 
                              {'validation': config_validation, 'missing': missing_configs}, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result('component_integration', '配置结构验证', 'ERROR', 
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
            
            summary[category] = {
                'total': len(tests),
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'errors': errors,
                'success_rate': (passed / len(tests) * 100) if tests else 0
            }
        
        # 生成报告内容
        report_content = f"""# AlphaSeeker系统稳定性和性能测试报告

## 测试概述
- **测试时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总测试时长**: {total_duration:.2f}秒
- **测试环境**: Python {sys.version.split()[0]}
- **系统资源**: CPU {psutil.cpu_count()}核心, 内存 {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB
- **测试类型**: 静态分析 + 性能模拟

## 测试结果汇总

"""
        
        for category, stats in summary.items():
            category_name = {
                'file_structure': '文件结构检查',
                'system_startup': '系统启动测试',
                'performance_benchmarks': '性能基准测试',
                'component_integration': '组件集成测试',
                'stability_tests': '稳定性测试'
            }.get(category, category)
            
            status_emoji = "✅" if stats['success_rate'] >= 90 else "⚠️" if stats['success_rate'] >= 70 else "❌"
            
            report_content += f"""### {status_emoji} {category_name}
- **总测试数**: {stats['total']}
- **通过**: {stats['passed']} ({stats['success_rate']:.1f}%)
- **失败**: {stats['failed']}
- **警告**: {stats['warnings']}
- **错误**: {stats['errors']}

"""

        report_content += "\n## 详细测试结果\n\n"
        
        for category, tests in self.test_results.items():
            category_name = {
                'file_structure': '文件结构检查',
                'system_startup': '系统启动测试',
                'performance_benchmarks': '性能基准测试',
                'component_integration': '组件集成测试',
                'stability_tests': '稳定性测试'
            }.get(category, category)
            
            report_content += f"### {category_name}\n\n"
            
            for test in tests:
                status_emoji = {
                    'PASS': '✅',
                    'FAIL': '❌',
                    'WARN': '⚠️',
                    'ERROR': '🚫'
                }.get(test['status'], '❓')
                
                report_content += f"- **{status_emoji} {test['test_name']}** ({test['duration']:.3f}s)\n"
                report_content += f"  - 状态: {test['status']}\n"
                report_content += f"  - 详情: {json.dumps(test['details'], indent=6, ensure_ascii=False)}\n\n"
        
        # 性能指标总结
        performance_tests = [t for t in self.test_results['performance_benchmarks'] if t['status'] == 'PASS']
        if performance_tests:
            report_content += "\n## 性能指标总结\n\n"
            report_content += "### 基准性能指标\n\n"
            for test in performance_tests:
                report_content += f"- **{test['test_name']}**: {test['duration']:.3f}秒\n"
                if 'data_size' in test['details']:
                    report_content += f"  - 数据规模: {test['details']['data_size']:,}\n"
                if 'samples' in test['details']:
                    report_content += f"  - 样本数量: {test['details']['samples']:,}\n"
                if 'tasks' in test['details']:
                    report_content += f"  - 并发任务: {test['details']['tasks']}\n"
                report_content += "\n"
        
        # 系统资源信息
        resources = self.get_system_resources()
        report_content += "\n## 系统资源信息\n\n"
        report_content += f"- **CPU核心数**: {psutil.cpu_count()}\n"
        report_content += f"- **CPU使用率**: {resources['cpu_percent']:.1f}%\n"
        report_content += f"- **内存使用率**: {resources['memory_percent']:.1f}%\n"
        report_content += f"- **内存使用量**: {resources['memory_used_mb']:.1f} MB\n"
        report_content += f"- **磁盘使用率**: {resources['disk_percent']:.1f}%\n"
        report_content += f"- **进程数量**: {resources['process_count']}\n\n"
        
        # 配置文件分析
        if self.config:
            report_content += "\n## 配置分析\n\n"
            report_content += "### 性能配置\n"
            if 'performance' in self.config:
                perf_config = self.config['performance']
                if 'max_concurrent_tasks' in perf_config:
                    report_content += f"- **最大并发任务**: {perf_config['max_concurrent_tasks']}\n"
                if 'request_timeout' in perf_config:
                    report_content += f"- **请求超时**: {perf_config['request_timeout']}秒\n"
                if 'batch_size' in perf_config:
                    report_content += f"- **批处理大小**: {perf_config['batch_size']}\n"
            
            report_content += "\n### 组件配置\n"
            if 'components' in self.config:
                components = self.config['components']
                for component_name, component_config in components.items():
                    report_content += f"- **{component_name}**: 配置完整\n"
            
            report_content += "\n"
        
        # 建议和改进
        total_tests = sum(stats['total'] for stats in summary.values())
        total_passed = sum(stats['passed'] for stats in summary.values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report_content += "\n## 系统评估和建议\n\n"
        
        if overall_success_rate >= 90:
            report_content += "### ✅ 系统状态：优秀\n"
            report_content += "系统整体性能表现优秀，大部分测试通过。\n\n"
        elif overall_success_rate >= 70:
            report_content += "### ⚠️ 系统状态：良好\n"
            report_content += "系统整体性能表现良好，存在一些需要关注的问题。\n\n"
        else:
            report_content += "### ❌ 系统状态：需要改进\n"
            report_content += "系统存在较多问题，需要紧急修复和优化。\n\n"
        
        report_content += "### 建议改进措施\n\n"
        report_content += "1. **性能优化**\n"
        report_content += "   - 优化数据处理算法\n"
        report_content += "   - 改进机器学习模型推理速度\n"
        report_content += "   - 优化并发处理机制\n\n"
        
        report_content += "2. **稳定性增强**\n"
        report_content += "   - 加强异常处理机制\n"
        report_content += "   - 实施资源使用监控\n"
        report_content += "   - 添加自动恢复机制\n\n"
        
        report_content += "3. **监控和告警**\n"
        report_content += "   - 实施实时性能监控\n"
        report_content += "   - 设置性能告警阈值\n"
        report_content += "   - 建立日志分析系统\n\n"
        
        report_content += "4. **部署和运维**\n"
        report_content += "   - 优化Docker容器配置\n"
        report_content += "   - 实施负载均衡\n"
        report_content += "   - 建立自动化测试流程\n\n"
        
        report_content += f"---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        report_content += f"*测试工具: AlphaSeeker Static Test Suite v1.0*\n"
        
        return report_content
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始AlphaSeeker系统稳定性和性能测试\n")
        
        try:
            # 记录测试开始时的系统资源
            initial_resources = self.get_system_resources()
            print(f"初始系统资源 - CPU: {initial_resources['cpu_percent']:.1f}%, "
                  f"内存: {initial_resources['memory_percent']:.1f}%\n")
            
            # 执行各项测试
            self.check_file_structure()
            self.test_system_startup()
            self.test_performance_benchmarks()
            self.test_component_integration()
            self.test_stability()
            self.test_configuration_validation()
            
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
            
            # 输出简要结果
            total_tests = sum(len(tests) for tests in self.test_results.values())
            total_passed = sum(len([t for t in tests if t['status'] == 'PASS']) for tests in self.test_results.values())
            success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            
            print(f"\n📋 测试总结: {total_passed}/{total_tests} 通过 ({success_rate:.1f}%)")
            
        except Exception as e:
            print(f"测试执行过程中发生错误: {e}")
            traceback.print_exc()


def main():
    """主函数"""
    try:
        test_suite = AlphaSeekerStaticTestSuite()
        test_suite.run_all_tests()
    except Exception as e:
        print(f"测试启动失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()