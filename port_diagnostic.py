#!/usr/bin/env python3
"""
AlphaSeeker 端口诊断工具
=========================

检查系统中端口占用状态、启动服务、检查LLM连接

作者: AlphaSeeker Team
版本: 1.0.0
"""

import os
import sys
import socket
import subprocess
import time
import requests
from pathlib import Path

def check_port_status(port):
    """检查端口占用状态"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"检查端口 {port} 时出错: {e}")
        return False

def find_process_using_port(port):
    """查找占用指定端口的进程"""
    try:
        result = subprocess.run(
            ["netstat", "-tulpn"], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        for line in result.stdout.split('\n'):
            if str(port) in line:
                parts = line.split()
                if ':' in parts[-1]:
                    pid_info = parts[-1]
                    print(f"端口 {port} 被占用: {pid_info}")
                    return True
        return False
    except Exception as e:
        print(f"查找进程时出错: {e}")
        return False

def test_llm_connection():
    """测试LLM连接"""
    llm_url = "http://localhost:11434"
    print(f"测试LLM连接: {llm_url}")
    
    try:
        response = requests.get(f"{llm_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ LLM连接成功!")
            print(f"可用模型: {[model['name'] for model in models.get('models', [])]}")
            return True
        else:
            print(f"❌ LLM连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LLM连接失败: {e}")
        print("请确保Ollama服务正在运行: ollama serve")
        return False

def check_requirements():
    """检查依赖包"""
    print("检查Python依赖包...")
    required_packages = [
        'fastapi', 'uvicorn', 'lightgbm', 'pandas', 
        'numpy', 'scikit-learn', 'requests', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包，请安装: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def start_alphaseeker():
    """启动AlphaSeeker系统"""
    print("启动AlphaSeeker系统...")
    
    # 检查端口8000是否被占用
    if check_port_status(8000):
        print("⚠️  端口8000已被占用")
        if find_process_using_port(8000):
            print("请先停止占用端口8000的进程")
            return False
    
    # 测试LLM连接
    if not test_llm_connection():
        print("⚠️  LLM服务未连接，系统将使用模拟模式")
    
    print("🚀 启动AlphaSeeker...")
    try:
        # 启动主应用
        subprocess.run([
            sys.executable, 
            "main_integration.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 系统停止")
    except Exception as e:
        print(f"启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("🔍 AlphaSeeker 端口诊断工具")
    print("=" * 50)
    
    # 检查依赖
    print("\n1. 检查依赖包:")
    if not check_requirements():
        print("\n请先安装缺失的依赖包")
        return
    
    # 检查端口状态
    print("\n2. 检查端口状态:")
    ports_to_check = [8000, 11434]  # 系统端口 + LLM端口
    for port in ports_to_check:
        if check_port_status(port):
            print(f"端口 {port}: ✅ 已开放")
        else:
            print(f"端口 {port}: ❌ 未开放")
    
    # 测试LLM连接
    print("\n3. 测试LLM连接:")
    test_llm_connection()
    
    # 端口占用检查
    print("\n4. 检查端口占用:")
    if check_port_status(8000):
        find_process_using_port(8000)
    
    print("\n" + "=" * 50)
    print("诊断完成!")
    print("=" * 50)

if __name__ == "__main__":
    # 检查是否有启动参数
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        # 直接启动系统
        start_alphaseeker()
    else:
        # 运行诊断
        main()
        print("\n要启动系统，请运行: python port_diagnostic.py start")