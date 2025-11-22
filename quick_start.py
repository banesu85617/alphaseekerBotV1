#!/usr/bin/env python3
"""
AlphaSeeker 一键启动脚本
=======================

自动解决所有问题并启动系统

作者: AlphaSeeker Team
版本: 1.0.0
"""

import os
import sys
import subprocess
import time
import signal
import socket
from pathlib import Path

def check_port_available(port):
    """检查端口是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result != 0
    except:
        return False

def install_dependencies():
    """安装依赖包"""
    print("📦 安装依赖包...")
    dependencies = [
        'fastapi', 'uvicorn', 'lightgbm', 'pandas', 
        'numpy', 'scikit-learn', 'requests', 'pyyaml',
        'ccxt', 'psutil', 'aiofiles'
    ]
    
    for dep in dependencies:
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', dep],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"✅ {dep}")
            else:
                print(f"❌ {dep} - {result.stderr}")
        except Exception as e:
            print(f"❌ {dep} - {e}")

def setup_environment():
    """设置环境"""
    print("🔧 设置环境...")
    
    # 创建必要目录
    dirs = ['data', 'logs', 'models', 'cache', 'config']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 创建目录: {dir_name}")
    
    # 创建.env文件
    env_content = """# AlphaSeeker 环境配置
ALPHA_SEEKER_HOST=0.0.0.0
ALPHA_SEEKER_PORT=8000
ALPHA_SEEKER_DEBUG=false
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL_NAME=llama2:13b
LLM_TIMEOUT=10.0
MAX_CONCURRENT_TASKS=32
BATCH_SIZE=100
ENABLE_CACHE=true
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    print("✅ 创建 .env 文件")

def start_ollama():
    """启动Ollama服务"""
    print("🤖 启动Ollama服务...")
    try:
        # 检查Ollama是否已运行
        response = subprocess.run(
            ['curl', '-s', 'http://localhost:11434/api/tags'],
            capture_output=True, timeout=5
        )
        
        if response.returncode == 0:
            print("✅ Ollama服务已运行")
            return True
    except:
        pass
    
    print("⚠️  Ollama服务未运行")
    print("请手动启动Ollama:")
    print("1. 终端1: ollama serve")
    print("2. 终端2: ollama run llama2:13b")
    return False

def start_alphaseeker():
    """启动AlphaSeeker"""
    print("🚀 启动AlphaSeeker系统...")
    
    # 检查端口
    if not check_port_available(8000):
        print("❌ 端口8000被占用，请先释放端口")
        return False
    
    try:
        print("\n" + "="*50)
        print("🚀 AlphaSeeker 启动中...")
        print("="*50)
        print("Web界面: http://localhost:8000")
        print("API文档: http://localhost:8000/docs")
        print("按 Ctrl+C 停止系统")
        print("="*50)
        
        # 启动系统
        subprocess.run([sys.executable, "main_integration.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 系统已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 AlphaSeeker 一键启动脚本")
    print("=" * 60)
    
    # 设置工作目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 安装依赖
    print("\n1. 安装依赖包:")
    install_dependencies()
    
    # 设置环境
    print("\n2. 设置环境:")
    setup_environment()
    
    # 检查Ollama
    print("\n3. 检查LLM服务:")
    start_ollama()
    
    # 启动系统
    print("\n4. 启动系统:")
    start_alphaseeker()
    
    print("\n" + "=" * 60)
    print("👋 感谢使用AlphaSeeker!")
    print("=" * 60)

if __name__ == "__main__":
    main()