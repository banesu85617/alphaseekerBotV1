#!/usr/bin/env python3
"""
快速测试真实数据源功能
测试各个数据源是否能正确获取当前市场价格
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_direct_data_sources():
    """直接测试各个数据源"""
    print("🔍 AlphaSeeker 2.0 真实数据源快速测试")
    print("=" * 60)
    
    try:
        # 测试ccxt导入
        import ccxt
        print("✅ CCXT 库导入成功")
        
        # 测试Binance数据源
        print("\n📊 测试 Binance 数据源...")
        binance = ccxt.binance()
        
        # 获取BTC价格
        btc_ticker = binance.fetch_ticker('BTC/USDT')
        btc_price = btc_ticker['last']
        print(f"💰 BTC 价格 (Binance): ${btc_price:,.2f}")
        print(f"📈 24h 变化: {btc_ticker['percentage']:.2f}%")
        
        # 获取ETH价格
        eth_ticker = binance.fetch_ticker('ETH/USDT')
        eth_price = eth_ticker['last']
        print(f"💎 ETH 价格 (Binance): ${eth_price:,.2f}")
        print(f"📈 24h 变化: {eth_ticker['percentage']:.2f}%")
        
        # 测试OKX数据源
        print("\n📊 测试 OKX 数据源...")
        okx = ccxt.okx()
        
        btc_okx = okx.fetch_ticker('BTC/USDT')
        print(f"💰 BTC 价格 (OKX): ${btc_okx['last']:,.2f}")
        
        eth_okx = okx.fetch_ticker('ETH/USDT')
        print(f"💎 ETH 价格 (OKX): ${eth_okx['last']:,.2f}")
        
        # 测试CoinGecko API
        print("\n📊 测试 CoinGecko 数据源...")
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd') as response:
                if response.status == 200:
                    data = await response.json()
                    cg_btc = data['bitcoin']['usd']
                    cg_eth = data['ethereum']['usd']
                    print(f"💰 BTC 价格 (CoinGecko): ${cg_btc:,.2f}")
                    print(f"💎 ETH 价格 (CoinGecko): ${cg_eth:,.2f}")
                else:
                    print(f"❌ CoinGecko API 错误: {response.status}")
        
        print("\n" + "=" * 60)
        print("✅ 所有数据源测试完成！")
        
        # 价格验证
        print("\n📋 价格验证结果:")
        avg_btc = (btc_price + btc_okx['last'] + cg_btc) / 3
        avg_eth = (eth_price + eth_okx['last'] + cg_eth) / 3
        
        print(f"💰 BTC 平均价格: ${avg_btc:,.2f}")
        print(f"💎 ETH 平均价格: ${avg_eth:,.2f}")
        
        # 检查是否为真实价格
        if btc_price > 50000:
            print("✅ BTC 价格显示为真实市场价格")
        else:
            print("❌ BTC 价格异常，可能是模拟数据")
            
        if eth_price > 1000:
            print("✅ ETH 价格显示为真实市场价格")
        else:
            print("❌ ETH 价格异常，可能是模拟数据")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_integration():
    """测试集成版本的数据源"""
    print("\n🔧 测试 AlphaSeeker 集成版本...")
    
    try:
        # 导入真实数据提供者
        from data_sources.real_data_provider import RealDataProvider
        
        provider = RealDataProvider()
        print("✅ 真实数据提供者初始化成功")
        
        # 获取BTC数据
        btc_data = await provider.get_market_data("BTC")
        if btc_data:
            print(f"💰 BTC 数据 (集成): ${btc_data.get('price', 'N/A')}")
            print(f"📈 数据源: {btc_data.get('source', 'N/A')}")
            print(f"⏰ 更新时间: {btc_data.get('timestamp', 'N/A')}")
        else:
            print("❌ BTC 数据获取失败")
            
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_direct_data_sources())
    asyncio.run(test_integration())