#!/usr/bin/env python3
"""
修正后的真实数据源测试
测试各个数据源是否能正确获取当前市场价格
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_direct_data_sources():
    """直接测试各个数据源"""
    print("🔍 AlphaSeeker 2.0 真实数据源修正测试")
    print("=" * 60)
    
    try:
        # 测试ccxt导入
        import ccxt
        print("✅ CCXT 库导入成功")
        
        # 由于Binance有地理限制，先测试OKX
        print("\n📊 测试 OKX 数据源 (主数据源)...")
        okx = ccxt.okx()
        
        # 获取BTC价格
        btc_ticker = okx.fetch_ticker('BTC/USDT')
        btc_price = btc_ticker['last']
        print(f"💰 BTC 价格 (OKX): ${btc_price:,.2f}")
        print(f"📈 24h 变化: {btc_ticker['percentage']:.2f}%")
        print(f"📊 24h 最高: ${btc_ticker['high']:,.2f}")
        print(f"📊 24h 最低: ${btc_ticker['low']:,.2f}")
        
        # 获取ETH价格
        eth_ticker = okx.fetch_ticker('ETH/USDT')
        eth_price = eth_ticker['last']
        print(f"💎 ETH 价格 (OKX): ${eth_price:,.2f}")
        print(f"📈 24h 变化: {eth_ticker['percentage']:.2f}%")
        
        # 测试其他主要币种
        print("\n📊 测试其他主要币种...")
        
        # ADA价格
        try:
            ada_ticker = okx.fetch_ticker('ADA/USDT')
            ada_price = ada_ticker['last']
            print(f"💎 ADA 价格 (OKX): ${ada_price:.4f}")
        except Exception as e:
            print(f"❌ ADA 获取失败: {e}")
        
        # SOL价格
        try:
            sol_ticker = okx.fetch_ticker('SOL/USDT')
            sol_price = sol_ticker['last']
            print(f"💎 SOL 价格 (OKX): ${sol_price:.2f}")
        except Exception as e:
            print(f"❌ SOL 获取失败: {e}")
        
        # 测试CoinGecko API (作为备用数据源)
        print("\n📊 测试 CoinGecko 数据源...")
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # 获取主要币种价格
            url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,cardano,solana&vs_currencies=usd&include_24hr_change=true'
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    cg_btc = data['bitcoin']['usd']
                    cg_btc_change = data['bitcoin']['usd_24h_change']
                    cg_eth = data['ethereum']['usd']
                    cg_eth_change = data['ethereum']['usd_24h_change']
                    
                    print(f"💰 BTC 价格 (CoinGecko): ${cg_btc:,.2f} (24h: {cg_btc_change:.2f}%)")
                    print(f"💎 ETH 价格 (CoinGecko): ${cg_eth:,.2f} (24h: {cg_eth_change:.2f}%)")
                    
                    # 检查其他币种
                    if 'cardano' in data:
                        cg_ada = data['cardano']['usd']
                        print(f"💎 ADA 价格 (CoinGecko): ${cg_ada:.4f}")
                    
                    if 'solana' in data:
                        cg_sol = data['solana']['usd']
                        print(f"💎 SOL 价格 (CoinGecko): ${cg_sol:.2f}")
                        
                else:
                    print(f"❌ CoinGecko API 错误: {response.status}")
        
        print("\n" + "=" * 60)
        print("✅ 数据源测试完成！")
        
        # 价格验证
        print("\n📋 价格验证结果:")
        print(f"💰 BTC 当前价格: ${btc_price:,.2f}")
        print(f"💎 ETH 当前价格: ${eth_price:,.2f}")
        
        # 检查是否为真实价格
        if btc_price > 50000:
            print("✅ BTC 价格显示为真实市场价格")
        else:
            print("❌ BTC 价格异常，可能是模拟数据")
            
        if eth_price > 1000:
            print("✅ ETH 价格显示为真实市场价格")
        else:
            print("❌ ETH 价格异常，可能是模拟数据")
            
        # 建议的数据源策略
        print("\n💡 推荐数据源策略:")
        print("1. 主要数据源: OKX (无地理限制)")
        print("2. 备用数据源: CoinGecko (免费API)")
        print("3. Binance: 暂不可用 (地理限制)")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_integration():
    """测试集成版本的数据源"""
    print("\n🔧 测试 AlphaSeeker 集成版本...")
    
    try:
        # 导入真实数据提供者函数
        from data_sources.real_data_provider import get_real_market_data
        
        print("✅ 真实数据提供者函数导入成功")
        
        # 获取BTC数据
        btc_data = await get_real_market_data("BTCUSDT")
        if btc_data:
            print(f"💰 BTC 数据 (集成): ${btc_data.get('price', 'N/A')}")
            print(f"📈 数据源: {btc_data.get('source', 'N/A')}")
            print(f"⏰ 更新时间: {btc_data.get('timestamp', 'N/A')}")
            print(f"✅ 是否真实数据: {btc_data.get('is_real_data', 'N/A')}")
        else:
            print("❌ BTC 数据获取失败")
            
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_multi_source_manager():
    """测试多数据源管理器"""
    print("\n🔧 测试多数据源管理器...")
    
    try:
        from data_sources.multi_source_manager import data_source_manager
        
        print("✅ 多数据源管理器导入成功")
        
        # 测试获取BTC数据
        btc_data = await data_source_manager.get_market_data("BTCUSDT")
        if btc_data:
            print(f"💰 BTC 数据 (管理器): ${btc_data.price}")
            print(f"📈 数据源: {btc_data.source}")
            print(f"⏰ 时间戳: {btc_data.timestamp}")
        else:
            print("❌ 多数据源管理器 BTC 数据获取失败")
            
    except Exception as e:
        print(f"❌ 多数据源管理器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行所有测试
    asyncio.run(test_direct_data_sources())
    asyncio.run(test_integration())
    asyncio.run(test_multi_source_manager())