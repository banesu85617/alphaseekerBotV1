#!/usr/bin/env python3
"""
测试AlphaSeeker真实数据获取
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_alphaseeker_data():
    """测试AlphaSeeker数据获取"""
    print("🧪 AlphaSeeker 真实数据获取测试")
    print("=" * 50)
    
    try:
        from data_sources.real_data_provider import get_real_market_data
        
        # 测试主要币种
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            print(f"\n📊 获取 {symbol} 数据...")
            data = await get_real_market_data(symbol)
            
            if data:
                print(f"✅ {symbol}: ${data['price']:,.2f}")
                print(f"📈 24h变化: {data['change_24h']:.2f}%")
                print(f"📊 数据源: {data['source']}")
                print(f"⏰ 时间: {data['timestamp']}")
                print(f"🔄 真实数据: {data['is_real_data']}")
            else:
                print(f"❌ {symbol}: 数据获取失败")
        
        print("\n" + "=" * 50)
        print("🎯 AlphaSeeker数据获取测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_alphaseeker_data())