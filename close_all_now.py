"""
Close All Open Positions in TWS - Auto Mode
============================================
"""

from ib_insync import IB, Stock, MarketOrder
import time

TWS_HOST = "127.0.0.1"
TWS_PORT = 7497  # Paper Trading
CLIENT_ID = 98

def main():
    ib = IB()
    
    try:
        print("Connecting to TWS...")
        ib.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
        print("✅ Connected!")
        
        # Get all positions
        positions = ib.positions()
        
        if not positions:
            print("\n📭 No open positions found.")
            return
        
        print(f"\n📊 Found {len(positions)} positions:\n")
        print("-" * 60)
        
        for pos in positions:
            symbol = pos.contract.symbol
            qty = pos.position
            avg_cost = pos.avgCost
            print(f"  {symbol}: {int(qty)} shares @ ${avg_cost:.2f}")
        
        print("-" * 60)
        print("\n🔄 Closing ALL positions (After Hours enabled)...\n")
        
        closed = 0
        for pos in positions:
            symbol = pos.contract.symbol
            qty = pos.position
            
            if qty == 0:
                continue
            
            action = "SELL" if qty > 0 else "BUY"  # Close opposite
            close_qty = abs(int(qty))
            
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                ib.qualifyContracts(contract)
                
                order = MarketOrder(action, close_qty)
                order.outsideRth = True  # Allow after hours trading!
                
                trade = ib.placeOrder(contract, order)
                ib.sleep(1)  # Wait for order to process
                
                print(f"  ✅ {action} {close_qty} {symbol} - Order placed (After Hours)")
                closed += 1
                
            except Exception as e:
                print(f"  ❌ Error closing {symbol}: {e}")
        
        print(f"\n✅ Sent close orders for {closed} positions!")
        
        # Wait and show status
        ib.sleep(3)
        remaining = ib.positions()
        open_count = sum(1 for p in remaining if p.position != 0)
        
        if open_count > 0:
            print(f"\n⚠️ {open_count} positions still open (orders may be pending)")
            for pos in remaining:
                if pos.position != 0:
                    print(f"   - {pos.contract.symbol}: {int(pos.position)}")
        else:
            print("\n✅ All positions closed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ib.disconnect()
        print("\n[TWS] Disconnected")


if __name__ == "__main__":
    main()
