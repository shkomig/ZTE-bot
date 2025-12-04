"""
Close All Open Positions in TWS
================================
This script connects to TWS and closes all open positions.
"""

from ib_insync import IB, Stock, MarketOrder
import time

TWS_HOST = "127.0.0.1"
TWS_PORT = 7497  # Paper Trading
CLIENT_ID = 98   # Different from bot

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
            
            action = "BUY" if qty < 0 else "SELL"  # Close opposite
            close_qty = abs(int(qty))
            
            if close_qty == 0:
                continue
                
            print(f"  {symbol}: {int(qty)} shares @ ${avg_cost:.2f}")
        
        print("-" * 60)
        
        # Ask for confirmation
        confirm = input("\n❓ Close ALL positions? (yes/no): ").strip().lower()
        
        if confirm != "yes":
            print("❌ Cancelled.")
            return
        
        print("\n🔄 Closing positions...\n")
        
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
                
                print(f"  ✅ {action} {close_qty} {symbol} - Order placed")
                closed += 1
                
            except Exception as e:
                print(f"  ❌ Error closing {symbol}: {e}")
        
        print(f"\n✅ Closed {closed} positions!")
        
        # Wait a moment and show updated positions
        ib.sleep(2)
        remaining = ib.positions()
        if remaining:
            print(f"\n⚠️ {len(remaining)} positions still open (may take a moment to fill)")
        else:
            print("\n✅ All positions closed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ib.disconnect()
        print("\n[TWS] Disconnected")


if __name__ == "__main__":
    main()
