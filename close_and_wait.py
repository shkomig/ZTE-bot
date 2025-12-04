"""
Close All Positions - Wait for Fill
====================================
"""
from ib_insync import IB, Stock, MarketOrder
import random
import time

def close_all_positions():
    ib = IB()
    
    try:
        client_id = random.randint(400, 499)
        ib.connect('127.0.0.1', 7497, clientId=client_id)
        print(f"[TWS] Connected (client {client_id})")
        
        positions = ib.positions()
        
        if not positions:
            print("[TWS] No positions")
            return
        
        print(f"\n[TWS] Closing {len(positions)} positions...")
        
        trades = []
        for pos in positions:
            if pos.position == 0:
                continue
                
            symbol = pos.contract.symbol
            qty = pos.position
            
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            
            action = 'SELL' if qty > 0 else 'BUY'
            order = MarketOrder(action, abs(qty))
            order.tif = 'GTC'  # Good Till Cancel
            
            trade = ib.placeOrder(contract, order)
            trades.append((symbol, trade))
            print(f"  {action} {abs(qty)} {symbol}")
            
            time.sleep(0.3)
        
        # Wait for fills
        print("\n[TWS] Waiting for fills (max 30 sec)...")
        for i in range(30):
            ib.sleep(1)
            
            all_filled = True
            for symbol, trade in trades:
                if trade.orderStatus.status not in ['Filled', 'Cancelled']:
                    all_filled = False
                    
            if all_filled:
                print("[TWS] All orders filled!")
                break
                
            if i % 5 == 0:
                print(f"  ...waiting ({i}s)")
        
        # Show results
        print("\n[TWS] Results:")
        for symbol, trade in trades:
            status = trade.orderStatus.status
            filled = trade.orderStatus.filled
            avg_price = trade.orderStatus.avgFillPrice
            print(f"  {symbol}: {status} - {filled} @ ${avg_price:.2f}")
        
        # Final check
        ib.sleep(2)
        remaining = [p for p in ib.positions() if p.position != 0]
        print(f"\n[TWS] Remaining positions: {len(remaining)}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        ib.disconnect()
        print("[TWS] Done")


if __name__ == "__main__":
    close_all_positions()
