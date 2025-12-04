"""
Close All Open Positions Script
================================
סוגר את כל הפוזיציות הפתוחות ב-TWS
"""

from ib_insync import IB, Stock, MarketOrder
import random
import time

def close_all_positions():
    ib = IB()
    
    try:
        # Connect to TWS
        client_id = random.randint(100, 999)
        ib.connect('127.0.0.1', 7497, clientId=client_id)
        print(f"[TWS] Connected with client ID {client_id}")
        
        # Get all positions
        positions = ib.positions()
        
        if not positions:
            print("[TWS] No open positions to close")
            return
        
        print(f"\n[TWS] Found {len(positions)} positions to close:")
        
        for pos in positions:
            symbol = pos.contract.symbol
            qty = pos.position
            
            if qty == 0:
                continue
                
            print(f"  - {symbol}: {qty} shares")
            
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            
            # Create market order (opposite direction)
            action = 'SELL' if qty > 0 else 'BUY'
            order = MarketOrder(action, abs(qty))
            
            # Place order
            trade = ib.placeOrder(contract, order)
            print(f"  [ORDER] {action} {abs(qty)} {symbol} @ MARKET")
            
            time.sleep(0.5)  # Wait between orders
        
        # Wait for orders to fill
        print("\n[TWS] Waiting for orders to fill...")
        time.sleep(3)
        
        # Cancel any open orders
        open_orders = ib.openOrders()
        if open_orders:
            print(f"\n[TWS] Cancelling {len(open_orders)} open orders...")
            for order in open_orders:
                ib.cancelOrder(order)
        
        print("\n[TWS] All positions closed!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        ib.disconnect()
        print("[TWS] Disconnected")


if __name__ == "__main__":
    close_all_positions()
