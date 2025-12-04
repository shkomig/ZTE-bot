"""
Close All Positions with Limit Orders
=====================================
Uses aggressive limit orders (at bid price for sells)
"""
from ib_insync import IB, Stock, LimitOrder
import random
import time

def close_all():
    ib = IB()
    client_id = random.randint(600, 699)
    
    try:
        ib.connect('127.0.0.1', 7497, clientId=client_id)
        print(f"[TWS] Connected (client {client_id})")
        
        positions = ib.positions()
        if not positions:
            print("[TWS] No positions")
            return
        
        print(f"\n[TWS] Closing {len(positions)} positions with LIMIT orders...")
        
        trades = []
        for pos in positions:
            if pos.position == 0:
                continue
            
            symbol = pos.contract.symbol
            qty = pos.position
            
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            
            # Get current market price
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(0.5)
            
            # Use bid for sells, ask for buys (aggressive)
            if qty > 0:  # Long position - SELL
                action = 'SELL'
                price = ticker.bid if ticker.bid > 0 else ticker.last
                if price <= 0:
                    price = pos.avgCost * 0.99  # 1% below entry as fallback
            else:  # Short position - BUY
                action = 'BUY'
                price = ticker.ask if ticker.ask > 0 else ticker.last
                if price <= 0:
                    price = pos.avgCost * 1.01  # 1% above entry as fallback
            
            ib.cancelMktData(contract)
            
            # Create limit order
            order = LimitOrder(action, abs(qty), round(price, 2))
            order.outsideRth = True  # Allow after hours
            order.tif = 'GTC'  # Good till cancel
            
            trade = ib.placeOrder(contract, order)
            trades.append((symbol, trade, price))
            print(f"  {action} {abs(qty)} {symbol} @ ${price:.2f} LIMIT")
            
            time.sleep(0.3)
        
        # Wait for fills
        print("\n[TWS] Waiting for fills (30 sec)...")
        for i in range(30):
            ib.sleep(1)
            
            filled_count = sum(1 for _, t, _ in trades if t.orderStatus.status == 'Filled')
            if filled_count == len(trades):
                print(f"[TWS] All {filled_count} orders filled!")
                break
            
            if i % 5 == 0:
                print(f"  {filled_count}/{len(trades)} filled...")
        
        # Final status
        print("\n[TWS] Final Status:")
        for symbol, trade, price in trades:
            status = trade.orderStatus.status
            filled = trade.orderStatus.filled
            avg = trade.orderStatus.avgFillPrice
            print(f"  {symbol}: {status} - {filled} filled @ ${avg:.2f}" if avg > 0 else f"  {symbol}: {status}")
        
        # Check remaining
        ib.sleep(2)
        remaining = [p for p in ib.positions() if p.position != 0]
        print(f"\n[TWS] Remaining positions: {len(remaining)}")
        
        if remaining:
            for p in remaining:
                print(f"  {p.contract.symbol}: {p.position}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        ib.disconnect()
        print("\n[TWS] Disconnected")


if __name__ == "__main__":
    close_all()
