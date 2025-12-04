"""Check market status and close with Limit orders if needed"""
from ib_insync import IB, Stock, MarketOrder, LimitOrder
import random
import time

ib = IB()
client_id = random.randint(500, 599)
ib.connect('127.0.0.1', 7497, clientId=client_id)

print(f"\n=== MARKET STATUS CHECK ===")

# Check if we can get market data
positions = ib.positions()
print(f"\nPositions: {len(positions)}")

for pos in positions:
    if pos.position == 0:
        continue
        
    symbol = pos.contract.symbol
    qty = pos.position
    
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    # Request market data
    ticker = ib.reqMktData(contract, '', False, False)
    ib.sleep(1)
    
    bid = ticker.bid if ticker.bid > 0 else "N/A"
    ask = ticker.ask if ticker.ask > 0 else "N/A"
    last = ticker.last if ticker.last > 0 else "N/A"
    
    print(f"  {symbol}: {qty} @ entry ${pos.avgCost:.2f}")
    print(f"    Market: Bid={bid} Ask={ask} Last={last}")
    
    ib.cancelMktData(contract)

# Check open orders
orders = ib.openOrders()
print(f"\nOpen Orders: {len(orders)}")
for o in orders:
    print(f"  {o}")

# Check executions today
executions = ib.executions()
print(f"\nExecutions today: {len(executions)}")
for e in executions:
    print(f"  {e}")

ib.disconnect()
print("\n=== DONE ===")
