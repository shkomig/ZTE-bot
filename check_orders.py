"""Quick check for open orders in TWS."""
from ib_insync import IB
import random

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=random.randint(400, 499))
ib.sleep(2)

# Get all open orders
orders = ib.reqAllOpenOrders()
print(f"\n=== OPEN ORDERS ({len(orders)}) ===\n")

for o in orders:
    symbol = o.contract.symbol
    action = o.order.action
    qty = o.order.totalQuantity
    order_type = o.order.orderType
    
    if order_type == "LMT":
        price = o.order.lmtPrice
        print(f"  {symbol}: {action} {qty} LIMIT @ ${price:.2f}")
    elif order_type == "STP":
        price = o.order.auxPrice
        print(f"  {symbol}: {action} {qty} STOP @ ${price:.2f}")
    else:
        print(f"  {symbol}: {action} {qty} {order_type}")

ib.disconnect()
print("\n=== DONE ===")
