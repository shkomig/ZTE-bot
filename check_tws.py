"""Check TWS Status"""
from ib_insync import IB
import random

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=random.randint(300, 399))

print("\n=== TWS STATUS ===")
print(f"\nAccount: {ib.managedAccounts()}")

positions = ib.positions()
print(f"\nOPEN POSITIONS ({len(positions)}):")
for p in positions:
    if p.position != 0:
        print(f"  {p.contract.symbol}: {p.position} shares @ ${p.avgCost:.2f}")

# Use reqAllOpenOrders for more reliable order checking
orders = ib.reqAllOpenOrders()
ib.sleep(0.5)
print(f"\nOPEN ORDERS ({len(orders)}):")
for o in orders:
    symbol = o.contract.symbol if hasattr(o, 'contract') else 'N/A'
    order_type = o.order.orderType
    price = o.order.auxPrice if order_type == "STP" else o.order.lmtPrice
    print(f"  {o.order.action} {int(o.order.totalQuantity)} {symbol} @ {order_type} ${price:.2f}")

ib.disconnect()
print("\n=== DONE ===")
