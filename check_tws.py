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

orders = ib.openOrders()
print(f"\nOPEN ORDERS ({len(orders)}):")
for o in orders:
    print(f"  {o.action} {o.totalQuantity} {o.contract.symbol if hasattr(o, 'contract') else 'N/A'}")

ib.disconnect()
print("\n=== DONE ===")
