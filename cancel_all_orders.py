"""Cancel all open orders in TWS."""
from ib_insync import IB
import random

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=random.randint(500, 599))
ib.sleep(2)

# Get all open orders
orders = ib.reqAllOpenOrders()
ib.sleep(1)

print(f"\n=== CANCELLING {len(orders)} ORDERS ===\n")

# Cancel each order
cancelled = 0
for o in orders:
    try:
        ib.cancelOrder(o.order)
        cancelled += 1
        print(f"  Cancelled: {o.contract.symbol} {o.order.orderType} #{o.order.orderId}")
    except Exception as e:
        print(f"  Error: {e}")
    ib.sleep(0.1)

ib.sleep(2)

# Verify
remaining = ib.reqAllOpenOrders()
print(f"\n=== CANCELLED {cancelled} ORDERS ===")
print(f"=== REMAINING: {len(remaining)} ===")
ib.disconnect()
