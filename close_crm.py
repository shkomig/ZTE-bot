"""
Close CRM position to make room for V3.7.7 hardened system
"""
from ib_insync import IB, Stock, MarketOrder
import time

# Connect to TWS
ib = IB()
print("Connecting to TWS...")
ib.connect('127.0.0.1', 7497, clientId=999)
print("Connected!")

# Get CRM contract
crm = Stock('CRM', 'SMART', 'USD')
ib.qualifyContracts(crm)

# Check current position
positions = ib.positions()
crm_position = None
for pos in positions:
    if pos.contract.symbol == 'CRM':
        crm_position = pos
        break

if not crm_position:
    print("No CRM position found!")
    ib.disconnect()
    exit(1)

print(f"\nCurrent CRM position: {int(crm_position.position)} shares @ ${crm_position.avgCost:.2f}")
quantity = int(abs(crm_position.position))

# Cancel existing CRM orders
print("\nCancelling existing CRM orders...")
all_orders = ib.openOrders()
for order in all_orders:
    if order.contract.symbol == 'CRM':
        print(f"  Cancelling: {order.order.action} {order.order.totalQuantity} CRM @ {order.order.orderType}")
        ib.cancelOrder(order.order)

time.sleep(1)

# Place market sell order
print(f"\nPlacing MARKET SELL order for {quantity} CRM shares...")
market_order = MarketOrder('SELL', quantity)
market_order.outsideRth = True

trade = ib.placeOrder(crm, market_order)
ib.sleep(2)

print(f"\nOrder placed!")
print(f"Status: {trade.orderStatus.status}")
print(f"Filled: {trade.orderStatus.filled}/{quantity}")

# Wait for fill
print("\nWaiting for order to fill...")
for i in range(30):
    ib.sleep(1)
    if trade.orderStatus.status == 'Filled':
        print(f"\n✅ Order filled at ${trade.orderStatus.avgFillPrice:.2f}")
        print(f"   Total: {trade.orderStatus.filled} shares")
        break
    elif i % 5 == 0:
        print(f"   Status: {trade.orderStatus.status} ({trade.orderStatus.filled}/{quantity})")

# Final status
print("\n" + "="*70)
print("FINAL STATUS:")
positions_after = ib.positions()
print(f"Open positions: {len([p for p in positions_after if abs(p.position) > 0])}")
for pos in positions_after:
    if abs(pos.position) > 0:
        print(f"  {pos.contract.symbol}: {int(pos.position)} shares @ ${pos.avgCost:.2f}")
print("="*70)

ib.disconnect()
print("\nDone!")
