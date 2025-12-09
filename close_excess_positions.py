"""
ZTE Close Excess Positions Script - EMERGENCY CLEANUP
======================================================
Created: 2025-12-09
Purpose: Close excess GOOGL and MU positions from duplicate order incident

INCIDENT DETAILS (09/12/2025):
- GOOGL: Has 135 shares, should have 15 → Sell 120
- MU: Has 180 shares, should have 20 → Sell 160

SAFETY FEATURES:
- Reads actual positions from TWS
- Calculates exact excess to sell
- Shows planned sales for confirmation
- Requires manual approval before execution
- Uses market orders for immediate fill

Run when market opens (9:30 AM ET)
"""

from ib_insync import IB, Stock, MarketOrder
import random
from datetime import datetime

def close_excess_positions():
    """
    Close excess positions with safety confirmation.
    """
    print("\n" + "="*70)
    print("  ZTE EXCESS POSITION CLEANUP - EMERGENCY")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S ET"))
    print("="*70)

    # Target positions (what we SHOULD have)
    TARGET_POSITIONS = {
        'GOOGL': 15,  # Keep 15 shares
        'MU': 20      # Keep 20 shares
    }

    # Step 1: Connect to TWS
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=random.randint(500, 599))
        print("✅ Connected to TWS (Paper Trading - Port 7497)")
    except Exception as e:
        print(f"❌ FAILED to connect to TWS: {e}")
        print("\n⚠️  Make sure TWS or IB Gateway is running!")
        return

    # Step 2: Get current positions from TWS
    print("\n📊 Fetching current positions from TWS...")
    current_positions = {}
    try:
        for pos in ib.positions():
            symbol = pos.contract.symbol
            qty = int(pos.position)
            avg_cost = pos.avgCost
            current_positions[symbol] = {
                'quantity': qty,
                'avg_cost': avg_cost
            }

        print(f"✅ Found {len(current_positions)} open positions")
    except Exception as e:
        print(f"❌ Failed to fetch positions: {e}")
        ib.disconnect()
        return

    # Step 3: Calculate excess to sell
    print("\n" + "="*70)
    print("POSITION ANALYSIS:")
    print("="*70)

    excess_to_sell = {}
    total_excess_value = 0

    for symbol, target_qty in TARGET_POSITIONS.items():
        current = current_positions.get(symbol)

        if not current:
            print(f"⚠️  {symbol}: No position found (expected {target_qty})")
            continue

        current_qty = current['quantity']
        avg_cost = current['avg_cost']

        if current_qty > target_qty:
            excess_qty = current_qty - target_qty
            excess_to_sell[symbol] = {
                'quantity': excess_qty,
                'current_qty': current_qty,
                'target_qty': target_qty,
                'avg_cost': avg_cost
            }
            excess_value = excess_qty * avg_cost
            total_excess_value += excess_value

            print(f"\n🔴 {symbol}:")
            print(f"   Current Position: {current_qty} shares @ ${avg_cost:.2f}")
            print(f"   Target Position:  {target_qty} shares")
            print(f"   EXCESS TO SELL:   {excess_qty} shares (${excess_value:,.2f})")
        else:
            print(f"\n✅ {symbol}: Position OK ({current_qty} shares, target {target_qty})")

    # Step 4: Safety check - if nothing to sell, exit
    if not excess_to_sell:
        print("\n" + "="*70)
        print("✅ NO EXCESS POSITIONS - Nothing to sell!")
        print("="*70)
        ib.disconnect()
        return

    # Step 5: Show execution plan and ask for confirmation
    print("\n" + "="*70)
    print("⚠️  EXECUTION PLAN - REVIEW CAREFULLY:")
    print("="*70)

    for symbol, data in excess_to_sell.items():
        print(f"\n  {symbol}:")
        print(f"    → SELL {data['quantity']} shares (Market Order)")
        print(f"    → Keep {data['target_qty']} shares")
        print(f"    → Estimated Value: ${data['quantity'] * data['avg_cost']:,.2f}")

    print(f"\n  💰 Total Excess Value: ${total_excess_value:,.2f}")
    print("\n" + "="*70)
    print("⚠️  This will execute MARKET ORDERS immediately!")
    print("⚠️  Orders will fill at current market price!")
    print("="*70)

    # CRITICAL: Get user confirmation
    response = input("\n👉 Proceed with execution? (y/n): ").strip().lower()

    if response not in ['y', 'yes']:
        print("\n❌ CANCELLED - No orders placed")
        ib.disconnect()
        return

    # Step 6: Execute sell orders
    print("\n" + "="*70)
    print("🚀 EXECUTING SELL ORDERS...")
    print("="*70)

    orders_placed = []

    for symbol, data in excess_to_sell.items():
        qty_to_sell = data['quantity']

        print(f"\n📉 {symbol}: Placing SELL order for {qty_to_sell} shares...")

        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)

            # Create market order
            order = MarketOrder('SELL', qty_to_sell)

            # Place order
            trade = ib.placeOrder(contract, order)
            orders_placed.append((symbol, qty_to_sell, trade))

            print(f"   ✅ Order placed: SELL {qty_to_sell} {symbol} (Market)")

        except Exception as e:
            print(f"   ❌ FAILED to place order: {e}")

    # Step 7: Wait for fills
    if orders_placed:
        print("\n⏳ Waiting 30 seconds for orders to fill...")
        ib.sleep(30)

    # Step 8: Check results
    print("\n" + "="*70)
    print("📊 ORDER RESULTS:")
    print("="*70)

    total_filled = 0
    total_proceeds = 0

    for symbol, qty, trade in orders_placed:
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        avg_price = trade.orderStatus.avgFillPrice

        if status == 'Filled':
            proceeds = filled * avg_price
            total_proceeds += proceeds
            total_filled += 1
            print(f"\n✅ {symbol}:")
            print(f"   Status: FILLED")
            print(f"   Quantity: {filled} shares")
            print(f"   Avg Price: ${avg_price:.2f}")
            print(f"   Proceeds: ${proceeds:,.2f}")
        else:
            print(f"\n⚠️  {symbol}:")
            print(f"   Status: {status}")
            print(f"   Filled: {filled} / {qty}")

    print(f"\n💰 Total Proceeds: ${total_proceeds:,.2f}")
    print(f"📈 Orders Filled: {total_filled} / {len(orders_placed)}")

    # Step 9: Show final positions
    print("\n" + "="*70)
    print("📂 FINAL POSITIONS:")
    print("="*70)

    ib.sleep(2)  # Brief pause for positions to update

    final_positions = ib.positions()
    for p in final_positions:
        symbol = p.contract.symbol
        qty = int(p.position)
        avg_cost = p.avgCost
        market_value = qty * avg_cost

        # Highlight the symbols we just cleaned up
        marker = "✅" if symbol in TARGET_POSITIONS else "  "
        print(f"{marker} {symbol}: {qty} shares @ ${avg_cost:.2f} = ${market_value:,.2f}")

    # Step 10: Account summary
    print("\n" + "="*70)
    print("💼 ACCOUNT SUMMARY:")
    print("="*70)

    try:
        summary = ib.accountSummary()
        for item in summary:
            if item.tag == 'NetLiquidation':
                print(f"💰 Net Liquidation: ${float(item.value):,.2f}")
            elif item.tag == 'TotalCashValue':
                print(f"💵 Cash: ${float(item.value):,.2f}")
            elif item.tag == 'GrossPositionValue':
                print(f"📊 Position Value: ${float(item.value):,.2f}")
    except:
        pass

    # Cleanup
    ib.disconnect()

    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ⚠️  ZTE EMERGENCY POSITION CLEANUP")
    print("="*70)
    print("\nThis script will close EXCESS positions from the duplicate order incident.")
    print("\nTarget cleanup:")
    print("  • GOOGL: Reduce from 135 → 15 shares (sell 120)")
    print("  • MU:    Reduce from 180 → 20 shares (sell 160)")
    print("\n⚠️  Requirements:")
    print("  • Market must be OPEN (9:30 AM - 4:00 PM ET)")
    print("  • TWS or IB Gateway must be running")
    print("  • Paper Trading account (Port 7497)")
    print("\n" + "="*70)

    response = input("\n👉 Continue? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        close_excess_positions()
    else:
        print("\n❌ Cancelled - No action taken\n")
