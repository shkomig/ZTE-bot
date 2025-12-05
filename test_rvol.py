from ib_insync import IB, Stock
import random
import pytz
from datetime import datetime

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=random.randint(400,499))

eastern = pytz.timezone('US/Eastern')
now = datetime.now(eastern)
market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
mins_since_open = (now - market_open).seconds / 60
time_factor = 390 / max(mins_since_open, 1)

print(f"ET Time: {now.strftime('%H:%M')}")
print(f"Minutes since open: {mins_since_open:.0f}")
print(f"Time factor: {time_factor:.2f}")
print("")

for symbol in ['AAPL', 'MSFT', 'NVDA', 'MA']:
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    # Get historical avg
    bars = ib.reqHistoricalData(
        contract, '', '20 D', '1 day', 'TRADES', True
    )
    avg_vol = sum(b.volume for b in bars[:-1]) / max(len(bars)-1, 1) if bars else 0
    
    # Get today volume
    ticker = ib.reqMktData(contract, '', False, False)
    ib.sleep(0.5)
    today_vol = ticker.volume or 0
    ib.cancelMktData(contract)
    
    # Calculate RVOL with time projection
    projected_vol = today_vol * time_factor
    rvol = projected_vol / avg_vol if avg_vol > 0 else 0
    
    print(f"{symbol}: Today={today_vol:,.0f} | Avg={avg_vol:,.0f} | Projected={projected_vol:,.0f} | RVOL={rvol:.2f}x")

ib.disconnect()
