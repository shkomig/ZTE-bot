import sys
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('tm', 'CORE_TRADING/trading_memory.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mem = mod.TradingMemory()

print("Searching for NVDA trades...")
results = mem.find_similar_trades("NVDA momentum MA_Crossover", 5)
print(f"Found {len(results)} similar trades")

for i, r in enumerate(results[:3]):
    outcome = r.get("outcome", "?")
    print(f"--- Trade {i+1} ({outcome}) ---")
    meta = r.get("metadata", {})
    print(f"  Symbol: {meta.get('symbol', '?')}")
    print(f"  Strategy: {meta.get('strategy', '?')}")
    print(f"  P/L: {meta.get('profit_pct', 0):.2f}%")
    print()

