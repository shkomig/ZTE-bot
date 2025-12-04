"""
Zero Trading Expert (ZTE) - Auto Trader Bot
===========================================
Autonomous trading bot that scans the market and executes trades via the ZTE API.

Features:
- Continuous market scanning
- Real-time data fetching (via yfinance)
- Automated analysis requests
- Paper trading execution and logging

Usage:
    python auto_trader.py
"""

import time
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import tools
from TOOLS.stock_data_fetcher import StockDataFetcher

# Configuration
API_URL = "http://localhost:5001/api"
SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "SPY", "QQQ", "AMZN", "GOOGL", "META"]
SCAN_INTERVAL = 60  # Seconds
PAPER_TRADING_FILE = "paper_trades.jsonl"

def log(message: str, color: str = "white"):
    """Print colored log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def save_trade(trade: Dict):
    """Save trade to paper trading file."""
    with open(PAPER_TRADING_FILE, "a") as f:
        f.write(json.dumps(trade) + "\n")

def main():
    print("\n" + "="*60)
    print("Zero Trading Expert (ZTE) - Auto Trader Bot")
    print("="*60)
    print(f"Watching Symbols: {', '.join(SYMBOLS)}")
    print(f"Scan Interval: {SCAN_INTERVAL} seconds")
    print(f"API URL: {API_URL}")
    print("="*60 + "\n")

    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    # Main Loop
    while True:
        try:
            log("Starting scan cycle...", "cyan")
            
            for symbol in SYMBOLS:
                try:
                    # 1. Fetch Data
                    # log(f"Fetching data for {symbol}...")
                    data = fetcher.get_stock_data(symbol, period="1mo")
                    
                    if not data:
                        log(f"Skipping {symbol} (No data)", "yellow")
                        continue
                        
                    # 2. Prepare Analysis Request
                    # We calculate a basic score here or let the API handle deep analysis
                    # For now, we pass the raw data we have
                    
                    payload = {
                        "symbol": symbol,
                        "price": data.current_price,
                        "atr": data.current_price * 0.02,  # Approx ATR if not calculated
                        "score": 50,  # Neutral start
                        "signals": [],
                        "context": f"RVOL: {data.rvol:.2f}, Change: {data.change_pct:.2f}%",
                        "technical": data.to_dict()
                    }
                    
                    # 3. Request Analysis
                    response = requests.post(f"{API_URL}/analyze", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        action = result.get("action", "HOLD")
                        confidence = result.get("confidence", 0.0)
                        
                        # Log result
                        if action in ["BUY", "SELL"]:
                            log(f"🔥 {symbol}: {action} ({confidence:.1%})", "green")
                            print(f"   Reason: {result.get('reasoning')[:100]}...")
                            
                            # Execute Paper Trade
                            if confidence >= 0.6:  # Threshold for execution
                                trade_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "symbol": symbol,
                                    "action": action,
                                    "price": data.current_price,
                                    "confidence": confidence,
                                    "reasoning": result.get("reasoning")
                                }
                                save_trade(trade_record)
                                log(f"✅ Trade Executed (Paper): {action} {symbol} @ {data.current_price}", "green")
                        else:
                            # log(f"{symbol}: {action} ({confidence:.1%})")
                            pass
                            
                    else:
                        log(f"API Error for {symbol}: {response.status_code}", "red")
                        
                except Exception as e:
                    log(f"Error processing {symbol}: {e}", "red")
                    
                # Small delay between symbols to be nice to APIs
                time.sleep(1)
            
            log("Scan cycle complete. Waiting...", "cyan")
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[Bot] Stopping...")
            break
        except Exception as e:
            log(f"Critical Error: {e}", "red")
            time.sleep(10)

if __name__ == "__main__":
    main()
