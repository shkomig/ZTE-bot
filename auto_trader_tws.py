"""
Zero Trading Expert (ZTE) - TWS Auto Trader Bot V2.0
====================================================
Advanced Day Trading Bot with RVOL, Session Rules, and Daily P&L Tracking.

Features:
- Real-time market data from TWS
- RVOL (Relative Volume) filter - only trade high volume
- Session-based trading rules (avoid lunch dead zone)
- Daily P&L tracking with max loss protection
- Trailing stop support
- Automated analysis via ZTE API
- Bracket orders with SL/TP

Requirements:
- TWS or IB Gateway running
- API connections enabled in TWS settings

Usage:
    python auto_trader_tws.py
"""

import time
import sys
import json
import random
import requests
import pytz
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# IB API
from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, util

# Phase 1 Technical Indicators
from CORE_TRADING.market_analyzer import MarketAnalyzer
from CORE_TRADING.pattern_detector import PatternDetector

# V3.4: Live Performance Tracking
from CORE_TRADING.live_performance import LivePerformanceTracker

# V3.7.2: Trading Memory for Learning from Trades
from CORE_TRADING.trading_memory import TradingMemory

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "http://localhost:5001/api"  # Fixed: Match api_server_trading.py port
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497  # 7497 = Paper, 7496 = Live
CLIENT_ID = random.randint(100, 999)  # Random client ID to avoid conflicts

# Premium Watchlist (42 stocks - MCP compliant)
SYMBOLS = [
    # Tech Giants (8)
    'NVDA', 'AMD', 'GOOGL', 'AMZN', 'META', 'MSFT', 'AAPL', 'TSLA',
    # Semiconductors (8)
    'AVGO', 'QCOM', 'MU', 'INTC', 'ARM', 'MRVL', 'AMAT', 'LRCX',
    # Software & Cloud (8)
    'CRM', 'PLTR', 'SNOW', 'NET', 'DDOG', 'ZS', 'CRWD', 'PANW',
    # Finance (6)
    'JPM', 'GS', 'V', 'MA', 'BAC', 'MS',
    # Consumer & Retail (5)
    'NKE', 'SBUX', 'HD', 'WMT', 'COST',
    # Healthcare & Biotech (4)
    'UNH', 'LLY', 'MRNA', 'ISRG',
    # ETFs (3)
    'SPY', 'QQQ', 'IWM'
]

# Sector Map for diversification
SECTOR_MAP = {
    'TECH': ['NVDA', 'AMD', 'GOOGL', 'AMZN', 'META', 'MSFT', 'AAPL', 'TSLA'],
    'SEMI': ['AVGO', 'QCOM', 'MU', 'INTC', 'ARM', 'MRVL', 'AMAT', 'LRCX'],
    'SOFTWARE': ['CRM', 'PLTR', 'SNOW', 'NET', 'DDOG', 'ZS', 'CRWD', 'PANW'],
    'FINANCE': ['JPM', 'GS', 'V', 'MA', 'BAC', 'MS'],
    'CONSUMER': ['NKE', 'SBUX', 'HD', 'WMT', 'COST'],
    'HEALTH': ['UNH', 'LLY', 'MRNA', 'ISRG'],
    'ETF': ['SPY', 'QQQ', 'IWM']
}

MAX_PER_SECTOR = 2  # Maximum positions per sector

SCAN_INTERVAL = 30  # Seconds between full scans
PAPER_TRADING_FILE = "paper_trades.jsonl"
DAILY_LOG_FILE = "daily_pnl.jsonl"

# ============================================================================
# RISK MANAGEMENT (MCP Compliant - DAY TRADING OPTIMIZED)
# ============================================================================
# V3.7.3: Optimized for Day Trading - realistic TP/SL based on actual daily moves

MAX_POSITION_VALUE = 5000    # Max $ per position
MAX_SL_PCT = 0.0125          # 1.25% Stop Loss (FIXED: was 0.75% - too tight!)
MAX_TP_PCT = 0.025           # 2.5% Take Profit (R:R = 1:2) - Realistic for volatility
MIN_CONFIDENCE = 0.50        # Minimum confidence to execute

# Daily Risk Limits
MAX_DAILY_LOSS_PCT = 0.03    # 3% max daily loss - STOP trading
MAX_TRADES_PER_DAY = 20      # Maximum trades per day
MAX_OPEN_POSITIONS = 10      # Maximum concurrent positions (Tier 1 + Tier 2)

# ============================================================================
# TIERED POSITION SYSTEM (V3.5)
# ============================================================================
# Tier 1 (positions 1-5): Standard requirements
# Tier 2 (positions 6-10): Stricter requirements for quality trades only

TIER1_POSITIONS = 5              # First 5 positions - standard requirements
TIER2_MIN_CONFIDENCE = 0.65      # Tier 2: Higher confidence required (65%)
TIER2_MIN_RVOL = 2.0             # Tier 2: Higher RVOL required (2.0x)
TIER2_MIN_PHASE1_SIGNALS = 2     # Tier 2: Minimum 2 Phase1 signals
TIER2_REQUIRE_STRONG_BUY = True  # Tier 2: Only STRONG_BUY actions

# ============================================================================
# DAY TRADING FILTERS
# ============================================================================

# Paper Trading Mode - TWS Paper doesn't return accurate volume data
# Set to True to bypass RVOL checks (for testing)
PAPER_TRADING_MODE = True    # ⚠️ Set to False for LIVE trading!
PAPER_DEFAULT_RVOL = 2.0     # Default RVOL to use in Paper mode

MIN_RVOL = 1.5               # Minimum Relative Volume (1.5x average)
MIN_PRICE = 5.0              # Minimum stock price
MAX_PRICE = 500.0            # Maximum stock price
MIN_SPREAD_PCT = 0.001       # Minimum spread (0.1%)
MAX_SPREAD_PCT = 0.005       # Maximum spread (0.5%) - avoid illiquid stocks

# Trailing Stop Settings (V3.7.3 - Day Trading Optimized)
TRAIL_ACTIVATION_PCT = 0.005  # Start trailing after 0.5% profit (earlier lock-in!)
TRAIL_DISTANCE_PCT = 0.003    # 0.3% trailing distance (tighter for day trading)

# ============================================================================
# TRADING SESSIONS (Eastern Time)
# ============================================================================

class TradingSession(Enum):
    PRE_MARKET = "pre_market"       # 04:00 - 09:30
    OPENING_BELL = "opening_bell"   # 09:30 - 10:00 (High volatility)
    MID_MORNING = "mid_morning"     # 10:00 - 11:30 (BEST setups)
    LUNCH_DEAD = "lunch_dead"       # 11:30 - 14:00 (AVOID!)
    AFTERNOON = "afternoon"         # 14:00 - 15:00 (Choppy)
    POWER_HOUR = "power_hour"       # 15:00 - 16:00 (Good momentum)
    AFTER_HOURS = "after_hours"     # 16:00 - 20:00


TRADEABLE_SESSIONS = [
    TradingSession.PRE_MARKET,      # 04:00 - 09:30 (Hot news plays!)
    TradingSession.OPENING_BELL,
    TradingSession.MID_MORNING,
    TradingSession.AFTERNOON,        # 14:00 - 15:00 (Can trade, less volatile)
    TradingSession.POWER_HOUR,
    TradingSession.AFTER_HOURS       # 16:00 - 20:00 (Earnings plays!)
]


# ============================================================================
# DATA CLASSES
# ============================================================================

import math
import numpy as np

def clean_for_json(obj):
    """Clean object for JSON serialization (handle NaN, Inf, numpy types)."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return [clean_for_json(v) for v in obj.tolist()]
    return obj

@dataclass
class DailyPnL:
    """Track daily profit/loss and enforce limits."""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    starting_balance: float = 100000.0  # Paper account default
    current_pnl: float = 0.0
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    stopped_trading: bool = False
    stop_reason: str = ""
    
    def update_trade(self, pnl: float, is_win: bool):
        """Update after a trade closes."""
        self.current_pnl += pnl
        self.realized_pnl += pnl
        self.trades_today += 1
        
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        
        # Check max drawdown
        if self.current_pnl < 0:
            drawdown_pct = abs(self.current_pnl) / self.starting_balance
            self.max_drawdown = max(self.max_drawdown, drawdown_pct)
            
            # STOP if max loss reached
            if drawdown_pct >= MAX_DAILY_LOSS_PCT:
                self.stopped_trading = True
                self.stop_reason = f"Daily max loss reached ({drawdown_pct:.1%})"
        
        # Check trade limit
        if self.trades_today >= MAX_TRADES_PER_DAY:
            self.stopped_trading = True
            self.stop_reason = f"Max trades reached ({MAX_TRADES_PER_DAY})"
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if self.stopped_trading:
            return False, self.stop_reason
        
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max trades reached ({MAX_TRADES_PER_DAY})"
        
        if self.max_drawdown >= MAX_DAILY_LOSS_PCT:
            return False, f"Daily max loss reached ({self.max_drawdown:.1%})"
        
        return True, "OK"
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "pnl": round(self.current_pnl, 2),
            "trades": self.trades_today,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": f"{self.win_rate:.1%}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "stopped": self.stopped_trading,
            "reason": self.stop_reason
        }


@dataclass
class Position:
    """Track an open position."""
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0  # For trailing
    lowest_price: float = 999999.0  # For trailing
    
    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """Update trailing stop based on price movement."""
        if self.action == "BUY":
            self.highest_price = max(self.highest_price, current_price)
            profit_pct = (current_price - self.entry_price) / self.entry_price
            
            if profit_pct >= TRAIL_ACTIVATION_PCT:
                new_stop = current_price * (1 - TRAIL_DISTANCE_PCT)
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = round(new_stop, 2)
                    return self.trailing_stop
        else:  # SELL
            self.lowest_price = min(self.lowest_price, current_price)
            profit_pct = (self.entry_price - current_price) / self.entry_price
            
            if profit_pct >= TRAIL_ACTIVATION_PCT:
                new_stop = current_price * (1 + TRAIL_DISTANCE_PCT)
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = round(new_stop, 2)
                    return self.trailing_stop
        
        return None
    
    @property
    def current_stop(self) -> float:
        """Get current active stop (trailing or original)."""
        if self.trailing_stop:
            if self.action == "BUY":
                return max(self.trailing_stop, self.stop_loss)
            else:
                return min(self.trailing_stop, self.stop_loss)
        return self.stop_loss


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_session() -> TradingSession:
    """Get the current trading session based on US Eastern Time."""
    import pytz
    
    # Convert to US Eastern Time
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    hour = now_et.hour
    minute = now_et.minute
    time_decimal = hour + minute / 60
    
    if time_decimal < 9.5:
        return TradingSession.PRE_MARKET
    elif time_decimal < 10:
        return TradingSession.OPENING_BELL
    elif time_decimal < 11.5:
        return TradingSession.MID_MORNING
    elif time_decimal < 14:
        return TradingSession.LUNCH_DEAD
    elif time_decimal < 15:
        return TradingSession.AFTERNOON
    elif time_decimal < 16:
        return TradingSession.POWER_HOUR
    else:
        return TradingSession.AFTER_HOURS


def is_good_trading_time() -> Tuple[bool, str]:
    """Check if current time is good for trading."""
    session = get_current_session()
    
    if session in TRADEABLE_SESSIONS:
        return True, f"✅ {session.value} - Good to trade"
    elif session == TradingSession.LUNCH_DEAD:
        return False, "⚠️ LUNCH DEAD ZONE - Avoid trading!"
    elif session == TradingSession.PRE_MARKET:
        return False, "⏰ Pre-market - Wait for open"
    elif session == TradingSession.AFTER_HOURS:
        return False, "🌙 After hours - Market closed"
    else:
        return False, f"⚠️ {session.value} - Not optimal"


def calculate_rvol(current_volume: int, avg_volume: int) -> float:
    """Calculate Relative Volume ratio."""
    if avg_volume == 0:
        return 1.0
    return round(current_volume / avg_volume, 2)


def check_spread(bid: float, ask: float, price: float) -> Tuple[bool, float]:
    """Check if spread is acceptable."""
    if bid <= 0 or ask <= 0:
        return True, 0.0  # Can't calculate, allow
    
    spread = ask - bid
    spread_pct = spread / price
    
    if spread_pct > MAX_SPREAD_PCT:
        return False, spread_pct
    elif spread_pct < MIN_SPREAD_PCT:
        return False, spread_pct  # Too tight, might be stale data
    
    return True, spread_pct


def log(message: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def save_trade(trade: Dict):
    """Save trade to file."""
    with open(PAPER_TRADING_FILE, "a") as f:
        f.write(json.dumps(trade) + "\n")


def save_daily_log(pnl: DailyPnL):
    """Save daily P&L to file."""
    with open(DAILY_LOG_FILE, "a") as f:
        f.write(json.dumps(pnl.to_dict()) + "\n")


# ============================================================================
# TWS TRADER CLASS
# ============================================================================

class TWSTrader:
    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = DailyPnL()
        self.avg_volumes: Dict[str, int] = {}  # Cache for average volumes
        
        # Phase 1: Advanced Technical Analysis
        self.market_analyzer = MarketAnalyzer()
        self.pattern_detector = PatternDetector()
        self.historical_cache: Dict[str, Dict] = {}  # Cache for historical data
        
        # V3.4: Live Performance Tracking (REAL Win Rate!)
        self.live_tracker = LivePerformanceTracker()
        
        # V3.7.2: Trading Memory for Learning from Trades
        self.trading_memory = TradingMemory()

        # API Health Check (CRITICAL FIX)
        self.api_last_check = datetime.now()
        self.api_healthy = False

    def connect(self) -> bool:
        """Connect to TWS."""
        try:
            self.ib.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
            self.connected = True
            print(f"[TWS] Connected to TWS on port {TWS_PORT}")
            
            # Get account info
            account = self.ib.managedAccounts()
            print(f"[TWS] Account: {account}")
            
            # Get account value for P&L tracking
            account_values = self.ib.accountSummary()
            for av in account_values:
                if av.tag == "NetLiquidation":
                    self.daily_pnl.starting_balance = float(av.value)
                    print(f"[TWS] Account Balance: ${float(av.value):,.2f}")
                    break
            
            # Load existing positions from TWS
            self._load_existing_positions()
            
            return True
        except Exception as e:
            print(f"[TWS] Connection failed: {e}")
            return False
    
    def _load_existing_positions(self):
        """Load existing positions from TWS at startup."""
        try:
            tws_positions = self.ib.positions()
            if not tws_positions:
                print("[TWS] No existing positions found")
                return
            
            count = 0
            for pos in tws_positions:
                symbol = pos.contract.symbol
                qty = int(pos.position)
                avg_cost = pos.avgCost
                
                if qty == 0:
                    continue
                
                # Determine action based on position direction
                action = "BUY" if qty > 0 else "SELL"
                qty = abs(qty)
                
                # Create position tracking (use avg_cost for estimates)
                # DAY TRADING: 1% SL, 2% TP (1:2 R/R ratio)
                self.positions[symbol] = Position(
                    symbol=symbol,
                    action=action,
                    quantity=qty,
                    entry_price=avg_cost,
                    stop_loss=avg_cost * ((1 - MAX_SL_PCT) if action == "BUY" else (1 + MAX_SL_PCT)),  # 1% SL
                    take_profit=avg_cost * ((1 + MAX_TP_PCT) if action == "BUY" else (1 - MAX_TP_PCT)),  # 2% TP
                    entry_time=datetime.now(),
                    highest_price=avg_cost,
                    lowest_price=avg_cost
                )
                count += 1
                print(f"[TWS] Loaded: {action} {qty} {symbol} @ ${avg_cost:.2f}")
            
            print(f"[TWS] Loaded {count} existing positions")
            
            # CRITICAL: Add SL/TP to existing positions at startup
            if count > 0:
                print(f"[TWS] 🛡️ Checking/Adding SL/TP for {count} positions...")
                self._add_sl_tp_to_existing_positions()
            
        except Exception as e:
            print(f"[TWS] Error loading positions: {e}")
    
    def _add_sl_tp_to_existing_positions(self):
        """Add SL/TP orders to all existing positions at startup."""
        try:
            # Get existing orders using reqAllOpenOrders (more reliable)
            all_orders = self.ib.reqAllOpenOrders()
            self.ib.sleep(0.5)
            
            existing_orders = {}
            for o in all_orders:
                symbol = o.contract.symbol
                if symbol not in existing_orders:
                    existing_orders[symbol] = {"has_sl": False, "has_tp": False}
                if o.order.orderType == "STP":
                    existing_orders[symbol]["has_sl"] = True
                elif o.order.orderType == "LMT":
                    existing_orders[symbol]["has_tp"] = True
            
            added_count = 0
            for symbol, position in self.positions.items():
                has_sl = existing_orders.get(symbol, {}).get("has_sl", False)
                has_tp = existing_orders.get(symbol, {}).get("has_tp", False)
                
                if has_sl and has_tp:
                    print(f"[TWS] ✅ {symbol}: SL/TP already exists")
                    continue
                
                # Calculate SL/TP based on 1% SL, 2% TP
                entry = position.entry_price
                if position.action == "BUY":
                    sl_price = round(entry * (1 - MAX_SL_PCT), 2)
                    tp_price = round(entry * (1 + MAX_TP_PCT), 2)
                else:  # SELL/SHORT
                    sl_price = round(entry * (1 + MAX_SL_PCT), 2)
                    tp_price = round(entry * (1 - MAX_TP_PCT), 2)
                
                # Only add what's missing
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                exit_action = "SELL" if position.action == "BUY" else "BUY"
                
                if not has_sl:
                    print(f"[TWS] ⚠️ {symbol}: Adding SL=${sl_price:.2f}")
                    sl_order = StopOrder(exit_action, position.quantity, sl_price)
                    sl_order.outsideRth = True
                    self.ib.placeOrder(contract, sl_order)
                
                if not has_tp:
                    print(f"[TWS] ⚠️ {symbol}: Adding TP=${tp_price:.2f}")
                    tp_order = LimitOrder(exit_action, position.quantity, tp_price)
                    tp_order.outsideRth = True
                    self.ib.placeOrder(contract, tp_order)
                
                position.stop_loss = sl_price
                position.take_profit = tp_price
                added_count += 1
                print(f"[TWS] ✅ {symbol}: SL/TP added successfully")
                
                self.ib.sleep(0.3)  # Small delay between orders
            
            print(f"[TWS] 🛡️ Added SL/TP to {added_count} positions")
            
        except Exception as e:
            print(f"[TWS] Error adding SL/TP: {e}")
            import traceback
            print(f"[TWS] {traceback.format_exc()}")
    
    def disconnect(self):
        """Disconnect from TWS."""
        if self.connected:
            save_daily_log(self.daily_pnl)
            self.ib.disconnect()
            self.connected = False
            print("[TWS] Disconnected")

    def check_api_health(self) -> bool:
        """
        Check if API server is responsive (cached for 30s).
        CRITICAL: Prevents placing orders when API is down.
        """
        # Cache API health checks for 30 seconds
        if (datetime.now() - self.api_last_check).seconds < 30:
            return self.api_healthy

        try:
            response = requests.get(f"{API_URL}/health", timeout=3)
            self.api_healthy = response.status_code == 200
            self.api_last_check = datetime.now()
            if not self.api_healthy:
                log(f"⚠️ API Health Check: HTTP {response.status_code}")
            return self.api_healthy
        except Exception as e:
            self.api_healthy = False
            self.api_last_check = datetime.now()
            log(f"⚠️ API Health Check Failed: {e}")
            return False

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time market data for a symbol."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1.5)  # Wait for data
            
            if ticker.last is None and ticker.close is None:
                return None
            
            price = float(ticker.last) if ticker.last else float(ticker.close)
            
            def safe_float(val, default):
                try:
                    if val is None or (hasattr(val, '__class__') and 'nan' in str(val).lower()):
                        return float(default)
                    return float(val)
                except:
                    return float(default)
            
            def safe_int(val, default):
                try:
                    if val is None:
                        return int(default)
                    return int(val)
                except:
                    return int(default)
            
            volume = safe_int(ticker.volume, 0)
            
            # Calculate RVOL using cached average or estimate
            avg_vol = self.avg_volumes.get(symbol, volume)  # Default to current
            if avg_vol == 0:
                avg_vol = volume
            rvol = calculate_rvol(volume, avg_vol)
            
            # Update average volume cache (simple moving avg)
            if symbol in self.avg_volumes:
                self.avg_volumes[symbol] = int((self.avg_volumes[symbol] * 0.9) + (volume * 0.1))
            else:
                self.avg_volumes[symbol] = volume
            
            data = {
                "symbol": symbol,
                "current_price": price,
                "bid": safe_float(ticker.bid, price * 0.999),
                "ask": safe_float(ticker.ask, price * 1.001),
                "volume": volume,
                "avg_volume": avg_vol,
                "rvol": rvol,
                "high": safe_float(ticker.high, price),
                "low": safe_float(ticker.low, price),
                "open": safe_float(ticker.open, price),
                "close": safe_float(ticker.close, price),
            }
            
            # Cancel market data subscription
            self.ib.cancelMktData(contract)
            
            return data
            
        except Exception as e:
            print(f"[TWS] Error getting data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, bars: int = 100) -> Optional[Dict]:
        """Get historical OHLCV data for Phase 1 indicators."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Request historical bars (5-minute for day trading)
            hist_bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='2 D',  # 2 days of data
                barSizeSetting='5 mins',
                whatToShow='TRADES',
                useRTH=False,  # Include extended hours
                formatDate=1
            )
            
            if not hist_bars or len(hist_bars) < 20:
                return None
            
            # Extract OHLCV arrays (last 'bars' entries)
            recent_bars = hist_bars[-bars:] if len(hist_bars) >= bars else hist_bars
            
            prices = [float(bar.close) for bar in recent_bars]
            highs = [float(bar.high) for bar in recent_bars]
            lows = [float(bar.low) for bar in recent_bars]
            volumes = [int(bar.volume) for bar in recent_bars]
            
            return {
                "prices": prices,
                "highs": highs,
                "lows": lows,
                "volumes": volumes,
                "bars_count": len(prices)
            }
            
        except Exception as e:
            print(f"[TWS] Historical data error for {symbol}: {e}")
            return None
    
    def analyze_with_phase1_indicators(self, symbol: str, data: Dict) -> Dict:
        """Run Phase 1 advanced technical analysis."""
        try:
            # Get historical data (use cache if fresh)
            cache_key = symbol
            cache_age = 300  # 5 minutes cache
            
            if cache_key in self.historical_cache:
                cache_entry = self.historical_cache[cache_key]
                if (datetime.now() - cache_entry["timestamp"]).seconds < cache_age:
                    hist_data = cache_entry["data"]
                else:
                    hist_data = self.get_historical_data(symbol)
                    if hist_data:
                        self.historical_cache[cache_key] = {
                            "data": hist_data,
                            "timestamp": datetime.now()
                        }
            else:
                hist_data = self.get_historical_data(symbol)
                if hist_data:
                    self.historical_cache[cache_key] = {
                        "data": hist_data,
                        "timestamp": datetime.now()
                    }
            
            if not hist_data or len(hist_data["prices"]) < 30:
                return {"phase1_signals": [], "indicators": {}}
            
            # Run MarketAnalyzer for Phase 1 indicators
            indicators = self.market_analyzer.analyze(
                prices=hist_data["prices"],
                highs=hist_data["highs"],
                lows=hist_data["lows"],
                volumes=hist_data["volumes"]
            )
            
            # Run PatternDetector
            patterns = self.pattern_detector.detect_all(
                prices=hist_data["prices"],
                highs=hist_data["highs"],
                lows=hist_data["lows"],
                volumes=hist_data["volumes"]
            )
            
            # Generate Phase 1 signals
            phase1_signals = []
            signal_strength = 0
            
            # RSI Divergence (High priority - 85% WR)
            if indicators.rsi_divergence == "bullish":
                phase1_signals.append("🟢 RSI_BULLISH_DIVERGENCE")
                signal_strength += 3
            elif indicators.rsi_divergence == "bearish":
                phase1_signals.append("🔴 RSI_BEARISH_DIVERGENCE")
                signal_strength -= 3
            
            # TSI (Trend Strength) - tsi_signal is a string: "oversold", "neutral", "overbought"
            if indicators.tsi is not None:
                if indicators.tsi > 25 and indicators.tsi_signal == "overbought":
                    phase1_signals.append("🔴 TSI_OVERBOUGHT")
                    signal_strength -= 1
                elif indicators.tsi < -25 and indicators.tsi_signal == "oversold":
                    phase1_signals.append("🟢 TSI_OVERSOLD")
                    signal_strength += 1
                elif indicators.tsi > 0:
                    phase1_signals.append("🟢 TSI_BULLISH")
                    signal_strength += 2
                elif indicators.tsi < 0:
                    phase1_signals.append("🔴 TSI_BEARISH")
                    signal_strength -= 2
            
            # Bollinger %B (Oversold/Overbought)
            if indicators.bb_percent is not None:
                if indicators.bb_percent < 0.05:
                    phase1_signals.append("🟢 BB_OVERSOLD")
                    signal_strength += 2
                elif indicators.bb_percent > 0.95:
                    phase1_signals.append("🔴 BB_OVERBOUGHT")
                    signal_strength -= 2
            
            # Volume Profile (VPOC, VAL, VAH)
            current_price = data["current_price"]
            if indicators.vpoc:
                dist_to_vpoc = abs(current_price - indicators.vpoc) / current_price
                if dist_to_vpoc < 0.005:  # Within 0.5% of VPOC
                    phase1_signals.append("⚡ AT_VPOC")
                    signal_strength += 1
                    
            if indicators.val and current_price <= indicators.val * 1.01:
                phase1_signals.append("🟢 AT_VAL_SUPPORT")
                signal_strength += 2
                
            if indicators.vah and current_price >= indicators.vah * 0.99:
                phase1_signals.append("🔴 AT_VAH_RESISTANCE")
                signal_strength -= 2
            
            # MACD Divergence
            if indicators.macd_divergence == "bullish":
                phase1_signals.append("🟢 MACD_BULLISH_DIVERGENCE")
                signal_strength += 2
            elif indicators.macd_divergence == "bearish":
                phase1_signals.append("🔴 MACD_BEARISH_DIVERGENCE")
                signal_strength -= 2
            
            # Add pattern signals
            for pattern in patterns[:5]:  # Top 5 patterns
                if pattern.confidence >= 0.7:
                    emoji = "🟢" if "BULLISH" in pattern.pattern_type.name else "🔴" if "BEARISH" in pattern.pattern_type.name else "⚡"
                    phase1_signals.append(f"{emoji} {pattern.pattern_type.name}")
            
            return {
                "phase1_signals": phase1_signals,
                "signal_strength": signal_strength,
                "indicators": {
                    "rsi": round(indicators.rsi, 1) if indicators.rsi else None,
                    "rsi_divergence": indicators.rsi_divergence,
                    "tsi": round(indicators.tsi, 1) if indicators.tsi else None,
                    "bb_percent": round(indicators.bb_percent, 2) if indicators.bb_percent is not None else None,
                    "vpoc": round(indicators.vpoc, 2) if indicators.vpoc else None,
                    "vah": round(indicators.vah, 2) if indicators.vah else None,
                    "val": round(indicators.val, 2) if indicators.val else None,
                    "macd_divergence": indicators.macd_divergence
                },
                "patterns_count": len(patterns),
                "recommendation": "BUY" if signal_strength >= 3 else "SELL" if signal_strength <= -3 else "NEUTRAL"
            }
            
        except Exception as e:
            import traceback
            print(f"[Phase1] Analysis error for {symbol}: {e}")
            traceback.print_exc()
            return {"phase1_signals": [], "indicators": {}, "signal_strength": 0}
    
    def check_trade_filters(self, data: Dict, phase1_signals: List[str] = None, zte_action: str = None) -> Tuple[bool, str]:
        """Check all filters before trading, including Tier requirements."""
        symbol = data["symbol"]
        price = data["current_price"]
        rvol = data.get("rvol", 1.0)
        bid = data.get("bid", price)
        ask = data.get("ask", price)

        # ============ CRITICAL FIX: CHECK ACTUAL TWS POSITIONS ============
        # MUST verify actual TWS positions, not just internal tracking
        # This prevents duplicate orders if bot restarts or API fails
        try:
            tws_positions = self.ib.positions()
            for pos in tws_positions:
                if pos.contract.symbol == symbol and abs(pos.position) > 0:
                    return False, f"Already in position (TWS: {int(pos.position)} shares)"
        except Exception as e:
            log(f"⚠️ Failed to check TWS positions: {e}")
            # Fail-safe: If we can't check positions, don't trade
            return False, "Cannot verify positions - safety block"
        # ==================================================================

        # Price filter
        if price < MIN_PRICE:
            return False, f"Price too low (${price:.2f} < ${MIN_PRICE})"
        if price > MAX_PRICE:
            return False, f"Price too high (${price:.2f} > ${MAX_PRICE})"
        
        # Basic RVOL filter (Tier 1 minimum)
        if rvol < MIN_RVOL:
            return False, f"Low volume (RVOL {rvol:.1f}x < {MIN_RVOL}x)"
        
        # Spread filter
        spread_ok, spread_pct = check_spread(bid, ask, price)
        if not spread_ok:
            return False, f"Bad spread ({spread_pct:.2%})"
        
        # Position limit - check if we need Tier 2 requirements
        current_positions = len(self.positions)
        if current_positions >= MAX_OPEN_POSITIONS:
            return False, f"Max positions reached ({MAX_OPEN_POSITIONS})"
        
        # Tier 2 requirements (positions 6-10)
        if current_positions >= TIER1_POSITIONS:
            # Stricter RVOL for Tier 2
            if rvol < TIER2_MIN_RVOL:
                return False, f"Tier2: Low RVOL ({rvol:.1f}x < {TIER2_MIN_RVOL}x)"
            
            # Require minimum Phase1 signals for Tier 2
            if phase1_signals is None or len(phase1_signals) < TIER2_MIN_PHASE1_SIGNALS:
                signal_count = len(phase1_signals) if phase1_signals else 0
                return False, f"Tier2: Need {TIER2_MIN_PHASE1_SIGNALS}+ Phase1 signals (got {signal_count})"
            
            # Require STRONG_BUY for Tier 2
            if TIER2_REQUIRE_STRONG_BUY and zte_action not in ["STRONG_BUY", "STRONG_SELL"]:
                return False, f"Tier2: Requires STRONG signal (got {zte_action})"
        
        # Already in position
        if symbol in self.positions:
            return False, "Already in position"
        
        # Sector diversification check
        if not self.check_sector_exposure(symbol):
            return False, f"Max positions in sector ({MAX_PER_SECTOR})"
        
        # Determine tier for logging
        tier = "Tier1" if current_positions < TIER1_POSITIONS else "Tier2"
        return True, f"OK ({tier})"
    
    def get_current_tier(self) -> str:
        """Get current tier based on position count."""
        if len(self.positions) < TIER1_POSITIONS:
            return "Tier1"
        return "Tier2"
    
    def check_sector_exposure(self, symbol: str) -> bool:
        """Check if we're overexposed to a sector."""
        # Find symbol's sector
        symbol_sector = None
        for sector, symbols in SECTOR_MAP.items():
            if symbol in symbols:
                symbol_sector = sector
                break
        
        if not symbol_sector:
            return True  # Unknown sector, allow
        
        # Count positions in same sector
        sector_positions = 0
        for pos_symbol in self.positions.keys():
            for sector, symbols in SECTOR_MAP.items():
                if pos_symbol in symbols and sector == symbol_sector:
                    sector_positions += 1
        
        # Max positions per sector
        return sector_positions < MAX_PER_SECTOR
    
    def get_symbol_sector(self, symbol: str) -> str:
        """Get the sector for a symbol."""
        for sector, symbols in SECTOR_MAP.items():
            if symbol in symbols:
                return sector
        return "UNKNOWN"
    
    def calculate_real_rvol(self, symbol: str) -> float:
        """Calculate RVOL using 20-day average volume from TWS."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Get 20 days of daily bars
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='20 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            
            if not bars or len(bars) < 5:
                return 1.0
            
            # Calculate average volume (exclude today)
            avg_volume = sum(bar.volume for bar in bars[:-1]) / len(bars[:-1])
            
            # Get today's volume
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(0.5)
            today_volume = ticker.volume or 0
            self.ib.cancelMktData(contract)
            
            if avg_volume == 0:
                return 1.0
            
            # Adjust for time of day (project full day volume)
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            
            if now > market_open:
                minutes_since_open = (now - market_open).seconds / 60
                expected_minutes = 390  # Full trading day
                time_factor = expected_minutes / max(minutes_since_open, 1)
                projected_volume = today_volume * time_factor
                rvol = projected_volume / avg_volume
            else:
                rvol = 1.0
            
            # Cache the average volume
            self.avg_volumes[symbol] = int(avg_volume)
            
            return round(min(rvol, 10.0), 2)  # Cap at 10x
            
        except Exception as e:
            return 1.0
    
    def scan_premarket_gaps(self) -> List[Dict]:
        """Scan for gap up/down stocks before market open."""
        gaps = []
        log("🌅 Scanning pre-market gaps...")
        
        # Limit scan time (max 5 stocks to avoid timeout)
        scan_symbols = SYMBOLS[:5]  # Only scan first 5 for speed
        
        for symbol in scan_symbols:
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                try:
                    self.ib.qualifyContracts(contract)
                except Exception:
                    continue  # Skip if qualification fails
                
                # Get yesterday's close
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='2 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    timeout=10  # Add timeout
                )
                
                if not bars or len(bars) < 2:
                    continue
                
                prev_close = bars[-1].close
                
                # Get pre-market price
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(0.3)
                
                premarket_price = ticker.last or ticker.close
                self.ib.cancelMktData(contract)
                
                if not premarket_price:
                    continue
                
                gap_pct = ((premarket_price - prev_close) / prev_close) * 100
                
                if abs(gap_pct) >= 2.0:  # 2% gap threshold
                    gaps.append({
                        'symbol': symbol,
                        'prev_close': round(prev_close, 2),
                        'premarket': round(premarket_price, 2),
                        'gap_pct': round(gap_pct, 2),
                        'direction': 'UP' if gap_pct > 0 else 'DOWN',
                        'sector': self.get_symbol_sector(symbol)
                    })
                    
            except Exception as e:
                continue
        
        # Sort by gap size
        gaps.sort(key=lambda x: abs(x['gap_pct']), reverse=True)
        
        if gaps:
            log(f"📊 Found {len(gaps)} gaps:")
            for g in gaps[:5]:
                emoji = "🟢" if g['direction'] == 'UP' else "🔴"
                log(f"   {emoji} {g['symbol']}: {g['gap_pct']:+.1f}% (${g['prev_close']} → ${g['premarket']})")
        
        return gaps[:10]  # Top 10 gaps
    
    def place_bracket_order(self, symbol: str, action: str, quantity: int, 
                            entry_price: float, stop_loss: float, take_profit: float,
                            zte_confidence: float = 0, zte_action: str = "",
                            zte_signals: List[str] = None, sector: str = "",
                            rvol: float = 0, phase1_signals: List[str] = None) -> bool:
        """Place a bracket order with SL and TP."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Create bracket order
            bracket = self.ib.bracketOrder(
                action=action,
                quantity=quantity,
                limitPrice=entry_price,
                takeProfitPrice=take_profit,
                stopLossPrice=stop_loss
            )
            
            # Enable after hours trading for all orders
            for order in bracket:
                order.outsideRth = True  # After Hours enabled!
                self.ib.placeOrder(contract, order)
            
            # Track position
            self.positions[symbol] = Position(
                symbol=symbol,
                action=action,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                highest_price=entry_price,
                lowest_price=entry_price
            )
            
            # V3.4: Record entry in Live Performance Tracker
            self.live_tracker.record_entry(
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                quantity=quantity,
                zte_confidence=zte_confidence,
                zte_action=zte_action,
                zte_signals=zte_signals or [],
                sector=sector or self.get_symbol_sector(symbol),
                rvol=rvol,
                phase1_signals=phase1_signals or []
            )
            
            print(f"[TWS] Bracket order placed: {action} {quantity} {symbol}")
            print(f"[TWS] Entry: ${entry_price:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
            
            return True
            
        except Exception as e:
            print(f"[TWS] Bracket order failed: {e}")
            # Fallback to market order WITH SL/TP
            market_success = self.place_market_order(symbol, action, quantity)
            if market_success:
                # Add SL/TP orders separately after market order
                print(f"[TWS] Adding SL/TP after market order...")
                self.place_sl_tp_orders(symbol, action, quantity, stop_loss, take_profit)
                # Track position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_time=datetime.now(),
                    highest_price=entry_price,
                    lowest_price=entry_price
                )
            return market_success
    
    def place_sl_tp_orders(self, symbol: str, action: str, quantity: int, 
                           stop_loss: float, take_profit: float) -> bool:
        """Place separate SL and TP orders for existing positions."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Determine opposite action for exit orders
            exit_action = "SELL" if action == "BUY" else "BUY"
            
            # Create Stop Loss order
            sl_order = StopOrder(exit_action, quantity, stop_loss)
            sl_order.outsideRth = True
            
            # Create Take Profit order  
            tp_order = LimitOrder(exit_action, quantity, take_profit)
            tp_order.outsideRth = True
            
            # Place orders
            self.ib.placeOrder(contract, sl_order)
            self.ib.placeOrder(contract, tp_order)
            
            print(f"[TWS] SL/TP orders placed: {exit_action} {quantity} {symbol}")
            print(f"[TWS] SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
            
            return True
            
        except Exception as e:
            print(f"[TWS] SL/TP orders failed: {e}")
            return False
    
    def update_trailing_stops(self):
        """Update trailing stops for all open positions."""
        for symbol, position in list(self.positions.items()):
            try:
                data = self.get_market_data(symbol)
                if data:
                    new_stop = position.update_trailing_stop(data["current_price"])
                    if new_stop:
                        log(f"📈 {symbol} Trailing stop updated to ${new_stop:.2f}")
            except Exception as e:
                pass  # Silently handle errors
    
    def get_tws_positions(self) -> Dict:
        """Get current positions from TWS."""
        positions = {}
        for pos in self.ib.positions():
            positions[pos.contract.symbol] = {
                "quantity": pos.position,
                "avg_cost": pos.avgCost
            }
        return positions
    
    def check_and_add_missing_sl_tp(self):
        """Check existing positions and add SL/TP if missing."""
        try:
            # Get ALL open orders from TWS using reqAllOpenOrders (more reliable)
            all_orders = self.ib.reqAllOpenOrders()
            self.ib.sleep(0.5)
            
            # Build a map of existing orders per symbol
            open_orders = {}
            for o in all_orders:
                symbol = o.contract.symbol
                if symbol not in open_orders:
                    open_orders[symbol] = {"has_sl": False, "has_tp": False, "sl_count": 0, "tp_count": 0}
                if o.order.orderType == "STP":
                    open_orders[symbol]["has_sl"] = True
                    open_orders[symbol]["sl_count"] += 1
                elif o.order.orderType == "LMT":
                    open_orders[symbol]["has_tp"] = True
                    open_orders[symbol]["tp_count"] += 1
            
            # Check TWS positions
            tws_positions = self.get_tws_positions()
            
            for symbol, tws_pos in tws_positions.items():
                pos_qty = abs(int(tws_pos["quantity"]))
                if pos_qty == 0:
                    continue
                    
                pos_action = "BUY" if tws_pos["quantity"] > 0 else "SELL"
                pos_entry = tws_pos["avg_cost"]
                
                # Update tracking if not exists
                if symbol not in self.positions:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        action=pos_action,
                        quantity=pos_qty,
                        entry_price=pos_entry,
                        stop_loss=pos_entry * ((1 - MAX_SL_PCT) if pos_action == "BUY" else (1 + MAX_SL_PCT)),
                        take_profit=pos_entry * ((1 + MAX_TP_PCT) if pos_action == "BUY" else (1 - MAX_TP_PCT)),
                        entry_time=datetime.now(),
                        highest_price=pos_entry,
                        lowest_price=pos_entry
                    )
                
                # Check if has SL/TP
                order_info = open_orders.get(symbol, {"has_sl": False, "has_tp": False, "sl_count": 0, "tp_count": 0})
                has_sl = order_info["has_sl"]
                has_tp = order_info["has_tp"]
                
                # Skip if already has both SL and TP
                if has_sl and has_tp:
                    continue
                
                log(f"🔍 {symbol}: SL={has_sl}, TP={has_tp}")
                
                # Only add what's missing
                if not has_sl or not has_tp:
                    log(f"⚠️ {symbol}: Missing {'SL' if not has_sl else ''}{'/' if not has_sl and not has_tp else ''}{'TP' if not has_tp else ''}")
                    
                    # Calculate SL/TP
                    if pos_action == "BUY":
                        sl_price = round(pos_entry * (1 - MAX_SL_PCT), 2)
                        tp_price = round(pos_entry * (1 + MAX_TP_PCT), 2)
                    else:  # SELL/SHORT
                        sl_price = round(pos_entry * (1 + MAX_SL_PCT), 2)
                        tp_price = round(pos_entry * (1 - MAX_TP_PCT), 2)
                    
                    # Place only missing orders
                    contract = Stock(symbol, 'SMART', 'USD')
                    self.ib.qualifyContracts(contract)
                    exit_action = "SELL" if pos_action == "BUY" else "BUY"
                    
                    if not has_sl:
                        sl_order = StopOrder(exit_action, pos_qty, sl_price)
                        sl_order.outsideRth = True
                        self.ib.placeOrder(contract, sl_order)
                        log(f"   ✅ Added SL: ${sl_price:.2f}")
                    
                    if not has_tp:
                        tp_order = LimitOrder(exit_action, pos_qty, tp_price)
                        tp_order.outsideRth = True
                        self.ib.placeOrder(contract, tp_order)
                        log(f"   ✅ Added TP: ${tp_price:.2f}")
                    
                    self.ib.sleep(0.3)
                        
        except Exception as e:
            log(f"❌ Error checking SL/TP: {e}")
    
    def sync_positions(self):
        """Sync internal positions with TWS."""
        tws_positions = self.get_tws_positions()
        
        # Remove closed positions from tracking
        for symbol in list(self.positions.keys()):
            if symbol not in tws_positions or tws_positions[symbol]["quantity"] == 0:
                closed_pos = self.positions.pop(symbol)
                log(f"📤 Position closed: {symbol}")
                
                # Get exit price for accurate tracking
                try:
                    market_data = self.get_market_data(symbol)
                    exit_price = market_data["current_price"] if market_data else closed_pos.entry_price
                except:
                    exit_price = closed_pos.entry_price
                
                # Calculate P&L
                if closed_pos.action == "BUY":
                    pnl_pct = ((exit_price - closed_pos.entry_price) / closed_pos.entry_price) * 100
                else:
                    pnl_pct = ((closed_pos.entry_price - exit_price) / closed_pos.entry_price) * 100
                
                pnl_usd = pnl_pct / 100 * closed_pos.entry_price * closed_pos.quantity
                is_win = pnl_pct > 0.1  # > 0.1% is a win
                
                # Determine exit reason
                if closed_pos.trailing_stop and exit_price <= closed_pos.trailing_stop:
                    exit_reason = "trailing_stop"
                elif exit_price <= closed_pos.stop_loss:
                    exit_reason = "stop_loss"
                elif exit_price >= closed_pos.take_profit:
                    exit_reason = "take_profit"
                else:
                    exit_reason = "manual" if abs(pnl_pct) < 0.1 else ("profit" if is_win else "loss")
                
                # Record exit in Live Performance Tracker
                self.live_tracker.record_exit(
                    symbol=symbol,
                    exit_price=exit_price,
                    reason=exit_reason
                )
                
                log(f"📊 {symbol} Exit: ${exit_price:.2f} | P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | {exit_reason}")
                
                # ============= V3.7.2 LEARNING FROM TRADES =============
                # Store trade in ChromaDB for learning!
                try:
                    sector = "UNKNOWN"
                    for sec, symbols in SECTOR_MAP.items():
                        if symbol in symbols:
                            sector = sec
                            break
                    
                    trade_data = {
                        "symbol": symbol,
                        "entry_price": closed_pos.entry_price,
                        "exit_price": exit_price,
                        "profit_pct": pnl_pct,
                        "strategy": exit_reason,
                        "signals": [],  # Could add Phase1 signals here
                        "atr": 0,
                        "score": closed_pos.confidence if hasattr(closed_pos, 'confidence') else 0,
                        "timestamp": datetime.now().isoformat(),
                        "context": f"Sector: {sector}, Action: {closed_pos.action}, Entry: ${closed_pos.entry_price:.2f}, Exit: ${exit_price:.2f}"
                    }
                    
                    doc_id = self.trading_memory.store_trade(trade_data)
                    if doc_id:
                        outcome = "WIN" if pnl_pct > 0 else "LOSS"
                        log(f"🧠 Trade learned: {symbol} ({outcome}) → Memory ID: {doc_id[:16]}...")
                except Exception as e:
                    log(f"⚠️ Failed to store trade for learning: {e}")
                # ========================================================
                
                # Update daily P&L
                self.daily_pnl.update_trade(pnl_usd, is_win)
        
        # Check and add missing SL/TP for existing positions
        self.check_and_add_missing_sl_tp()


# ============================================================================
# MAIN TRADING LOOP
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  Zero Trading Expert (ZTE) - TWS Auto Trader V3.5")
    print("  🎯 PHASE 1 INDICATORS: RSI Div, TSI, BB%, VPOC, MACD Div")
    print("  🚀 ENHANCED: Real RVOL, Gap Scanner, Sector Diversification")
    print("  📊 NEW: Tiered Position System (10 positions)")
    print("="*70)
    print(f"  TWS Connection: {TWS_HOST}:{TWS_PORT}")
    print(f"  Watching: {len(SYMBOLS)} stocks")
    print(f"  Scan Interval: {SCAN_INTERVAL}s")
    print(f"  Min Confidence: {MIN_CONFIDENCE:.0%}")
    print(f"  Min RVOL: {MIN_RVOL}x")
    print(f"  Max Daily Loss: {MAX_DAILY_LOSS_PCT:.1%}")
    print(f"  Max Positions: {MAX_OPEN_POSITIONS} (Tier1: {TIER1_POSITIONS} | Tier2: {MAX_OPEN_POSITIONS - TIER1_POSITIONS})")
    print(f"  Tier2 Requirements: RVOL {TIER2_MIN_RVOL}x | Conf {TIER2_MIN_CONFIDENCE:.0%} | Phase1 {TIER2_MIN_PHASE1_SIGNALS}+")
    print(f"  Max Per Sector: {MAX_PER_SECTOR}")
    print(f"  📊 Phase 1 Indicators: ACTIVE")
    print("="*70 + "\n")
    
    # Initialize TWS connection
    trader = TWSTrader()
    
    if not trader.connect():
        print("[ERROR] Failed to connect to TWS. Exiting.")
        return
    
    # Pre-market gap scan (if in pre-market)
    gap_stocks = []
    current_session = get_current_session()
    if current_session == TradingSession.PRE_MARKET:
        gap_stocks = trader.scan_premarket_gaps()
    
    try:
        cycle = 0
        last_gap_scan = datetime.now()
        
        while True:
            cycle += 1
            
            # Check if trading allowed (daily limits)
            can_trade, reason = trader.daily_pnl.can_trade()
            if not can_trade:
                log(f"🛑 TRADING STOPPED: {reason}")
                log("Waiting for tomorrow or manual reset...")
                time.sleep(300)  # Wait 5 min before checking again
                continue
            
            # Check trading session
            session_ok, session_msg = is_good_trading_time()
            current_session = get_current_session()
            
            log(f"=== Cycle {cycle} | {current_session.value.upper()} ===")
            log(session_msg)
            
            # Refresh gap scan every 15 minutes in pre-market
            if current_session == TradingSession.PRE_MARKET:
                if (datetime.now() - last_gap_scan).seconds > 900:  # 15 min
                    gap_stocks = trader.scan_premarket_gaps()
                    last_gap_scan = datetime.now()
            
            # Show daily P&L status
            pnl = trader.daily_pnl
            log(f"📊 Daily P&L: ${pnl.current_pnl:+.2f} | Trades: {pnl.trades_today} | Win Rate: {pnl.win_rate:.0%}")
            
            # Show sector exposure
            sector_exposure = {}
            for pos_symbol in trader.positions.keys():
                sector = trader.get_symbol_sector(pos_symbol)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + 1
            if sector_exposure:
                log(f"📈 Sector Exposure: {sector_exposure}")
            
            # Always sync positions and check SL/TP (even in bad sessions)
            trader.update_trailing_stops()
            trader.sync_positions()
            
            if not session_ok:
                log(f"Waiting {SCAN_INTERVAL}s for better session...")
                time.sleep(SCAN_INTERVAL)
                continue
            
            # Prioritize gap stocks at open, otherwise use balanced sector rotation
            if gap_stocks and current_session in [TradingSession.OPENING_BELL, TradingSession.MID_MORNING]:
                gap_symbols = [g['symbol'] for g in gap_stocks[:5]]
                scan_list = gap_symbols + [s for s in SYMBOLS if s not in gap_symbols]
                log(f"🌅 Prioritizing gap stocks: {gap_symbols}")
            else:
                # 🔀 V3.5.3: Balanced Sector Rotation (Round-Robin)
                # This ensures fair distribution across all sectors instead of always
                # scanning TECH first. Each cycle starts with a different sector.
                sectors = list(SECTOR_MAP.keys())
                
                # Rotate sector order based on cycle number
                sector_offset = cycle % len(sectors)
                rotated_sectors = sectors[sector_offset:] + sectors[:sector_offset]
                
                # Build scan list by taking one stock from each sector in rotation
                scan_list = []
                max_stocks_per_sector = max(len(SECTOR_MAP[s]) for s in sectors)
                
                for i in range(max_stocks_per_sector):
                    for sector in rotated_sectors:
                        sector_stocks = SECTOR_MAP[sector]
                        if i < len(sector_stocks):
                            scan_list.append(sector_stocks[i])
                
                # Log the first stock from each sector (shows the rotation)
                first_per_sector = [SECTOR_MAP[s][0] for s in rotated_sectors if SECTOR_MAP[s]]
                log(f"🔀 Sector rotation order: {' → '.join(rotated_sectors[:4])}...")
                log(f"   First in queue: {first_per_sector[:5]}")
            
            # Scan symbols
            signals_found = 0
            phase1_hits = 0
            
            for symbol in scan_list:
                try:
                    # 1. Get market data from TWS
                    data = trader.get_market_data(symbol)
                    
                    if not data:
                        continue
                    
                    price = data["current_price"]
                    
                    # 2. Calculate REAL RVOL (or use default in Paper mode)
                    if PAPER_TRADING_MODE:
                        # Paper Trading: TWS doesn't return accurate volume
                        # Use default RVOL to allow trading
                        rvol = PAPER_DEFAULT_RVOL
                    elif symbol not in trader.avg_volumes or cycle % 20 == 1:
                        # Refresh real RVOL every 20 cycles or if not cached
                        rvol = trader.calculate_real_rvol(symbol)
                    else:
                        # Use quick estimate from current data
                        rvol = data.get("rvol", 1.0)
                    
                    data["rvol"] = rvol  # Update data dict
                    
                    # Quick filter - only log interesting stocks
                    if rvol >= MIN_RVOL:
                        sector = trader.get_symbol_sector(symbol)
                        log(f"🔍 {symbol} [{sector}]: ${price:.2f} (RVOL: {rvol:.1f}x)")
                    
                    # 3. Quick filter (basic checks only, full Tier check after API)
                    # Only check basic filters first, not Tier requirements
                    current_positions = len(trader.positions)
                    if current_positions >= MAX_OPEN_POSITIONS:
                        if rvol >= MIN_RVOL:
                            log(f"   ⏭️ Skipped: Max positions ({MAX_OPEN_POSITIONS})")
                        continue
                    if symbol in trader.positions:
                        continue
                    if not trader.check_sector_exposure(symbol):
                        if rvol >= MIN_RVOL:
                            log(f"   ⏭️ Skipped: Max sector positions ({MAX_PER_SECTOR})")
                        continue
                    
                    # 4. ⚡ Run Phase 1 Advanced Analysis
                    phase1_analysis = trader.analyze_with_phase1_indicators(symbol, data)
                    phase1_signals = phase1_analysis.get("phase1_signals", [])
                    signal_strength = phase1_analysis.get("signal_strength", 0)
                    indicators = phase1_analysis.get("indicators", {})
                    
                    # Log Phase 1 signals if any
                    if phase1_signals:
                        phase1_hits += 1
                        log(f"   📊 Phase1: {' | '.join(phase1_signals[:3])}")
                        if indicators.get("rsi"):
                            log(f"   📈 RSI: {indicators['rsi']} | TSI: {indicators.get('tsi', 'N/A')} | BB%: {indicators.get('bb_percent', 'N/A')}")
                        if indicators.get("vpoc"):
                            log(f"   📊 VPOC: ${indicators['vpoc']} | VAL: ${indicators.get('val', 'N/A')} | VAH: ${indicators.get('vah', 'N/A')}")
                    
                    # 4. Build enhanced payload with Phase 1 data
                    all_signals = [f"RVOL_{rvol:.1f}x"] + phase1_signals
                    phase1_context = f" | Phase1 Strength: {signal_strength:+d}" if signal_strength != 0 else ""
                    
                    # Clean indicators for JSON (remove NaN/Inf)
                    clean_indicators = clean_for_json(indicators)
                    
                    payload = {
                        "symbol": symbol,
                        "price": float(price) if price and not np.isnan(price) else 0.0,
                        "atr": float(price * 0.02) if price and not np.isnan(price) else 0.0,
                        "score": 50 + (signal_strength * 5),  # Boost score based on Phase 1
                        "signals": all_signals,
                        "context": f"Day Trade | RVOL: {rvol:.1f}x | Session: {current_session.value}{phase1_context}",
                        "technical": clean_indicators  # Pass cleaned Phase 1 indicators to API
                    }
                    payload = clean_for_json(payload)  # Clean entire payload
                    
                    # 5. Check Phase 1 recommendation for filtering
                    phase1_rec = phase1_analysis.get("recommendation", "NEUTRAL")

                    # CRITICAL: Check API health before making request
                    if not trader.check_api_health():
                        log(f"   ⏭️ API server down - skipping {symbol}")
                        continue

                    response = requests.post(f"{API_URL}/analyze", json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        action = result.get("action", "HOLD")
                        confidence = result.get("confidence", 0.0)
                        adjustments = result.get("adjustments", {})
                        
                        # Boost confidence if Phase 1 agrees
                        if phase1_rec == action and action != "NEUTRAL":
                            confidence = min(confidence + 0.15, 0.95)
                            log(f"   ⚡ Phase1 confirmation: +15% confidence")
                        
                        if action in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
                            signals_found += 1
                            tier = trader.get_current_tier()
                            log(f"🔥 {symbol}: {action} ({confidence:.1%}) RVOL={rvol:.1f}x [{tier}]")
                            print(f"   Reason: {result.get('reasoning', '')[:80]}...")
                            
                            # ⚡ V3.5 TIERED POSITION SYSTEM
                            # Determine minimum requirements based on current tier
                            current_positions = len(trader.positions)
                            is_tier2 = current_positions >= TIER1_POSITIONS
                            
                            # Set thresholds based on tier
                            if is_tier2:
                                min_confidence = TIER2_MIN_CONFIDENCE
                                min_rvol = TIER2_MIN_RVOL
                                min_signals = TIER2_MIN_PHASE1_SIGNALS
                                require_strong = TIER2_REQUIRE_STRONG_BUY
                            else:
                                min_confidence = MIN_CONFIDENCE
                                min_rvol = MIN_RVOL
                                min_signals = 1
                                require_strong = False
                            
                            # Check tier requirements
                            tier_passed = True
                            tier_reason = ""
                            
                            if confidence < min_confidence:
                                tier_passed = False
                                tier_reason = f"Confidence {confidence:.1%} < {min_confidence:.0%}"
                            elif rvol < min_rvol:
                                tier_passed = False
                                tier_reason = f"RVOL {rvol:.1f}x < {min_rvol:.1f}x"
                            elif len(phase1_signals) < min_signals:
                                tier_passed = False
                                tier_reason = f"Phase1 signals {len(phase1_signals)} < {min_signals}"
                            elif require_strong and action not in ["STRONG_BUY", "STRONG_SELL"]:
                                tier_passed = False
                                tier_reason = f"Tier2 requires STRONG signal (got {action})"
                            
                            if not tier_passed:
                                log(f"   ⏭️ {tier} requirements not met: {tier_reason}")
                                continue
                            
                            # Execute trade - passed all tier requirements
                            # Calculate dynamic quantity
                            quantity = max(1, int(MAX_POSITION_VALUE / price))
                            
                            # Calculate SL/TP (MCP rules)
                            sl_multiplier = adjustments.get("sl_multiplier", 1.0)
                            tp_multiplier = adjustments.get("tp_multiplier", 2.0)
                            
                            atr = price * 0.02
                            sl_from_atr = atr * sl_multiplier * 0.5
                            tp_from_atr = atr * tp_multiplier * 0.5
                            
                            sl_from_pct = price * MAX_SL_PCT
                            tp_from_pct = price * MAX_TP_PCT
                            
                            stop_loss_dist = min(sl_from_atr, sl_from_pct)
                            take_profit_dist = min(tp_from_atr, tp_from_pct)
                            
                            if action in ["BUY", "STRONG_BUY"]:
                                stop_loss = round(price - stop_loss_dist, 2)
                                take_profit = round(price + take_profit_dist, 2)
                            else:
                                stop_loss = round(price + stop_loss_dist, 2)
                                take_profit = round(price - take_profit_dist, 2)
                            
                            # Save trade record
                            trade_record = {
                                "timestamp": datetime.now().isoformat(),
                                "symbol": symbol,
                                "action": action,
                                "price": price,
                                "quantity": quantity,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "confidence": confidence,
                                "rvol": rvol,
                                "session": current_session.value,
                                "reasoning": result.get("reasoning"),
                                "tier": tier
                            }
                            save_trade(trade_record)
                            
                            # Execute bracket order
                            success = trader.place_bracket_order(
                                symbol=symbol,
                                action=action if action in ["BUY", "SELL"] else ("BUY" if "BUY" in action else "SELL"),
                                quantity=quantity,
                                entry_price=price,
                                stop_loss=stop_loss,
                                take_profit=take_profit
                            )
                            
                            if success:
                                sl_pct = abs(stop_loss - price) / price * 100
                                tp_pct = abs(take_profit - price) / price * 100
                                log(f"✅ EXECUTED [{tier}]: {action} {quantity} {symbol} @ ${price:.2f}")
                                log(f"   📉 SL: ${stop_loss:.2f} (-{sl_pct:.1f}%) | 📈 TP: ${take_profit:.2f} (+{tp_pct:.1f}%)")
                            else:
                                log(f"❌ Order failed for {symbol}")
                    else:
                        log(f"API Error for {symbol}: {response.status_code}")
                        
                except Exception as e:
                    log(f"Error processing {symbol}: {e}")
                
                # Small delay between symbols
                time.sleep(0.3)
            
            # Summary with Tier info
            current_tier = trader.get_current_tier()
            tier_slots = f"{TIER1_POSITIONS - min(len(trader.positions), TIER1_POSITIONS)} Tier1" if len(trader.positions) < TIER1_POSITIONS else f"{MAX_OPEN_POSITIONS - len(trader.positions)} Tier2"
            log(f"📋 Signals: {signals_found} | Phase1 Hits: {phase1_hits} | Positions: {len(trader.positions)}/{MAX_OPEN_POSITIONS} ({tier_slots} available)")
            
            # Show open positions
            if trader.positions:
                log("📂 Open Positions:")
                for symbol, pos in trader.positions.items():
                    log(f"   {symbol}: {pos.action} {pos.quantity} @ ${pos.entry_price:.2f} | SL: ${pos.current_stop:.2f}")
            
            log(f"⏳ Next scan in {SCAN_INTERVAL}s...")
            print()
            time.sleep(SCAN_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n[Bot] Stopping...")
    finally:
        # Save daily log
        save_daily_log(trader.daily_pnl)
        log(f"📊 Final Daily P&L: ${trader.daily_pnl.current_pnl:+.2f}")
        trader.disconnect()


if __name__ == "__main__":
    util.startLoop()
    main()
