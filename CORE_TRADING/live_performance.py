"""
Live Performance Tracker for Zero Trading Expert
=================================================
Tracks REAL trading performance in real-time.
Unlike the RAG memory, this tracks actual trade outcomes.

The win rate from this tracker is THE REAL WIN RATE!
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LiveTrade:
    """A single live trade record."""
    timestamp: str
    symbol: str
    action: str  # BUY or SELL
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int = 0
    profit_pct: Optional[float] = None
    profit_usd: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED_WIN, CLOSED_LOSS, CLOSED_SCRATCH
    
    # ZTE analysis at entry
    zte_confidence: float = 0.0
    zte_action: str = ""
    zte_signals: List[str] = field(default_factory=list)
    
    # Validation
    zte_correct: Optional[bool] = None  # Was ZTE's recommendation correct?
    
    # Additional context
    sector: str = ""
    rvol: float = 0.0
    phase1_signals: List[str] = field(default_factory=list)


class LivePerformanceTracker:
    """
    Tracks real trading performance for accurate Win Rate calculation.
    
    This is the ONLY source of truth for actual trading performance.
    The RAG memory (98.7%) is NOT a real win rate!
    """
    
    def __init__(self, file_path: str = None):
        if file_path is None:
            file_path = Path(__file__).parent.parent / "MEMORY" / "live_performance.jsonl"
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing trades
        self.trades: List[LiveTrade] = []
        self._load_trades()
        
        print(f"[LIVE_TRACKER] Initialized with {len(self.trades)} trades")
        self._print_summary()
    
    def _load_trades(self):
        """Load trades from JSONL file."""
        if not self.file_path.exists():
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Convert dict to LiveTrade
                        trade = LiveTrade(
                            timestamp=data.get('timestamp', ''),
                            symbol=data.get('symbol', ''),
                            action=data.get('action', ''),
                            entry_price=data.get('entry_price', 0),
                            exit_price=data.get('exit_price'),
                            quantity=data.get('quantity', 0),
                            profit_pct=data.get('profit_pct'),
                            profit_usd=data.get('profit_usd'),
                            status=data.get('status', 'OPEN'),
                            zte_confidence=data.get('zte_confidence', 0),
                            zte_action=data.get('zte_action', ''),
                            zte_signals=data.get('zte_signals', []),
                            zte_correct=data.get('zte_correct'),
                            sector=data.get('sector', ''),
                            rvol=data.get('rvol', 0),
                            phase1_signals=data.get('phase1_signals', [])
                        )
                        self.trades.append(trade)
        except Exception as e:
            print(f"[LIVE_TRACKER] Error loading trades: {e}")
    
    def _save_trade(self, trade: LiveTrade):
        """Append a single trade to the file."""
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(trade)) + '\n')
        except Exception as e:
            print(f"[LIVE_TRACKER] Error saving trade: {e}")
    
    def _save_all_trades(self):
        """Rewrite all trades to file (for updates)."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for trade in self.trades:
                    f.write(json.dumps(asdict(trade)) + '\n')
        except Exception as e:
            print(f"[LIVE_TRACKER] Error saving trades: {e}")
    
    def _print_summary(self):
        """Print performance summary."""
        stats = self.get_performance_stats()
        if stats['total_trades'] > 0:
            print(f"[LIVE_TRACKER] REAL Win Rate: {stats['win_rate']:.1%} "
                  f"({stats['wins']}/{stats['total_closed']} closed trades)")
            print(f"[LIVE_TRACKER] ZTE Accuracy: {stats['zte_accuracy']:.1%}")
    
    # ===== RECORDING TRADES =====
    
    def record_entry(self, symbol: str, action: str, entry_price: float,
                    quantity: int, zte_confidence: float = 0,
                    zte_action: str = "", zte_signals: List[str] = None,
                    sector: str = "", rvol: float = 0,
                    phase1_signals: List[str] = None) -> LiveTrade:
        """
        Record a new trade entry.
        
        Args:
            symbol: Stock symbol (e.g., "NVDA")
            action: "BUY" or "SELL"
            entry_price: Entry price
            quantity: Number of shares
            zte_confidence: ZTE's confidence score at entry
            zte_action: ZTE's recommended action
            zte_signals: List of signals from ZTE
            sector: Stock sector
            rvol: Relative volume at entry
            phase1_signals: Phase 1 indicator signals
            
        Returns:
            The created LiveTrade object
        """
        trade = LiveTrade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            quantity=quantity,
            status="OPEN",
            zte_confidence=zte_confidence,
            zte_action=zte_action,
            zte_signals=zte_signals or [],
            sector=sector,
            rvol=rvol,
            phase1_signals=phase1_signals or []
        )
        
        self.trades.append(trade)
        self._save_trade(trade)
        
        print(f"[LIVE_TRACKER] 📝 Entry recorded: {action} {quantity} {symbol} @ ${entry_price:.2f}")
        print(f"[LIVE_TRACKER]    ZTE: {zte_action} ({zte_confidence:.0%})")
        
        return trade
    
    def record_exit(self, symbol: str, exit_price: float,
                   reason: str = "manual") -> Optional[LiveTrade]:
        """
        Record a trade exit and calculate performance.
        
        Args:
            symbol: Stock symbol
            exit_price: Exit price
            reason: Exit reason (manual, stop_loss, take_profit, trailing_stop)
            
        Returns:
            The updated LiveTrade object, or None if not found
        """
        # Find the open trade for this symbol
        trade = None
        for t in reversed(self.trades):
            if t.symbol == symbol and t.status == "OPEN":
                trade = t
                break
        
        if not trade:
            print(f"[LIVE_TRACKER] ⚠️ No open trade found for {symbol}")
            return None
        
        # Calculate profit/loss
        trade.exit_price = exit_price
        
        if trade.action == "BUY":
            trade.profit_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:  # SELL (short)
            trade.profit_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        
        trade.profit_usd = trade.profit_pct / 100 * trade.entry_price * trade.quantity
        
        # Determine status
        if trade.profit_pct > 0.1:
            trade.status = "CLOSED_WIN"
        elif trade.profit_pct < -0.1:
            trade.status = "CLOSED_LOSS"
        else:
            trade.status = "CLOSED_SCRATCH"  # Break-even
        
        # Validate ZTE's recommendation
        if trade.zte_action:
            # ZTE was correct if:
            # 1. It said BUY and we made profit
            # 2. It said SELL/SKIP and we would have lost
            if trade.action == "BUY":
                trade.zte_correct = (
                    (trade.zte_action == "BUY" and trade.profit_pct > 0) or
                    (trade.zte_action in ["SELL", "SKIP", "HOLD"] and trade.profit_pct < 0)
                )
            else:
                trade.zte_correct = (
                    (trade.zte_action == "SELL" and trade.profit_pct > 0) or
                    (trade.zte_action in ["BUY", "SKIP", "HOLD"] and trade.profit_pct < 0)
                )
        
        # Save updated trade
        self._save_all_trades()
        
        emoji = "✅" if trade.profit_pct > 0 else "❌"
        print(f"[LIVE_TRACKER] {emoji} Exit recorded: {symbol} @ ${exit_price:.2f}")
        print(f"[LIVE_TRACKER]    P/L: {trade.profit_pct:+.2f}% (${trade.profit_usd:+.2f})")
        print(f"[LIVE_TRACKER]    ZTE Correct: {trade.zte_correct}")
        
        return trade
    
    # ===== PERFORMANCE STATS =====
    
    def get_performance_stats(self) -> Dict:
        """
        Get REAL performance statistics.
        
        This is the ACTUAL Win Rate, not the RAG memory fake 98.7%!
        """
        closed_trades = [t for t in self.trades if t.status != "OPEN"]
        
        if not closed_trades:
            return {
                'total_trades': len(self.trades),
                'open_trades': len([t for t in self.trades if t.status == "OPEN"]),
                'total_closed': 0,
                'wins': 0,
                'losses': 0,
                'scratches': 0,
                'win_rate': 0.0,
                'avg_profit_pct': 0.0,
                'total_profit_usd': 0.0,
                'zte_accuracy': 0.0,
                'note': 'No closed trades yet - tracking in progress'
            }
        
        wins = len([t for t in closed_trades if t.status == "CLOSED_WIN"])
        losses = len([t for t in closed_trades if t.status == "CLOSED_LOSS"])
        scratches = len([t for t in closed_trades if t.status == "CLOSED_SCRATCH"])
        
        # ZTE accuracy
        zte_validated = [t for t in closed_trades if t.zte_correct is not None]
        zte_correct = len([t for t in zte_validated if t.zte_correct])
        
        # Calculate averages
        profits = [t.profit_pct for t in closed_trades if t.profit_pct is not None]
        total_usd = sum(t.profit_usd or 0 for t in closed_trades)
        
        return {
            'total_trades': len(self.trades),
            'open_trades': len([t for t in self.trades if t.status == "OPEN"]),
            'total_closed': len(closed_trades),
            'wins': wins,
            'losses': losses,
            'scratches': scratches,
            'win_rate': wins / len(closed_trades) if closed_trades else 0.0,
            'avg_profit_pct': sum(profits) / len(profits) if profits else 0.0,
            'total_profit_usd': total_usd,
            'zte_accuracy': zte_correct / len(zte_validated) if zte_validated else 0.0,
            'zte_validated_count': len(zte_validated)
        }
    
    def get_daily_stats(self, date: datetime = None) -> Dict:
        """Get stats for a specific day."""
        if date is None:
            date = datetime.now()
        
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_trades = [
            t for t in self.trades
            if day_start.isoformat() <= t.timestamp < day_end.isoformat()
        ]
        
        closed_today = [t for t in day_trades if t.status != "OPEN"]
        wins = len([t for t in closed_today if t.status == "CLOSED_WIN"])
        
        return {
            'date': day_start.strftime('%Y-%m-%d'),
            'trades': len(day_trades),
            'closed': len(closed_today),
            'wins': wins,
            'losses': len([t for t in closed_today if t.status == "CLOSED_LOSS"]),
            'win_rate': wins / len(closed_today) if closed_today else 0.0,
            'total_pnl': sum(t.profit_usd or 0 for t in closed_today)
        }
    
    def get_sector_performance(self) -> Dict[str, Dict]:
        """Get performance breakdown by sector."""
        sectors = {}
        
        closed_trades = [t for t in self.trades if t.status != "OPEN" and t.sector]
        
        for trade in closed_trades:
            if trade.sector not in sectors:
                sectors[trade.sector] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            
            sectors[trade.sector]['trades'] += 1
            if trade.status == "CLOSED_WIN":
                sectors[trade.sector]['wins'] += 1
            sectors[trade.sector]['pnl'] += trade.profit_usd or 0
        
        # Calculate win rates
        for sector in sectors:
            if sectors[sector]['trades'] > 0:
                sectors[sector]['win_rate'] = sectors[sector]['wins'] / sectors[sector]['trades']
        
        return sectors
    
    def get_signal_performance(self) -> Dict[str, Dict]:
        """Get performance breakdown by Phase 1 signals."""
        signals = {}
        
        closed_trades = [t for t in self.trades if t.status != "OPEN"]
        
        for trade in closed_trades:
            for signal in trade.phase1_signals:
                if signal not in signals:
                    signals[signal] = {'trades': 0, 'wins': 0}
                
                signals[signal]['trades'] += 1
                if trade.status == "CLOSED_WIN":
                    signals[signal]['wins'] += 1
        
        # Calculate win rates
        for signal in signals:
            if signals[signal]['trades'] > 0:
                signals[signal]['win_rate'] = signals[signal]['wins'] / signals[signal]['trades']
        
        return signals


# ===== TESTING =====

if __name__ == "__main__":
    print("Testing Live Performance Tracker...")
    
    tracker = LivePerformanceTracker()
    
    # Example: Record a trade
    trade = tracker.record_entry(
        symbol="NVDA",
        action="BUY",
        entry_price=140.50,
        quantity=100,
        zte_confidence=0.72,
        zte_action="BUY",
        zte_signals=["RSI_BULLISH_DIVERGENCE", "TSI_BULLISH"],
        sector="TECH",
        rvol=2.3,
        phase1_signals=["RSI_DIV", "BB_OVERSOLD"]
    )
    
    # Example: Close the trade
    # tracker.record_exit("NVDA", exit_price=143.25, reason="take_profit")
    
    print("\n--- Performance Stats ---")
    stats = tracker.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
