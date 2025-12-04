"""
Trade Log Importer for Zero Trading Expert
==========================================
Imports trade history from Pro-Gemini-Trade CSV files.

Source: C:\Vs-Pro\pro-gemini-traed\data\trade_history.csv
Format: timestamp,symbol,strategy,action,price,quantity,order_type,tp_price,sl_price
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
import json

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from CORE_TRADING.trading_memory import TradingMemory


class TradeLogImporter:
    """
    Imports trade logs from Pro-Gemini-Trade into ZTE memory.
    """
    
    DEFAULT_CSV_PATH = "C:/Vs-Pro/pro-gemini-traed/data/trade_history.csv"
    DEFAULT_LOGS_PATH = "C:/Vs-Pro/pro-gemini-traed/logs"
    
    def __init__(self, memory: TradingMemory = None):
        """
        Initialize the importer.
        
        Args:
            memory: TradingMemory instance (creates new if not provided)
        """
        self.memory = memory or TradingMemory()
        self.imported_count = 0
        self.skipped_count = 0
        self.errors = []
    
    def import_csv(self, csv_path: str = None) -> Dict[str, Any]:
        """
        Import trades from CSV file.
        
        Args:
            csv_path: Path to CSV file (uses default if not provided)
            
        Returns:
            Dict with import statistics
        """
        csv_path = Path(csv_path or self.DEFAULT_CSV_PATH)
        
        if not csv_path.exists():
            return {
                "success": False,
                "error": f"CSV file not found: {csv_path}",
                "imported": 0
            }
        
        print(f"[IMPORTER] Reading {csv_path}...")
        
        trades = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read CSV: {e}",
                "imported": 0
            }
        
        print(f"[IMPORTER] Found {len(trades)} trades")
        
        # Group trades by symbol to match entries with exits
        trades_by_symbol = {}
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Process and import trades
        for symbol, symbol_trades in trades_by_symbol.items():
            self._process_symbol_trades(symbol, symbol_trades)
        
        print(f"[IMPORTER] Import complete: {self.imported_count} imported, {self.skipped_count} skipped")
        
        return {
            "success": True,
            "imported": self.imported_count,
            "skipped": self.skipped_count,
            "errors": self.errors[:10],  # First 10 errors
            "total_in_file": len(trades),
            "unique_symbols": len(trades_by_symbol)
        }
    
    def _process_symbol_trades(self, symbol: str, trades: List[Dict]):
        """Process trades for a single symbol."""
        
        for trade in trades:
            try:
                # Extract data from CSV row
                timestamp = trade.get('timestamp', '')
                strategy = trade.get('strategy', 'Unknown')
                action = trade.get('action', 'BUY')
                price = self._parse_float(trade.get('price', 0))
                quantity = self._parse_float(trade.get('quantity', 0))
                order_type = trade.get('order_type', 'MARKET')
                tp_price = self._parse_float(trade.get('tp_price', 0))
                sl_price = self._parse_float(trade.get('sl_price', 0))
                
                # Skip sells (we only want entries with their results)
                if action.upper() == 'SELL':
                    self.skipped_count += 1
                    continue
                
                # Skip if price is 0 or invalid
                if price <= 0:
                    self.skipped_count += 1
                    continue
                
                # Calculate potential profit/loss from bracket orders
                # For now, we'll estimate based on TP/SL ratio
                if tp_price > 0 and sl_price > 0 and price > 0:
                    potential_gain = (tp_price - price) / price * 100
                    potential_loss = (price - sl_price) / price * 100
                    risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
                else:
                    potential_gain = 0
                    potential_loss = 0
                    risk_reward = 0
                
                # Create trade data for memory
                trade_data = {
                    "symbol": symbol,
                    "entry_price": price,
                    "exit_price": tp_price if tp_price > 0 else price,  # Assume TP hit
                    "profit_pct": potential_gain if potential_gain > 0 else -potential_loss,
                    "strategy": strategy,
                    "signals": [strategy],  # Strategy name as signal
                    "atr": abs(tp_price - sl_price) / 2 if tp_price and sl_price else 0,
                    "score": 50,  # Default score
                    "timestamp": timestamp,
                    "context": f"Order type: {order_type}, Qty: {quantity}, TP: ${tp_price:.2f}, SL: ${sl_price:.2f}, R:R={risk_reward:.2f}"
                }
                
                # Store in memory
                self.memory.store_trade(trade_data)
                self.imported_count += 1
                
            except Exception as e:
                self.errors.append(f"{symbol}: {str(e)}")
                self.skipped_count += 1
    
    def _parse_float(self, value) -> float:
        """Parse a value to float, handling various formats."""
        if value is None or value == '':
            return 0.0
        
        try:
            # Handle string values like "MARKET"
            if isinstance(value, str):
                value = value.strip()
                if value.upper() in ['MARKET', '', 'N/A', 'NONE']:
                    return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def import_logs(self, logs_path: str = None) -> Dict[str, Any]:
        """
        Import additional context from log files.
        
        Args:
            logs_path: Path to logs directory
            
        Returns:
            Dict with import statistics
        """
        logs_path = Path(logs_path or self.DEFAULT_LOGS_PATH)
        
        if not logs_path.exists():
            return {
                "success": False,
                "error": f"Logs directory not found: {logs_path}",
                "processed": 0
            }
        
        log_files = list(logs_path.glob("*.log"))
        print(f"[IMPORTER] Found {len(log_files)} log files")
        
        # For now, just count them - full parsing can be added later
        return {
            "success": True,
            "log_files_found": len(log_files),
            "note": "Log parsing available for future enhancement"
        }
    
    def export_to_jsonl(self, output_path: str = None) -> str:
        """
        Export imported trades to JSONL format for training.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to created file
        """
        output_path = output_path or str(
            Path(__file__).parent.parent / "DATASETS" / "imported_trades.jsonl"
        )
        
        # Get all trades from memory
        stats = self.memory.get_stats()
        total_trades = stats.get('successful_trades', 0) + stats.get('failed_trades', 0)
        
        if total_trades == 0:
            print("[IMPORTER] No trades to export")
            return ""
        
        # Query all trades
        all_trades = []
        
        # Get winning trades
        if self.memory.successful_trades and self.memory.successful_trades.count() > 0:
            results = self.memory.successful_trades.get(limit=1000)
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents']):
                    all_trades.append({
                        "document": doc,
                        "metadata": results['metadatas'][i] if results.get('metadatas') else {},
                        "outcome": "win"
                    })
        
        # Get losing trades
        if self.memory.failed_trades and self.memory.failed_trades.count() > 0:
            results = self.memory.failed_trades.get(limit=1000)
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents']):
                    all_trades.append({
                        "document": doc,
                        "metadata": results['metadatas'][i] if results.get('metadatas') else {},
                        "outcome": "loss"
                    })
        
        # Write to JSONL
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for trade in all_trades:
                f.write(json.dumps(trade, ensure_ascii=False) + '\n')
        
        print(f"[IMPORTER] Exported {len(all_trades)} trades to {output_path}")
        
        return str(output_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about imported trades."""
        
        stats = self.memory.get_stats()
        wins = stats.get('successful_trades', 0)
        losses = stats.get('failed_trades', 0)
        total = wins + losses
        
        win_rate = wins / total if total > 0 else 0
        
        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": f"{win_rate:.1%}",
            "memory_stats": stats
        }


# ===== CLI Interface =====

def main():
    """Command-line interface for the importer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import Pro-Gemini-Trade logs to ZTE")
    parser.add_argument('--csv', type=str, default=TradeLogImporter.DEFAULT_CSV_PATH,
                       help='Path to trade_history.csv')
    parser.add_argument('--export', action='store_true',
                       help='Export to JSONL after import')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics only (no import)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Trade Log Importer - Zero Trading Expert")
    print("="*60)
    
    importer = TradeLogImporter()
    
    if args.stats:
        stats = importer.get_statistics()
        print("\n--- Memory Statistics ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Import from CSV
    result = importer.import_csv(args.csv)
    
    print("\n--- Import Result ---")
    for key, value in result.items():
        if key != 'errors':
            print(f"  {key}: {value}")
    
    if result.get('errors'):
        print(f"\n  Errors ({len(result['errors'])} shown):")
        for err in result['errors'][:5]:
            print(f"    - {err}")
    
    # Export if requested
    if args.export and result.get('success'):
        print("\n--- Exporting to JSONL ---")
        output = importer.export_to_jsonl()
        print(f"  Exported to: {output}")
    
    # Show final statistics
    print("\n--- Final Statistics ---")
    stats = importer.get_statistics()
    for key, value in stats.items():
        if key != 'memory_stats':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

