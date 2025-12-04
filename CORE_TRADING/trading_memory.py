"""
Trading Memory System for Zero Trading Expert
==============================================
Advanced RAG system with trading-specific collections.

V2.0 - Enhanced with:
- Metadata filtering (sector, market condition, date range)
- Composite scoring (similarity + profit + recency)
- Better search accuracy for trading decisions

Collections:
- trading_patterns: Technical chart patterns (Head & Shoulders, etc.)
- successful_trades: Winning trades for learning
- failed_trades: Losing trades for learning  
- market_conditions: Market context snapshots
- technical_knowledge: Indicators, strategies, concepts

⚠️ IMPORTANT: The win rate from this RAG memory is NOT a real trading win rate!
   It only reflects the ratio of imported winning vs losing trades.
   For REAL performance tracking, use LivePerformanceTracker.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import uuid

# ChromaDB
import chromadb
from chromadb.config import Settings

# Add Zero Agent path for imports
ZERO_PATH = Path(__file__).parent.parent.parent / "ZERO"
sys.path.insert(0, str(ZERO_PATH))


class TradingMemory:
    """
    Trading-specialized RAG Memory System.
    Extends Zero Agent's memory with trading-specific collections.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize Trading Memory.
        
        Args:
            db_path: Path to ChromaDB storage. Defaults to MEMORY/chroma_trading_db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "MEMORY" / "chroma_trading_db"
        else:
            db_path = Path(db_path)
        
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create trading-specific collections
        self.trading_patterns = self._get_or_create_collection("trading_patterns")
        self.successful_trades = self._get_or_create_collection("successful_trades")
        self.failed_trades = self._get_or_create_collection("failed_trades")
        self.market_conditions = self._get_or_create_collection("market_conditions")
        self.technical_knowledge = self._get_or_create_collection("technical_knowledge")
        
        print(f"[TRADING_MEMORY] Initialized at {db_path}")
        self._print_stats()
    
    def _get_or_create_collection(self, name: str):
        """Get or create a collection."""
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"description": f"ZTE {name}"}
            )
        except Exception as e:
            print(f"[WARN] Collection creation error for {name}: {e}")
            return None
    
    def _print_stats(self):
        """Print memory statistics."""
        stats = self.get_stats()
        print(f"[TRADING_MEMORY] Stats: {stats}")
    
    # ===== TRADE STORAGE =====
    
    def store_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Store a trade result for learning.
        
        Args:
            trade_data: Dict with keys:
                - symbol: Stock symbol (e.g., "TSLA")
                - entry_price: Entry price
                - exit_price: Exit price (if closed)
                - profit_pct: Profit/Loss percentage
                - strategy: Strategy that triggered the trade
                - signals: List of signals that agreed
                - atr: ATR at entry
                - score: Scanner score
                - timestamp: Trade timestamp
                - context: Additional context (gap, rvol, etc.)
                
        Returns:
            Document ID
        """
        symbol = trade_data.get("symbol", "UNKNOWN")
        profit_pct = trade_data.get("profit_pct", 0)
        
        # Determine collection based on outcome
        collection = self.successful_trades if profit_pct > 0 else self.failed_trades
        
        # Create document
        doc = f"""
Trade: {symbol}
Entry: ${trade_data.get('entry_price', 0):.2f}
Exit: ${trade_data.get('exit_price', 0):.2f}
P/L: {profit_pct:+.2f}%
Strategy: {trade_data.get('strategy', 'Unknown')}
Signals: {', '.join(trade_data.get('signals', []))}
ATR: {trade_data.get('atr', 0):.2f}
Score: {trade_data.get('score', 0)}
Context: {trade_data.get('context', '')}
Time: {trade_data.get('timestamp', datetime.now().isoformat())}
"""
        
        doc_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            collection.add(
                documents=[doc.strip()],
                metadatas=[{
                    "symbol": symbol,
                    "profit_pct": profit_pct,
                    "strategy": trade_data.get("strategy", ""),
                    "score": trade_data.get("score", 0),
                    "timestamp": trade_data.get("timestamp", datetime.now().isoformat()),
                    "outcome": "win" if profit_pct > 0 else "loss"
                }],
                ids=[doc_id]
            )
            print(f"[TRADING_MEMORY] Stored {'winning' if profit_pct > 0 else 'losing'} trade: {symbol}")
            return doc_id
        except Exception as e:
            print(f"[ERROR] Failed to store trade: {e}")
            return ""
    
    def find_similar_trades(self, query: str, n_results: int = 5,
                           symbol: str = None, sector: str = None,
                           min_profit: float = None, max_days_old: int = None,
                           market_condition: str = None) -> List[Dict]:
        """
        Find similar past trades with ADVANCED RAG filtering and scoring.
        
        V2.0 Enhancements:
        - Metadata filtering (sector, profit, date)
        - Composite scoring: similarity * 0.5 + profit * 0.3 + recency * 0.2
        - Better relevance for trading decisions
        
        Args:
            query: Search query (e.g., "TSLA gap up momentum")
            n_results: Number of results to return
            symbol: Filter by symbol (optional)
            sector: Filter by sector (optional) - TECH, SEMI, SOFTWARE, etc.
            min_profit: Minimum profit percentage (optional)
            max_days_old: Only trades from last N days (optional)
            market_condition: Filter by market condition (optional)
            
        Returns:
            List of similar trades with metadata, sorted by composite score
        """
        results = []
        
        # Build metadata filter
        where_filter = {}
        if symbol:
            where_filter['symbol'] = symbol
        if sector:
            where_filter['sector'] = sector
        if market_condition:
            where_filter['market_condition'] = market_condition
        
        # Search both winning and losing trades
        for collection in [self.successful_trades, self.failed_trades]:
            if collection is None:
                continue
            
            count = collection.count()
            if count == 0:
                continue
            
            try:
                # Query with optional filters
                search_results = collection.query(
                    query_texts=[query],
                    n_results=min(n_results * 3, count),  # Get more for scoring
                    where=where_filter if where_filter else None,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if search_results and search_results['documents']:
                    for i, doc in enumerate(search_results['documents'][0]):
                        metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                        distance = search_results['distances'][0][i] if search_results.get('distances') else 1.0
                        
                        # Apply additional filters
                        if min_profit is not None:
                            profit = metadata.get('profit_pct', 0)
                            if profit < min_profit:
                                continue
                        
                        if max_days_old is not None:
                            trade_date = metadata.get('timestamp', '')
                            if trade_date:
                                try:
                                    trade_dt = datetime.fromisoformat(trade_date.replace('Z', '+00:00'))
                                    if datetime.now() - trade_dt > timedelta(days=max_days_old):
                                        continue
                                except:
                                    pass
                        
                        # Calculate composite score
                        similarity_score = 1 / (1 + distance)  # 0-1, higher is better
                        profit_score = min(metadata.get('profit_pct', 0) / 10, 1.0)  # Normalize to 0-1
                        recency_score = self._calculate_recency_score(metadata.get('timestamp', ''))
                        
                        composite_score = (
                            similarity_score * 0.5 +
                            profit_score * 0.3 +
                            recency_score * 0.2
                        )
                        
                        results.append({
                            "document": doc,
                            "metadata": metadata,
                            "distance": distance,
                            "similarity_score": similarity_score,
                            "profit_score": profit_score,
                            "recency_score": recency_score,
                            "composite_score": composite_score,
                            "outcome": "win" if collection == self.successful_trades else "loss"
                        })
            except Exception as e:
                print(f"[WARN] Search error: {e}")
        
        # Sort by composite score (higher is better)
        results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return results[:n_results]
    
    def _calculate_recency_score(self, timestamp: str) -> float:
        """Calculate recency score (0-1). More recent = higher score."""
        if not timestamp:
            return 0.5  # Default for unknown dates
        
        try:
            trade_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            days_old = (datetime.now() - trade_dt).days
            
            # Exponential decay: 1.0 for today, ~0.5 for 30 days, ~0.1 for 90 days
            return max(0.1, 1.0 * (0.98 ** days_old))
        except:
            return 0.5
    
    # ===== PATTERN STORAGE =====
    
    def store_pattern(self, pattern_name: str, description: str, 
                      success_rate: float = None, examples: List[str] = None):
        """
        Store a trading pattern.
        
        Args:
            pattern_name: Name (e.g., "Head and Shoulders")
            description: Full description
            success_rate: Historical success rate (0-1)
            examples: List of example symbols/dates
        """
        if self.trading_patterns is None:
            return
        
        doc = f"""
Pattern: {pattern_name}
Description: {description}
Success Rate: {success_rate*100:.1f}% if success_rate else 'Unknown'
Examples: {', '.join(examples) if examples else 'None'}
"""
        
        doc_id = f"pattern_{pattern_name.lower().replace(' ', '_')}"
        
        try:
            # Check if exists and update, or add new
            self.trading_patterns.upsert(
                documents=[doc.strip()],
                metadatas=[{
                    "pattern_name": pattern_name,
                    "success_rate": success_rate or 0,
                    "type": "pattern"
                }],
                ids=[doc_id]
            )
            print(f"[TRADING_MEMORY] Stored pattern: {pattern_name}")
        except Exception as e:
            print(f"[ERROR] Failed to store pattern: {e}")
    
    def find_patterns(self, query: str, n_results: int = 3) -> List[Dict]:
        """Find relevant patterns."""
        if self.trading_patterns is None:
            return []
        
        count = self.trading_patterns.count()
        if count == 0:
            return []
        
        try:
            results = self.trading_patterns.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )
            
            if not results or not results['documents']:
                return []
            
            return [{
                "document": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results.get('distances') else 0
            } for i, doc in enumerate(results['documents'][0])]
        except Exception as e:
            print(f"[WARN] Pattern search error: {e}")
            return []
    
    # ===== KNOWLEDGE STORAGE =====
    
    def store_knowledge(self, topic: str, content: str, category: str = "general"):
        """
        Store technical knowledge.
        
        Args:
            topic: Topic name (e.g., "RSI Indicator")
            content: Full content/explanation
            category: Category (indicators, strategies, concepts, risk)
        """
        if self.technical_knowledge is None:
            return
        
        doc = f"{topic}: {content}"
        doc_id = f"knowledge_{topic.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.technical_knowledge.add(
                documents=[doc],
                metadatas=[{
                    "topic": topic,
                    "category": category,
                    "type": "knowledge"
                }],
                ids=[doc_id]
            )
            print(f"[TRADING_MEMORY] Stored knowledge: {topic}")
        except Exception as e:
            print(f"[ERROR] Failed to store knowledge: {e}")
    
    def search_knowledge(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search technical knowledge."""
        if self.technical_knowledge is None:
            return []
        
        count = self.technical_knowledge.count()
        if count == 0:
            return []
        
        try:
            results = self.technical_knowledge.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )
            
            if not results or not results['documents']:
                return []
            
            return [{
                "document": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results.get('distances') else 0
            } for i, doc in enumerate(results['documents'][0])]
        except Exception as e:
            print(f"[WARN] Knowledge search error: {e}")
            return []
    
    # ===== MARKET CONDITIONS =====
    
    def store_market_condition(self, condition: Dict[str, Any]):
        """
        Store market condition snapshot.
        
        Args:
            condition: Dict with keys like spy_trend, vix_level, sector_leaders, etc.
        """
        if self.market_conditions is None:
            return
        
        timestamp = datetime.now().isoformat()
        
        doc = f"""
Market Snapshot: {timestamp}
SPY Trend: {condition.get('spy_trend', 'Unknown')}
VIX Level: {condition.get('vix_level', 'Unknown')}
Sector Leaders: {', '.join(condition.get('sector_leaders', []))}
Market Sentiment: {condition.get('sentiment', 'Neutral')}
Notes: {condition.get('notes', '')}
"""
        
        doc_id = f"market_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.market_conditions.add(
                documents=[doc.strip()],
                metadatas=[{
                    "timestamp": timestamp,
                    "spy_trend": condition.get('spy_trend', ''),
                    "vix_level": str(condition.get('vix_level', '')),
                    "sentiment": condition.get('sentiment', 'Neutral')
                }],
                ids=[doc_id]
            )
            print(f"[TRADING_MEMORY] Stored market condition")
        except Exception as e:
            print(f"[ERROR] Failed to store market condition: {e}")
    
    # ===== STATISTICS =====
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        try:
            return {
                "trading_patterns": self.trading_patterns.count() if self.trading_patterns else 0,
                "successful_trades": self.successful_trades.count() if self.successful_trades else 0,
                "failed_trades": self.failed_trades.count() if self.failed_trades else 0,
                "market_conditions": self.market_conditions.count() if self.market_conditions else 0,
                "technical_knowledge": self.technical_knowledge.count() if self.technical_knowledge else 0
            }
        except Exception as e:
            print(f"[WARN] Stats error: {e}")
            return {}
    
    def get_win_rate(self) -> float:
        """
        Calculate win rate from stored trades.
        
        ⚠️ WARNING: This is NOT a real trading win rate!
        This only shows the ratio of winning vs losing trades in the RAG memory.
        These are historical/imported trades, not live trading results.
        
        For REAL performance, use LivePerformanceTracker.get_performance_stats()
        """
        try:
            wins = self.successful_trades.count() if self.successful_trades else 0
            losses = self.failed_trades.count() if self.failed_trades else 0
            total = wins + losses
            
            if total == 0:
                return 0.0
            
            return wins / total
        except:
            return 0.0
    
    def get_stats_with_warning(self) -> Dict:
        """
        Get stats with clear warning about what they mean.
        """
        stats = self.get_stats()
        wins = stats.get('successful_trades', 0)
        losses = stats.get('failed_trades', 0)
        total = wins + losses
        
        return {
            **stats,
            'rag_win_rate': wins / total if total > 0 else 0,
            'total_trades_in_memory': total,
            'warning': 'This is RAG memory stats, NOT real trading performance!',
            'note': 'Use LivePerformanceTracker for actual trading win rate'
        }


# ===== TESTING =====

if __name__ == "__main__":
    print("Testing Trading Memory System...")
    
    memory = TradingMemory()
    
    # Test storing knowledge
    memory.store_knowledge(
        topic="RSI Indicator",
        content="RSI (Relative Strength Index) measures momentum on a scale of 0-100. "
                "Values above 70 indicate overbought conditions, below 30 indicate oversold.",
        category="indicators"
    )
    
    # Test storing a pattern
    memory.store_pattern(
        pattern_name="Double Bottom",
        description="A bullish reversal pattern forming a 'W' shape. "
                   "Entry on breakout above the middle peak with stop below the bottoms.",
        success_rate=0.65,
        examples=["AAPL 2024-01-15", "MSFT 2024-02-20"]
    )
    
    # Test storing a trade
    memory.store_trade({
        "symbol": "TSLA",
        "entry_price": 245.50,
        "exit_price": 252.30,
        "profit_pct": 2.77,
        "strategy": "MA_Crossover",
        "signals": ["MA_CROSS", "VWAP", "VOLUME"],
        "atr": 5.2,
        "score": 78,
        "context": "Gap up 3.5%, RVOL 2.8x, Strong momentum"
    })
    
    # Test search
    print("\n--- Knowledge Search ---")
    results = memory.search_knowledge("RSI overbought")
    for r in results:
        print(f"  Found: {r['metadata'].get('topic', 'Unknown')}")
    
    print("\n--- Similar Trades Search ---")
    results = memory.find_similar_trades("TSLA momentum gap")
    for r in results:
        print(f"  Found: {r['metadata'].get('symbol', 'Unknown')} - {r['outcome']}")
    
    print("\n--- Final Stats ---")
    print(memory.get_stats())
    print(f"Win Rate: {memory.get_win_rate()*100:.1f}%")

