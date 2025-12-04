"""
Trading Orchestrator for Zero Trading Expert
=============================================
Main analysis engine that combines all components.

Uses:
- Zero's Tree-of-Thought for strategy evaluation
- Market Analyzer for technical analysis
- Pattern Detector for chart patterns
- Trading Memory for historical context

Provides:
- Full trade analysis with confidence scores
- Risk-adjusted recommendations
- Similar trade retrieval
- Multi-strategy evaluation
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Add Zero Agent to path for imports
ZERO_PATH = Path(__file__).parent.parent.parent / "ZERO"
sys.path.insert(0, str(ZERO_PATH))

# Import local components
from .trading_memory import TradingMemory
from .market_analyzer import MarketAnalyzer, TechnicalIndicators
from .pattern_detector import PatternDetector, DetectedPattern, PatternSignal

# Try to import Sentiment Agent
try:
    from .sentiment_agent import SentimentAgent, SentimentResult
    SENTIMENT_AVAILABLE = True
    print("[ORCHESTRATOR] Sentiment Agent imported successfully")
except ImportError as e:
    SENTIMENT_AVAILABLE = False
    print(f"[ORCHESTRATOR] Sentiment Agent not available: {e}")

# Try to import Zero's ToT (optional - graceful fallback)
try:
    from CORE.tot_reasoning import TreeOfThought
    TOT_AVAILABLE = True
    print("[ORCHESTRATOR] Zero ToT imported successfully")
except ImportError:
    TOT_AVAILABLE = False
    print("[ORCHESTRATOR] Zero ToT not available - using built-in reasoning")

# Try to import Ollama for LLM
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class TradingThought:
    """A single trading strategy thought."""
    id: int
    strategy: str
    reasoning: str
    score: float  # 0-10
    risk_level: str  # low, medium, high
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "reasoning": self.reasoning,
            "score": self.score,
            "risk_level": self.risk_level
        }


@dataclass
class TradeAnalysis:
    """Complete trade analysis result."""
    symbol: str
    action: str  # BUY, SELL, HOLD, SKIP
    confidence: float  # 0-1
    
    thoughts: List[TradingThought]
    selected_thought: int
    reasoning: str
    
    technical_summary: Dict[str, Any]
    patterns: List[Dict]
    similar_trades: List[Dict]
    
    adjustments: Dict[str, float]  # sl_multiplier, tp_multiplier, position_size
    risk_assessment: Dict[str, Any]
    
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "selected": self.selected_thought,
            "reasoning": self.reasoning,
            "technical": self.technical_summary,
            "patterns": self.patterns,
            "similar_trades": self.similar_trades,
            "adjustments": self.adjustments,
            "risk": self.risk_assessment,
            "timestamp": self.timestamp
        }


class TradingOrchestrator:
    """
    Main Trading Analysis Orchestrator.
    Combines all analysis components with Tree-of-Thought reasoning.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Trading Orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.memory = TradingMemory()
        self.analyzer = MarketAnalyzer(self.config.get("technical", {}))
        self.pattern_detector = PatternDetector(self.config.get("patterns", {}))
        
        # Initialize Sentiment Agent (NEW!)
        self.sentiment_agent = None
        if SENTIMENT_AVAILABLE:
            try:
                self.sentiment_agent = SentimentAgent(self.config.get("sentiment", {}))
                print("[ORCHESTRATOR] Sentiment Agent initialized")
            except Exception as e:
                print(f"[ORCHESTRATOR] Sentiment Agent init failed: {e}")
        
        # LLM settings
        self.ollama_url = self.config.get("ollama_url", "http://localhost:11434")
        self.model = self.config.get("model", "llama3.1:8b")
        
        # Analysis settings
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.num_thoughts = self.config.get("tot_strategies", 3)
        
        print("[ORCHESTRATOR] Initialized")
    
    def analyze(self, 
                symbol: str,
                price: float,
                atr: float = None,
                score: int = None,
                signals: List[str] = None,
                context: str = None,
                prices: List[float] = None,
                highs: List[float] = None,
                lows: List[float] = None,
                volumes: List[float] = None,
                technical_data: Dict = None) -> TradeAnalysis:
        """
        Perform full trade analysis.
        
        Args:
            symbol: Stock symbol
            price: Current price
            atr: Average True Range (optional)
            score: Scanner score from Pro-Gemini (optional)
            signals: List of triggered signals (optional)
            context: Additional context string (optional)
            prices: Historical prices (optional)
            highs/lows/volumes: Historical OHLCV data (optional)
            technical_data: Technical indicators from Pro-Gemini (NEW!)
            
        Returns:
            TradeAnalysis with complete analysis
        """
        timestamp = datetime.now().isoformat()
        signals = signals or []
        
        print(f"[ORCHESTRATOR] Analyzing {symbol}...")
        
        # Step 1: Use Technical Data from Pro-Gemini if available
        if technical_data:
            # Build technical summary from Pro-Gemini data
            rsi = technical_data.get('rsi')
            sma_50 = technical_data.get('sma_50')
            sma_200 = technical_data.get('sma_200')
            price_vs_sma50 = technical_data.get('price_vs_sma50', 0)
            price_vs_sma200 = technical_data.get('price_vs_sma200', 0)
            
            # Determine bias from real data
            bullish_signals = []
            bearish_signals = []
            
            if rsi:
                if rsi < 30:
                    bullish_signals.append(f"RSI oversold ({rsi:.0f})")
                elif rsi > 70:
                    bearish_signals.append(f"RSI overbought ({rsi:.0f})")
                elif rsi < 50:
                    bullish_signals.append(f"RSI bullish zone ({rsi:.0f})")
                else:
                    bearish_signals.append(f"RSI bearish zone ({rsi:.0f})")
            
            if price_vs_sma50:
                if price_vs_sma50 > 0:
                    bullish_signals.append(f"Above SMA50 (+{price_vs_sma50:.1f}%)")
                else:
                    bearish_signals.append(f"Below SMA50 ({price_vs_sma50:.1f}%)")
            
            if price_vs_sma200:
                if price_vs_sma200 > 0:
                    bullish_signals.append(f"Above SMA200 (+{price_vs_sma200:.1f}%)")
                else:
                    bearish_signals.append(f"Below SMA200 ({price_vs_sma200:.1f}%)")
            
            # Calculate bias
            bias = "neutral"
            confidence = 0.5
            if len(bullish_signals) > len(bearish_signals):
                bias = "bullish"
                confidence = 0.5 + (len(bullish_signals) - len(bearish_signals)) * 0.1
            elif len(bearish_signals) > len(bullish_signals):
                bias = "bearish"
                confidence = 0.5 - (len(bearish_signals) - len(bullish_signals)) * 0.1
            
            technical_summary = {
                "bias": bias,
                "confidence": min(confidence, 0.9),
                "rsi": rsi,
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
                "data_source": "Pro-Gemini"
            }
            print(f"[ORCHESTRATOR] Technical: {bias} (RSI={rsi}, Bullish={len(bullish_signals)}, Bearish={len(bearish_signals)})")
        else:
            # Fallback to internal analyzer
            indicators = self.analyzer.analyze(
                symbol=symbol if prices is None else None,
                prices=prices,
                highs=highs,
                lows=lows,
                volumes=volumes,
                current_price=price
            )
            technical_summary = self.analyzer.get_summary(indicators)
            if atr is None:
                atr = indicators.atr
        
        # Step 2: Pattern Detection
        if prices:
            patterns = self.pattern_detector.detect_all(prices, highs, lows, volumes)
        else:
            patterns = []
        
        pattern_summary = self.pattern_detector.summarize_patterns(patterns)
        
        # Step 3: Sentiment Analysis (NEW!)
        sentiment_data = None
        if self.sentiment_agent:
            try:
                sentiment_result = self.sentiment_agent.analyze(symbol)
                sentiment_data = {
                    "score": sentiment_result.score,
                    "label": sentiment_result.label,
                    "confidence": sentiment_result.confidence,
                    "news_count": sentiment_result.news_count,
                    "headlines": sentiment_result.headlines[:2]
                }
                print(f"[ORCHESTRATOR] Sentiment: {sentiment_result.label} ({sentiment_result.score:+.2f}) from {sentiment_result.news_count} news")
                
                # Add to technical summary
                technical_summary["sentiment"] = sentiment_data
                
                # Adjust bullish/bearish signals based on sentiment
                if sentiment_result.label == "bullish" and sentiment_result.confidence > 0.5:
                    technical_summary.setdefault("bullish_signals", []).append(f"News sentiment bullish ({sentiment_result.score:+.2f})")
                elif sentiment_result.label == "bearish" and sentiment_result.confidence > 0.5:
                    technical_summary.setdefault("bearish_signals", []).append(f"News sentiment bearish ({sentiment_result.score:+.2f})")
            except Exception as e:
                print(f"[ORCHESTRATOR] Sentiment analysis failed: {e}")
        
        # Step 4: Find Similar Trades
        query = f"{symbol} {' '.join(signals)} {context or ''}"
        print(f"[ORCHESTRATOR] Searching similar trades with query: {query}")
        similar_trades = self.memory.find_similar_trades(query, n_results=5)
        print(f"[ORCHESTRATOR] Found {len(similar_trades)} similar trades")
        
        # Step 4: Generate Trading Thoughts (ToT)
        thoughts = self._generate_thoughts(
            symbol=symbol,
            price=price,
            atr=atr,
            score=score,
            signals=signals,
            context=context,
            technical=technical_summary,
            patterns=pattern_summary,
            similar_trades=similar_trades
        )
        
        # Step 5: Evaluate and Select Best Thought
        selected_idx, reasoning = self._evaluate_thoughts(thoughts, technical_summary)
        
        # Step 6: Determine Action and Confidence
        action, confidence = self._determine_action(
            thoughts=thoughts,
            selected_idx=selected_idx,
            technical=technical_summary,
            patterns=pattern_summary,
            similar_trades=similar_trades
        )
        
        # Step 7: Calculate Adjustments
        adjustments = self._calculate_adjustments(
            action=action,
            confidence=confidence,
            technical=technical_summary,
            atr=atr,
            similar_trades=similar_trades
        )
        
        # Step 8: Risk Assessment
        risk_assessment = self._assess_risk(
            symbol=symbol,
            price=price,
            atr=atr,
            technical=technical_summary,
            patterns=pattern_summary
        )
        
        # Create result
        analysis = TradeAnalysis(
            symbol=symbol,
            action=action,
            confidence=confidence,
            thoughts=thoughts,
            selected_thought=selected_idx,
            reasoning=reasoning,
            technical_summary=technical_summary,
            patterns=[p.to_dict() for p in patterns[:3]],  # Top 3 patterns
            similar_trades=similar_trades[:3],
            adjustments=adjustments,
            risk_assessment=risk_assessment,
            timestamp=timestamp
        )
        
        print(f"[ORCHESTRATOR] Analysis complete: {action} ({confidence:.0%})")
        
        return analysis
    
    def _generate_thoughts(self, symbol: str, price: float, atr: float,
                          score: int, signals: List[str], context: str,
                          technical: Dict, patterns: Dict,
                          similar_trades: List) -> List[TradingThought]:
        """Generate multiple trading strategies using ToT approach."""
        
        thoughts = []
        
        # Thought 1: Follow the technical signals
        tech_signal = technical.get("bias", "neutral")
        tech_confidence = technical.get("confidence", 0.5)
        
        if tech_signal == "bullish":
            thoughts.append(TradingThought(
                id=1,
                strategy="Enter LONG based on technical signals",
                reasoning=f"Technical analysis shows {tech_signal} bias with {len(technical.get('bullish_signals', []))} bullish signals: {', '.join(technical.get('bullish_signals', [])[:3])}",
                score=7.0 + tech_confidence * 2,
                risk_level="medium"
            ))
        elif tech_signal == "bearish":
            thoughts.append(TradingThought(
                id=1,
                strategy="SKIP - Technical signals are bearish",
                reasoning=f"Technical analysis shows {tech_signal} bias with {len(technical.get('bearish_signals', []))} bearish signals",
                score=3.0,
                risk_level="high"
            ))
        else:
            thoughts.append(TradingThought(
                id=1,
                strategy="WAIT for clearer signal",
                reasoning="Technical signals are mixed/neutral",
                score=5.0,
                risk_level="medium"
            ))
        
        # Thought 2: Pattern-based strategy
        if patterns.get("count", 0) > 0:
            dominant = patterns.get("dominant_signal", "neutral")
            if dominant == "bullish":
                thoughts.append(TradingThought(
                    id=2,
                    strategy="Enter LONG based on bullish pattern",
                    reasoning=f"Detected {patterns['bullish_count']} bullish patterns with highest confidence {patterns.get('highest_confidence', 0):.0%}",
                    score=7.5 + patterns.get('highest_confidence', 0) * 2,
                    risk_level="medium"
                ))
            else:
                thoughts.append(TradingThought(
                    id=2,
                    strategy="Caution - Bearish/neutral patterns detected",
                    reasoning=f"Detected {patterns['bearish_count']} bearish patterns",
                    score=4.0,
                    risk_level="high"
                ))
        else:
            thoughts.append(TradingThought(
                id=2,
                strategy="No clear pattern - Use other signals",
                reasoning="No significant chart patterns detected",
                score=5.0,
                risk_level="medium"
            ))
        
        # Thought 3: Strategy based on similar historical trades
        if similar_trades:
            wins = [t for t in similar_trades if t.get("outcome") == "win"]
            losses = [t for t in similar_trades if t.get("outcome") == "loss"]
            win_rate = len(wins) / len(similar_trades) if similar_trades else 0
            
            if win_rate >= 0.6:
                thoughts.append(TradingThought(
                    id=3,
                    strategy=f"Enter based on historical success ({win_rate:.0%} win rate)",
                    reasoning=f"Found {len(similar_trades)} similar trades with {win_rate:.0%} success rate",
                    score=6.5 + win_rate * 3,
                    risk_level="low" if win_rate > 0.7 else "medium"
                ))
            else:
                thoughts.append(TradingThought(
                    id=3,
                    strategy="Caution - Similar trades had poor results",
                    reasoning=f"Historical win rate for similar setups: {win_rate:.0%}",
                    score=3.5,
                    risk_level="high"
                ))
        else:
            thoughts.append(TradingThought(
                id=3,
                strategy="New setup - No historical reference",
                reasoning="No similar trades found in memory",
                score=5.0,
                risk_level="medium"
            ))
        
        # Add scanner score influence if available
        if score is not None and len(thoughts) < self.num_thoughts:
            if score >= 70:
                thoughts[0].score += 1.0
                thoughts[0].reasoning += f" Scanner score: {score}/100 (high)"
            elif score < 40:
                thoughts[0].score -= 1.0
                thoughts[0].reasoning += f" Scanner score: {score}/100 (low)"
        
        return thoughts
    
    def _evaluate_thoughts(self, thoughts: List[TradingThought], 
                          technical: Dict) -> tuple:
        """Evaluate thoughts and select the best one."""
        
        if not thoughts:
            return 0, "No strategies available"
        
        # Find highest scoring thought
        best_idx = 0
        best_score = thoughts[0].score
        
        for i, thought in enumerate(thoughts):
            if thought.score > best_score:
                best_score = thought.score
                best_idx = i
        
        selected = thoughts[best_idx]
        
        # Build reasoning
        reasoning = f"Selected strategy {selected.id}: {selected.strategy}. "
        reasoning += f"Score: {selected.score:.1f}/10. "
        reasoning += f"Risk Level: {selected.risk_level}. "
        reasoning += selected.reasoning
        
        return best_idx, reasoning
    
    def _determine_action(self, thoughts: List[TradingThought], 
                         selected_idx: int, technical: Dict,
                         patterns: Dict, similar_trades: List) -> tuple:
        """
        Determine final action and confidence.
        V3.7: More conservative confidence calculation based on REAL data.
        """
        
        if not thoughts:
            return "SKIP", 0.3
        
        selected = thoughts[selected_idx]
        strategy = selected.strategy.upper()
        
        # Determine action from strategy text
        if "LONG" in strategy or "ENTER" in strategy and "BEARISH" not in strategy:
            action = "BUY"
        elif "SHORT" in strategy or "SELL" in strategy:
            action = "SELL"
        elif "SKIP" in strategy or "CAUTION" in strategy:
            action = "SKIP"
        elif "WAIT" in strategy:
            action = "HOLD"
        else:
            action = "HOLD"
        
        # ============= V3.7 REALISTIC CONFIDENCE =============
        # Start with base 50% - neutral
        base_confidence = 0.50
        confidence_factors = []
        
        # Factor 1: Technical Analysis (most important for day trading)
        tech_bias = technical.get("bias", "neutral")
        rsi = technical.get("rsi")
        
        if tech_bias == "bullish" and action == "BUY":
            base_confidence += 0.15
            confidence_factors.append("Tech bullish +15%")
        elif tech_bias == "bearish" and action == "BUY":
            base_confidence -= 0.20
            confidence_factors.append("Tech bearish -20%")
        
        # RSI extreme zones
        if rsi:
            if rsi < 30 and action == "BUY":
                base_confidence += 0.10
                confidence_factors.append(f"RSI oversold ({rsi:.0f}) +10%")
            elif rsi > 70 and action == "BUY":
                base_confidence -= 0.15
                confidence_factors.append(f"RSI overbought ({rsi:.0f}) -15%")
        
        # Factor 2: Historical trades (reduced weight - database is skewed)
        if similar_trades:
            wins = len([t for t in similar_trades if t.get("outcome") == "win"])
            total = len(similar_trades)
            win_rate = wins / total if total > 0 else 0.5
            
            # Only add small bonus, don't rely too much on skewed data
            if win_rate >= 0.8 and total >= 3:
                base_confidence += 0.05
                confidence_factors.append(f"History {win_rate:.0%} +5%")
            elif win_rate <= 0.4:
                base_confidence -= 0.10
                confidence_factors.append(f"History poor -10%")
        
        # Factor 3: Pattern alignment
        pattern_signal = patterns.get("dominant_signal", "neutral")
        if action == "BUY" and pattern_signal == "bullish":
            base_confidence += 0.05
            confidence_factors.append("Pattern bullish +5%")
        elif action == "BUY" and pattern_signal == "bearish":
            base_confidence -= 0.10
            confidence_factors.append("Pattern bearish -10%")
        
        # Factor 4: Data quality
        data_source = technical.get("data_source", "internal")
        if data_source == "Pro-Gemini":
            # Real data from Pro-Gemini is more reliable
            base_confidence += 0.05
            confidence_factors.append("Real data +5%")
        
        # Factor 5: SENTIMENT ANALYSIS (NEW! - biggest impact per research)
        sentiment = technical.get("sentiment")
        if sentiment:
            sent_score = sentiment.get("score", 0)
            sent_label = sentiment.get("label", "neutral")
            sent_conf = sentiment.get("confidence", 0)
            
            if action == "BUY":
                if sent_label == "bullish" and sent_conf > 0.5:
                    # Strong bullish sentiment = big confidence boost
                    boost = min(0.15, sent_score * 0.2)
                    base_confidence += boost
                    confidence_factors.append(f"Sentiment bullish +{boost*100:.0f}%")
                elif sent_label == "bearish" and sent_conf > 0.5:
                    # Bearish sentiment against BUY = reduce confidence
                    penalty = min(0.20, abs(sent_score) * 0.25)
                    base_confidence -= penalty
                    confidence_factors.append(f"Sentiment bearish -{penalty*100:.0f}%")
        
        # Clamp confidence to realistic range (35% - 90%)
        # Wider range to allow sentiment to have bigger impact
        confidence = max(0.35, min(0.90, base_confidence))
        
        print(f"[ORCHESTRATOR] Confidence calc: {' | '.join(confidence_factors)} = {confidence:.0%}")
        
        return action, round(confidence, 2)
    
    def _calculate_adjustments(self, action: str, confidence: float,
                              technical: Dict, atr: float,
                              similar_trades: List) -> Dict[str, float]:
        """Calculate position sizing and SL/TP adjustments."""
        
        adjustments = {
            "sl_multiplier": 1.0,
            "tp_multiplier": 1.5,
            "position_size": 1.0
        }
        
        if action in ["SKIP", "HOLD"]:
            adjustments["position_size"] = 0.0
            return adjustments
        
        # Adjust based on confidence
        if confidence >= 0.8:
            adjustments["position_size"] = 1.0
            adjustments["tp_multiplier"] = 2.0  # Larger target for high confidence
        elif confidence >= 0.6:
            adjustments["position_size"] = 0.75
            adjustments["tp_multiplier"] = 1.5
        else:
            adjustments["position_size"] = 0.5
            adjustments["tp_multiplier"] = 1.2
        
        # Adjust SL based on volatility
        atr_pct = technical.get("atr_percent", 0)
        if atr_pct > 5:  # High volatility
            adjustments["sl_multiplier"] = 1.5
        elif atr_pct < 2:  # Low volatility
            adjustments["sl_multiplier"] = 0.8
        
        # Adjust based on historical results
        if similar_trades:
            # If similar trades had large losses, widen SL
            losses = [t for t in similar_trades if t.get("outcome") == "loss"]
            if len(losses) > len(similar_trades) / 2:
                adjustments["sl_multiplier"] *= 1.2
                adjustments["position_size"] *= 0.8
        
        return adjustments
    
    def _assess_risk(self, symbol: str, price: float, atr: float,
                    technical: Dict, patterns: Dict) -> Dict[str, Any]:
        """Assess overall risk of the trade."""
        
        risk_factors = []
        risk_score = 50  # Base score
        
        # RSI risk
        rsi = technical.get("rsi", 50)
        if rsi > 80 or rsi < 20:
            risk_factors.append(f"RSI extreme ({rsi:.0f})")
            risk_score += 15
        
        # Volatility risk
        atr_pct = technical.get("atr_percent", 0)
        if atr_pct > 6:
            risk_factors.append(f"High volatility ({atr_pct:.1f}%)")
            risk_score += 20
        
        # Pattern risk
        if patterns.get("bearish_count", 0) > patterns.get("bullish_count", 0):
            risk_factors.append("Bearish patterns present")
            risk_score += 10
        
        # Trend risk
        if technical.get("trend") == "bearish":
            risk_factors.append("Against trend")
            risk_score += 15
        
        # Volume risk
        if technical.get("volume_signal") == "low":
            risk_factors.append("Low volume")
            risk_score += 10
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "HIGH"
        elif risk_score >= 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommended_max_loss": f"${50 * (100 / max(risk_score, 50)):.0f}",
            "notes": "Consider reducing position size for high-risk trades"
        }
    
    def record_result(self, trade_data: Dict[str, Any]):
        """Record a trade result for learning."""
        self.memory.store_trade(trade_data)


# ===== TESTING =====

if __name__ == "__main__":
    print("Testing Trading Orchestrator...")
    
    # Sample config
    config = {
        "technical": {
            "rsi_oversold": 30,
            "rsi_overbought": 70
        },
        "patterns": {
            "sensitivity": 0.5,
            "min_confidence": 0.6
        },
        "min_confidence": 0.5,
        "tot_strategies": 3
    }
    
    orchestrator = TradingOrchestrator(config)
    
    # Generate sample data
    import numpy as np
    np.random.seed(42)
    
    prices = [100 + i * 0.3 + np.random.randn() * 1.5 for i in range(60)]
    highs = [p + abs(np.random.randn()) for p in prices]
    lows = [p - abs(np.random.randn()) for p in prices]
    volumes = [1000000 + np.random.randint(-200000, 500000) for _ in prices]
    
    # Run analysis
    analysis = orchestrator.analyze(
        symbol="TSLA",
        price=prices[-1],
        atr=3.5,
        score=78,
        signals=["MA_CROSS", "VWAP", "VOLUME"],
        context="Gap up 4.2%, RVOL 2.8x",
        prices=prices,
        highs=highs,
        lows=lows,
        volumes=volumes
    )
    
    print("\n" + "="*60)
    print("ANALYSIS RESULT")
    print("="*60)
    print(f"\nSymbol: {analysis.symbol}")
    print(f"Action: {analysis.action}")
    print(f"Confidence: {analysis.confidence:.0%}")
    
    print(f"\n--- Thoughts (ToT) ---")
    for thought in analysis.thoughts:
        print(f"  [{thought.id}] {thought.strategy} (Score: {thought.score:.1f})")
    
    print(f"\nSelected: Thought {analysis.selected_thought + 1}")
    print(f"Reasoning: {analysis.reasoning}")
    
    print(f"\n--- Adjustments ---")
    print(f"  SL Multiplier: {analysis.adjustments['sl_multiplier']:.2f}x")
    print(f"  TP Multiplier: {analysis.adjustments['tp_multiplier']:.2f}x")
    print(f"  Position Size: {analysis.adjustments['position_size']:.0%}")
    
    print(f"\n--- Risk Assessment ---")
    print(f"  Risk Level: {analysis.risk_assessment['risk_level']}")
    print(f"  Risk Score: {analysis.risk_assessment['risk_score']}")
    print(f"  Risk Factors: {analysis.risk_assessment['risk_factors']}")
    
    print("\n" + "="*60)
    
    # Test JSON output
    print("\n--- JSON Output (partial) ---")
    result = analysis.to_dict()
    print(json.dumps({
        "action": result["action"],
        "confidence": result["confidence"],
        "adjustments": result["adjustments"]
    }, indent=2))

