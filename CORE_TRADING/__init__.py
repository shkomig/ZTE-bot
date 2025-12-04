"""
CORE_TRADING - Zero Trading Expert Core Components
===================================================
Trading-specialized components that extend Zero Agent's capabilities.

Components:
- trading_memory.py: Extended RAG for trading data
- trading_orchestrator.py: Main analysis engine
- market_analyzer.py: Technical analysis
- pattern_detector.py: Chart pattern recognition
- sentiment_agent.py: News & sentiment analysis (NEW!)
"""

from pathlib import Path

# Add Zero Agent's CORE to path for imports
import sys
ZERO_CORE_PATH = Path(__file__).parent.parent.parent / "ZERO" / "CORE"
if ZERO_CORE_PATH.exists():
    sys.path.insert(0, str(ZERO_CORE_PATH.parent))

__version__ = "1.1.0"  # Updated for sentiment agent
__all__ = [
    "TradingMemory",
    "TradingOrchestrator", 
    "MarketAnalyzer",
    "PatternDetector",
    "SentimentAgent"
]

