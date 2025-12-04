"""
Sentiment Agent for Zero Trading Expert
========================================
Analyzes news and social sentiment for stocks.

Features:
- News fetching from Finnhub API (free tier)
- Basic sentiment scoring
- FinBERT integration (optional)
- Caching to reduce API calls

Based on research showing +921% improvement with sentiment analysis!
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    symbol: str
    score: float  # -1 (very bearish) to +1 (very bullish)
    label: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    news_count: int
    headlines: List[str]
    source: str  # "finnhub", "finbert", "keywords"
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "news_count": self.news_count,
            "headlines": self.headlines[:3],  # Top 3
            "source": self.source,
            "timestamp": self.timestamp
        }


class SentimentAgent:
    """
    Sentiment Analysis Agent.
    Uses multiple sources for comprehensive sentiment scoring.
    """
    
    # Keyword-based sentiment (fallback)
    BULLISH_KEYWORDS = [
        'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'up', 'high', 'record',
        'beat', 'exceed', 'strong', 'growth', 'profit', 'upgrade', 'buy', 'bullish',
        'breakthrough', 'success', 'positive', 'optimistic', 'boom', 'outperform',
        'exceed expectations', 'revenue growth', 'earnings beat', 'new high'
    ]
    
    BEARISH_KEYWORDS = [
        'drop', 'fall', 'plunge', 'crash', 'decline', 'down', 'low', 'loss',
        'miss', 'weak', 'sell', 'bearish', 'warning', 'concern', 'risk', 'cut',
        'downgrade', 'negative', 'pessimistic', 'slump', 'underperform', 'layoff',
        'bankruptcy', 'lawsuit', 'investigation', 'recall', 'miss expectations'
    ]
    
    def __init__(self, config: Dict = None):
        """
        Initialize Sentiment Agent.
        
        Args:
            config: Configuration with API keys
        """
        self.config = config or {}
        
        # Finnhub API (free tier: 60 calls/minute)
        self.finnhub_key = self.config.get("finnhub_api_key") or os.getenv("FINNHUB_API_KEY")
        
        # Cache for API responses (reduce calls)
        self._cache = {}
        self._cache_duration = timedelta(minutes=15)
        
        # Try to load FinBERT model
        self.finbert_model = None
        self.finbert_tokenizer = None
        self._init_finbert()
        
        print(f"[SENTIMENT] Agent initialized")
        print(f"[SENTIMENT] Finnhub API: {'Yes' if self.finnhub_key else 'No'}")
        print(f"[SENTIMENT] FinBERT: {'Yes' if self.finbert_model else 'No (using keywords)'}")
    
    def _init_finbert(self):
        """Try to initialize FinBERT model."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Try to load FinBERT (lightweight sentiment model)
            model_name = "ProsusAI/finbert"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=-1  # CPU
            )
        except Exception as e:
            print(f"[SENTIMENT] FinBERT not available: {e}")
            self.finbert_model = None
    
    def _get_cache_key(self, symbol: str) -> str:
        """Generate cache key."""
        return f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
    
    def _is_cached(self, symbol: str) -> bool:
        """Check if result is cached and valid."""
        key = self._get_cache_key(symbol)
        if key in self._cache:
            cached_time = self._cache[key].get("_cached_at")
            if cached_time and datetime.now() - cached_time < self._cache_duration:
                return True
        return False
    
    def _get_cached(self, symbol: str) -> Optional[SentimentResult]:
        """Get cached result."""
        key = self._get_cache_key(symbol)
        if key in self._cache:
            data = self._cache[key]
            return SentimentResult(**{k: v for k, v in data.items() if k != "_cached_at"})
        return None
    
    def _set_cache(self, symbol: str, result: SentimentResult):
        """Cache result."""
        key = self._get_cache_key(symbol)
        data = result.to_dict()
        data["_cached_at"] = datetime.now()
        self._cache[key] = data
    
    def fetch_news_finnhub(self, symbol: str) -> List[Dict]:
        """
        Fetch news from Finnhub API.
        
        Free tier: 60 calls/minute
        Returns: List of news articles
        """
        if not self.finnhub_key or not REQUESTS_AVAILABLE:
            return []
        
        try:
            # Get news from last 7 days
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "token": self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                news = response.json()
                return news[:20]  # Limit to 20 articles
            else:
                print(f"[SENTIMENT] Finnhub error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[SENTIMENT] Finnhub fetch error: {e}")
            return []
    
    def analyze_with_finbert(self, texts: List[str]) -> List[Dict]:
        """
        Analyze texts with FinBERT model.
        
        Returns: List of {label, score} dicts
        """
        if not self.finbert_model or not texts:
            return []
        
        try:
            results = self.finbert_pipeline(texts[:10])  # Limit for speed
            return results
        except Exception as e:
            print(f"[SENTIMENT] FinBERT error: {e}")
            return []
    
    def analyze_with_keywords(self, text: str) -> Tuple[float, str]:
        """
        Simple keyword-based sentiment analysis.
        
        Returns: (score, label)
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, "neutral"
        
        score = (bullish_count - bearish_count) / total
        
        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"
        
        return score, label
    
    def analyze(self, symbol: str) -> SentimentResult:
        """
        Perform full sentiment analysis for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            
        Returns:
            SentimentResult with score, label, confidence
        """
        # Check cache first
        if self._is_cached(symbol):
            cached = self._get_cached(symbol)
            if cached:
                print(f"[SENTIMENT] {symbol}: Using cached result")
                return cached
        
        print(f"[SENTIMENT] Analyzing {symbol}...")
        
        # Fetch news
        news = self.fetch_news_finnhub(symbol)
        headlines = [article.get("headline", "") for article in news if article.get("headline")]
        
        if not headlines:
            # No news available - return neutral
            result = SentimentResult(
                symbol=symbol,
                score=0.0,
                label="neutral",
                confidence=0.3,
                news_count=0,
                headlines=[],
                source="no_data",
                timestamp=datetime.now().isoformat()
            )
            return result
        
        # Analyze sentiment
        scores = []
        source = "keywords"
        
        if self.finbert_model:
            # Use FinBERT
            finbert_results = self.analyze_with_finbert(headlines)
            for r in finbert_results:
                label = r.get("label", "neutral").lower()
                conf = r.get("score", 0.5)
                
                if label == "positive":
                    scores.append(conf)
                elif label == "negative":
                    scores.append(-conf)
                else:
                    scores.append(0)
            source = "finbert"
        else:
            # Use keyword analysis
            for headline in headlines:
                score, _ = self.analyze_with_keywords(headline)
                scores.append(score)
            source = "keywords"
        
        # Calculate aggregate score
        if scores:
            avg_score = sum(scores) / len(scores)
            
            # Determine label
            if avg_score > 0.15:
                label = "bullish"
            elif avg_score < -0.15:
                label = "bearish"
            else:
                label = "neutral"
            
            # Confidence based on consistency
            if len(scores) >= 3:
                same_sign = sum(1 for s in scores if (s > 0) == (avg_score > 0))
                confidence = same_sign / len(scores)
            else:
                confidence = 0.5
        else:
            avg_score = 0.0
            label = "neutral"
            confidence = 0.3
        
        result = SentimentResult(
            symbol=symbol,
            score=round(avg_score, 3),
            label=label,
            confidence=round(confidence, 2),
            news_count=len(headlines),
            headlines=headlines[:5],
            source=source,
            timestamp=datetime.now().isoformat()
        )
        
        # Cache result
        self._set_cache(symbol, result)
        
        print(f"[SENTIMENT] {symbol}: {label} ({avg_score:+.2f}) from {len(headlines)} articles")
        
        return result
    
    def analyze_batch(self, symbols: List[str]) -> Dict[str, SentimentResult]:
        """
        Analyze multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbol to SentimentResult
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.analyze(symbol)
        return results
    
    def get_market_sentiment(self) -> Dict:
        """
        Get overall market sentiment from major indices/ETFs.
        
        Returns:
            Dict with market sentiment indicators
        """
        market_symbols = ["SPY", "QQQ", "DIA", "IWM"]
        sentiments = []
        
        for symbol in market_symbols:
            result = self.analyze(symbol)
            sentiments.append(result.score)
        
        if sentiments:
            avg = sum(sentiments) / len(sentiments)
            if avg > 0.1:
                market_mood = "risk_on"
            elif avg < -0.1:
                market_mood = "risk_off"
            else:
                market_mood = "neutral"
        else:
            avg = 0
            market_mood = "unknown"
        
        return {
            "market_score": round(avg, 3),
            "market_mood": market_mood,
            "timestamp": datetime.now().isoformat()
        }


# Test function
if __name__ == "__main__":
    # Test the sentiment agent
    agent = SentimentAgent()
    
    # Test single stock
    result = agent.analyze("AAPL")
    print(f"\nResult: {result.to_dict()}")
    
    # Test market sentiment
    market = agent.get_market_sentiment()
    print(f"\nMarket: {market}")

