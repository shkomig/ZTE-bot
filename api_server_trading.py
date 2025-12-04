"""
Zero Trading Expert (ZTE) - API Server
======================================
FastAPI server for trading analysis on port 5001.

Endpoints:
    POST /api/analyze         - Main analysis endpoint
    POST /api/memory/trade    - Store trade results
    POST /api/knowledge/add   - Add knowledge manually
    POST /api/knowledge/pdf   - Upload PDF
    GET  /api/memory/stats    - Memory statistics
    GET  /api/similar-trades  - Find similar past trades
    GET  /api/health          - Health check
    GET  /docs                - API documentation

Usage:
    python api_server_trading.py
    
    Or via start_zte.bat
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio
import os

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import ZTE components
from CORE_TRADING.trading_memory import TradingMemory
from CORE_TRADING.market_analyzer import MarketAnalyzer
from CORE_TRADING.pattern_detector import PatternDetector
from CORE_TRADING.trading_orchestrator import TradingOrchestrator
from CORE_TRADING.live_performance import LivePerformanceTracker

# Load config
try:
    import yaml
    with open(Path(__file__).parent / "config.yaml", 'r') as f:
        CONFIG = yaml.safe_load(f)
except:
    CONFIG = {
        "server": {"host": "0.0.0.0", "port": 5001},
        "models": {"primary": "zero-trading-expert", "fallback": "llama3.1:8b"},
        "analysis": {"min_confidence": 0.5, "tot_strategies": 3}
    }


# ==================== Pydantic Models ====================

class AnalyzeRequest(BaseModel):
    """Request model for /api/analyze"""
    symbol: str = Field(..., description="Stock symbol (e.g., 'TSLA')")
    price: float = Field(..., description="Current price")
    atr: Optional[float] = Field(None, description="Average True Range")
    score: Optional[int] = Field(None, description="Scanner score (0-100)")
    signals: Optional[List[str]] = Field(default=[], description="Triggered signals")
    context: Optional[str] = Field(None, description="Additional context")
    technical: Optional[Dict[str, Any]] = Field(default=None, description="Technical indicators from Pro-Gemini")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "TSLA",
                "price": 245.50,
                "atr": 3.2,
                "score": 78,
                "signals": ["MA_CROSS", "VWAP", "VOLUME"],
                "context": "Gap up 4.2%, RVOL 3.5x",
                "technical": {"rsi": 65, "sma_50": 240.0, "sma_200": 220.0}
            }
        }


class AnalyzeResponse(BaseModel):
    """Response model for /api/analyze"""
    symbol: str
    action: str  # BUY, SELL, HOLD, SKIP
    confidence: float
    thoughts: List[Dict]
    selected: int
    reasoning: str
    adjustments: Dict[str, float]
    risk: Dict[str, Any]
    similar_trades: List[Dict]
    timestamp: str


class TradeResultRequest(BaseModel):
    """Request model for /api/memory/trade"""
    symbol: str
    entry_price: float
    exit_price: Optional[float] = None  # None for pending trades
    profit_pct: Optional[float] = 0
    strategy: str
    outcome: Optional[str] = "pending"  # pending, win, loss
    signals: Optional[List[str]] = []
    atr: Optional[float] = 0
    score: Optional[int] = 0
    context: Optional[str] = ""


class KnowledgeRequest(BaseModel):
    """Request model for /api/knowledge/add"""
    topic: str
    content: str
    category: Optional[str] = "general"


class SimilarTradesRequest(BaseModel):
    """Request model for /api/similar-trades"""
    query: str
    n_results: Optional[int] = 5


# ==================== FastAPI App ====================

app = FastAPI(
    title="Zero Trading Expert (ZTE)",
    description="AI-powered trading analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
memory: TradingMemory = None
orchestrator: TradingOrchestrator = None


# ==================== Lifecycle ====================

@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    global memory, orchestrator
    
    print("\n" + "="*60)
    print("Zero Trading Expert (ZTE) - Starting...")
    print("="*60)
    
    # Initialize memory
    memory = TradingMemory()
    
    # Initialize orchestrator with config (including sentiment!)
    orchestrator = TradingOrchestrator({
        "technical": CONFIG.get("technical", {}),
        "patterns": CONFIG.get("patterns", {}),
        "sentiment": CONFIG.get("sentiment", {}),  # <-- Finnhub API Key!
        "min_confidence": CONFIG.get("analysis", {}).get("min_confidence", 0.5),
        "tot_strategies": CONFIG.get("analysis", {}).get("tot_strategies", 3),
        "model": CONFIG.get("models", {}).get("primary", "llama3.1:8b")
    })
    
    print(f"\n[ZTE] Server ready on port {CONFIG.get('server', {}).get('port', 5001)}")
    print(f"[ZTE] Memory stats: {memory.get_stats()}")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    print("\n[ZTE] Shutting down...")


# ==================== Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint with system info."""
    return {
        "system": "Zero Trading Expert (ZTE)",
        "version": "1.0.0",
        "status": "running",
        "port": CONFIG.get("server", {}).get("port", 5001),
        "endpoints": {
            "analyze": "POST /api/analyze",
            "memory_trade": "POST /api/memory/trade",
            "knowledge_add": "POST /api/knowledge/add",
            "memory_stats": "GET /api/memory/stats",
            "similar_trades": "POST /api/similar-trades",
            "health": "GET /api/health",
            "docs": "GET /docs"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_initialized": memory is not None,
        "orchestrator_initialized": orchestrator is not None
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_trade(request: AnalyzeRequest):
    """
    Main analysis endpoint.
    Analyzes a potential trade and returns recommendation with confidence.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # Run analysis with technical data from Pro-Gemini
        analysis = orchestrator.analyze(
            symbol=request.symbol,
            price=request.price,
            atr=request.atr,
            score=request.score,
            signals=request.signals or [],
            context=request.context,
            technical_data=request.technical  # NEW: Pass technical indicators
        )
        
        # Convert to response
        return AnalyzeResponse(
            symbol=analysis.symbol,
            action=analysis.action,
            confidence=analysis.confidence,
            thoughts=[t.to_dict() for t in analysis.thoughts],
            selected=analysis.selected_thought,
            reasoning=analysis.reasoning,
            adjustments=analysis.adjustments,
            risk=analysis.risk_assessment,
            similar_trades=analysis.similar_trades[:3],
            timestamp=analysis.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/memory/trade")
async def store_trade_result(request: TradeResultRequest):
    """
    Store a trade result for learning.
    Call this after a trade closes to build historical knowledge.
    Also accepts pending trades for tracking.
    """
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    
    try:
        # Determine outcome
        outcome = request.outcome or "pending"
        if request.exit_price is not None and request.profit_pct:
            outcome = "win" if request.profit_pct > 0 else "loss"
        
        trade_data = {
            "symbol": request.symbol,
            "entry_price": request.entry_price,
            "exit_price": request.exit_price,
            "profit_pct": request.profit_pct or 0,
            "strategy": request.strategy,
            "outcome": outcome,
            "signals": request.signals,
            "atr": request.atr,
            "score": request.score,
            "context": request.context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Only store completed trades in memory for learning
        if outcome != "pending":
            doc_id = memory.store_trade(trade_data)
        else:
            doc_id = f"pending_{request.symbol}_{datetime.now().strftime('%H%M%S')}"
            print(f"[ZTE] Tracking pending trade: {request.symbol}")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "outcome": outcome,
            "message": f"Trade {'tracked' if outcome == 'pending' else 'stored'}: {request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store trade: {str(e)}")


@app.post("/api/knowledge/add")
async def add_knowledge(request: KnowledgeRequest):
    """
    Add knowledge to the system manually.
    """
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    
    try:
        memory.store_knowledge(
            topic=request.topic,
            content=request.content,
            category=request.category
        )
        
        return {
            "success": True,
            "topic": request.topic,
            "category": request.category,
            "message": "Knowledge stored successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add knowledge: {str(e)}")


@app.post("/api/knowledge/pdf")
async def upload_pdf(file: UploadFile = File(...), category: str = Form(None)):
    """
    Upload and process a PDF document.
    """
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    
    try:
        # Save temporarily
        temp_path = Path(__file__).parent / "MEMORY" / f"temp_{file.filename}"
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Import and process
        from TOOLS.pdf_loader import PDFLoader
        loader = PDFLoader(memory)
        result = loader.load_pdf(str(temp_path), category)
        
        # Cleanup
        temp_path.unlink()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.get("/api/memory/stats")
async def get_memory_stats():
    """
    Get memory statistics.
    ⚠️ WARNING: The win_rate shown here is from IMPORTED DATA, not real trading!
    """
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    
    stats = memory.get_stats()
    win_rate = memory.get_win_rate()
    
    return {
        "collections": stats,
        "win_rate": f"{win_rate:.1%}",
        "win_rate_warning": "⚠️ This is imported data ratio, NOT real trading performance!",
        "total_trades": stats.get("successful_trades", 0) + stats.get("failed_trades", 0),
        "total_knowledge": stats.get("technical_knowledge", 0)
    }


@app.get("/api/live-performance")
async def get_live_performance():
    """
    Get REAL live trading performance statistics.
    This tracks actual trades executed by the bot.
    """
    try:
        tracker = LivePerformanceTracker()
        stats = tracker.get_performance_stats()
        daily = tracker.get_daily_stats()
        sector = tracker.get_sector_performance()
        signal = tracker.get_signal_performance()
        
        return {
            "overall": stats,
            "today": daily.get(datetime.now().strftime("%Y-%m-%d"), {}),
            "by_date": daily,
            "by_sector": sector,
            "by_signal": signal,
            "note": "This is REAL trading performance from live execution"
        }
    except Exception as e:
        return {
            "overall": {"error": str(e)},
            "note": "Live performance tracking not yet initialized"
        }


@app.post("/api/similar-trades")
async def find_similar_trades(request: SimilarTradesRequest):
    """
    Find similar past trades based on query.
    """
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    
    try:
        trades = memory.find_similar_trades(
            query=request.query,
            n_results=request.n_results
        )
        
        return {
            "query": request.query,
            "count": len(trades),
            "trades": trades
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/patterns")
async def get_patterns(symbol: str = None):
    """
    Get stored patterns (optionally filtered by symbol).
    """
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    
    query = symbol or "trading pattern"
    patterns = memory.find_patterns(query, n_results=10)
    
    return {
        "query": query,
        "count": len(patterns),
        "patterns": patterns
    }


@app.get("/api/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """
    Get sentiment analysis for a stock symbol.
    Uses news from Finnhub API and keyword/FinBERT analysis.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if not hasattr(orchestrator, 'sentiment_agent') or orchestrator.sentiment_agent is None:
        raise HTTPException(status_code=503, detail="Sentiment Agent not available")
    
    try:
        result = orchestrator.sentiment_agent.analyze(symbol.upper())
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@app.get("/api/sentiment/market/overview")
async def get_market_sentiment():
    """
    Get overall market sentiment from major indices.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if not hasattr(orchestrator, 'sentiment_agent') or orchestrator.sentiment_agent is None:
        raise HTTPException(status_code=503, detail="Sentiment Agent not available")
    
    try:
        return orchestrator.sentiment_agent.get_market_sentiment()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market sentiment failed: {str(e)}")


# ==================== Main ====================

def main():
    """Run the API server."""
    host = CONFIG.get("server", {}).get("host", "0.0.0.0")
    port = CONFIG.get("server", {}).get("port", 5001)
    debug = CONFIG.get("server", {}).get("debug", False)
    
    print(f"\nStarting ZTE API Server on {host}:{port}")
    print(f"Documentation: http://localhost:{port}/docs\n")
    
    uvicorn.run(
        "api_server_trading:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()

