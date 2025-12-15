# 📈 Zero Trading Expert (ZTE) - Master Control Protocol

**Last Updated:** December 15, 2025
**Current Version:** V3.7.7 (Hardening Phase - Quality over Quantity)
**Status:** 🟢 OPERATIONAL - Process Supervision + Hardened Filters
**API Port:** 5002 ✅ MIGRATED (Docker conflict resolved)
**TWS Port:** 7497 (Paper Trading)

### 📊 Current Status:
| Metric | Value | Notes |
|--------|-------|-------|
| **Process Supervisor** | ✅ launcher.py | Auto-restart on crashes |
| **RAG Memory** | Disabled | ChromaDB compatibility issue |
| **Sentiment Engine** | **FinBERT** | Fully integrated |
| **Max Positions** | **5** | V3.7.7: Reduced from 10 for quality |
| **Min Confidence** | **75%** | V3.7.7: Hardened from 50% |
| **Min Phase1 Signals** | **3** | V3.7.7: Hardened from 2 |

---

## 🛡️ **Current System Configuration (V3.7.7 - HARDENED)**

### Critical Safety Features:

| Feature | Status | Description |
|---------|--------|-------------|
| **Process Supervisor** | ✅ NEW | launcher.py monitors & auto-restarts both processes |
| **API Health Check** | ✅ ACTIVE | Checks API availability before each trade cycle |
| **Connection Recovery** | ✅ ACTIVE | Auto-reconnects to TWS every 5 minutes |
| **Duplicate Order Prevention** | ✅ ACTIVE | Checks pending orders before entry & exit |
| **Port Configuration** | ✅ MIGRATED | Bot → 5002, API → 5002 (Docker conflict resolved) |
| **Stop-Loss** | ✅ WIDENED | 2.0% (was 1.25% - gives quality trades room) |
| **Take-Profit** | ✅ FIXED | 2.5% (maintains 1:1.25 R:R) |

### Risk Management Settings (V3.7.7 - HARDENED):

```python
# CURRENT SETTINGS (V3.7.7 - Quality over Quantity)
API_URL = "http://127.0.0.1:5002"      # Port 5002: Docker conflict resolved
MAX_SL_PCT = 0.02                      # 2.0% Stop Loss (wider for quality trades)
MAX_TP_PCT = 0.025                     # 2.5% Take Profit (R:R 1:1.25)
MAX_POSITION_VALUE = 5000              # Max $ per position
MIN_CONFIDENCE = 0.75                  # 75% minimum confidence (was 50%)
MIN_PHASE1_SIGNALS = 3                 # Minimum 3 Phase1 signals (was 2)
MAX_OPEN_POSITIONS = 5                 # 5 positions max (was 10)
MAX_DAILY_LOSS_PCT = 0.03              # 3% max daily loss
MAX_TRADES_PER_DAY = 20                # Maximum trades per day
MIN_RVOL = 1.5                         # Minimum 1.5x relative volume
```

### Position System (V3.7.7 - Simplified):

**All 5 positions now use same strict requirements:**

| Requirement | Value | Notes |
|-------------|-------|-------|
| **Confidence** | ≥75% | Hardened from 50% |
| **RVOL** | ≥1.5x | Minimum relative volume |
| **Phase1 Signals** | ≥3 | Hardened from 2 |
| **Max Per Sector** | 2 | Diversification required |

**No more tiers - focus on quality over quantity!**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ZERO TRADING EXPERT (ZTE) - V3.7.5                    │
│                         PORT: 5002 ✅ (MIGRATED)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐         │
│  │  Zero CORE     │    │ Trading Brain  │    │ Trading Memory │         │
│  │  (fallback)    │    │                │    │  (Disabled)    │         │
│  │                │    │                │    │                │         │
│  │  • ToT(builtin)│◄──►│ • Analyzer     │◄──►│ • Patterns     │         │
│  │  • Keywords    │    │ • Patterns     │    │ • Trades       │         │
│  │  • Router      │    │ • Risk Calc    │    │ • Knowledge    │         │
│  └────────────────┘    └────────────────┘    └────────────────┘         │
│           │                    │                     │                   │
│           │            ┌──────────────┐              │                   │
│           │            │  SENTIMENT   │◄─── Finnhub API (FinBERT)       │
│           │            │    AGENT     │     Real-time news              │
│           │            └──────────────┘                                  │
│           │                    │                     │                   │
│           └────────────────────┼─────────────────────┘                   │
│                                ▼                                         │
│                    ┌────────────────────┐                               │
│                    │  API Server        │                               │
│                    │  FastAPI :5002     │◄─── ✅ HEALTH CHECK           │
│                    └────────────────────┘                               │
│                                │                                         │
└────────────────────────────────┼─────────────────────────────────────────┘
                                 │ REST API
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUTO TRADER TWS (V3.7.4)                             │
│                                                                          │
│  Scanner ──► Phase1 ──► [Health Check] ──► [Position Check] ──► Trade   │
│              Analysis      ✅ NEW!           ✅ NEW!                     │
│              (RSI Div,                                                   │
│               TSI, BB%,                                                  │
│               VPOC, MACD)                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Safety Mechanisms:

#### 1. API Health Check
```python
def check_api_health(self) -> bool:
    """Check if API server is responsive (cached for 30s)."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False
```

#### 2. TWS Position Verification
```python
# CRITICAL: Check actual TWS positions before orders
tws_positions = self.ib.positions()
for pos in tws_positions:
    if pos.contract.symbol == symbol and abs(pos.position) > 0:
        return False, f"Already in position (TWS: {int(pos.position)} shares)"
```

#### 3. Process Supervisor (V3.7.7 - NEW)
```python
# launcher.py - Monitors both API Server and Trading Bot
# Auto-restarts either process if it crashes
# Graceful shutdown on Ctrl+C

PROCESSES = {
    'api_server': {
        'command': ['python', 'api_server_trading.py'],
        'startup_delay': 5,  # Wait for API to be ready
        'restart_delay': 3
    },
    'trader_bot': {
        'command': ['python', 'auto_trader_tws.py'],
        'startup_delay': 0,
        'restart_delay': 3
    }
}
```

**Usage:**
```bash
# Start entire system with supervision
python launcher.py

# Press Ctrl+C to shutdown gracefully
```

**Features:**
- Starts API Server first, waits 5 seconds for initialization
- Starts Trading Bot after API is ready
- Monitors both processes every 10 seconds
- Auto-restarts crashed processes (with 3s delay)
- Logs all restart events
- Status report every 5 minutes
- Graceful shutdown: stops Bot first (prevent new trades), then API

---

## 📁 File Structure

```
C:\AI-ALL-PRO\ZERO-TRADING-EXPERT\
├── MCP.md                          # This document - CURRENT STATE ONLY
├── INCIDENTS.md                    # Historical incident reports (SEE THIS FOR HISTORY)
├── update.md                       # Development roadmap & active tasks
├── launcher.py                     # ⭐ V3.7.7: Process Supervisor (NEW!)
├── api_server_trading.py           # API Server (Port 5002)
├── auto_trader_tws.py              # Trading Bot (V3.7.7 - Hardened)
├── close_excess_positions.py       # Emergency cleanup script
├── config.yaml                     # Configuration
├── requirements.txt                # Dependencies
│
├── CORE_TRADING\
│   ├── trading_orchestrator.py     # Main orchestrator
│   ├── market_analyzer.py          # Phase 1 Technical Analysis
│   ├── pattern_detector.py         # Pattern detection
│   ├── sentiment_agent.py          # Sentiment analysis (FinBERT)
│   ├── trading_memory.py           # RAG for trading
│   └── live_performance.py         # Real performance tracking
│
├── TOOLS\
│   ├── pdf_loader.py               # Load PDFs
│   ├── trade_log_importer.py       # Import trades
│   └── tws_data_fetcher.py         # Market data from TWS
│
└── MEMORY\
    └── chroma_trading_db\          # ChromaDB (54 items - clean)
```

---

## 🎯 Phase 1 Technical Indicators (ACTIVE)

All Phase 1 indicators are integrated and operational:

| Indicator | Status | Description | Win Rate |
|-----------|--------|-------------|----------|
| **RSI Divergence** | ✅ ACTIVE | Detects bullish/bearish divergence | 85-86% |
| **TSI** | ✅ ACTIVE | True Strength Index (momentum) | High |
| **Bollinger %B** | ✅ ACTIVE | Oversold/Overbought detection | High |
| **Volume Profile** | ✅ ACTIVE | VPOC, VAL, VAH support/resistance | High |
| **MACD Divergence** | ✅ ACTIVE | Histogram divergence detection | High |

### Phase 1 Signals Generated:
- 🟢 RSI_BULLISH_DIVERGENCE
- 🔴 RSI_BEARISH_DIVERGENCE
- 🟢 TSI_OVERSOLD / TSI_BULLISH
- 🔴 TSI_OVERBOUGHT / TSI_BEARISH
- 🟢 BB_OVERSOLD
- 🔴 BB_OVERBOUGHT
- ⚡ AT_VPOC
- 🟢 AT_VAL_SUPPORT
- 🔴 AT_VAH_RESISTANCE
- 🟢 MACD_BULLISH_DIVERGENCE
- 🔴 MACD_BEARISH_DIVERGENCE

---

## 🔌 API Reference

### Base URL: `http://127.0.0.1:5002/api`

### GET /api/health
Health check endpoint.
```json
{
  "status": "healthy",
  "timestamp": "2025-12-09T...",
  "memory_initialized": true,
  "orchestrator_initialized": true
}
```

### POST /api/analyze
Main analysis endpoint with sentiment and Phase 1 indicators.

**Request:**
```json
{
  "symbol": "TSLA",
  "price": 245.50,
  "atr": 3.2,
  "score": 78,
  "signals": ["MA_CROSS", "VWAP"],
  "context": "Gap up 4.2%, RVOL 3.5x",
  "technical": {
    "rsi": 65,
    "sma_50": 240.0,
    "sma_200": 220.0
  }
}
```

**Response:**
```json
{
  "symbol": "TSLA",
  "action": "BUY",  // or STRONG_BUY, SELL, STRONG_SELL, HOLD, SKIP
  "confidence": 0.72,
  "reasoning": "Strong momentum with bullish sentiment...",
  "adjustments": {
    "sl_multiplier": 1.0,
    "tp_multiplier": 2.0,
    "position_size": 1.0
  },
  "risk_assessment": {
    "risk_level": "MEDIUM",
    "risk_score": 55
  },
  "similar_trades": [...]
}
```

### GET /api/sentiment/{symbol}
Sentiment analysis for a symbol.

### GET /api/live-performance
Real trading performance statistics.

### POST /api/memory/trade
Store completed trade for learning.

### GET /api/memory/stats
RAG memory statistics.

---

## 📊 Premium Watchlist (42 Stocks)

### 7 Sectors:
```python
SECTOR_MAP = {
    'TECH': ['NVDA', 'AMD', 'GOOGL', 'AMZN', 'META', 'MSFT', 'AAPL', 'TSLA'],
    'SEMI': ['AVGO', 'QCOM', 'MU', 'INTC', 'ARM', 'MRVL', 'AMAT', 'LRCX'],
    'SOFTWARE': ['CRM', 'PLTR', 'SNOW', 'NET', 'DDOG', 'ZS', 'CRWD', 'PANW'],
    'FINANCE': ['JPM', 'GS', 'V', 'MA', 'BAC', 'MS'],
    'CONSUMER': ['NKE', 'SBUX', 'HD', 'WMT', 'COST'],
    'HEALTH': ['UNH', 'LLY', 'MRNA', 'ISRG'],
    'ETF': ['SPY', 'QQQ', 'IWM']
}
```

**Sector Diversification:** Max 2 positions per sector

---

## 🗺️ Development Roadmap

See `update.md` for detailed roadmap and active tasks.

### Current Focus:
- ✅ **Phase 1 Complete:** Technical Indicators (RSI Div, TSI, BB%, VPOC, MACD)
- ✅ **System Stabilized:** All critical safety fixes applied
- 🔄 **Baseline Collection:** Gathering 50-100 real trades for performance baseline
- ⏳ **Phase 2 Pending:** Multi-Strategy Engine (planned Jan 2026)

### Phase Roadmap:
| Phase | Description | Status | Target Date |
|-------|-------------|--------|-------------|
| Phase 1 | Technical Indicators | ✅ Complete | Dec 2025 |
| Phase 2 | Multi-Strategy Engine | ⏳ Pending | Jan 2026 |
| Phase 3 | ML/AI Predictions | ⏳ Pending | Feb 2026 |
| Phase 4 | Multi-Agent Architecture | ⏳ Pending | Mar 2026 |

---

## ⚙️ Configuration (config.yaml)

```yaml
server:
  host: "0.0.0.0"
  port: 5002  # Migrated from 5001 (Docker conflict)
  debug: false

models:
  primary: "zero-trading-expert"
  fallback: "llama3.1:8b"
  ollama_url: "http://localhost:11434"

analysis:
  min_confidence: 0.5
  tot_strategies: 3
  timeout: 30

risk:
  max_sl_pct: 0.0125  # 1.25% FIXED
  max_tp_pct: 0.025   # 2.5% FIXED
  max_daily_loss_pct: 0.03
  max_trades_per_day: 20

memory:
  chroma_path: "./MEMORY/chroma_trading_db"
  max_similar_trades: 5

sentiment:
  finnhub_api_key: "YOUR_KEY_HERE"
  cache_minutes: 15
  max_news_items: 5

technical:
  rsi_oversold: 30
  rsi_overbought: 70
  macd_fast: 12
  macd_slow: 26
  bollinger_period: 20
```

---

## 🚀 Quick Start (V3.7.7 - RECOMMENDED)

### Prerequisites:
1. TWS or IB Gateway running on port 7497 (Paper Trading)
2. Virtual environment activated (if using venv)

### Launch Entire System (RECOMMENDED):
```bash
# V3.7.7: Use the process supervisor (auto-restarts on crashes)
python launcher.py

# Press Ctrl+C to shutdown gracefully
```

**The launcher will:**
- Start API Server first (port 5002)
- Wait 5 seconds for API to initialize
- Start Trading Bot
- Monitor both processes and auto-restart on crashes
- Log all restart events

### Manual Launch (Legacy - Not Recommended):
```bash
# Terminal 1: Start API Server
python api_server_trading.py

# Terminal 2: Start Trading Bot (wait 5 seconds after API starts)
python auto_trader_tws.py
```

**Note:** Manual launch doesn't have auto-restart. If either process crashes, you must manually restart it.

### Verify System Health:
Look for these startup messages:
```
[2025-12-15 10:00:00] [INFO] === ZTE System Supervisor Starting ===
[2025-12-15 10:00:00] [INFO] Starting API Server...
[2025-12-15 10:00:00] [INFO] API Server started (PID: 12345)
[2025-12-15 10:00:05] [INFO] Starting Trading Bot...
[2025-12-15 10:00:05] [INFO] Trading Bot started (PID: 12346)
[2025-12-15 10:00:05] [INFO] === All systems running ===

[TWS] Connected to TWS on port 7497
[TWS] Account Balance: ${balance}
[TWS] Loaded 0 existing positions

🛡️ HARDENED REQUIREMENTS (V3.7.7):
   Min Confidence: 75% (was 50%)
   Min Phase1 Signals: 3+ (was 2)
   Min RVOL: 1.5x
   Stop Loss: 2.0% (wider for quality)
```

During operation, watch for hardened filters:
```
✅ API server healthy
⏭️ Requirements not met: Confidence 68.5% < 75.0%
⏭️ Requirements not met: Phase1 signals 2 < 3
📉 SL: ${price} (-2.0%) | 📈 TP: ${price} (+2.5%)  # V3.7.7: Wider SL
```

---

## 🔧 Troubleshooting

### Check API Health:
```bash
curl http://127.0.0.1:5002/api/health
```

### Check TWS Connection:
- Verify TWS/IB Gateway is running on port 7497
- Check Paper Trading mode is enabled
- Ensure API connections are enabled in TWS settings

### Verify Safety Features:
1. **API Health Check:** Bot should skip symbols if API is down
2. **Position Verification:** Bot should block duplicate orders
3. **SL/TP Levels:** Check logs show 1.25%/2.5% (not 0.75%/1.5%)

---

## 📝 Important Notes

- **Historical Incidents:** See `INCIDENTS.md` for past issues and fixes
- **Development Roadmap:** See `update.md` for active tasks and future plans
- **This Document:** Represents ONLY the current running state (V3.7.4)
- **Clean Slate:** System restarted with 0 positions after incident cleanup
- **Safety First:** All critical safety features are now active

---

## 📊 Performance Tracking

### Current Metrics:
- **Total Trades:** 0 (fresh start)
- **Win Rate:** N/A (collecting baseline)
- **Positions:** 0 (clean slate)
- **RAG Items:** 54 (technical knowledge only)

### Collection Goal:
Execute 50-100 paper trades to establish baseline metrics before Phase 2.

---

> **Last System Check:** December 9, 2025
> **Status:** 🟢 OPERATIONAL
> **Safety Features:** ✅ ALL ACTIVE
> **Ready for Trading:** YES (Paper Mode)

**For historical context and incident details, see `INCIDENTS.md`**
