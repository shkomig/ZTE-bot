# Zero Trading Expert (ZTE) Bot 🤖📈

**Advanced Day Trading Bot with AI-Powered Analysis and Real-Time Execution**

Version: **V3.7.6** (Production Ready)
Status: 🟢 **ACTIVE**

---

## 🎯 Overview

Zero Trading Expert is a sophisticated automated trading system that combines:
- **Real-time market data** from Interactive Brokers TWS
- **AI-powered analysis** via FastAPI orchestrator
- **Advanced technical indicators** (Phase 1: RSI Div, TSI, BB%, VPOC, MACD Div)
- **Risk management** with automatic SL/TP and trailing stops
- **Intelligent session-based trading** (avoids lunch dead zone)
- **Sector diversification** with tiered position system

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Interactive Brokers TWS or IB Gateway (running)
- API connections enabled in TWS

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure settings
# Edit config.yaml with your preferences
```

### Running the System

**Step 1: Start the API Server**
```bash
python api_server_trading.py
```

**Step 2: Start the Trading Bot**
```bash
python auto_trader_tws.py
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ZTE Trading System                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐      ┌──────────────────┐         │
│  │  auto_trader    │─────▶│  api_server      │         │
│  │  _tws.py        │◀─────│  _trading.py     │         │
│  │                 │      │                  │         │
│  │ - Scans stocks  │      │ - AI Analysis    │         │
│  │ - Executes      │      │ - Orchestrator   │         │
│  │ - Risk Mgmt     │      │ - RAG Memory     │         │
│  └────────┬────────┘      └──────────────────┘         │
│           │                                              │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                    │
│  │   TWS/Gateway   │                                    │
│  │   (Port 7497)   │                                    │
│  └─────────────────┘                                    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🛡️ Risk Management

### Position Limits
- **Max Positions**: 10 (Tier 1: 5, Tier 2: 5)
- **Max Per Sector**: 2 positions
- **Max Daily Loss**: 3%
- **Max Trades/Day**: 20

### Stop Loss & Take Profit
- **SL**: 1.25% (ATR-adjusted)
- **TP**: 2.5%
- **Trailing Stop**: Activated at +1.5% profit

### Filters
- **Min RVOL**: 1.5x (Tier 2: 2.0x)
- **Min Confidence**: 60% (Tier 2: 70%)
- **Min Phase1 Signals**: Tier 2 requires 2+ signals
- **Spread Filter**: Max 0.5%

---

## 📈 Trading Sessions

| Session | Time (ET) | Strategy |
|---------|-----------|----------|
| **Pre-Market** | 04:00 - 09:30 | Gap plays, news reactions |
| **Opening Bell** | 09:30 - 10:00 | High volatility, momentum |
| **Mid-Morning** | 10:00 - 11:30 | ⭐ BEST setups |
| **Lunch Dead** | 11:30 - 14:00 | ❌ AVOID (no trading) |
| **Afternoon** | 14:00 - 15:00 | Cautious, lower volume |
| **Power Hour** | 15:00 - 16:00 | Strong momentum moves |
| **After Hours** | 16:00 - 20:00 | Earnings plays |

---

## 🔧 Key Features

### Phase 1 Technical Indicators
- ✅ RSI Bullish/Bearish Divergence
- ✅ TSI (True Strength Index) Oversold/Overbought
- ✅ Bollinger Bands % Position
- ✅ VPOC (Volume Point of Control)
- ✅ MACD Divergence
- ✅ Chart Patterns (Head & Shoulders, Double Top/Bottom)

### V3.7.6 Updates
- 🛡️ **Logic Hardening**: Prevents duplicate entry and exit orders
- 🔄 **Auto-Reconnect**: Recovers from TWS connection drops
- 📝 **File Logging**: Comprehensive runtime logs for debugging
- ⚡ **Connection Health Check**: Every 5 minutes
- 🎯 **Exception Handling**: Graceful recovery from crashes

---

## 📁 Project Structure

```
ZERO-TRADING-EXPERT/
├── auto_trader_tws.py          # Main trading bot
├── api_server_trading.py       # FastAPI orchestrator
├── config.yaml                 # Configuration (port 5002)
├── check_tws.py                # TWS connection tester
├── CORE_TRADING/
│   ├── trading_orchestrator.py # AI decision engine
│   ├── market_analyzer.py      # Technical analysis
│   ├── pattern_detector.py     # Chart patterns
│   └── trading_memory.py       # RAG learning (disabled)
├── MCP.md                      # Complete system documentation
├── INCIDENTS.md                # Issue tracking & resolutions
├── update.md                   # Version history & roadmap
├── daily_pnl.jsonl             # P&L tracking
├── paper_trades.jsonl          # Trade history
└── bot_runtime.log             # Runtime logs
```

---

## 📊 Performance Tracking

The system tracks performance in real-time:

- **Daily P&L**: Recorded to `daily_pnl.jsonl`
- **Trade History**: Saved to `paper_trades.jsonl`
- **Runtime Logs**: Written to `bot_runtime.log`
- **Live Tracker**: Real-time win rate and performance metrics

### Recent Performance (Dec 2025)
- **Best Day**: Dec 11 - $163.98 profit (6 trades, 33% win rate)
- **Total Trades**: 247+ executed
- **System Uptime**: Stable with auto-recovery

---

## 🔍 Monitoring

### Check Bot Status
```bash
# View live logs
tail -f bot_runtime.log

# Check API health
curl http://127.0.0.1:5002/api/health

# Check TWS connection
python check_tws.py

# View open positions
grep "Open Positions" bot_runtime.log | tail -1
```

### Health Indicators
- ✅ Bot logs every 30 seconds
- ✅ API returns `{"status":"healthy"}`
- ✅ TWS connected on port 7497
- ✅ Phase1 signals detected

---

## ⚙️ Configuration

Key settings in `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 5002  # API server port

trading:
  max_positions: 10
  max_daily_loss_pct: 0.03
  scan_interval: 30  # seconds

risk:
  stop_loss_pct: 0.0125  # 1.25%
  take_profit_pct: 0.025  # 2.5%
  trailing_stop_distance: 0.015  # 1.5%
```

---

## 🐛 Troubleshooting

### Bot stops unexpectedly
✅ **FIXED** in V3.7.6 - Auto-reconnect and exception handling added

### API server not responding
```bash
# Check if running
netstat -ano | findstr ":5002"

# Restart
python api_server_trading.py
```

### Duplicate orders
✅ **FIXED** in V3.7.6 - Pending order checks added

### TWS connection issues
```bash
# Test connection
python check_tws.py

# Check TWS settings:
# - API enabled
# - Port 7497 for paper trading
# - Socket port configured
```

---

## 📚 Documentation

- **[MCP.md](MCP.md)** - Complete technical documentation
- **[INCIDENTS.md](INCIDENTS.md)** - Issue tracking and resolutions
- **[update.md](update.md)** - Version history and roadmap

---

## 🔐 Safety Features

- ✅ **Paper Trading Mode**: Test strategies safely
- ✅ **Daily Loss Limit**: Auto-stops at -3%
- ✅ **Position Limits**: Max 10 concurrent positions
- ✅ **Sector Limits**: Max 2 per sector
- ✅ **Session Filters**: Avoids dangerous market conditions
- ✅ **Duplicate Prevention**: Checks pending orders before entry
- ✅ **Connection Recovery**: Auto-reconnects on failure

---

## 🎓 How It Works

1. **Scan**: Every 30 seconds, scans configured stocks
2. **Analyze**: Phase1 indicators + AI orchestrator analysis
3. **Filter**: RVOL, spread, position limits, session checks
4. **Execute**: Places bracket orders with SL/TP
5. **Monitor**: Updates trailing stops, checks exits
6. **Learn**: Records trades for future analysis (RAG)

---

## 📞 Support

For issues or questions:
- Check [INCIDENTS.md](INCIDENTS.md) for known issues
- Review logs in `bot_runtime.log`
- Verify TWS connection with `check_tws.py`

---

## ⚠️ Disclaimer

This is an automated trading system. Use at your own risk.
- Always test in paper trading first
- Monitor performance regularly
- Understand the risks of automated trading
- Past performance does not guarantee future results

---

## 📝 License

Proprietary - All Rights Reserved

---

**Built with ❤️ using Python, FastAPI, and ib_insync**

*Last Updated: December 11, 2025*
