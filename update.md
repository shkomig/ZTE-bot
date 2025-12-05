# 📋 ZTE Development Roadmap - תוכנית עבודה להמשך פיתוח

**תאריך יצירה:** 05/12/2025  
**גרסה נוכחית:** V3.5.2  
**סטטוס:** 🟢 פעיל ויציב

---

## 📊 סיכום מצב נוכחי

### ✅ מה הושלם:
| שלב | תיאור | תאריך השלמה |
|-----|--------|-------------|
| Phase 1-6 | Foundation to Integration | נובמבר 2025 |
| Phase 7 | Sentiment Analysis (Finnhub) | נובמבר 2025 |
| Phase 8 | Premium Watchlist (42 stocks) | נובמבר 2025 |
| Phase 9 | Day Trading Optimization | דצמבר 2025 |
| Phase 10 | Scanner Upgrades V3.3 | 03/12/2025 |
| **Phase 1 V4** | Technical Indicators (RSI Div, TSI, BB, VP, MACD) | 02/12/2025 |
| Bug Fixes | Paper Trading Fix, RAG Cleanup, Duplicate Orders | 05/12/2025 |

### 📈 מערכת נוכחית:
- **42 מניות** ב-7 סקטורים
- **10 פוזיציות מקסימום** (Tier1: 5, Tier2: 5)
- **5 אינדיקטורי Phase1** פעילים
- **54 פריטי ידע טכני** ב-RAG (נוקה מעסקאות לא אמינות)
- **Paper Trading Mode** עם RVOL=2.0 ברירת מחדל

---

## 🎯 Phase 2: Multi-Strategy Engine (ינואר 2026)

### 🔴 עדיפות קריטית

| # | משימה | תיאור | זמן משוער | קבצים |
|---|--------|--------|-----------|--------|
| 2.1 | **Strategy Registry** | מערכת רישום אסטרטגיות | 2-3 ימים | `strategy_registry.py` |
| 2.2 | **Base Strategy Class** | מחלקת בסיס אבסטרקטית | 1 יום | `strategies/base_strategy.py` |

### 🟠 עדיפות גבוהה

| # | משימה | תיאור | זמן משוער | קבצים |
|---|--------|--------|-----------|--------|
| 2.3 | **Mean Reversion Strategy** | Bollinger Bands + TSI | 3-4 ימים | `strategies/mean_reversion.py` |
| 2.4 | **Momentum Strategy** | Trend Following + MACD | 3-4 ימים | `strategies/momentum.py` |
| 2.5 | **Breakout Strategy** | Volume + Price Breakout | 3-4 ימים | `strategies/breakout.py` |

### 🟡 עדיפות בינונית

| # | משימה | תיאור | זמן משוער | קבצים |
|---|--------|--------|-----------|--------|
| 2.6 | **Strategy Selector** | בחירה דינמית לפי תנאי שוק | 2-3 ימים | `strategy_selector.py` |
| 2.7 | **Backtesting Framework** | בדיקת אסטרטגיות על היסטוריה | 4-5 ימים | `backtester.py` |

### 📁 מבנה קבצים מוצע:
```
CORE_TRADING/
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py      # Abstract base class
│   ├── mean_reversion.py     # Bollinger + TSI
│   ├── momentum.py           # Trend Following
│   ├── breakout.py           # Volume Breakout
│   └── gap_fill.py           # Gap Trading (חדש!)
├── strategy_selector.py      # בחירת אסטרטגיה
├── strategy_registry.py      # רישום ומעקב
└── backtester.py             # בדיקות היסטוריות
```

### 📝 פירוט טכני - Mean Reversion:
```python
class MeanReversionStrategy(BaseStrategy):
    """
    Entry Conditions:
    - BB% < 0.05 (below lower band)
    - TSI < -25 (oversold)
    - RSI < 35
    - Volume > 1.5x average
    
    Exit Conditions:
    - BB% > 0.5 (back to middle)
    - TSI cross above signal
    - 2% Take Profit / 1% Stop Loss
    """
```

---

## 🧠 Phase 3: ML/AI Predictions (פברואר 2026)

### 🔴 עדיפות קריטית

| # | משימה | תיאור | זמן משוער | דיוק צפוי |
|---|--------|--------|-----------|-----------|
| 3.1 | **LSTM Price Prediction** | מודל חיזוי מחירים | 1-2 שבועות | 70-80% |
| 3.2 | **Feature Engineering** | יצירת features למודל | 3-4 ימים | - |

### 🟠 עדיפות גבוהה

| # | משימה | תיאור | זמן משוער | דיוק צפוי |
|---|--------|--------|-----------|-----------|
| 3.3 | **Sentiment ML Enhancement** | שיפור סנטימנט עם ML | 1 שבוע | +33% Sharpe |
| 3.4 | **Ensemble Model** | שילוב מספר מודלים | 1 שבוע | +15% accuracy |

### 🟡 עדיפות בינונית

| # | משימה | תיאור | זמן משוער | דיוק צפוי |
|---|--------|--------|-----------|-----------|
| 3.5 | **Pattern Recognition CNN** | זיהוי דפוסי גרפים | 2 שבועות | Chart Patterns |
| 3.6 | **Reinforcement Learning** | למידה מחיזוק | לטווח ארוך | Optimization |

### 📝 ארכיטקטורת LSTM מוצעת:
```python
# Model Architecture
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(60, features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Next-day return prediction
])

# Features:
# - OHLCV (5)
# - RSI, TSI, MACD, BB% (4)
# - Volume Profile (3)
# - Sentiment Score (1)
# Total: 13 features × 60 days = 780 inputs
```

---

## 👥 Phase 4: Multi-Agent Architecture (מרץ 2026)

### 🔴 עדיפות קריטית

| # | סוכן | תפקיד | סטטוס נוכחי | זמן משוער |
|---|------|--------|-------------|-----------|
| 4.1 | **Technical Analyst Agent** | ניתוח טכני מעמיק | חלקי (Phase1) | 1 שבוע |
| 4.2 | **Risk Manager Agent** | ניהול סיכונים מתקדם | בסיסי | 1 שבוע |

### 🟠 עדיפות גבוהה

| # | סוכן | תפקיד | סטטוס נוכחי | זמן משוער |
|---|------|--------|-------------|-----------|
| 4.3 | **Fundamental Analyst Agent** | ניתוח פונדמנטלי | ❌ לא קיים | 2 שבועות |
| 4.4 | **Portfolio Manager Agent** | אופטימיזציית תיק | ❌ לא קיים | 2 שבועות |

### 🟡 עדיפות בינונית

| # | סוכן | תפקיד | סטטוס נוכחי | זמן משוער |
|---|------|--------|-------------|-----------|
| 4.5 | **Bull Researcher** | חיפוש סיבות לקנייה | ❌ לא קיים | 1 שבוע |
| 4.6 | **Bear Researcher** | חיפוש סיבות למכירה | ❌ לא קיים | 1 שבוע |
| 4.7 | **Debate Moderator** | הכרעה בין Bull/Bear | ❌ לא קיים | 3-4 ימים |

### 📁 מבנה קבצים מוצע:
```
CORE_TRADING/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Abstract base class
│   ├── technical_analyst.py    # Technical analysis agent
│   ├── fundamental_analyst.py  # Fundamental analysis agent
│   ├── sentiment_agent.py      # ✅ קיים!
│   ├── risk_manager.py         # Risk management agent
│   ├── portfolio_manager.py    # Portfolio optimization
│   ├── bull_researcher.py      # Bullish thesis
│   ├── bear_researcher.py      # Bearish thesis
│   └── debate_moderator.py     # Final decision
├── agent_orchestrator.py       # Coordinates all agents
└── consensus_engine.py         # Voting/consensus mechanism
```

---

## 🔧 משימות תחזוקה שוטפות

### ⚡ עדיפות מיידית (השבוע)

| # | משימה | תיאור | זמן |
|---|--------|--------|-----|
| M1 | **מעקב פוזיציות LIVE** | לוודא SL/TP מופעלים | יומי |
| M2 | **RAG Learning** | לוודא עסקאות נשמרות ל-RAG | יומי |
| M3 | **RVOL Calibration** | כיול RVOL ל-Live Trading | לפני LIVE |

### 📊 משימות שבועיות

| # | משימה | תיאור |
|---|--------|--------|
| W1 | **סקירת ביצועים** | Win Rate, P/L, Drawdown |
| W2 | **בדיקת לוגים** | זיהוי שגיאות ובאגים |
| W3 | **עדכון MCP** | תיעוד שינויים |
| W4 | **Git Push** | שמירת גרסאות |

### 🔄 משימות חודשיות

| # | משימה | תיאור |
|---|--------|--------|
| M1 | **Backtest אסטרטגיות** | בדיקת ביצועים היסטוריים |
| M2 | **עדכון Watchlist** | הוספה/הסרה של מניות |
| M3 | **אופטימיזציית פרמטרים** | SL/TP, Confidence thresholds |

---

## 📈 KPIs ויעדים

### יעדים לסוף Q1 2026:

| מדד | נוכחי | יעד Q1 | יעד שנתי |
|-----|-------|--------|----------|
| **Win Rate** | N/A (מחכה לנתונים) | 65%+ | 75%+ |
| **Sharpe Ratio** | לא נמדד | 1.5+ | 2.0+ |
| **Max Drawdown** | לא נמדד | <15% | <10% |
| **Daily Trades** | 0-5 | 5-10 | 10-15 |
| **Strategies** | 1 | 3+ | 5+ |
| **RAG Items** | 54 | 200+ | 500+ |

---

## 🚀 סדר עדיפויות - מה עכשיו?

### 🔥 השבוע (5-12 דצמבר 2025):

1. **[יום 1-2]** מעקב פוזיציות - לוודא המערכת יציבה
2. **[יום 3-4]** התחלת Phase 2.1 - Strategy Registry
3. **[יום 5-7]** Phase 2.2 - Base Strategy Class

### 📅 השבוע הבא (12-19 דצמבר):

1. Phase 2.3 - Mean Reversion Strategy
2. בדיקות על Paper Trading

### 📅 סוף דצמבר:

1. Phase 2.4-2.5 - Momentum + Breakout
2. Phase 2.6 - Strategy Selector
3. אינטגרציה ל-auto_trader_tws.py

---

## 📝 תבנית עבודה למשימה חדשה

```markdown
## [מספר משימה] - שם המשימה

**תאריך התחלה:** DD/MM/YYYY
**סטטוס:** ⬜ לא התחיל / 🔄 בתהליך / ✅ הושלם

### מטרה:
[תיאור המטרה]

### שלבים:
- [ ] שלב 1
- [ ] שלב 2
- [ ] שלב 3

### קבצים לשנות:
- `path/to/file1.py`
- `path/to/file2.py`

### בדיקות:
- [ ] Unit Tests
- [ ] Paper Trading Test
- [ ] Integration Test

### הערות:
[הערות נוספות]
```

---

## 🔗 קישורים מהירים

| מה | איפה |
|----|------|
| MCP ראשי | `MCP.md` |
| בוט מסחר | `auto_trader_tws.py` |
| Market Analyzer | `CORE_TRADING/market_analyzer.py` |
| Pattern Detector | `CORE_TRADING/pattern_detector.py` |
| Sentiment Agent | `CORE_TRADING/sentiment_agent.py` |
| Trading Memory | `CORE_TRADING/trading_memory.py` |
| Config | `config.yaml` |
| GitHub | `https://github.com/shkomig/ZTE-bot.git` |

---

> **עודכן לאחרונה:** 05/12/2025  
> **גרסה הבאה:** V3.6.0 (Multi-Strategy)  
> **מפתח:** @shkomig