# מחקר מעמיק: סורק מניות עם שילוב LLM וסוכנים אוטונומיים

## תקציר מנהלים

סורקי מניות (Stock Screeners) עוברים מהפכה עם כניסה של מודלי שפה גדולים וסוכנים אוטונומיים. בעבר, סוררים סיכמו מספרים פשוטים (P/E, ROE, וכו'). כיום, הם יכולים לקרוא דוחות כספיים מורכבים, לנתח חדשות וסנטימנט, ולהשוות אלפי מניות בשניות.

---

## 1. היתרונות של LLMs בסקינג מניות

### 1.1 קריאה וניתוח של דוחות כספיים

GPT-4 הראה יכולת לבצע ניתוח יחסים פיננסיים (Ratio Analysis) ולזהות מגמות מדוחות כספיים עם ביטחון גבוה. השימוש בטכניקת Chain-of-Thought (CoT) מאפשר למודלים לתת הסברים מפורשים ודירוגי ביטחון עבור כל החלטה.

### 1.2 גילוי גורמים כלכליים

GPT-4 יכול לייצר גורמים (factors) בעלי הנמקה כלכלית שוקלטו ולא רק סטטיסטיים, שיכולים להישמש במודלים כמותיים להשקעה.

### 1.3 זיהוי אופות מחיר

מודלים כמו Quantformer (Transformer מותאם לנתונים כמותיים) מראים ביצועים עדיפים בזיהוי אופות מחיר בהשוואה ל-SVMs וCNNs.

### 1.4 שיפור מרשים בביצועים

סקר 2025 על LLMs בשווקי מניות (84 מחקרים) הראה:
- שיפור של **33.8% ב-Sortino ratio** של MarketSenseAI 2.0 לעומת השוק
- יכולות משופרות בניתוח קביעות סנטימנט
- ניתוח כמותי אוטומטי של דוחות כספיים בקנה מידה

---

## 2. סקירת בשוק: סוררים AI מובילים

### 2.1 Zen Ratings

**מאפיינים:**
- בדירוג AI מבוסס 20+ שנות של NeuroNetwork Training
- ניתוח 100+ גורמים במידע שש קטגוריות: Financials, Growth, Momentum, Safety, Sentiment, Value
- תוצאות מדידה: מניות דורג "A" השיגו **32.52% תשואה שנתית ממוצעת**, "B" **19.88%**

**עלות:** Free basic plan, $19.50/month premium (annual), trial $1 לשבועות 2

### 2.2 TrendSpider

**מאפיינים:**
- אוטומציה מולטי-טיים-פריים של ניתוח מגמות
- ניתוח אוטומטי של support/resistance zones
- >700 Smart Watch Lists שנוצרו באופן אוטומטי
- AI Strategy Lab לזיהוי סיגנלי כניסה

**עלות:** Starts at $54/month

### 2.3 Trade Ideas עם Holly AI

**מאפיינים:**
- בינה מלאכותית וירטואלית (Holly) המספקת המלצות
- סנכרון עם כל ברוקר
- Trading ישיר מ-Chart עם Click 1

**עלות:** Starts at $89/month

### 2.4 Tickeron

**מאפיינים:**
- זיהוי אוטומטי של אופות ב-Chart (סוקן אלפי מניות כל דקה)
- דירוג הסתברות וודאות עבור כל אופה
- Virtual Agents מותאמים למסחר אוטומטי
- עדכונים עד 85% בביצועים לבוטים טק

**עלות:** Free limited plan, $30/month (annual), Paid plans available

### 2.5 Seeking Alpha Premium

**מאפיינים:**
- דוחות AI המסכמים כתיבות של אנליסטים
- ניתוח Value, Momentum, Growth, Safety
- סקרנר שקול עם מאות גורמים

**עלות:** $4.95 for 1 month (trial), $299/year renewal

---

## 3. ארכיטקטורות Multi-Agent לסקינג

### 3.1 MarketSenseAI 2.0

מסגרת GPT-4 מתקדמת בפריסה בשנת 2024:

#### סוכנים מתמחים:
| סוכן | תפקיד |
|------|--------|
| **Fundamental Analyst** | ניתוח דוחות כספיים וDCF Valuations |
| **Sentiment Analyst** | סנטימנט ממניות, חדשות, רשתות חברתיות |
| **News Analyst** | זיהוי שינויים בטון וידע עתידי |
| **Technical Analyst** | תבניות ואינדיקטורים טכניים |
| **Macro Analyst** | הקשר כלכלי וrisk-off/on regimes |

#### תוצאות:
- **S&P 500 Backtests (2024):** 33.8% Sortino Ratio > Market
- דיוק בבחירת מניות גבוהות תשואה משמעותי

### 3.2 Multi-Agent Collaboration לניתוח מניות (CrewAI)

מערכת קוד פתוח לסקינג מדורג:

```
Data Retriever Agent
    ↓
Insights Analyst Agent
    ↓
Sentiment Analyst Agent
    ↓
Quantitative Analyst Agent
    ↓
Report Generation Agent
```

#### סוכנים וכלים:
- yfinance: הורדת נתוני היסטוריה
- SerperDevTool: חיפוש ספציפי לנתונים
- WebsiteSearchTool: מיצוי מידע מאתרים
- Process.sequential: ביצוע בסדר עוקב

### 3.3 FinRobot

פלטפורמה AI Agent מקיפה מבית AI4Finance:

**סוכנים מובנים:**
- Market Forecaster (חיזוי כיווני תנועה)
- Annual Report Analyzer (ניתוח דוחות שנתיים)
- Trade Strategist (יצירת אסטרטגיות)
- Multimodal LLM Agent (עם visualization)

**API תמיכה:** Finnhub, FinnLP, Financial Modeling Prep (FMP)

---

## 4. טכניקות Machine Learning לבחירת מניות

### 4.1 Quantformer - Transformer לנתונים כמותיים

ארכיטקטורה חדשה לניתוח כמותי של מניות:

**מאפיינים:**
- Normalization Z-score עבור כל timestep
- Ranking וportfolio optimization מבוסס בחירה
- Backtested על 4,600+ מניות (14 שנים, 2010-2023)

**תוצאות:**
- פחות overfitting מ-SVMs
- ביצועים יתירים על מודלים תיאוריים בסיסיים
- Frequency: Daily, Weekly, Monthly strategies

### 4.2 עיבוד בחירת גורמים (Feature Selection)

טכניקות מובילות כפי שנמצאו במחקר 2024-2025:

| שיטה | תיאור | דיוק |
|------|--------|------|
| **Random Forest + Tuning** | RFC עם hyperparameter optimization | 97%+ |
| **RFE + RFC** | Recursive Feature Elimination | 90-97% |
| **Boruta + RFC** | אלגוריתם Boruta לגורמים | 84-91% |

**מאפיין:** RFC עם fine-tuning הראה ביצועים קונסיסטנטיים בכל סוגי הנתונים.

### 4.3 Hybrid Deep Learning Models

CNN-BiLSTM עם Attention Mechanism:

```
CNN Layer (Feature Extraction)
    ↓
BiLSTM Layers (Temporal Dependencies)
    ↓
Attention Mechanism (Focus on Key Steps)
    ↓
Dense Output Layer
```

**תוצאות בתחזוקת מחיר:**
- R² = 0.9580 (CNN-BiLSTM-AM)
- RMSE = 23.005
- MAPE = 1.094%
- ביצועים עדיפים ל-LSTM בלבד

### 4.4 Combined Machine Learning Weighting

שיטה של משקלול דינמי עבור Ensemble Models:

**שתי גישות:**
1. **Static Weighting:** על סמך מטריקות הערכה (MAE, RMSE)
2. **Dynamic Weighting:** על סמך Information Coefficient (IC) Time-Series

**תוצאה:** Ensemble Strategies התגברו על Single Models בתשואות Backtested.

---

## 5. כלים וממשקי API

### 5.1 מקורות נתונים

| מקור | סוגי נתונים | Frequency |
|------|----------|-----------|
| **yfinance** | OHLCV, Fundamentals | Daily/Real-time |
| **Finnhub** | Real-time Quotes, News | 1 Sec - Real-time |
| **AlphaVantage** | Time Series, Indicators | Real-time |
| **Financial Modeling Prep** | Financial Statements, Ratios | Updated Daily |
| **AlphaResearch** | SEC Filings, Earnings Calls | Daily Updates |
| **EDGAR** | SEC Filings (10-K, 10-Q) | Filing Date |

### 5.2 Backtesting Frameworks

| Framework | שפה | ביצועים |
|-----------|-----|---------|
| **Backtrader** | Python | מהר, טוב לתיעוד |
| **QuantConnect** | Python/C# | Cloud-based, נתונים רבים |
| **Zipline** | Python | קל ללמוד, מאוד דומה ל-Backtrader |
| **FinRL** | Python | עם RL, וארטוב פיננסי |

### 5.3 API שילוב עם LLMs

**OpenAI:** GPT-4, GPT-4-turbo עבור ניתוח
**Anthropic Claude:** התפלגות פחות הלוצינציה
**LangChain:** Orchestration בין LLMs לכלים
**CrewAI:** Multi-Agent Orchestration
**AutoGen:** Microsoft's Multi-Agent Framework

---

## 6. NLP לניתוח סנטימנט וחדשות

### 6.1 Sentiment Models

**FinBERT vs BERT:**
- FinBERT: Fine-tuned ספציפי למטבע פיננסי
- BERT: General NLP Model

**FinGPT:** סקומפטיביציו GPT מה-Foundation Models
- עלות Fine-tuning: <$300
- Accuracy משופר על FinBERT בתרחישים פיננסיים

### 6.2 מחקר מעשי: LLM-Enhanced Trading

תוצאות עבור ניתוח סנטימנט מבוסס LLM על מניות:

| מניה | Strategy Ratio בסיס | With Sentiment | ביצוע |
|------|---------|--------|-------|
| TSLA | 0.34 | 3.47 | +921% 🚀 |
| AAPL | בסיסי | משופר | משמעותי |
| AMZN | בסיסי | משופר | משמעותי |

**Win Ratio עבור TSLA:** 32.2% → 57.0%

### 6.3 מקורות נתוני סנטימנט

- ממשקי חדשות בזמן אמת (Finnhub, Reuters, CNBC)
- Twitter/X (API v2)
- Reddit (PRAW)
- Earnings Calls (Transcripts)
- SEC MD&A Sections
- Analyst Reports

---

## 7. מקרי שימוש וביישומים

### 7.1 סקינג בחירה ערך (Value Screening)

**קריטריונים:**
- P/E < 12, Price/Book < 1.5
- Dividend Yield > 3%
- ROE > 10%
- Debt/Equity < 1

**LLM Role:** ניתוח איכות ניהול מהדוחות
**תוצאה:** שיטוק משופר בהשוואה לנתונים בלבד

### 7.2 גדילה זיהוי מעלה (Growth Screening)

**קריטריונים:**
- Revenue Growth > 15% YoY
- EPS Growth > 10% YoY
- PEG Ratio < 2.0
- R&D Spend > Industry Average

**LLM Role:** הערכת טיוטת המוצר החדש מחדשות ו-MD&A
**תוצאה:** יעדי גדילה זיהוי בשלב מוקדם

### 7.3 סקינג טכני + סנטימנט

**אינדיקטורים טכניים:**
- RSI > 50 (מומנטום)
- MACD Crossover
- Break של Resistance Level
- Volume Confirmation

**NLP Sentiment:** Bullish News / Social Mentions
**סוכן Coordinator:** Integration & Deal-Breaking Signal

---

## 8. פרויקטים GitHub ופתוח קוד

### 8.1 StockScreener-MCP (Local LLM)

```
Python + Ollama (Qwen3) + LangChain + BeautifulSoup
```
**תכונות:**
- Company Details (Price, Market Cap, P/E, ROE, ROCE)
- Profit Analysis (Quarterly & Yearly)
- Shareholding Pattern Analysis
- Tool Integration ב-MCP

**קישור:** https://github.com/ambideXtrous9/StockScreener-MCP-with-Ollama-and-Langchain

### 8.2 AI_Agent_Trader (Multi-Agent Stock Analysis)

```
CrewAI Orchestration + SerperDev + Web Scraping
```

**Agents:**
1. Stock Picker Agent (Trend Identification)
2. Data Analyst Agent (Market Data Processing)
3. Strategy Architect Agent (Strategy Development)
4. Execution Strategist (Execution Planning)
5. Risk Architect (Risk Assessment & Mitigation)

**Techniques:**
- Linear Regression, Moving Averages, RSI (Trending)
- ARIMA, LSTM (Forecasting)
- Genetic Algorithms, Simulated Annealing (Optimization)
- VaR, Monte Carlo (Risk Management)

### 8.3 Stock-Market-Sentiment-Analysis-NLP (Academic)

```
NLP + GloVe Embeddings + TensorFlow/Keras
```

**מתודולוגיה:**
- Sentiment Classification (Positive/Negative)
- Text Preprocessing (Tokenization, Stopword Removal)
- Model Evaluation (Accuracy ~78%, AUC ~0.78)

### 8.4 EDGAR-CRAWLER (SEC Filings)

```
Open-Source Toolkit for SEC Filings Parsing
```

**תכונות:**
- Download Raw SEC Filings
- Parse לـ Structured JSON
- Bootstrap Financial NLP Experiments

---

## 9. אתגרים וציפיות

### 9.1 אתגרים טכניים

| אתגר | פתרון |
|------|--------|
| **Hallucinations** | Multi-source Verification, RAG |
| **Latency** | Batch Processing, Caching |
| **Data Quality** | Multiple Data Providers |
| **Overfitting** | Cross-validation, Out-of-sample Testing |
| **Concept Drift** | Continuous Retraining |

### 9.2 סוגיות רגולטוריות

- Compliance עם SEC Rules
- Disclosure Requirements
- Explainability ב-AI Models
- Attribution ו-Auditability

### 9.3 המלצות בעיצום

1. **לשלוח Paper Trading תמיד** לפני מסחר אמיתי
2. **לאמת נתונים ממקורות מרובים**
3. **ליישם Stop-Losses ו-Limits**
4. **לנטר Performance Continuously**
5. **למשוקלל בין Automation וHuman Judgment**

---

## 10. קישורים למשאבים

### 10.1 מאמרים ומחקרים (PDF)

1. **MarketSenseAI 2.0: Enhancing Stock Analysis Through LLMs**
   - https://arxiv.org/html/2502.00415v2

2. **Large Language Models in Equity Markets** (Survey - 84 Papers)
   - https://pdfs.semanticscholar.org/e66a/65d55da55b7beee81e0cc6809968b17dc32c.pdf

3. **Quantformer: From Attention to Profit**
   - https://arxiv.org/abs/2404.00424

4. **Factor-based Stock Selection Using ML Methods**
   - https://www.atlantis-press.com/proceedings/icdeba-24/126008611

5. **The New Quant: LLMs in Investing** (Survey)
   - https://arxiv.org/html/2510.05533v1

6. **Stock Prediction Using NLP Sentiment Analysis** (COVID-19 Study)
   - https://fount.aucegypt.edu/cgi/viewcontent.cgi?article=2577&context=etds

7. **Hybrid Deep Learning for Stock Price Prediction**
   - https://www.scitepress.org/Papers/2024/132142/132142.pdf

8. **Data-Driven Neural Networks in Stock Forecasting** (Review)
   - https://www.sciencedirect.com/science/article/pii/S1566253524003944

9. **Combined ML for Stock Selection** (Dynamic Weighting)
   - https://arxiv.org/abs/2508.18592

10. **Deep Learning for Risk-Aligned Portfolio Investing**
    - https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0330547

11. **Multifactor Prediction Model for Stock Market Analysis**
    - https://www.nature.com/articles/s41598-025-88734-6

12. **LLM Impact on Stock Prices** (MD&A Analysis)
    - https://papers.ssrn.com/sol3/Delivery.cfm/a2e75db6-9b36-4af3-8ac4-5caf1843cf5b-MECA.pdf

### 10.2 פרויקטים GitHub

| פרויקט | תיאור | קישור |
|---------|--------|--------|
| **FinRobot** | AI Agent Platform for Finance | https://github.com/AI4Finance-Foundation/FinRobot |
| **TradingAgents** | Multi-Agent Trading Framework | https://github.com/TauricResearch/TradingAgents |
| **AI_Agent_Trader** | Multi-Agent Stock Analysis | https://github.com/AloshkaD/AI_Agent_Trader |
| **StockScreener-MCP** | Local LLM Stock Analysis | https://github.com/ambideXtrous9/StockScreener-MCP-with-Ollama-and-Langchain |
| **EDGAR-Crawler** | SEC Filings Parser | https://github.com/lefterisloukas/edgar-crawler |
| **Stock-Sentiment-NLP** | Sentiment Analysis with GloVe | https://github.com/MayCooper/Stock-Market-Sentiment-Analysis-NLP-GloVe-TensorFlow |

### 10.3 סוררים וכלים זמינים

| סורק | מאפיינים | עלות |
|------|---------|------|
| **Zen Ratings** | 100+ Factors, A-Grade 32.52% ROI | Free / $19.50/mo |
| **TrendSpider** | Automated Technical Analysis | $54/mo |
| **Trade Ideas** | Holly AI Assistant | $89/mo |
| **Tickeron** | Pattern Recognition AI | Free / $30/mo |
| **Seeking Alpha** | AI Summaries & Analysis | $4.95 trial / $299/yr |
| **Stock Rover** | 600+ Metrics, 8,500 stocks | Free / $27.99/mo |
| **TradingView** | Global Scanner | Free / $239/mo |
| **AlphaResearch** | SEC Filings Analysis | $49.99/mo |

### 10.4 מודלים וספריות

**Transformers Library:** https://huggingface.co/transformers/

**FinBERT:** https://huggingface.co/ProsusAI/finbert

**FinGPT:** https://github.com/AI4Finance-Foundation/FinGPT

**LangChain:** https://www.langchain.com/

**CrewAI:** https://www.crewai.com/

---

## 11. מסקנות

מערכות סקינג מניות עם LLM וסוכנים משנות באופן דרמטי את תהליך בחירת מניות:

### יתרונות מרכזיים:
✅ **Scalability:** ניתוח אלפי מניות בשניות
✅ **Nuance:** הבנה עמוקה של טקסטים כספיים מורכבים
✅ **Speed:** סנכרוני סעיפים להחלטה תוך דקות
✅ **Consistency:** ביצוע אוטומטי ללא הטיות אנושיות
✅ **Integration:** איחוד נתונים מקורות רבים

### עדיין חשוב:
⚠️ **Validation:** אימות נתונים מובנה-תמיד
⚠️ **Oversight:** שמירה על אלמנט אנושי בהחלטות
⚠️ **Backtesting:** בדיקה קפדנית של אסטרטגיות
⚠️ **Risk Management:** יישום נכון של סיכון קידום

---

*מסמך זה נוצר למטרות מחקר ולמידה בלבד. אין להסתמך עליו כייעוץ השקעות.*

*תאריך: נובמבר 2025*