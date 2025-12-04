# Research Summary: Algorithmic Trading with LLMs & Agents
**Date:** November 2024
**Topic:** Integration of Generative AI in Financial Trading Systems

## 1. Executive Summary
The integration of Large Language Models (LLMs) into algorithmic trading marks a shift from purely quantitative strategies to **quantitative-semantic strategies**. Systems can now reason about market events, not just react to price changes.

## 2. Core Technologies
### A. Financial LLMs
* **FinBERT / FinGPT:** Models fine-tuned on financial texts (earnings calls, news, reports).
* **Reasoning:** Unlike traditional NLP (positive/negative), LLMs can explain *why* an event is bullish or bearish.

### B. Autonomous Agents
An agent loop consists of:
1.  **Observer:** Fetches market data + news.
2.  **Planner (LLM):** Formulates a strategy based on inputs.
3.  **Executor:** Connects to Broker API to place orders.
4.  **Risk Manager:** Validates orders against portfolio limits.

## 3. Key Strategies
* **Sentiment Arbitrage:** Reacting to news faster and more accurately than human traders.
* **Macro-Analysis:** Reading central bank minutes (Fed reports) to predict interest rate moves.
* **Multi-Modal Trading:** Combining Chart Image recognition (Vision models) with text analysis.

## 4. Risks & Mitigations
| Risk | Description | Mitigation |
| :--- | :--- | :--- |
| **Hallucination** | LLM inventing false data | Use RAG (Retrieval Augmented Generation) to ground data. |
| **Latency** | Inference time is slow | Avoid HFT; Focus on Swing/Intraday trading. |
| **Context Limit** | Cannot read all history | Use Vector Databases (Pinecone/Chroma) for long-term memory. |

## 5. Recommended Resources (ArXiv)
* [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/pdf/2306.06031.pdf)
* [Can ChatGPT Forecast Stock Price Movements?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788)
* [A Survey of Large Language Models in Finance](https://arxiv.org/pdf/2306.03784.pdf)

---
*Note: This document is for educational purposes only and does not constitute financial advice.*