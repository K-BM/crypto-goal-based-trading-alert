# crypto-goal-based-trading-alert

# Goal-Based Trading Alert 🚀📈
A smart, real-time Bitcoin (BTC) trading assistant that sends you tailored trading alerts based on your personal profit and risk goals — directly via Telegram.

---

## 🔍 What It Does
This MVP product allows users to define a trading goal such as:
> _"My goal: +3% profit on BTC within 2 days, low risk"_
And in response, the assistant will send a structured alert:

🎯 Goal-Based BTC Trade Alert:
Target Profit: +3%
Entry: 63,100 USDT
Exit: 65,000 USDT
Time Horizon: 1h - 4h
Risk Estimate: Low 
📊 Attached: Chart with key levels

---

## 🔁 Data Flow from Binance API
Here's how data moves through the system starting from the Binance API:
1. User sends a goal:
e.g., "My goal is +3% on BTC in next 2 days, low risk" via Telegram

2. parser.py processes the input:
Extracts target, asset, time window, risk level → returns a structured goal

3. signal_engine.py:
**Receives the parsed goal
**Calls Binance API to:
    - Get real-time price of BTC (or other asset)
    - Pull historical OHLCV data for trend analysis
**Evaluates the market and suggests:
    - An entry price (e.g., current or upcoming support)
    - An exit target (based on % profit)
    - Risk profile (volatility, momentum, etc.)
    - Time horizon (based on goal)

4. chart.py:
**Pulls the same historical price data from Binance
**Plots an annotated chart:
    - Marking entry, exit, trend lines, etc.
**Returns image or buffer

5. notifier.py:
**Formats the goal + signal + chart
**Sends the result via Telegram (or optionally SMS/email in future)

---
## High-Level Architecture (Modular Overview)
                    ┌──────────────────────┐
                    │   Telegram (User)    │
                    └─────────┬────────────┘
                              │ User input (goal)
                              ▼
                    ┌──────────────────────┐
                    │   FastAPI Backend     │   <-- app/main.py
                    └─────────┬────────────┘
                              │
         ┌────────────────────┼─────────────────────┐
         │                    │                     │
         ▼                    ▼                     ▼
┌────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│   parser.py     │   │  signal_engine.py │   │     notifier.py       │
│ NLP goal parser │   │  Trend/entry/exit │   │  Send alert to user  │
└────┬───────────┘   └────┬─────────────┘   └────────────┬────────┘
     │                    │                                │
     ▼                    ▼                                │
Parsed user goal     Fetch real-time price                │
                     + analyze trend                      │
                     from Binance API                     │
                     + calculate signals                  │
                             │                            │
                             ▼                            │
                     ┌────────────────┐                   │
                     │  chart.py      │                   │
                     │ Draw annotated │                   │
                     │ BTC chart      │                   │
                     └────────────────┘                   │
                             │                            │
                             ▼                            ▼
                      ┌────────────────────────────────────┐
                      │   Alert = goal + signal + chart    │
                      └────────────────────────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Telegram / SMS / Email   │
                          └──────────────────────────┘

---

## 🧠 Features

- ✨ **NLP-based goal parsing** from natural language inputs  
- 📊 **Real-time BTC market data** & technical analysis  
- 🧮 **Smart signal engine** (entry/exit, risk estimate, horizon)  
- 📉 **Annotated chart generation** with entry & exit levels  
- 📲 **Telegram notifications** (SMS/Email in future versions)  

---

## 📦 Tech Stack

| Layer          | Tool/Lib                        |
|----------------|---------------------------------|
| Language       | Python 3.10+                    |
| Framework      | FastAPI                         |
| Messaging      | Telegram Bot API                |
| Market Data    | Binance API, CoinGecko API      |
| Indicators     | pandas-ta, TA-Lib               |
| NLP            | spaCy, Regex                    |
| Charting       | Matplotlib / Plotly             |
| Scheduler      | APScheduler / Cron              |

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/K-BM/crypto-goal-based-trading-alert.git
cd crypto-goal-based-trading-alert
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables
Create a .env file with the following:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```
### 4. Run the Bot
```bash
uvicorn app.main:app --reload
```

🗺️ Roadmap
 Telegram Bot MVP
 Goal parsing (NLP)
 Real-time BTC signal engine
 Chart annotations
 SMS/Email integration
 ETH and other asset support
 User dashboard & backend DB