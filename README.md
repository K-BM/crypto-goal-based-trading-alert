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