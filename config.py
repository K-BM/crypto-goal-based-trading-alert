import os
from dotenv import load_dotenv
import numpy as np
from binance.client import Client

# Load environment variables
load_dotenv()

TELEGRAM_CONFIG = {
    'bot_token': '7843017667:AAG7IRyQUhx_5yzN6HZ8vmCSFQdLtpUmAzA',
    'chat_id': 6182519076,  # Your confirmed numeric ID
    'parse_mode': 'HTML'  # Recommended for reliability
}

# Binance Configuration (keep your existing setup)
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY'),  # Also move these to .env
    'api_secret': os.getenv('BINANCE_API_SECRET'),
    'symbol': 'BTCUSDT',
    'start_date': "1 Jan 2012",
    'timeframe': Client.KLINE_INTERVAL_1HOUR
}

# Prediction Parameters (unchanged)
THRESHOLD_PCT = np.arange(0.1, 3, 0.1)