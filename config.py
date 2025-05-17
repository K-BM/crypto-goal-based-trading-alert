import numpy as np
from binance.client import Client

# Binance Configuration
BINANCE_CONFIG = {
    'api_key': 'your_api_key_here',
    'api_secret': 'your_api_secret_here',
    'symbol': 'BTCUSDT',
    'start_date': "1 Jan 2012",
    'timeframe': Client.KLINE_INTERVAL_1HOUR
}

EMAIL_CONFIG = {
    'sender': 'your.email@gmail.com',  # Your Gmail
    'password': 'your_app_password',
    'receiver': 'your.email@gmail.com',  # Same as sender
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

# Prediction Parameters
THRESHOLD_PCT = np.arange(0.1, 3, 0.05)