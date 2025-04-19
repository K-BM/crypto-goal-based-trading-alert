import pandas as pd
import time
from datetime import datetime, timedelta
from binance.client import Client
import pytz

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

# Initialize Binance client
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
client = Client(API_KEY, API_SECRET)
riyadh = pytz.timezone("Asia/Riyadh")

def get_full_1h_data(symbol="BTCUSDT", start_str="1 Jan 2015", end_time=None):
    if end_time is None:
        end_time = datetime.now(riyadh)

    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    data = []

    print("Fetching data... This may take a while ⏳")

    while True:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR,
                                    startTime=start_ts, limit=1000)
        if not candles:
            break

        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Riyadh')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        data.append(df[['open', 'high', 'low', 'close', 'volume']])

        start_ts = int((df.index[-1] + timedelta(hours=1)).timestamp() * 1000)

        if df.index[-1] >= end_time:
            break

        time.sleep(0.3)

    all_data = pd.concat(data)
    all_data = all_data[~all_data.index.duplicated(keep='first')]
    return all_data

def add_technical_indicators(df):
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()

    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['ma_9'] = SMAIndicator(close=df['close'], window=9).sma_indicator()
    df['ma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['ma_100'] = SMAIndicator(close=df['close'], window=100).sma_indicator()
    df['ma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()

    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = bb.bollinger_wband()

    return df