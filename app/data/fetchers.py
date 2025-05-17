from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import time

class BinanceDataFetcher:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
    
    def fetch_hourly_data(self, symbol, start_str, end_date=None):
        """Fetch raw hourly candlestick data from Binance"""
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
        
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        all_data = []
        
        while True:
            candles = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1HOUR,
                startTime=start_ts,
                limit=1000
            )
            
            if not candles:
                break
                
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            all_data.append(df)
            
            last_timestamp = df.index[-1]
            if last_timestamp >= end_date:
                break
                
            start_ts = int((last_timestamp + timedelta(hours=1)).timestamp() * 1000)
            time.sleep(0.1)
        
        return pd.concat(all_data).drop_duplicates()