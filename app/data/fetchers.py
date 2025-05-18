import pandas as pd
import time
from datetime import datetime, timedelta
from binance.client import Client
from config import BINANCE_CONFIG

class BinanceDataFetcher:
    def __init__(self):
        self.client = Client(
            api_key=BINANCE_CONFIG['api_key'],
            api_secret=BINANCE_CONFIG['api_secret']
        )

    def fetch_daily_data(self, symbol, start_str, end_date=None):
        """Fetch raw daily candlestick data from Binance"""
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
        
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        all_data = []
        
        while True:
            try:
                candles = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1DAY,
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
                
                # Get the last timestamp in the dataframe
                last_timestamp = df.index[-1]
                
                # Check if we've reached the end date
                if last_timestamp >= end_date:
                    break
                    
                # Set the new start time to the next day after the last candle
                start_ts = int((last_timestamp + timedelta(days=1)).timestamp() * 1000)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_data:
            raise ValueError("No data fetched from Binance API")
            
        full_df = pd.concat(all_data).drop_duplicates()
        # Ensure we don't exceed the end_date
        return full_df