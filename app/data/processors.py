import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

class DataProcessor:
    @staticmethod
    def calculate_hourly_indicators(hourly_df):
        """Calculate all technical indicators on hourly data"""
        df = hourly_df.copy()
        
        # Shifted prices to avoid lookahead bias
        df['close_shifted'] = df['close'].shift(1)
        df['high_shifted'] = df['high'].shift(1)
        df['low_shifted'] = df['low'].shift(1)
        
        # Momentum Indicators
        df['rsi'] = RSIIndicator(df['close_shifted'], window=14).rsi()
        stoch = StochasticOscillator(
            high=df['high_shifted'],
            low=df['low_shifted'],
            close=df['close_shifted'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Trend Indicators
        macd = MACD(df['close_shifted'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Moving Averages
        for window in [9, 20, 50, 100, 200]:
            df[f'ma_{window}'] = SMAIndicator(df['close_shifted'], window=window).sma_indicator()
        
        # Volatility Indicators
        bb = BollingerBands(df['close_shifted'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        
        # Lag Features
        for feature in ['rsi', 'stoch_k', 'macd', 'ma_9', 'bb_width']:
            for lag in [1, 3, 6, 12, 24]:
                df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        
        # Market Context Features
        df['hourly_return'] = df['close'].pct_change()
        df['volatility_24h'] = df['hourly_return'].rolling(24).std()
        df['volume_ma_24h'] = df['volume'].rolling(24).mean()
        
        return df.dropna()

    @staticmethod
    def create_daily_dataset(hourly_df, threshold_pct):
        """Resample to daily and create targets"""
        # Get current UTC time
        current_utc = pd.Timestamp.now(tz='UTC')
        current_utc_hour = current_utc.hour
        
        # Calculate hours to nearest midnight (forward or backward)
        if current_utc_hour < 12:
            # Before noon UTC - subtract hours to reach previous midnight
            hours_to_midnight = -current_utc_hour
        else:
            # After noon UTC - add hours to reach next midnight
            hours_to_midnight = 24 - current_utc_hour
        
        # Adjust the dataframe index
        hourly_df.index = hourly_df.index + pd.Timedelta(hours=hours_to_midnight)
        print(hourly_df.index[-1])
        
        # Aggregate OHLCV
        daily_df = hourly_df[['open', 'high', 'low', 'close', 'volume']].resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Resample indicators
        indicator_cols = [col for col in hourly_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        for col in indicator_cols:
            daily_df[col] = hourly_df[col].resample('1D').last()
        
        # Create target
        daily_df['target'] = (daily_df['high'] >= daily_df['open'] * (1 + threshold_pct/100)).astype(int)
        
        # Additional features
        daily_df['overnight_return'] = (daily_df['open'] - daily_df['close'].shift(1)) / daily_df['close'].shift(1)
        daily_df['intraday_range'] = (daily_df['high'] - daily_df['low']) / daily_df['open']
        
        return daily_df.dropna()