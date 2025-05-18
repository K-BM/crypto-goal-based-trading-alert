import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands


class DataProcessor:
    @staticmethod
    def calculate_daily_indicators(daily_df):
        """
        Calculate all technical indicators on daily data with proper temporal alignment
        to prevent lookahead bias.

        Args:
            daily_df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with calculated indicators
        """
        df = daily_df.copy()

        # ===== SHIFTED PRICES (PREVENT LOOKAHEAD) =====
        df['close_shifted'] = df['close'].shift(1)  # Yesterday's close
        df['high_shifted'] = df['high'].shift(1)    # Yesterday's high
        df['low_shifted'] = df['low'].shift(1)      # Yesterday's low

        # ===== MOMENTUM INDICATORS =====
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

        # ===== TREND INDICATORS =====
        macd = MACD(df['close_shifted'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # ===== MOVING AVERAGES =====
        for window in [9, 20, 50, 100, 200]:
            df[f'ma_{window}'] = SMAIndicator(df['close_shifted'], window=window).sma_indicator()

        # ===== VOLATILITY INDICATORS =====
        bb = BollingerBands(df['close_shifted'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()

        # ===== LAG FEATURES =====
        for feature in ['rsi', 'stoch_k', 'macd', 'ma_9', 'bb_width']:
            for lag in [1, 3, 5, 7, 14]:  # 1d, 3d, 5d, 1w, 2w lags
                df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

        # ===== MARKET CONTEXT =====
        # Today's open vs yesterday's close
        df['overnight_return'] = (df['open'] - df['close_shifted']) / df['close_shifted']

        # Intraday metrics (using today's open, which is known)
        df['intraday_range'] = (df['high'] - df['low']) / df['open']
        df['daily_return'] = (df['close'] - df['open']) / df['open']
        df['volatility_7d'] = df['daily_return'].rolling(7).std()
        df['volume_ma_7d'] = df['volume'].rolling(7).mean()

        return df.dropna()

    @staticmethod
    def create_daily_dataset(daily_df, threshold_pct, direction='both'):
        """
        Create the final dataset(s) with target variable(s) for given thresholds

        Args:
            daily_df (pd.DataFrame): DataFrame with indicator data
            threshold_pct (float or list of floats): Percentage threshold(s) for target
            direction (str): 'increase', 'decrease', or 'both'

        Returns:
            pd.DataFrame or dict of pd.DataFrame: If threshold_pct is a list, returns dict keyed by threshold,
            else returns a single DataFrame with target column
        """
        df = daily_df.copy()

        # Make sure threshold_pct is iterable
        if isinstance(threshold_pct, (int, float)):
            threshold_pct = [threshold_pct]

        datasets = {}
        for threshold in threshold_pct:
            temp_df = df.copy()
            # ===== TARGET CREATION =====
            if direction == 'increase':
                temp_df['target'] = (temp_df['high'] >= temp_df['open'] * (1 + threshold / 100)).astype(int)
            elif direction == 'decrease':
                temp_df['target'] = (temp_df['low'] <= temp_df['open'] * (1 - threshold / 100)).astype(int)
            elif direction == 'both':
                temp_df['target'] = ((temp_df['high'] >= temp_df['open'] * (1 + threshold / 100)) | 
                                    (temp_df['low'] <= temp_df['open'] * (1 - threshold / 100))).astype(int)
            else:
                raise ValueError("Direction must be either 'increase', 'decrease', or 'both'")
            
            datasets[threshold] = temp_df

        # Return single DataFrame if only one threshold was provided
        if len(threshold_pct) == 1:
            return datasets[threshold_pct[0]]
        return datasets
