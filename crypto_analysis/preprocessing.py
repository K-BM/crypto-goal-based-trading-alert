import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, features):
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def create_sequences(df, time_steps, forecast_horizon):
    X, y = [], []
    feature_columns = [col for col in df.columns if col != 'close']

    for i in range(time_steps, len(df) - forecast_horizon + 1):
        X.append(df.iloc[i - time_steps + 1:i + 1][feature_columns].values)
        y.append(df.iloc[i + forecast_horizon - 1]['close'])

    return np.array(X), np.array(y)