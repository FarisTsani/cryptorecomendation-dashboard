# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(coin_df, tvl_df):
    return coin_df, tvl_df


def merge_and_clean_data(df, df_tvl):
    """Merge coin data with TVL data and handle missing values."""
    df['date'] = pd.to_datetime(df['date'])
    merged_df = pd.merge(df, df_tvl, on=['coin', 'date'], how='left')
    merged_df['marketCap'] = merged_df['tvl'].fillna(merged_df['marketCap'])
    merged_df = merged_df.drop(columns=['tvl'])
    merged_df = merged_df.rename(columns={'marketCap': 'tvl'})
    return merged_df

def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def feature_engineering(data):
    """Add SMA, RSI, and target columns."""
    data = data.sort_values(by=['coin', 'date']).reset_index(drop=True)
    data['close_t_plus_7'] = data.groupby('coin')['close'].shift(-7)
    data['target'] = (data['close_t_plus_7'] > data['close']).astype(int)
    data['SMA7'] = data.groupby('coin')['close'].transform(lambda x: x.rolling(window=7).mean())
    data['SMA14'] = data.groupby('coin')['close'].transform(lambda x: x.rolling(window=14).mean())
    data['RSI14'] = data.groupby('coin')['close'].apply(lambda x: compute_rsi(x, period=14)).reset_index(drop=True)
    data = data.fillna({'SMA7': 0, 'SMA14': 0, 'RSI14': 0})
    return data

def encode_coin_column(data):
    """Encode 'coin' column as numerical labels."""
    le = LabelEncoder()
    data['coin'] = le.fit_transform(data['coin'])
    return data
