import numpy as np
import pandas as pd

def kalman_denoise(data, process_noise=1e-5, measurement_noise=1e-3):
    """
    Applies a 1D Kalman Filter to smooth the data causally.

    Args:
        data (np.array): 1D array of price data.
        process_noise (float): 'Q' - How much the system (price) naturally varies.
                               Higher = closer fit to data (less smoothing).
        measurement_noise (float): 'R' - How much noise is in the observation.
                                   Higher = more smoothing (trusts model over data).

    Returns:
        np.array: The smoothed data.
    """
    n_iter = len(data)
    sz = (n_iter,)

    # Allocate space for arrays
    xhat = np.zeros(sz)      # a posteri estimate of x (the smoothed price)
    P = np.zeros(sz)         # a posteri error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # Kalman gain

    # Initial Guesses
    # We assume the first measurement is close to the truth
    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # 1. Time Update (Prediction)
        # We predict the next price is the same as the last (Random Walk)
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + process_noise

        # 2. Measurement Update (Correction)
        # Calculate Kalman Gain
        K[k] = Pminus[k] / (Pminus[k] + measurement_noise)

        # Update estimate with new measurement (z = data[k])
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])

        # Update error estimate
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

def create_feature_set(df, symbol):
  df = df.copy()
  #show columns
  print(f"Columns for {symbol}: {df.columns.tolist()}")
  df.dropna(inplace=True)

  df['ret_1m'] = np.log(df['close'] / df['close'].shift(1))
  df["volatility_20"] = (
    df["ret_1m"]
      .rolling(20)
      .std()
  )
  df['ret_5m'] = np.log(df['close'] / df['close'].shift(5))
  df['ret_15m'] = np.log(df['close'] / df['close'].shift(15))
  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
  df['ema_dist'] = (df['close'] - df['ema_50']) / df['ema_50']


  if 'time' in df.columns:
      df['date'] = pd.to_datetime(df['time'])
      minute_of_day = df['date'].dt.hour * 60 + df['date'].dt.minute
      df['tod_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
      df['tod_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
      dow = df['date'].dt.dayofweek
      df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
      df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

  nan_rows = df[df.isnull().any(axis=1)]
  nan_count = len(nan_rows)
  print(f"NaN Count: {nan_count}")

  df.dropna(inplace=True)

  feature_cols = []
  if 'tod_sin' in df.columns and symbol == 'EURUSD':
      feature_cols.extend(['tod_sin', 'tod_cos', 'dow_sin', 'dow_cos'])

  feature_cols.extend(['ret_1m','ret_5m','ret_15m','volatility_20','ema_dist'])
  features = df[feature_cols]
  close_prices = df['close']
  dates = df['date'] if 'date' in df.columns else None

  print('Features Shape: ',features.shape)
  print('Close Prices Shape: ',close_prices.shape)

  return features ,close_prices,dates

def normalize_score(value, min_value =  -2.0, max_value = 2.0):

    if max_value == min_value:
        raise ValueError("max_value and min_value cannot be the same")
    
    score = ((value - min_value) / (max_value - min_value)) * 100
    return max(0, min(100, score)) 
