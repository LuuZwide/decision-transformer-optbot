import numpy as np

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

def create_feature_set(df):
  df = df.copy()
  df['raw_return'] = np.log(df['close'] / df['close'].shift(1))

  #Kalman denoise
  kalman_denoised_data = kalman_denoise(df['close'].values)
  df['kalman_denoised'] = kalman_denoised_data
  df['kalman_ret'] = np.log(df['kalman_denoised'] / df['kalman_denoised'].shift(1))

  df['divergence'] = (df['close'] - df['kalman_denoised'])/df['kalman_denoised']

  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
  df['ema_dist'] = (df['close'] - df['ema_50']) / df['ema_50']

  df.dropna(inplace=True)
  features = df[['raw_return','kalman_ret','ema_dist','divergence']]
  #features = df['raw_return']

  close_prices = df['close']

  return features ,close_prices
