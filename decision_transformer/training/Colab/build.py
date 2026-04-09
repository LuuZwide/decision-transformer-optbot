from decision_transformer.training.Colab.utils import create_feature_set
from decision_transformer.training.Colab.ChartEnv import ChartEnv
import os
import pandas as pd
import numpy as np
from pathlib import Path


DATAFILES_DIR = Path(__file__).resolve().parent / "datafiles"


def build_charts():
    symbols = ['EURUSD', 'GBPUSD','USDJPY','USDCHF']
    df_charts = {}

    # build charts
    for file in os.listdir(DATAFILES_DIR):
        if file.endswith(".pkl"):
            symbol = file.replace(".pkl", "")
            df_charts[symbol] = pd.read_pickle(DATAFILES_DIR / file)
    

    #build datasets
    env_charts = {}
    env_close_prices = {}

    env_test_charts = {}
    env_close_test_prices = {}

    for symbol in symbols:
        chart = df_charts[symbol]
        feature_df,close_df = create_feature_set(chart)

        train_size = int(len(feature_df) * 0.9)

        train_data, train_close_data = feature_df[:train_size ], close_df[:train_size ]
        test_data, test_close_data = feature_df[train_size:],close_df[train_size:]

        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        train_close_data.reset_index(drop=True, inplace=True)
        test_close_data.reset_index(drop=True, inplace=True)

        env_charts[symbol] = train_data
        env_close_prices[symbol] = train_close_data

        env_test_charts[symbol] = test_data
        env_close_test_prices[symbol] = test_close_data
    

    env_charts = pd.concat(env_charts, axis=1)
    env_charts = env_charts.astype(np.float32).dropna().values
    env_test_charts = pd.concat(env_test_charts, axis=1)
    env_test_charts = env_test_charts.astype(np.float32).dropna().values

    return env_charts, env_close_prices, env_test_charts, env_close_test_prices