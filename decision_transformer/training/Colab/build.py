from decision_transformer.training.Colab.utils import create_feature_set
from decision_transformer.training.Colab.ChartEnv import ChartEnv
import os
import pandas as pd


def build_env():
    symbols = ['EURUSD', 'GBPUSD','USDJPY']
    df_charts = {}

    # build charts
    for file in os.listdir("/home/lnxumalo/lustre/Experiment04/odt_sand/opt/decision-transformer-optbot/decision_transformer/training/Colab/datafiles"):
        if file.endswith(".pkl"):
            symbol = file.replace(".pkl", "")
            df_charts[symbol] = pd.read_pickle(os.path.join("/home/lnxumalo/lustre/Experiment04/odt_sand/opt/decision-transformer-optbot/decision_transformer/training/Colab/datafiles", file))
    
    #for symbol in df_charts.keys():
    #    print(symbol,df_charts[symbol].shape)

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
    
    #for symbol in symbols:
    #    print('Train : ',symbol, env_charts[symbol].shape,env_close_prices[symbol].shape )
    #    print('Test : ',symbol, env_test_charts[symbol].shape, env_close_test_prices[symbol].shape)
    #    print('\n')

    train_env = ChartEnv(chart_dict = env_charts, close_prices= env_close_prices , symbols = symbols,timesteps = 1, episode_length = 1000, recurrent= False, random_start=True) 
    test_env = ChartEnv(chart_dict = env_test_charts, close_prices= env_close_test_prices , symbols = symbols,timesteps = 1, episode_length = 1000, recurrent= False, random_start=True)

    return train_env, test_env

build_env()