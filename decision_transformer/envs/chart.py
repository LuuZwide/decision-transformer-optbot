# This class - loads data from polygon 
# Transforms data into chart
# outputs chart data

# Main function is return_current_chart as a vector not DICT
# checks date a symbols
# returns the latest chart info for all symbols

# Other functions
# 1. Get latest candels from polygon for all symbols
# 2. Transform data using wavelet transform
# 3. add stat values 

# Process
# For each symbol in symbols:
#   Get latest data from polygon - convert datetime (oldest to newest) asc order
#   denoise data
#   On last symbol add data column
#   Drop na records
#   add to dict 
#
# Return vector not dict
import pandas as pd
import requests
from datetime import datetime, timedelta
from polygon import RESTClient
import utils
import numpy as np
import socket

FOREX_CLIENT = RESTClient("v40nosrbRie8q4wWrvpMu0f2pUB16Edr")

class Chart():
    def __init__(self, symbols):

        self.symbols = symbols

    def process(self):
        live_chart_dict = {}
        live_close_price_dict = {}
        
        for symbol in self.symbols:
            try: 
                symbol_chart = self.get_polygon_chart(symbol = symbol) # Return latest chart Data
                features, close_prices = self.create_feature_set(symbol_chart) 

                live_chart_dict[symbol] = features
                live_close_price_dict[symbol] = close_prices
           
            except RuntimeError as e:
                print(f"Error processing symbol {symbol}: {e}")
                continue
    
        if not live_chart_dict:
            raise RuntimeError("No valid chart data could be retrieved for any symbol.")
        
        live_chart = pd.concat(live_chart_dict, axis=1).dropna().values
        live_chart = live_chart.astype(np.float64)
        return live_chart, live_close_price_dict
            
    
    def get_polygon_chart(self, symbol = None):
        today = datetime.today()
        chart_to_date = today.strftime('%Y-%m-%d') # to today
        chart_from_date = today - timedelta(days=(1)) #From Yesterday
        chart_from_date = chart_from_date.strftime('%Y-%m-%d')
        
        try: 
            df = self.generateCandleSticks(client=FOREX_CLIENT, start_date = chart_from_date, end_date = chart_to_date, limit = 3000, symbol = symbol )
            return df
        except requests.exceptions.ConnectionError as e:
            if utils.is_dns_error(e):
                print(f"DNS error occurred while fetching data for {symbol}")
                raise RuntimeError(f"DNS error occurred while fetching data for {symbol}') from e")
            
            print(f"Connection error occurred while fetching data for {symbol}: {e}")
            raise RuntimeError(f"Connection error occurred while fetching data for {symbol}') from e")
        except Exception as e:
            print(f"An error occurred while fetching data for {symbol}: {e}")
            raise RuntimeError(f"An error occurred while fetching data for {symbol}') from e")

    def generateCandleSticks(self,client, start_date  , end_date ,limit , symbol= None):
        symbol = 'C:' + symbol # type: ignore
        response =  client.get_aggs(ticker = symbol ,multiplier = 1,timespan = 'minute',
                                  from_= start_date, to = end_date, sort = 'asc',limit = limit) # Oldest -> Latest
        
        df = pd.DataFrame(response)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        final_df = df[[ 'open', 'high', 'low', 'close','date']]
        print(symbol,final_df.values[-1] )
        return final_df
    
    def create_feature_set(self, df):
        
        close_prices = df['close'].values
        df['raw_return'] = np.log(df['close'] / df['close'].shift(1))

        kalman_denoised_data = utils.kalman_denoise(df['close'].values)
        df['kalman_denoised'] = kalman_denoised_data
        df['kalman_ret'] = np.log(df['kalman_denoised'] / df['kalman_denoised'].shift(1))

        df['divergence'] = (df['close'] - df['kalman_denoised'])/df['kalman_denoised']
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_dist'] = (df['close'] - df['ema_50']) / df['ema_50']

        df.dropna(inplace=True)
        features = df[['raw_return','kalman_ret','ema_dist','divergence']]

        return features ,close_prices
    
    
       