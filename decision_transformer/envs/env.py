import portfolio
import numpy as np
import chart
import utils
from datetime import datetime
import os
import meta
import json

class Env():

    def __init__(self, symbols,p_user):
        self.index = 0 # basically counts as counter 
        self.symbols = symbols
        self.portfolio = portfolio.portfolio(self.symbols)
        self.timesteps = 1

        self.port_values = np.ones((10000,1)) 
        self.port_diffs = np.zeros((10000,len(self.symbols)))
        self.actions = np.zeros((10000,len(self.symbols)))
        self.current_values = np.ones((10000,1))
        self.state_chart = np.zeros((10000,len(self.symbols)*4)) 
        self.chart_obj = chart.Chart(self.symbols)
        self.value = 1

        self.mean, self.std = utils.read_from_csv(dir = './info/ChartStats/')
        
        if p_user == 'live':
            self.threshold = 0.997
        else: 
            self.threshold = 0.98 

        self.current_value = 1
        self.cols = len(self.symbols)
        self.action_dict = {}

    
    def reset(self, load_from = 'R'):
        # R -> Reset
        # T -> Today
        # Y -> Yesterday

        self.action_dict = {}
        yesterday = utils.get_previous_weekday(datetime.now()).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        dir = './info/PortfolioStats/' + today + '/'

        if load_from == 'T' : #today
            dir = './info/PortfolioStats/' + today + '/'
            print('Loading from today dir : ', dir)
            self.load_state(dir)
            _ = self.portfolio.reset( dir, True)
        elif load_from == 'Y' : #Yesterday
            dir = './info/PortfolioStats/' + yesterday + '/'
            print('Loading from yesterday dir : ', dir)
            self.load_state(dir)
            _ = self.portfolio.reset( dir, True)
        else:
            self.port_values = np.zeros((10000,1))
            self.port_diffs = np.zeros((10000,len(self.symbols)))
            self.current_values = np.zeros((10000,1))
            self.actions = np.zeros((10000,len(self.symbols)))
            self.index = self.timesteps # Start at the timestep index
            _ = self.portfolio.reset( dir, False)
            meta.close_all(self.symbols)

        return 

    def get_recurrent_state(self, index):
        print('index : ', index)
        
        sequence = self.state_chart[index]
        sequence = np.reshape(sequence, (1,self.timesteps,self.cols*4))

        port_values = self.port_values[index]
        port_sequence = np.reshape(port_values, (1,self.timesteps,1))

        port_diffs_values = self.port_diffs[index]
        port_diff_sequence = np.reshape(port_diffs_values, (1,self.timesteps,len(self.symbols)))

        current_values = self.current_values[index]
        current_value_sequence = np.reshape(current_values, (1,self.timesteps,1))

        current_position = []
        for symbol in self.symbols:
            if self.portfolio.bought[symbol]:
                current_position.append(1)
            elif self.portfolio.selling[symbol]:
                current_position.append(-1)
            else:
                current_position.append(0)

        self.current_position = np.array(current_position)
        current_position = np.tile(self.current_position, (1, self.timesteps, 1))   

        state = np.concatenate((port_sequence,current_value_sequence,sequence,port_diff_sequence,current_position), axis=2).astype(np.float64)
        
        return state  

    def calculate_reward(self,action):
        self.action_dict = dict(zip(self.symbols, np.squeeze(action)))
        
        self.close_prices = {}
        for symbol in self.symbols:
            self.close_prices[symbol] = self.close_prices_dict[symbol][-1] 
        
        port_diffs, current_value = self.portfolio.update_value(close_values = self.close_prices, actions = self.action_dict )
        self.current_value = current_value

        self.value = self.portfolio.value
        self.port_values[self.index] = self.portfolio.value
        self.current_values[self.index] = current_value
        self.port_diffs[self.index] = np.clip(np.array(list(port_diffs.values())),-1,1)
        self.actions[self.index] = np.array(list(self.action_dict.values()))
    
    def return_current_state(self):
        self.chart,self.close_prices_dict = self.chart_obj.process() 
        self.state_chart[self.index] = self.chart[-1]
        print('State chart : ', self.state_chart[self.index])
        state = self.get_recurrent_state(self.index)
        self.index += 1
        state = state.flatten()
        return state
    
    def step(self, action):
        
        self.calculate_reward(action)
        print('Current value : ', self.current_value)
        done = False

        if ((self.current_value < self.threshold)):
            done = True
            print('__________done_______________')
            meta.close_all(self.symbols)

        return done
    
    def save_env(self):
        today = datetime.now().strftime("%Y-%m-%d")
        dir = './info/PortfolioStats/' + today + '/'
        #If directory does not exist create it
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.save_state(dir)
        self.portfolio.save_values(dir)

        return
    
    def save_state(self, dir):
        env_dic = {
            'index': self.index,
            'current' : self.current_value,
            'current_value': self.current_values.tolist(),
            'port_values': self.port_values.tolist(),
            'port_diffs': self.port_diffs.tolist(),
            'actions': self.actions.tolist()
        }
        try:
            with open(dir + 'env.json', 'w') as f:
                json.dump(env_dic, f)
            return
        except FileNotFoundError:
            with open(dir + 'env.json', 'w') as f:
               json.dump(env_dic, f)
        return  
    
    def load_state(self, dir):
        try:
            with open(dir + 'env.json', 'r') as f:
                env_dic = json.load(f)
                self.index = env_dic['index']
                self.current_value = env_dic['current']
                self.current_values = np.array(env_dic['current_value'])
                self.port_values = np.array(env_dic['port_values'])
                self.port_diffs = np.array(env_dic['port_diffs'])
                self.actions = np.array(env_dic['actions'])
        except FileNotFoundError:
            print("Env file not found. Starting with default values.")
            self.reset()
        return
    
    def reset_if_needed(self):
        if self.index >= 1440:
            print('Resetting env after 1440 steps')
            self.save_env()
            self.reset(load_from = 'R')
    

 