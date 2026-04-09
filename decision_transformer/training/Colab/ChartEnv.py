import numpy as np
import pandas as pd
import gym
from gym import spaces
import random
from decision_transformer.training.Colab.portfolio import portfolio

class ChartEnv(gym.Env):
  metadata = {'render_modes': ['console']}
  
  def __init__(self, chart,close_prices, symbols ,timesteps = 20, episode_length = 4*60, recurrent= False, random_start = False):
    super(ChartEnv, self).__init__()
    self.chart = chart
    self.close_prices_dict = close_prices
    self.symbols = symbols
    self.chart_len,self.cols = self.chart.shape
    #print(f"DEBUG: ChartEnv __init__: self.cols = {self.cols}, len(self.symbols) = {len(self.symbols)}") # Debug print

    self.unreal_pnl_threshold = -0.2
    #print('columns ', self.cols)

    self.random_start = random_start
    self.recurrent = recurrent
    self.timesteps = timesteps
    self.episode_length = episode_length
    self.index = 0
    self.portfolio = portfolio(self.symbols)
    self.done = False
    self.episode_counter = -1
    self.threshold = 0.997
    self.current_value = 1
    self.port_value = 1
    self.sma_prices_dict = {}
    self.ema_prices_dict = {}
    self.unreal_pnl = 0
    self.current_position = []

    self.action_space = spaces.Box(low=-1, high=1,shape=(len(self.symbols),), dtype=np.float32)
    if not self.recurrent:
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(26,), dtype=np.float64)
    else:
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(26,), dtype=np.float64)

    self.port_values = np.zeros((self.chart_len + 1,1)) # Increased size by 1
    self.current_values = np.zeros((self.chart_len + 1,1))
    self.port_diffs = np.zeros((self.chart_len + 1,len(self.symbols))) # Increased size by 1
    self.actions = np.zeros((self.chart_len + 1,len(self.symbols))) # Increased size by 1
    self.counter = 0

  def get_recurrent_state(self, index):

    sequence = self.chart[index]
    sequence = np.reshape(sequence, (1,self.timesteps,self.cols))
    #print(f"DEBUG: get_recurrent_state: sequence shape: {sequence.shape}")

    port_values = self.port_values[index]
    port_sequence = np.reshape(port_values, (1,self.timesteps,1))
    #print(f"DEBUG: get_recurrent_state: port_sequence shape: {port_sequence.shape}")

    port_diffs_values = np.array(self.port_diffs[index])
    port_diff_sequence = np.reshape(port_diffs_values, (1,self.timesteps,len(self.symbols)))
    #print(f"DEBUG: get_recurrent_state: port_diff_sequence shape: {port_diff_sequence.shape}")

    current_values = self.current_values[index]
    current_value_sequence = np.reshape(current_values, (1,self.timesteps,1))
    #print(f"DEBUG: get_recurrent_state: current_value_sequence shape: {current_value_sequence.shape}")

    current_position = []
    for symbol in self.symbols:
        if self.portfolio.bought[symbol]:
            current_position.append(1)
        elif self.portfolio.selling[symbol]:
            current_position.append(-1)
        else:
            current_position.append(0)

    self.current_position = np.array(current_position)
    current_position_sequence = np.tile(self.current_position, (1, self.timesteps, 1)) # Renamed to avoid shadowing
    #print(f"DEBUG: get_recurrent_state: current_position_sequence shape: {current_position_sequence.shape}")

    state = np.concatenate((port_sequence,current_value_sequence,sequence,port_diff_sequence,current_position_sequence), axis=2).astype(np.float64)
    #print(f"DEBUG: get_recurrent_state: state shape before flatten/squeeze: {state.shape}") # Debug print
    return state

  def calculate_reward(self, action):

    lock_back_window = 20
    start_index = max(0, self.counter - lock_back_window)

    recent_values = self.port_values[self.index - start_index: self.index]
    recent_returns = np.diff(recent_values, axis=0)

    if len(recent_returns) > 0:
      volatility = np.std(recent_returns)
    else:
      volatility = 0.0

    self.action_dict = dict(zip(self.symbols, np.squeeze(action))) #{'a': 1, 'b': 2, 'c': 3}
    close_prices = {}
    for symbol in self.symbols:
      close_prices[symbol] = self.close_prices_dict[symbol].iloc[self.index]

    reward, port_diffs_dict, current_value = self.portfolio.update_value(close_values = close_prices, action_dict = self.action_dict, volatility = volatility)
    self.current_value = current_value
    self.port_value = self.portfolio.get_value()
    #print(np.log(current_value))

    self.port_values[self.index + 1] = self.portfolio.get_value()
    self.current_values[self.index + 1] = current_value
    self.port_diffs[self.index + 1] = np.clip(np.array(list(port_diffs_dict.values())),-1,1)
    self.actions[self.index + 1] = np.array(list(self.action_dict.values()))

    self.unreal_pnl = sum(self.portfolio.percentage_diff_dict.values())
    return reward

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    _ = self.portfolio.reset()
    self.action_dict = {}
    self.port_value = 1
    self.port_values = np.zeros((self.chart_len + 1,1)) # Increased size by 1
    self.current_values = np.zeros((self.chart_len + 1,1))
    self.actions = np.zeros((self.chart_len + 1,len(self.symbols))) # Increased size by 1
    self.port_diffs = np.zeros((self.chart_len + 1,len(self.symbols))) # Increased size by 1
    self.unreal_pnl = 0
    self.current_position = []

    end_index = self.chart_len - self.episode_length

    if self.random_start:
      self.index = random.randint(self.timesteps,end_index)
    else:
      self.index +=1
      if self.index > end_index:
        self.index = 100

    self.start_index = self.index
    state = self.get_recurrent_state(self.index)

    if not self.recurrent:
      state = state.flatten()
      #print(f"DEBUG: reset: state shape after flatten: {state.shape}") # Debug print
    else:
      state = np.squeeze(state)
    self.prev_action = None

    self.counter = 0
    return state

  def step(self, action):

    reward = self.calculate_reward(action)

    close_prices = {}
    for symbol in self.symbols:
      close_prices[symbol] = self.close_prices_dict[symbol].iloc[self.index]

    next_index = self.index + 1
    self.counter += 1

    terminated = self.current_value < 0.997 
    truncated = self.counter >= self.episode_length or next_index >= self.chart_len

    if next_index < self.chart_len:
      self.index = next_index

    next_state = self.get_recurrent_state(self.index)
    if not self.recurrent:
      next_state = next_state.flatten()
      #print(f"DEBUG: step: next_state shape after flatten: {next_state.shape}") # Debug print
    else:
      next_state = np.squeeze(next_state)

    self.prev_action = action
    actions = []
    actions = np.array(np.round(list(self.action_dict.values()),2))

    total_trans = []
    total_trans = np.array(list(self.portfolio.total_trans.values()))

    #log info variables

    #log s_counter and b_counter
    info = {
        'current_value': self.current_value,
        'total_trans': total_trans,
        'reward': reward,
        'close_dict' : close_prices,
        'index' : self.index,
        'action_dict' : actions,
        'port_value' : self.port_value
    }
    
    return next_state, reward, terminated, info

  def close(self):
    pass