import numpy as np
import pandas as pd
import gym
from gym import spaces
import random
from gymnasium.spaces import MultiDiscrete, Discrete
from decision_transformer.training.Colab.portfolio import portfolio

class ChartEnv(gym.Env):
  metadata = {'render.modes': ['console']}

  def __init__(self, chart_dict, close_prices, symbols, timesteps = 20, episode_length = 4*60, recurrent= False, random_start = False, dates_dict=None, noise_level=0.0):
    super(ChartEnv, self).__init__()

    self.chart = chart_dict
    self.close_prices_dict = close_prices
    self.dates_dict = dates_dict
    #self.chart = self.chart.dropna().values
    #self.chart = self.chart.astype(np.float64)
    #self.chart_dict = chart_dict
    self.symbols = symbols
    self.chart_len,self.cols = self.chart.shape
    self.unreal_pnl_threshold = -0.2
    self.volatility = 1
    self.random_start = random_start
    self.recurrent = recurrent
    self.timesteps = timesteps
    self.episode_length = episode_length
    self.noise_level = noise_level
    self.index = 100
    self.portfolio = portfolio(self.symbols)
    self.done = False
    self.episode_counter = -1
    self.threshold = 0.99
    self.current_value = 1
    self.port_value = 1
    self.sma_prices_dict = {}
    self.ema_prices_dict = {}
    self.unreal_pnl = 0
    self.current_position = []

    # MODIFIED: Change action space to allow 2 discrete actions (0, 1) per symbol
    # self.action_space = MultiDiscrete([2] * len(self.symbols))

    obs_dim = 2 + self.cols + 2*len(self.symbols)
    print('obs_dim : ', obs_dim)

    self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(obs_dim,), dtype=np.float32)

    self.port_values = np.ones((self.chart_len + 1,1))
    self.current_values = np.ones((self.chart_len + 1,1))
    self.port_diffs = np.zeros((self.chart_len + 1,len(self.symbols)))
    self.actions = np.zeros((self.chart_len + 1,len(self.symbols)))
    self.counter = 1

  def get_recurrent_state(self, index):
    sequence = self.chart[index]
    if self.noise_level > 0:
        sequence = sequence + np.random.normal(0, self.noise_level, sequence.shape)
    sequence = np.reshape(sequence, (1,self.timesteps,self.cols))

    port_values = self.port_values[index]
    port_sequence = np.reshape(port_values, (1,self.timesteps,1))

    port_diffs_values = np.array(self.port_diffs[index])
    port_diff_sequence = np.reshape(port_diffs_values, (1,self.timesteps,len(self.symbols)))

    current_values = self.current_values[index]
    current_val_sequence = np.reshape(current_values, (1,self.timesteps,1))

    current_position = []
    for symbol in self.symbols:
        if self.portfolio.bought[symbol]:
            current_position.append(1)
        elif self.portfolio.selling[symbol]:
            current_position.append(-1)
        else:
            current_position.append(0)

    self.current_position = np.array(current_position)
    #print('current position : ', self.current_position)
    current_position_sequence = np.tile(self.current_position, (1, self.timesteps, 1))

    state = np.concatenate((current_val_sequence,port_sequence,port_diff_sequence,current_position_sequence,sequence), axis=2).astype(np.float32)
    return state

  def calculate_reward(self, action):
    look_back_window = 20
    start_index = max(0, self.index - look_back_window)

    recent_values = self.current_values[start_index: self.index]
    recent_returns = np.diff(recent_values, axis=0)

    if len(recent_returns) > 1:
      self.volatility = np.std(recent_returns)
    else:
      self.volatility = 0.0

    self.action_dict = dict(zip(self.symbols, np.squeeze(action)))
    close_prices = {}
    for symbol in self.symbols:
      close_prices[symbol] = self.close_prices_dict[symbol].iloc[self.index]

    current_hour = 12
    if self.dates_dict is not None and self.symbols[0] in self.dates_dict:
        current_hour = self.dates_dict[self.symbols[0]].iloc[self.index].hour

    raw_reward, port_diffs_dict, current_value = self.portfolio.update_value(close_values = close_prices, action_dict = self.action_dict, current_hour = current_hour)
    self.current_value = current_value
    self.port_value = self.portfolio.get_value()

    self.port_values[self.index + 1] = self.portfolio.get_value()
    self.current_values[self.index + 1] = current_value
    self.port_diffs[self.index + 1] = np.clip(np.array(list(port_diffs_dict.values())),-1,1)
    self.actions[self.index + 1] = np.array(list(self.action_dict.values()))

    self.unreal_pnl = sum(self.portfolio.percentage_diff_dict.values())

    reward = raw_reward
    return reward

  def reset(self, seed = None):
    _ = self.portfolio.reset()
    self.action_dict = {}
    self.port_value = 1
    self.port_values = np.ones((self.chart_len + 1,1))
    self.current_values = np.ones((self.chart_len + 1,1))

    self.actions = np.zeros((self.chart_len + 1,len(self.symbols)))
    self.port_diffs = np.zeros((self.chart_len + 1,len(self.symbols)))
    self.unreal_pnl = 0
    self.current_position = []

    end_index = self.chart_len - self.episode_length - 10

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
    else:
      state = np.squeeze(state)
    self.prev_action = None

    self.counter = 1
    return state,{}

  def step(self, action):
    trunc = False
    done = False
    reward = self.calculate_reward(action)

    if(self.counter == self.episode_length) or (self.current_value < self.threshold):
      for symbol in self.symbols:
        self.portfolio.trans_sum[symbol] += self.portfolio.percentage_diff_dict[symbol]  # type: ignore
      reward = np.tanh( 100 * np.log(self.current_value / self.portfolio.value))

      if (self.counter == self.episode_length):
        trunc = True
      if (self.current_value < self.threshold):
        done = True

    self.index += 1

    next_state = self.get_recurrent_state(self.index)
    if not self.recurrent:
      next_state = next_state.flatten()
    else:
      next_state = np.squeeze(next_state)

    self.counter += 1
    self.prev_action = action
    actions = []
    actions = np.array(np.round(list(self.action_dict.values()),2))

    total_trans = []
    total_trans = np.array(list(self.portfolio.total_trans.values()))

    trans_sum = []
    trans_sum = np.array(np.round(list(self.portfolio.trans_sum.values()),3))

    #if self.current_value < 0.995:
    #  done = True

    info = {
        'current_value': self.current_value,
        'total_trans': total_trans,
        'reward': reward,
        'index' : self.index,
        'action_dict' : actions,
        'port_value' : self.port_value,
        'trans_sum' : trans_sum
    }
    return next_state, reward, done,trunc, info

  def close(self):
    pass

