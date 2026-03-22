import math
import numpy as np

class portfolio():
  def __init__(self,symbols):
      self.value = 1
      self.prev_reward = 0
      self.percentage_diff_dict = dict.fromkeys(symbols, 0)
      self.bought =  dict.fromkeys(symbols, False)
      self.selling = dict.fromkeys(symbols, False)
      self.threshold_value = 0.1
      self.updating = dict.fromkeys(symbols, False)
      self.b_counters = dict.fromkeys(symbols, 0.0)
      self.s_counters = dict.fromkeys(symbols, 0.0)
      self.n_counters = dict.fromkeys(symbols, 0.0)
      self.total_trans = dict.fromkeys(symbols, 0.0)
      self.bought_values = {}
      self.b_counter = 0
      self.s_counter = 0
      self.n_counter = 0
      self.stop_loss = -0.1
      self.selling_values = {}
      self.sum_interest = dict.fromkeys(symbols, 0)
      self.port_changes = {}
      self.leverage = 1 #Essentially trading with 200 dollars
      self.symbols = symbols
      self.closed = False
      self.done = False
      self.spread_dict = {
            'EURUSD': 0.00010,  # ~1.0 pip
            'GBPUSD': 0.00012,  # ~1.2 pips
            'USDJPY': 0.010,    # ~1.0 pip (JPY pairs use 0.01)
            'USDCHF': 0.00015,  # ~1.5 pips
            'AUDCAD': 0.00018,  # ~1.8 pips
            'AUDCHF': 0.00020,  # ~2.0 pips
            'EURCAD': 0.00020,  # ~2.0 pips
            'AUDUSD': 0.00012   # ~1.2 pips
      }
      self.prev_value = 1


  def calculate_returns(self,close_price, type, bought_value,selling_value): # S or B
    if (type == 'S'):
      port_change_diff = (selling_value - close_price)/selling_value
    if (type == 'B'):
      port_change_diff = (close_price - bought_value)/bought_value

    port_change = port_change_diff * self.leverage
    percentage_diff = port_change_diff * 100
    return percentage_diff, port_change

  def add_spread(self, close_price, symbol):
    bid_price = close_price
    ask_price = close_price + self.spread_dict[symbol]
    return  bid_price, ask_price

  def update_value(self, close_values, action_dict, volatility): # close_value is the exit value
    for symbol in self.symbols:
      percentage_diff = 0

      close_value = close_values[symbol]
      action = action_dict[symbol]

      if (self.selling[symbol] and self.bought[symbol]):
        print("something wrong 1 ")

      if (action > self.threshold_value) and self.bought[symbol]: #Update port
        bought_value  = self.bought_values[symbol]
        bid_price, _ = self.add_spread(close_value, symbol) # Calculate Bid
        percentage_diff, port_change = self.calculate_returns(bid_price, 'B',bought_value, -1)
        self.port_changes[symbol] = port_change
        self.percentage_diff_dict[symbol] = percentage_diff

      if (action < -1*self.threshold_value) and self.selling[symbol]: #Update port
        selling_value = self.selling_values[symbol]
        _, ask_price = self.add_spread(close_value, symbol) # Calculate Ask
        percentage_diff, port_change = self.calculate_returns(ask_price, 'S', -1 ,selling_value)
        self.port_changes[symbol] = port_change
        self.percentage_diff_dict[symbol] = percentage_diff

      if (action > self.threshold_value) and not self.bought[symbol]: # First buy
        if self.selling[symbol]: #Close the sell trade
          self.selling[symbol] = False
          self.s_counters[symbol] +=0.5
          selling_value = self.selling_values[symbol]
          #Exit as ask_price
          _, ask_price = self.add_spread(close_value, symbol)
          percentage_diff, port_change = self.calculate_returns(ask_price, 'S', -1, selling_value)
          self.value *= (1 + port_change)
          self.port_changes[symbol] = 0
          self.percentage_diff_dict[symbol] = 0

        if not self.bought[symbol]:
          self.bought[symbol] = True
          self.b_counters[symbol] += 0.5
          #Buy at ask price
          _,ask_price = self.add_spread(close_value, symbol)
          self.bought_values[symbol] = ask_price
          percentage_diff, port_change = self.calculate_returns(close_value, 'B', ask_price, -1 )
          self.port_changes[symbol] = 0
          self.percentage_diff_dict[symbol] = 0

      if (action < -1*self.threshold_value) and not self.selling[symbol] : # First Sell
        if self.bought[symbol]: #Close the buy trade
          self.b_counters[symbol] += 0.5
          self.bought[symbol] = False
          bought_value = self.bought_values[symbol]
          #Exit at bid price
          bid_price, _ = self.add_spread(close_value, symbol)
          percentage_diff, port_change = self.calculate_returns(bid_price, 'B', bought_value, -1)
          self.value *= (1 + port_change)
          self.port_changes[symbol] = 0
          self.percentage_diff_dict[symbol] = 0

        if not self.selling[symbol]:
          self.selling[symbol] = True
          self.s_counters[symbol] +=0.5
          #Enter at bid price
          bid_price, _ = self.add_spread(close_value, symbol)
          self.selling_values[symbol] = bid_price
          percentage_diff, port_change = self.calculate_returns(bid_price, 'S', -1 , close_value)
          self.port_changes[symbol] = 0
          self.percentage_diff_dict[symbol] = 0

      if ((action < self.threshold_value) and (action > -1*self.threshold_value)) and self.bought[symbol]: # Close the buy
        bought_value = self.bought_values[symbol]
        self.b_counters[symbol] += 0.5
        #Exit at bid_price
        self.b_counter += 1
        bid_price, _ = self.add_spread(close_value, symbol)
        percentage_diff, port_change = self.calculate_returns(bid_price, 'B',bought_value,-1)
        self.value *= (1 + port_change)
        self.port_changes[symbol] = 0
        self.percentage_diff_dict[symbol] = 0
        self.bought[symbol] = False
        self.selling[symbol] = False

      if ((action < self.threshold_value) and  (action > -1*self.threshold_value)) and self.selling[symbol]: #Close the sell
        self.closed = True
        self.s_counters[symbol] +=0.5
        selling_value = self.selling_values[symbol]
        #Exit at ask_price
        self.s_counter += 1
        _, ask_price = self.add_spread(close_value, symbol)
        percentage_diff, port_change = self.calculate_returns(ask_price, 'S',-1,selling_value)
        self.value *= (1 + port_change)
        self.port_changes[symbol] = 0
        self.percentage_diff_dict[symbol] = 0
        self.selling[symbol] = False
        self.bought[symbol] = False

      if ((action < self.threshold_value) and  (action > -1*self.threshold_value))  and not ( self.bought[symbol]  or self.selling[symbol] ):
        self.updating[symbol] = False
        self.n_counter += 1
      else:
        self.updating[symbol] = True

      if (self.selling[symbol] and self.bought[symbol]):
        print("something wrong 2 ")

    sum_changes = sum(self.percentage_diff_dict.values())

    if sum_changes < self.stop_loss:
      self.done = True

    active_changes = [
    v for s, v in self.port_changes.items()
    if self.bought[s] or self.selling[s]
    ]

    sum_port_changes = np.mean(active_changes) if active_changes else 0
    current_value = self.value * (1 + sum_port_changes)
    current_value = max(current_value, 1e-8)

    reward = np.log(current_value / self.prev_value)
    risk_pen = 0.1 * volatility

    for symbol in self.symbols:
      self.total_trans[symbol] = self.b_counters[symbol] + self.s_counters[symbol] + self.n_counters[symbol]

    reward = reward

    self.prev_value = current_value
    self.prev_reward = reward
    return reward,self.percentage_diff_dict, current_value


  def log_return(self, value):
    return math.log(value)

  def sharpe_return(self, value, rist_free, portfolio_std):
    sharpe_ratio = (value - rist_free)/portfolio_std
    return sharpe_ratio

  def get_value(self):
    return self.value

  def reset(self):
    self.prev_reward = 0
    self.value = 1
    self.flying_value = 1
    self.trade_counter = 0
    self.bought_values = {}
    self.selling_values = {}
    self.b_counters = dict.fromkeys(self.symbols, 0.0)
    self.s_counters = dict.fromkeys(self.symbols, 0.0)
    self.n_counters = dict.fromkeys(self.symbols, 0.0)
    self.sum_interest = dict.fromkeys(self.symbols, 0)
    self.closed = False
    self.updating = dict.fromkeys(self.symbols, False)
    self.non_trades = 0
    self.percentage_diff_dict = dict.fromkeys(self.symbols, 0)
    self.bought = dict.fromkeys(self.symbols, False)
    self.b_counter = 0
    self.s_counter = 0
    self.n_counter = 0
    self.done = False
    self.threshold_value = 0.1
    self.selling = dict.fromkeys(self.symbols, False)
    self.port_changes = dict.fromkeys(self.symbols, 0)
    self.prev_value = 1
    self.stop_loss = -0.1
    self.total_trans = dict.fromkeys(self.symbols, 0.0)
    return self.value

  def set_threshold(self, threshold):
    self.threshold_value = threshold
    return self.threshold_value

  def get_threshold(self):
    return self.threshold_value