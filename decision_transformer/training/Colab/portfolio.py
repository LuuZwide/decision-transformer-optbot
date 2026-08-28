import math
import numpy as np

class portfolio():
  def __init__(self,symbols):
      self.value = 1
      self.current_value = 1
      self.prev_reward = 0
      self.percentage_diff_dict = dict.fromkeys(symbols, 0)
      self.bought =  dict.fromkeys(symbols, False)
      self.selling = dict.fromkeys(symbols, False)
      self.threshold_value = 0.1
      self.buy_threshold = 0.5
      self.prev_current_value = 1
      self.sell_threshold = 0.5
      self.updating = dict.fromkeys(symbols, False)
      self.b_counters = dict.fromkeys(symbols, 0.0)
      self.s_counters = dict.fromkeys(symbols, 0.0)
      self.n_counters = dict.fromkeys(symbols, 0.0)
      self.total_trans = dict.fromkeys(symbols, 0.0)
      self.trans_sum = dict.fromkeys(symbols, 0.0)
      self.bought_values = dict.fromkeys(symbols, 0.0)
      self.b_counter = 0.0
      self.s_counter = 0.0
      self.n_counter = 0.0
      self.stop_loss = -0.1
      self.selling_values =dict.fromkeys(symbols, 0)
      self.sum_interest = dict.fromkeys(symbols, 0)
      self.port_changes = {}
      self.leverage = 1 #Essentially trading with 200 dollars
      self.symbols = symbols
      self.closed = False
      self.done = False
      self.spread_dict = {
          'EURUSD': 0.00001,   # 1.0 pip effective
          'GBPUSD': 0.00001,   # 1.0 pip effective
          'USDJPY': 0.001,     # 1.0 pip effective
          'USDCHF': 0.00001,   # 1.0 pip effective
          'AUDUSD': 0.00001    # 1.0 pip effective
      }
      self.commission_dict = {
          'EURUSD': 0.00015,
          'GBPUSD': 0.00015,
          'USDJPY': 0.00015,
          'USDCHF': 0.00015,
          'AUDUSD': 0.00015
      }
      self.prev_value = 1
      self.can_trade = dict.fromkeys(symbols, True)


  def calculate_returns(self,close_price, type, bought_value,selling_value): # S or B
    if (type == 'S'):
      port_change_diff = (selling_value - close_price)/selling_value
    if (type == 'B'):
      port_change_diff = (close_price - bought_value)/bought_value

    port_change = (port_change_diff * self.leverage)
    percentage_diff = port_change_diff * 100
    return percentage_diff, port_change

  def add_spread(self, close_price, symbol, current_hour=12):
    spread = self.spread_dict[symbol]
    # Corrected: Divide spread by 2 as the values in spread_dict represent total spread.
    bid_price = close_price #- (spread / 2) # Calculate bid from mid-price
    ask_price = close_price #+ (spread / 2) # Calculate ask from mid-price
    return  bid_price, ask_price

  def update_value(self, close_values, action_dict, current_hour=12): # close_value is the exit value
    realized_port_change = 0
    for symbol in self.symbols:
      percentage_diff = 0.0

      close_value = close_values[symbol]
      action = action_dict[symbol]

      if (self.selling[symbol] and self.bought[symbol]):
        print("something wrong 1 ")

      if (action > self.buy_threshold) and self.bought[symbol]: #Update port
        bought_value  = self.bought_values[symbol]
        bid_price, _ = self.add_spread(close_value, symbol, current_hour) # Calculate Bid
        percentage_diff, port_change = self.calculate_returns(bid_price, 'B',bought_value, -1)
        self.port_changes[symbol] = port_change
        self.percentage_diff_dict[symbol] = percentage_diff

      if (action < self.sell_threshold) and self.selling[symbol]: #Update port
        selling_value = self.selling_values[symbol]
        _, ask_price = self.add_spread(close_value, symbol, current_hour) # Calculate Ask
        percentage_diff, port_change = self.calculate_returns(ask_price, 'S', -1 ,selling_value)
        self.port_changes[symbol] = port_change
        self.percentage_diff_dict[symbol] = percentage_diff

      if (action > self.buy_threshold) and not self.bought[symbol]: # First buy
        if self.selling[symbol]: #Close the sell trade
          self.selling[symbol] = False
          self.s_counters[symbol] +=0.5
          selling_value = self.selling_values[symbol]
          #Exit as ask_price
          _, ask_price = self.add_spread(close_value, symbol, current_hour)
          percentage_diff, port_change = self.calculate_returns(ask_price, 'S', -1, selling_value)
          realized_port_change += port_change - self.commission_dict[symbol]
          self.trans_sum[symbol] += percentage_diff # type: ignore
          self.port_changes[symbol] = 0
          self.percentage_diff_dict[symbol] = 0
        else:
          if not self.bought[symbol]:
            self.bought[symbol] = True
            self.b_counters[symbol] += 0.5
            #Buy at ask price
            bid_price, ask_price = self.add_spread(close_value, symbol)
            self.bought_values[symbol] = ask_price
            percentage_diff, port_change = self.calculate_returns(bid_price, 'B', ask_price, -1)
            self.port_changes[symbol] = port_change
            self.percentage_diff_dict[symbol] = percentage_diff

      if (action < self.sell_threshold) and not self.selling[symbol] : # First Sell
        if self.bought[symbol]: #Close the buy trade
          self.b_counters[symbol] += 0.5
          self.bought[symbol] = False
          bought_value = self.bought_values[symbol]
          #Exit at bid price
          bid_price, _ = self.add_spread(close_value, symbol, current_hour)
          percentage_diff, port_change = self.calculate_returns(bid_price, 'B', bought_value, -1)
          realized_port_change += port_change - self.commission_dict[symbol]
          self.trans_sum[symbol] += percentage_diff  # type: ignore
          self.port_changes[symbol] = 0
          self.percentage_diff_dict[symbol] = 0
        else:
          if not self.selling[symbol]:
            self.selling[symbol] = True
            self.s_counters[symbol] +=0.5
            #Enter at bid price
            bid_price, ask_price = self.add_spread(close_value, symbol)
            self.selling_values[symbol] = bid_price
            percentage_diff, port_change = self.calculate_returns(ask_price, 'S', -1 , bid_price)
            self.port_changes[symbol] = port_change
            self.percentage_diff_dict[symbol] = percentage_diff

      if (self.selling[symbol] and self.bought[symbol]):
        print("something wrong 2 ")

    sum_changes = sum(self.percentage_diff_dict.values())

    if sum_changes < self.stop_loss:
      self.done = True

    active_changes = [
      v for s, v in self.port_changes.items()
      if self.bought[s] or self.selling[s]
    ]

    sum_port_changes = np.sum(active_changes) if active_changes else 0
    current_value = self.value * (1 + sum_port_changes + realized_port_change)
    self.value *= (1 + realized_port_change)
    reward = 100 * np.log( self.value/ self.prev_value)

    reward = np.tanh(reward)

    for symbol in self.symbols:
      self.total_trans[symbol] = self.b_counters[symbol] + self.s_counters[symbol] + self.n_counters[symbol]

    # FIXED: Store current_value as prev_value for the next step calculation
    self.prev_current_value = current_value
    self.prev_value = self.value
    self.prev_reward = reward
    return reward, self.percentage_diff_dict, current_value

  def sharpe_return(self, value, rist_free, portfolio_std):
    sharpe_ratio = (value - rist_free)/portfolio_std
    return sharpe_ratio

  def get_value(self):
    return self.value

  def reset(self):
    self.value = 1
    self.prev_reward = 0
    self.current_value = 1
    self.flying_value = 1
    self.trade_counter = 0
    self.bought_values = {}
    self.selling_values = {}
    self.prev_current_value = 1

    self.trans_sum = dict.fromkeys(self.symbols, 0)
    self.b_counters = dict.fromkeys(self.symbols, 0.0)
    self.s_counters = dict.fromkeys(self.symbols, 0.0)
    self.n_counters = dict.fromkeys(self.symbols, 0.0)
    self.sum_interest = dict.fromkeys(self.symbols, 0.0)
    self.updating = False
    self.closed = False
    self.bought_values = dict.fromkeys(self.symbols, 0.0)
    self.selling_values = dict.fromkeys(self.symbols, 0.0)
    self.updating = dict.fromkeys(self.symbols, False)
    self.non_trades = 0
    self.percentage_diff_dict = dict.fromkeys(self.symbols, 0)
    self.bought = dict.fromkeys(self.symbols, False)
    self.b_counter = 0.0
    self.s_counter = 0.0
    self.n_counter = 0.0
    self.done = False
    self.threshold_value = 0.1
    self.selling = dict.fromkeys(self.symbols, False)
    self.port_changes = dict.fromkeys(self.symbols, 0)
    self.prev_value = 1
    self.stop_loss = -0.1
    self.total_trans = dict.fromkeys(self.symbols, 0.0)
    return self.current_value

  def set_threshold(self, threshold):
    self.threshold_value = threshold
    return self.threshold_value

  def get_threshold(self):
    return self.threshold_value