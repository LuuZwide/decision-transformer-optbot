
import meta
import json

class portfolio():

    def __init__(self,symbols):
        self.symbols = symbols
        self.leverage = 1
        self.bought =  dict.fromkeys(symbols, False)
        self.bought_values = {}
        self.percentage_diff_dict = dict.fromkeys(symbols, 0)
        self.selling = dict.fromkeys(symbols, False)
        self.percentage_diff_dict = {}
        self.port_changes = {}
        self.selling_values = {}
        self.threshold_value = 0.1
        self.updating = dict.fromkeys(symbols, False)
        self.counter = 0
        self.spread_dict = {'EURUSD': 0.0001, 'GBPUSD': 0.00012, 'USDJPY': 0.01, 'USDCHF' : 0.00015}
        self.initial_equity = meta.get_initial_equity()


    def calculate_returns(self,close_price, type, bought_value,selling_value):
        if (type == 'S'):
            port_change = (selling_value - close_price)/selling_value
        if (type == 'B'):
            port_change = (close_price - bought_value)/bought_value
        
        port_change = port_change * self.leverage
        percentage_diff = port_change * 100
        return percentage_diff, port_change
    
    def add_spread(self, close_price, symbol):
        bid_price = close_price
        ask_price = close_price + self.spread_dict[symbol]*0.4
        return  bid_price, ask_price
    
    def update_value(self, close_values, actions):
        for symbol in self.symbols:
            percentage_diff = 0
            self.counter += 1

            close_value = close_values[symbol]
            action = actions[symbol]

            if (action > self.threshold_value) and self.bought[symbol]: #Update port
                bought_value  = self.bought_values[symbol]
                bid_price, _ = self.add_spread(close_value, symbol) 
                percentage_diff, port_change = self.calculate_returns(bid_price, 'B',bought_value, -1)
                self.port_changes[symbol] = port_change
                self.percentage_diff_dict[symbol] = percentage_diff

            if (action < -1*self.threshold_value) and self.selling[symbol]: #Update port
                selling_value = self.selling_values[symbol]
                _, ask_price = self.add_spread(close_value, symbol) 
                percentage_diff, port_change = self.calculate_returns(ask_price, 'S', -1 ,selling_value)
                self.port_changes[symbol] = port_change
                self.percentage_diff_dict[symbol] = percentage_diff

            if (action > self.threshold_value) and not self.bought[symbol]: # First buy
                if self.selling[symbol]:#Close the sell trade
                    self.selling[symbol] = False
                    selling_value = self.selling_values[symbol]
                    #Exit at ask_price
                    _, ask_price = self.add_spread(close_value, symbol)
                    percentage_diff, port_change = self.calculate_returns(ask_price, 'S', -1, selling_value)
                    self.value += port_change
                    self.port_changes[symbol] = 0
                    self.percentage_diff_dict[symbol] = 0               
                    #Meta
                    meta.closePositions(symbol)
                
                if not self.bought[symbol]:
                    self.bought[symbol] = True
                    #Buy at ask_price
                    _,ask_price = self.add_spread(close_value, symbol)
                    self.bought_values[symbol] = ask_price
                    percentage_diff, port_change = self.calculate_returns(close_value, 'B', ask_price, -1 )
                    self.port_changes[symbol] = 0
                    self.percentage_diff_dict[symbol] = 0
                    meta.BUY(symbol, self.initial_equity, self.counter)       

            if (action < -1*self.threshold_value) and not self.selling[symbol] : # First Sell
                if self.bought[symbol]: #Close the buy trade
                    self.bought[symbol] = False
                    bought_value = self.bought_values[symbol]
                    #Exit at bid_price
                    bid_price, _ = self.add_spread(close_value, symbol)
                    percentage_diff, port_change = self.calculate_returns(bid_price, 'B', bought_value, -1)
                    self.value += port_change
                    self.port_changes[symbol] = 0
                    self.percentage_diff_dict[symbol] = 0
                    meta.closePositions(symbol)
                
                if not self.selling[symbol]:
                    self.selling[symbol] = True
                    #Enter at bid_price
                    bid_price, _ = self.add_spread(close_value, symbol)                   
                    self.selling_values[symbol] = bid_price
                    percentage_diff, port_change = self.calculate_returns(bid_price, 'S', -1 , close_value)
                    self.port_changes[symbol] = 0
                    self.percentage_diff_dict[symbol] = 0  
                    meta.SELL(symbol, self.initial_equity, self.counter)

            if ((action < self.threshold_value) and (action > -1*self.threshold_value)) and self.bought[symbol]: # Close the buy
                bought_value = self.bought_values[symbol]
                #Exit at bid_price
                bid_price, _ = self.add_spread(close_value, symbol)
                percentage_diff, port_change = self.calculate_returns(bid_price, 'B',bought_value,-1)
                self.value += port_change
                self.port_changes[symbol] = 0
                self.percentage_diff_dict[symbol] = 0
                self.bought[symbol] = False
                self.selling[symbol] = False   
                meta.closePositions(symbol)   

            if ((action < self.threshold_value) and  (action > -1*self.threshold_value)) and self.selling[symbol]: #Close the sell
                self.closed = True
                #Exit at ask_price
                _, ask_price = self.add_spread(close_value, symbol)
                selling_value = self.selling_values[symbol]
                percentage_diff, port_change= self.calculate_returns(ask_price, 'S',-1,selling_value)
                self.value += port_change
                self.port_changes[symbol] = 0
                self.percentage_diff_dict[symbol] = 0
                self.selling[symbol] = False
                self.bought[symbol] = False       
                meta.closePositions(symbol) 

        active_changes = [
            v for s, v in self.port_changes.items()
            if self.bought[s] or self.selling[s]
            ]
        sum_port_changes = sum(active_changes) / len(active_changes) if active_changes else 0
        current_value = self.value + sum_port_changes
        current_value = max(current_value, 1e-8) #Prevent negative or zero value

        return self.percentage_diff_dict, current_value    
    
    def reset(self,directory, load_from_previous = False):
        self.threshold_value = 0.1
        if load_from_previous:
            self.load_values(directory)
        else:
            self.value = 1
            self.flying_value = 1
            self.trade_counter = 0
            self.bought_values = {}
            self.selling_values = {}
            self.updating = False
            self.closed = False
            self.updating = dict.fromkeys(self.symbols, False)
            self.non_trades = 0
            self.percentage_diff_dict = dict.fromkeys(self.symbols, 0)
            self.bought = dict.fromkeys(self.symbols, False)
            self.selling = dict.fromkeys(self.symbols, False)
            self.port_changes = dict.fromkeys(self.symbols, 0)
            self.b_counter = 0
            self.s_counter = 0      
            self.initial_equity = meta.get_initial_equity()   
        
        return self.value
    
    def save_values(self, dir):
        portfolio_dict = {
            'value': self.value,
            'bought': self.bought,
            'bought_values': self.bought_values,
            'selling': self.selling,
            'selling_values': self.selling_values,
            'percentage_diff_dict': self.percentage_diff_dict,
            'port_changes': self.port_changes,
            'trade_counter': self.trade_counter,
            'counter': self.counter
        }
        try:
            with open(dir + 'portfolio.json', 'w') as f:
                json.dump(portfolio_dict, f)
        except FileNotFoundError:
            with open(dir + 'portfolio.json', 'w') as f:
                json.dump(portfolio_dict, f)
        return
    
    def load_values(self, dir):
        try:
            with open(dir + 'portfolio.json', 'r') as f:
                portfolio_dict = json.load(f)
                self.value = portfolio_dict['value']
                self.bought = portfolio_dict['bought']
                self.bought_values = portfolio_dict['bought_values']
                self.selling = portfolio_dict['selling']
                self.selling_values = portfolio_dict['selling_values']
                self.percentage_diff_dict = portfolio_dict['percentage_diff_dict']
                self.port_changes = portfolio_dict['port_changes']
                self.trade_counter = portfolio_dict['trade_counter']
                self.counter = portfolio_dict['counter']
        except FileNotFoundError:
            print("Portfolio file not found. Starting with default values.")
        return