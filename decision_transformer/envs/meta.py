import MetaTrader5 as mt5
import utils

def login(username, password, server, retry_count = 0):
        username = username
        password = password
        server = server
        print('username : ', username)
        print('password : ', password)
        print('server : ', server)

        if not mt5.initialize(login=username, server=server,password=password):# type: ignore 
              print("login failed, error code =",mt5.last_error())# type: ignore 
              quit()
        else: 
              #Set equity values only if this is the first start
              if (retry_count == 0):
                    initial_equity = mt5.account_info().equity # type: ignore
                    print('Equity set to :', initial_equity)
               
def do_test(symbols):
    meta_symbols = symbols
    for symbol in symbols:
        
        if not mt5.symbol_select(meta_symbols[symbol], True): # type: ignore
            print(f"Failed to select {meta_symbols[0]}, error code =", mt5.last_error()) # type: ignore
            return False
        else:
            tick_info = mt5.symbol_info_tick(meta_symbols[symbol])  # type: ignore
            if tick_info:
                print('Passed Connection...')
                return True
            else:
                print(f"Failed to get tick info for {meta_symbols[symbol]}, error code =", mt5.last_error()) # type: ignore   
                return False

def get_initial_equity():
    initial_equity = mt5.account_info().equity # type: ignore
    return initial_equity

def get_current_equity():
    current_equity = mt5.account_info().equity # type: ignore
    return current_equity

def can_act(initial_equity):
    #check equity
    current_equity = get_current_equity()
    can_act = True
    threshold = 0.6 * initial_equity
    print('Current Equity : ', current_equity, ' Threshold : ', threshold)
    #First Test ... we can add more test 
    # Like limit of steps and shit....
    if(current_equity < threshold):
          print('Equity Threshold broken!!.....')
          can_act = False
        
    return can_act

def retry(env_type,symbols,username, password,server,fail_count):
    if(fail_count < 5):
        #Increase fail count
        fail_count += 1     
        
        #login and test
        login(username, password,server,fail_count)
        result = do_test(symbols)
        return result
    
def closePositions(symbol):
        positions = mt5.positions_get(symbol=symbol) # type: ignore
        if positions != None:  
              for position in positions:
                    position_id = position.ticket
                    print(symbol, ' Postion Type :',position.type, 'Ticket:', position_id)
                    pos_type = position.type  # 0 = buy, 1 = sell
                    
                    if pos_type == mt5.ORDER_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(symbol).bid# type: ignore
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(symbol).ask# type: ignore

                    request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": symbol,
                                "volume": 0.01,
                                "type": order_type,
                                "position": position.ticket,
                                "price": price,
                                "deviation": 20,
                                "magic": 0,
                                "comment": "Close position",
                                "type_filling": mt5.ORDER_FILLING_IOC,
                                } 
                    result = mt5.order_send(request)# type: ignore
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Failed to close position #{position.ticket}, retcode={result.retcode}")
                    else:
                        print(f"Position #{position.ticket} closed successfully") 
        return

def close_all(symbols):
    for symbol in symbols:
          closePositions(symbol)
     
    return 

def getRequest(type, price,magic,symbol, volume = 0.01,deviation = 20): 
    request = {
        "action"        : mt5.TRADE_ACTION_DEAL,
        "magic"         : magic + 1,
        "symbol"        : symbol,
        "volume"        : volume,
        "price"         : price,
        "type"          : type,
        "deviation"     : deviation ,
        "type_time"     : mt5.ORDER_TIME_GTC,
        "type_filling"  : mt5.ORDER_FILLING_IOC, 
        "comment"       : "This is a test order ",   
    }
    return request

def BUY(symbol, initial_equity, counter):
    type = mt5.ORDER_TYPE_BUY
    print('symbol : ', symbol)
    price = mt5.symbol_info_tick(symbol).ask # type: ignore
    print('TICK INFO : ',  mt5.symbol_info_tick(symbol))# type: ignore
    request = getRequest(type,price, counter, symbol)
    print('REQUEST INFO : ', request)
    
    if can_act(initial_equity):
        result = mt5.order_send(request) # type: ignore
    else:
        raise RuntimeError('Equity threshold broken, cannot execute order')

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("order failed , retcode = {}".format(result.retcode))
        #utils.wait_minute(minutes = 1, seconds = 10)
        raise RuntimeError('Order failed, check logs for details : ' + str(result.retcode))
         
    return price 

def SELL(symbol, initial_equity, counter):
    type = mt5.ORDER_TYPE_SELL
    print('symbol : ', symbol)
    price = mt5.symbol_info_tick(symbol).bid  # type: ignore
    print('TICK INFO : ',  mt5.symbol_info_tick(symbol)) # type: ignore
    request = getRequest(type,price, counter, symbol)
    print('REQUEST INFO : ', request)

    if can_act(initial_equity):
        result = mt5.order_send(request) # type: ignore
    else:
        raise RuntimeError('Equity threshold broken, cannot execute order')
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("order failed , retcode = {}".format(result.retcode))
        #utils.wait_minute(minutes = 1, seconds = 10)
        raise RuntimeError('Order failed, check logs for details : ' + str(result.retcode))

    return price


