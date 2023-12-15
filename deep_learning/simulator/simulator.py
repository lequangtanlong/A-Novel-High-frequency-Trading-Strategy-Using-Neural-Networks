import os
import time
from traders.trader import Trader
from utils.data_utils import TradingDataLoader

if __name__ == '__main__':
    data_path = os.path.abspath(__file__ + "/../utils/data")
    days = [file for file in sorted(os.listdir(data_path)) if not file.startswith('.')]
    model = 0
    for day in days:
        print("Trading day: " + str(day))
        data_loader = TradingDataLoader(f"{data_path }/{day}")
        orders = []
        trader = Trader(init_balance=10000, pred_horizon=20, model=model)
        curr_tick = data_loader.step()
        speed = float('inf')
        while(curr_tick[1]):
            market_price = ((curr_tick[0]['Bid'] * curr_tick[0]['AskVolume'] + 
                        curr_tick[0]['Ask'] * curr_tick[0]['BidVolume']) / 
                        (curr_tick[0]['AskVolume'] + curr_tick[0]['BidVolume']))

            t_orders = trader.respond(curr_tick[0])[0]
            if t_orders != None:
                for o in t_orders:
                    orders.append(o)
                
            unfilled = []
            while (len(orders) > 0):
                order = orders.pop(0)
                if (order['type'] == 'BID'):
                    if (order['price'] >= market_price):
                        order['price'] = market_price
                        order['quantity'] = 1
                        
                        trader.filled_order(order)
                    else:
                        unfilled.append(order)
                else:
                    if (order['price'] <= market_price):
                        order['price'] = market_price
                        order['quantity'] = 1
                        
                        trader.filled_order(order)
                    else:
                        unfilled.append(order)
            orders = unfilled

            # Next tick
            if (speed != float('inf')):
                time.sleep(curr_tick[1]/speed)
            curr_tick = data_loader.step()

        trader.print_vals()
