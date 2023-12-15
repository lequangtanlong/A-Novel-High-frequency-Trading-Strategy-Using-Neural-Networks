import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from traders.trader import Trader
from utils.data_utils import TradingDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    file_name = 'btcusdt_20230613'
    path = os.path.abspath(__file__ + f'../../data/{file_name}.csv')
    model = 0
    if (model == 0):
        dir_file_name = f'ann_{file_name}'
    elif (model == 1):
        dir_file_name = f'rnn_{file_name}'
    else:
        dir_file_name = f'lstm_{file_name}'
    dir_path = os.path.abspath(__file__ + f'/../../../../grid_trading_algorithm_using_deep_learning/data/direction_{dir_file_name}.txt')

    trader = Trader(pred_horizon=20, model=model)
    data_loader = TradingDataLoader(path)
    curr_tick = data_loader.step()

    f = open(dir_path, 'a')
    while(curr_tick[1]):
        f.write(f'{trader.respond(curr_tick[0])[1]}\n')
        curr_tick = data_loader.step()
    f.close() 