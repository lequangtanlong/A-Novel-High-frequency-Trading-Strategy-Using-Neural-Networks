import collections
import numpy as np
import os
import statistics
import sys
import time
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
from train import ANN, RNN, LSTM
from traders.abstract_trader import AbstractTrader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Trader(AbstractTrader):
    def __init__(self, init_balance=0, pred_horizon=100, model=0):
        self.init_balance = init_balance
        self.balance = 0.0
        self.pred_horizon = pred_horizon
        self.stock = 0
        self.potential_stock = 0
        self.tick_num = 0
        self.orders = []
        if (model == 0):
            self.model = ANN.load_from_checkpoint(os.path.abspath(__file__ + "/../checkpoints/ann.ckpt"), input_size=7, hidden_size=100, map_location=device)    
        elif (model == 1):
            self.model = RNN.load_from_checkpoint(os.path.abspath(__file__ + "/../checkpoints/rnn.ckpt"), input_size=7,  hidden_size=100, num_layers=10, map_location=device)
        else:
            self.model = LSTM.load_from_checkpoint(os.path.abspath(__file__ + "/../checkpoints/lstm.ckpt"), input_size=7, seq_len=50, num_layers=10, map_location=device)
        self.model.eval()
        self.model.eval()
        self.ticks = collections.deque()
        self.micros = collections.deque()
        self.emas = collections.deque()
        self.total = 0
        self.buy_orders = collections.deque()
        self.sell_orders = collections.deque()
        self.ema = 0
        self.ones = 0
        self.twos = 0
        self.diffs = []
        self.smooth = 2/(self.pred_horizon+1.0)
        self.bid_norm_vals = {'max_p': 0, 'min_p': float('inf'), 
                              'max_v': 0, 'min_v': float('inf')}
        self.ask_norm_vals = {'max_p': 0, 'min_p': float('inf'), 
                              'max_v': 0, 'min_v': float('inf')}
        self.mic_norm_vals = {'max_p': 0, 'min_p': float('inf')}
        self.ema_norm_vals = {'max_p': 0, 'min_p': float('inf')}
        self.ready = False
        self.eval_times = [0.0, 0.0]

        self.tick_size = 0.01
        self.maker_fee=-0.00005
        self.taker_fee=0.0007
        self.position = 0.0
        self.fee = 0.0
        self.trade_num = 0
        self.trade_qty = 0.0
        self.trade_amount = 0.0
        self.timestamp = []
        self.mid = []
        self.balance_ = []
        self.position_ = []
        self.fee_ = []
        self.trade_num_ = []
        self.trade_qty_ = []
        self.trade_amount_ = []

    def respond(self, tick):
        new_orders = []

        self.tick_num += 1
        microprice = ((tick['Bid'] * tick['AskVolume'] + 
                    tick['Ask'] * tick['BidVolume']) / 
                    (tick['AskVolume'] + tick['BidVolume']))
        for order in list(self.buy_orders):
            if (order[0] == self.tick_num):
                self.diffs.append(-(microprice - order[1]))
                new_orders.append({'type': 'BID', 'price': microprice, 'quantity': tick['AskVolume']})
                self.buy_orders.popleft()
            else: 
                break
        for order in list(self.sell_orders):
            if (order[0] == self.tick_num):
                self.diffs.append(microprice - order[1])
                new_orders.append({'type': 'ASK', 'price': microprice, 'quantity': tick['BidVolume']})
                self.sell_orders.popleft()
            else: 
                break
        
        self.ticks.append(tick)
        self.micros.append(microprice)
        self.total += microprice

        trend = -1

        if (len(self.ticks) <= 100):
            self.emas.append((self.total + microprice) / (len(self.ticks)+1))
            self.__update_norm_vals__()
        else:
            self.emas.append((microprice * self.smooth) + (self.emas[-1]*(1-self.smooth)))
            self.micros.popleft()
            self.emas.popleft()
            self.ticks.popleft()
            self.__update_norm_vals__()
            inputs, timespans = self.__gen_inputs__()

            eval_start = time.time()
            output = self.model(inputs)
            eval_end = time.time()
            self.eval_times[0] += (eval_end - eval_start)
            self.eval_times[1] += 1

            direction = torch.argmax(output).item()
            certainty = torch.exp(torch.div(output, 1000))[0][direction].item() / sum(torch.exp(torch.div(output, 1000))[0])

            if (certainty >= 0.333):
                trend = direction
                if (direction == 2):
                    self.twos += 1
                    self.potential_stock += tick['AskVolume']
                    self.sell_orders.append((self.tick_num + self.pred_horizon, microprice))
                    new_orders.append({'type': 'BID', 'price': microprice, 'quantity': tick['AskVolume']})
                else:
                    self.ones += 1
                    self.potential_stock -= tick['BidVolume']
                    self.buy_orders.append((self.tick_num + self.pred_horizon, microprice))
                    new_orders.append({'type': 'ASK', 'price': microprice, 'quantity': tick['BidVolume']})
        
        if len(new_orders) == 0:
            return None, trend
        else:
            return new_orders, trend

    def filled_order(self, order):
        if (order['type'] == 'BID'):
            self.stock += order['quantity']
            self.balance -= order['price'] * order['quantity']
        else:
            self.stock -= order['quantity']
            self.balance += order['price'] * order['quantity']

    def print_vals(self):
        print("ones: " + str(self.ones) + " twos: " + str(self.twos))
        print("net worth: " + str(self.balance + (self.stock * self.micros[-1])))
        print("balance: " + str(self.balance))
        print("quantity: " + str(self.stock))
        print("profit: " + str(self.balance - self.init_balance))
        print("std_dev: " + str(statistics.stdev(self.diffs)))

        return ([self.balance - self.init_balance, self.ones, self.twos, statistics.stdev(self.diffs)])

    def save_vals(self, name):
        with open(name, 'w') as f:
            f.write(f"ones: {self.ones} twos:  {self.twos}\n")
            f.write(f"net worth: {self.balance + (self.stock * self.micros[-1])}\n")
            f.write(f"balance: {self.balance}\n")
            f.write(f"quantity: {self.stock}\n")
            f.write(f"profit: {self.balance - self.init_balance}\n")
            f.write(f"std_dev: {statistics.stdev(self.diffs)}\n")

        return ([self.balance - self.init_balance, self.ones, self.twos, statistics.stdev(self.diffs)])

    def __gen_inputs__(self):
        inputs = []
        timespans = []
        for i, tick in enumerate(self.ticks):
            if (i < 50): 
                continue

            inputs.append([
                (tick['Bid'] - self.bid_norm_vals['min_p']) / (self.bid_norm_vals['max_p'] - self.bid_norm_vals['min_p']),
                (tick['Ask'] - self.ask_norm_vals['min_p']) / (self.ask_norm_vals['max_p'] - self.ask_norm_vals['min_p']),
                (tick['AskVolume'] - self.ask_norm_vals['min_v']) / (self.ask_norm_vals['max_v'] - self.ask_norm_vals['min_v']),
                (tick['BidVolume'] - self.bid_norm_vals['min_v']) / (self.bid_norm_vals['max_v'] - self.bid_norm_vals['min_v'])
            ])
            timespans.append(tick['Local time'])
        
        for i, mic in enumerate(self.micros):
            if (i < 50): 
                continue

            inputs[i-50].append(
                (mic - self.mic_norm_vals['min_p']) / (self.mic_norm_vals['max_p'] - self.mic_norm_vals['min_p'])
            )

        for i, e in enumerate(self.emas):
            if (i < 50): 
                continue

            inputs[i-50].append(
                (e - self.ema_norm_vals['min_p']) / (self.ema_norm_vals['max_p'] - self.ema_norm_vals['min_p'])
            )
            inputs[i-50].append(self.micros[-1] - self.micros[i])

        return (torch.Tensor(np.asarray([inputs])).float(), 
                torch.Tensor(np.asarray([timespans])).float())
    
    def __update_norm_vals__(self):
        self.bid_norm_vals['max_p'] = max(self.bid_norm_vals['max_p'], self.ticks[-1]['Bid'])
        self.bid_norm_vals['min_p'] = min(self.bid_norm_vals['min_p'], self.ticks[-1]['Bid'])
        self.bid_norm_vals['max_v'] = max(self.bid_norm_vals['max_v'], self.ticks[-1]['BidVolume'])
        self.bid_norm_vals['min_v'] = min(self.bid_norm_vals['min_v'], self.ticks[-1]['BidVolume'])

        self.ask_norm_vals['max_p'] = max(self.ask_norm_vals['max_p'], self.ticks[-1]['Ask'])
        self.ask_norm_vals['min_p'] = min(self.ask_norm_vals['min_p'], self.ticks[-1]['Ask'])
        self.ask_norm_vals['max_v'] = max(self.ask_norm_vals['max_v'], self.ticks[-1]['AskVolume'])
        self.ask_norm_vals['min_v'] = min(self.ask_norm_vals['min_v'], self.ticks[-1]['AskVolume'])

        self.mic_norm_vals['max_p'] = max(self.mic_norm_vals['max_p'], self.micros[-1])
        self.mic_norm_vals['min_p'] = min(self.mic_norm_vals['min_p'], self.micros[-1])

        self.ema_norm_vals['max_p'] = max(self.ema_norm_vals['max_p'], self.emas[-1])
        self.ema_norm_vals['min_p'] = min(self.ema_norm_vals['min_p'], self.emas[-1])
