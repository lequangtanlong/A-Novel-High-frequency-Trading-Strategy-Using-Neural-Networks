from pytorch_lightning.loggers import TensorBoardLogger
from torch.autograd import Variable
from torchmetrics.functional import accuracy
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import random
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)[:,:,[1,2,3,4,5,6,9]]
        self.y = torch.tensor(y,dtype=torch.long)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.len
    
    def get_labels(self):
        return self.y

class DataModule(pl.LightningDataModule):
    def __init__(self, data_type='', window=10, batch_size=1, pred_horizon=10, alpha=0.0002):
        super().__init__()
        self.window = window
        self.batch_size = batch_size
        self.pred_horizon = pred_horizon
        self.data_type = data_type
        self.alpha = alpha

    def setup(self, stage=None):
        train_dfs, test_dfs = self.load_data(train_days=12, test_days=3)
        X_train, y_train = self.process_data(train_dfs)
        X_test, y_test = self.process_data(test_dfs)

        self.train_data = timeseries(X_train, y_train)
        self.test_data = timeseries(X_test, y_test)

    def load_data(self, train_days, test_days):
        path_to_data = os.path.abspath(__file__ + "../../../utils/data")
        train_dfs = []
        test_dfs = []
        for i, f in enumerate(sorted(os.listdir(path_to_data))):
            df = pd.read_csv(f"{path_to_data}/{f}")
            n = random.randint(0, df.shape[0]-200000)
            df = df[n:n+200000]

            if (i < train_days):
                train_dfs.append(df)
            elif (i < train_days+test_days):
                test_dfs.append(df)
            else:
              break

        return (train_dfs, test_dfs)

    def process_data(self, dfs):
        df = self.normalise(np.array(self.convert_to_seconds(dfs[0])))
        X, y = self.sequence_data(df, self.window)
        for i in range(1, len(dfs)):
            new_df = self.normalise(np.array(self.convert_to_seconds(dfs[i])))
            new_X, new_y = self.sequence_data(new_df, self.window)
            X = np.concatenate((X, new_X), axis=0)
            y = np.concatenate((y, new_y), axis=0)

        return X, y

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, sampler=ImbalancedDatasetSampler(self.train_data), batch_size=self.batch_size, num_workers=8)

        return train_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=8)

        return test_dataloader

    def normalise(self, data):
        ind = [1,2,3,4,5,6,8]
        mc = data[:,ind]
        data[:,ind] = (mc - mc.min()) / (mc.max() - mc.min())

        return data

    def convert_to_seconds(self, df):
        df = self.convert_to_microprice(df, moving_avg_size=self.pred_horizon)
        time_in_seconds = []
        start = 0.0
        i = 0
        for index, row in df.iterrows():
            time = float(row['Local time'])
            if (i == 0):
                start = time
            i += 1
            time_in_seconds.append([row['Microprice'], time - start])
        time_in_seconds = np.array(time_in_seconds)
        df['Local time'] = time_in_seconds[:,1]

        return df

    def convert_to_microprice(self, df, moving_avg_size=5, smoothing=2):
        df['Microprice'] = (df['Bid']*df['AskVolume'] + df['Ask']*df['BidVolume']) / (df['AskVolume'] + df['BidVolume'])
        MicroExpMovingAvg = []
        mult = smoothing / (1.0 + moving_avg_size)
        sma = 0.0
        for i in range(len(df)):
            if (i < moving_avg_size):
                sma += df.iloc[i]['Microprice']
                MicroExpMovingAvg.append(sma/float(i+1))
            else:
                MicroExpMovingAvg.append((df.iloc[i]['Microprice'] * mult) + (MicroExpMovingAvg[i-1]) * (1.0 - mult))

        df['NormMovingAvg'] = MicroExpMovingAvg
        df['MicroExpMovingAvg'] = MicroExpMovingAvg
        df['OrderBookImbalance'] = (df['BidVolume'] - df['AskVolume']) / (df['BidVolume'] + df['AskVolume'])
        return df

    def get_dir(self, curr_mvavg, next_mvavg):
        lt = (next_mvavg - curr_mvavg) / curr_mvavg
        if (lt > self.alpha):
            direction = 2
        elif (lt < -self.alpha):
            direction = 0
        else:
            direction = 1

        return direction

    def sequence_data(self, data, window):
        X = []
        Y = []
        stag = 0
        up = 0
        dwn = 0
        L = []
        for i in range(0, len(data)-window):
            j = i+window-1
            next_dir_idx = j+self.pred_horizon
            curr_dir_idx = j-self.pred_horizon
            curr_mvavg = data[j][7]
            curr_time = data[j][0]

            curr_dir = 0
            if (next_dir_idx < len(data)):
                if (curr_dir_idx >= 0):
                    curr_dir = self.get_dir(data[curr_dir_idx][7], curr_mvavg)
                    
                label = self.get_dir(curr_mvavg, data[next_dir_idx][7])

                if (label == 2):
                    up += 1
                elif (label == 0):
                    dwn += 1
                else:
                    stag += 1

                L.append([curr_time, label])
                seq = data[i:i+window]
                inputs_seq = []
                for k in range(0, len(seq)):
                    inputs_seq.append(np.append(seq[k], data[j][7]-seq[k][7]))

                X.append(inputs_seq)
                Y.append(label)

        print("Labels Done")
        return np.array(X), np.array(Y)
    
class ANN(pl.LightningModule):
    def __init__(self, input_size=2, hidden_size=100, output_size=3):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.test_step_outputs = []  
    
    def forward(self, x):
        out = self.fc1(x)
        out = out[:, -1]
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.elu3(out)
        out = self.fc4(out)
        return out
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        total = 0
        correct = 0
        true_pos_up = 0
        true_pos_dwn = 0
        true_pos_stag = 0
        false_pos_up = 0
        false_neg_up = 0
        false_pos_dwn = 0
        false_neg_dwn = 0
        false_pos_stag = 0
        false_neg_stag = 0
        up = 0
        dwn = 0
        stag = 0

        for i in range(len(logits)):
            pred = logits[i].argmax(dim=0, keepdim=True)

            if (y[i] == 2):
                up += 1
            elif (y[i] == 0):
                dwn += 1
            else:   
                stag += 1

            if (pred[0] == y[i]):
                correct += 1
                if (y[i] == 2): 
                    true_pos_up += 1
                elif (y[i] == 0):
                    true_pos_dwn += 1
                else:
                    true_pos_stag += 1
            else: 
                if (pred[0] == 2): 
                    false_neg_up += 1
                    if (y[i] == 0):
                        false_pos_dwn += 1
                    elif (y[i] == 1):
                        false_pos_stag += 1
                elif (pred[0] == 0):
                    false_neg_dwn += 1
                    if (y[i] == 2):
                        false_pos_up += 1
                    elif (y[i] == 1):
                        false_pos_stag += 1
                else:
                    false_neg_stag += 1
                    if (y[i] == 2):
                        false_pos_up += 1
                    elif (y[i] == 0):
                        false_pos_dwn += 1

            total += 1

        metrics = { 'correct': correct, 
                    'total': total,
                    'true_pos_up': true_pos_up,
                    'true_pos_dwn': true_pos_dwn,
                    'true_pos_stag': true_pos_stag,
                    'false_pos_up': false_pos_up,
                    'false_neg_up': false_neg_up,
                    'false_pos_dwn': false_pos_dwn,
                    'false_neg_dwn': false_neg_dwn,
                    'false_pos_stag': false_pos_stag,
                    'false_neg_stag': false_neg_stag,
                    'up': up,
                    'dwn': dwn,
                    'stag': stag}
        
        self.test_step_outputs.append(metrics)

        return metrics

    def on_test_epoch_end(self):
        correct = sum([x['correct'] for x in self.test_step_outputs])
        total = sum([x['total'] for x in self.test_step_outputs])
        true_pos_up = sum([x['true_pos_up'] for x in self.test_step_outputs])
        false_pos_up = sum([x['false_pos_up'] for x in self.test_step_outputs])
        false_neg_up = sum([x['false_neg_up'] for x in self.test_step_outputs])
        true_pos_dwn = sum([x['true_pos_dwn'] for x in self.test_step_outputs])
        false_pos_dwn = sum([x['false_pos_dwn'] for x in self.test_step_outputs])
        false_neg_dwn = sum([x['false_neg_dwn'] for x in self.test_step_outputs])
        true_pos_stag = sum([x['true_pos_stag'] for x in self.test_step_outputs])
        false_pos_stag = sum([x['false_pos_stag'] for x in self.test_step_outputs])
        false_neg_stag = sum([x['false_neg_stag'] for x in self.test_step_outputs])
        up = sum([x['up'] for x in self.test_step_outputs])
        dwn = sum([x['dwn'] for x in self.test_step_outputs])
        stag = sum([x['stag'] for x in self.test_step_outputs])

        with open('ann_metrics_results.txt', 'w') as f:
            f.write(f'total: {total}\n')
            f.write(f'correct: {correct}\n')
            f.write(f'true_pos_up: {true_pos_up}\n')
            f.write(f'true_pos_dwn: {true_pos_dwn}\n')
            f.write(f'true_pos_stag: {true_pos_stag}\n')
            f.write(f'false_pos_up: {false_pos_up}\n')
            f.write(f'false_pos_dwn: {false_pos_dwn}\n')
            f.write(f'false_pos_stag: {false_pos_stag}\n')
            f.write(f'false_neg_up: {false_neg_up}\n')
            f.write(f'false_neg_dwn: {false_neg_dwn}\n')
            f.write(f'false_neg_stag: {false_neg_stag}\n')
            f.write(f'up: {up}\n')
            f.write(f'dwn: {dwn}\n')
            f.write(f'stag: {stag}\n')

        self.test_step_outputs.clear()

        res = 100 * correct / total
        try:
            precision_up = true_pos_up / (true_pos_up + false_pos_up)
        except ZeroDivisionError:
            precision_up = 0
        try:
            precision_dwn = true_pos_dwn / (true_pos_dwn + false_pos_dwn)
        except ZeroDivisionError:
            precision_dwn = 0
        try:
            precision_stag = true_pos_stag / (true_pos_stag + false_pos_stag) 
        except ZeroDivisionError:
            precision_stag = 0
        try:
            recall_up = true_pos_up / (true_pos_up + false_neg_up)
        except ZeroDivisionError:
            recall_up = 0
        try:
            recall_dwn = true_pos_dwn / (true_pos_dwn + false_neg_dwn)
        except ZeroDivisionError:
            recall_dwn = 0
        try:
            recall_stag = true_pos_stag / (true_pos_stag + false_neg_stag) 
        except ZeroDivisionError:
            recall_stag = 0 
        precision_macro_avg = (precision_up + precision_dwn + precision_stag) / 3
        recall_macro_avg = (recall_up + recall_dwn + recall_stag) / 3
        f1_score_macro_avg = 2 * (precision_macro_avg * recall_macro_avg) / (precision_macro_avg + recall_macro_avg)

        precision_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + false_pos_up + false_pos_dwn + false_pos_stag)
        recall_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + false_neg_up + false_neg_dwn + false_neg_stag)
        f1_score_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + (false_pos_up + false_pos_dwn + false_pos_stag + false_neg_up + false_neg_dwn + false_neg_stag) / 2)

        with open('ann_results.txt', 'w') as f:
            f.write(f'Accuracy: {res}\n')
            f.write(f'Precision macro average: {100 *precision_macro_avg}\n')
            f.write(f'Recall macro average: {100 * recall_macro_avg}\n')
            f.write(f'F1-score macro average: {100 * f1_score_macro_avg}\n')
            f.write(f'Precision micro average: {100 * precision_micro_avg}\n')
            f.write(f'Recall micro average: {100 * recall_micro_avg}\n')
            f.write(f'F1-score micro average: {100 * f1_score_micro_avg}\n')
            f.write(f'Precision up: {100 * precision_up}\n')
            f.write(f'Precision down: {100 * precision_dwn}\n')
            f.write(f'Precision stag: {100 * precision_stag}\n')
            f.write(f'Recall up: {100 * recall_up}\n')
            f.write(f'Recall down: {100 * recall_dwn}\n')
            f.write(f'Recall stag: {100 * recall_stag}\n')
            
        return {'overall_accuracy': res}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

class RNN(pl.LightningModule):
    def __init__(self, input_size=2, hidden_size=100, num_layers=1, output_size=3):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)
        self.test_step_outputs = []
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device))
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.logger.experiment.add_scalar("Loss/Train", loss)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        total = 0
        correct = 0
        true_pos_up = 0
        true_pos_dwn = 0
        true_pos_stag = 0
        false_pos_up = 0
        false_neg_up = 0
        false_pos_dwn = 0
        false_neg_dwn = 0
        false_pos_stag = 0
        false_neg_stag = 0
        up = 0
        dwn = 0
        stag = 0

        for i in range(len(logits)):
            pred = logits[i].argmax(dim=0, keepdim=True)

            if (y[i] == 2):
                up += 1
            elif (y[i] == 0):
                dwn += 1
            else:   
                stag += 1

            if (pred[0] == y[i]):
                correct += 1
                if (y[i] == 2): 
                    true_pos_up += 1
                elif (y[i] == 0):
                    true_pos_dwn += 1
                else:
                    true_pos_stag += 1
            else: 
                if (pred[0] == 2): 
                    false_neg_up += 1
                    if (y[i] == 0):
                        false_pos_dwn += 1
                    elif (y[i] == 1):
                        false_pos_stag += 1
                elif (pred[0] == 0):
                    false_neg_dwn += 1
                    if (y[i] == 2):
                        false_pos_up += 1
                    elif (y[i] == 1):
                        false_pos_stag += 1
                else:
                    false_neg_stag += 1
                    if (y[i] == 2):
                        false_pos_up += 1
                    elif (y[i] == 0):
                        false_pos_dwn += 1

            total += 1

        metrics = { 'correct': correct, 
                    'total': total,
                    'true_pos_up': true_pos_up,
                    'true_pos_dwn': true_pos_dwn,
                    'true_pos_stag': true_pos_stag,
                    'false_pos_up': false_pos_up,
                    'false_neg_up': false_neg_up,
                    'false_pos_dwn': false_pos_dwn,
                    'false_neg_dwn': false_neg_dwn,
                    'false_pos_stag': false_pos_stag,
                    'false_neg_stag': false_neg_stag,
                    'up': up,
                    'dwn': dwn,
                    'stag': stag}
        
        self.test_step_outputs.append(metrics)

        return metrics

    def on_test_epoch_end(self):
        correct = sum([x['correct'] for x in self.test_step_outputs])
        total = sum([x['total'] for x in self.test_step_outputs])
        true_pos_up = sum([x['true_pos_up'] for x in self.test_step_outputs])
        false_pos_up = sum([x['false_pos_up'] for x in self.test_step_outputs])
        false_neg_up = sum([x['false_neg_up'] for x in self.test_step_outputs])
        true_pos_dwn = sum([x['true_pos_dwn'] for x in self.test_step_outputs])
        false_pos_dwn = sum([x['false_pos_dwn'] for x in self.test_step_outputs])
        false_neg_dwn = sum([x['false_neg_dwn'] for x in self.test_step_outputs])
        true_pos_stag = sum([x['true_pos_stag'] for x in self.test_step_outputs])
        false_pos_stag = sum([x['false_pos_stag'] for x in self.test_step_outputs])
        false_neg_stag = sum([x['false_neg_stag'] for x in self.test_step_outputs])
        up = sum([x['up'] for x in self.test_step_outputs])
        dwn = sum([x['dwn'] for x in self.test_step_outputs])
        stag = sum([x['stag'] for x in self.test_step_outputs])

        with open('rnn_metrics_results.txt', 'w') as f:
            f.write(f'total: {total}\n')
            f.write(f'correct: {correct}\n')
            f.write(f'true_pos_up: {true_pos_up}\n')
            f.write(f'true_pos_dwn: {true_pos_dwn}\n')
            f.write(f'true_pos_stag: {true_pos_stag}\n')
            f.write(f'false_pos_up: {false_pos_up}\n')
            f.write(f'false_pos_dwn: {false_pos_dwn}\n')
            f.write(f'false_pos_stag: {false_pos_stag}\n')
            f.write(f'false_neg_up: {false_neg_up}\n')
            f.write(f'false_neg_dwn: {false_neg_dwn}\n')
            f.write(f'false_neg_stag: {false_neg_stag}\n')
            f.write(f'up: {up}\n')
            f.write(f'dwn: {dwn}\n')
            f.write(f'stag: {stag}\n')

        self.test_step_outputs.clear()

        res = 100 * correct / total
        try:
            precision_up = true_pos_up / (true_pos_up + false_pos_up)
        except ZeroDivisionError:
            precision_up = 0
        try:
            precision_dwn = true_pos_dwn / (true_pos_dwn + false_pos_dwn)
        except ZeroDivisionError:
            precision_dwn = 0
        try:
            precision_stag = true_pos_stag / (true_pos_stag + false_pos_stag) 
        except ZeroDivisionError:
            precision_stag = 0
        try:
            recall_up = true_pos_up / (true_pos_up + false_neg_up)
        except ZeroDivisionError:
            recall_up = 0
        try:
            recall_dwn = true_pos_dwn / (true_pos_dwn + false_neg_dwn)
        except ZeroDivisionError:
            recall_dwn = 0
        try:
            recall_stag = true_pos_stag / (true_pos_stag + false_neg_stag) 
        except ZeroDivisionError:
            recall_stag = 0 
        precision_macro_avg = (precision_up + precision_dwn + precision_stag) / 3
        recall_macro_avg = (recall_up + recall_dwn + recall_stag) / 3
        f1_score_macro_avg = 2 * (precision_macro_avg * recall_macro_avg) / (precision_macro_avg + recall_macro_avg)

        precision_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + false_pos_up + false_pos_dwn + false_pos_stag)
        recall_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + false_neg_up + false_neg_dwn + false_neg_stag)
        f1_score_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + (false_pos_up + false_pos_dwn + false_pos_stag + false_neg_up + false_neg_dwn + false_neg_stag) / 2)

        with open('rnn_results.txt', 'w') as f:
            f.write(f'Accuracy: {res}\n')
            f.write(f'Precision macro average: {100 *precision_macro_avg}\n')
            f.write(f'Recall macro average: {100 * recall_macro_avg}\n')
            f.write(f'F1-score macro average: {100 * f1_score_macro_avg}\n')
            f.write(f'Precision micro average: {100 * precision_micro_avg}\n')
            f.write(f'Recall micro average: {100 * recall_micro_avg}\n')
            f.write(f'F1-score micro average: {100 * f1_score_micro_avg}\n')
            f.write(f'Precision up: {100 * precision_up}\n')
            f.write(f'Precision down: {100 * precision_dwn}\n')
            f.write(f'Precision stag: {100 * precision_stag}\n')
            f.write(f'Recall up: {100 * recall_up}\n')
            f.write(f'Recall down: {100 * recall_dwn}\n')
            f.write(f'Recall stag: {100 * recall_stag}\n')
            
        return {'overall_accuracy': res}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

class LSTM(pl.LightningModule):
    def __init__(self, input_size=2, hidden_size=100, seq_len=10, num_layers=1, batch_size=1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm1d(self.seq_len)

        self.test_step_outputs = []

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(self.bn(x))
        lstm_out = lstm_out[:,-1,:]
        lstm_out = self.relu(self.fc1(lstm_out))
        predictions = self.fc2(lstm_out)

        return predictions

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.logger.experiment.add_scalar("Loss/Train", loss)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        total = 0
        correct = 0
        true_pos_up = 0
        true_pos_dwn = 0
        true_pos_stag = 0
        false_pos_up = 0
        false_neg_up = 0
        false_pos_dwn = 0
        false_neg_dwn = 0
        false_pos_stag = 0
        false_neg_stag = 0
        up = 0
        dwn = 0
        stag = 0

        for i in range(len(logits)):
            pred = logits[i].argmax(dim=0, keepdim=True)

            if (y[i] == 2):
                up += 1
            elif (y[i] == 0):
                dwn += 1
            else:   
                stag += 1

            if (pred[0] == y[i]):
                correct += 1
                if (y[i] == 2): 
                    true_pos_up += 1
                elif (y[i] == 0):
                    true_pos_dwn += 1
                else:
                    true_pos_stag += 1
            else: 
                if (pred[0] == 2): 
                    false_neg_up += 1
                    if (y[i] == 0):
                        false_pos_dwn += 1
                    elif (y[i] == 1):
                        false_pos_stag += 1
                elif (pred[0] == 0):
                    false_neg_dwn += 1
                    if (y[i] == 2):
                        false_pos_up += 1
                    elif (y[i] == 1):
                        false_pos_stag += 1
                else:
                    false_neg_stag += 1
                    if (y[i] == 2):
                        false_pos_up += 1
                    elif (y[i] == 0):
                        false_pos_dwn += 1

            total += 1

        metrics = { 'correct': correct, 
                    'total': total,
                    'true_pos_up': true_pos_up,
                    'true_pos_dwn': true_pos_dwn,
                    'true_pos_stag': true_pos_stag,
                    'false_pos_up': false_pos_up,
                    'false_neg_up': false_neg_up,
                    'false_pos_dwn': false_pos_dwn,
                    'false_neg_dwn': false_neg_dwn,
                    'false_pos_stag': false_pos_stag,
                    'false_neg_stag': false_neg_stag,
                    'up': up,
                    'dwn': dwn,
                    'stag': stag}
        
        self.test_step_outputs.append(metrics)

        return metrics

    def on_test_epoch_end(self):
        correct = sum([x['correct'] for x in self.test_step_outputs])
        total = sum([x['total'] for x in self.test_step_outputs])
        true_pos_up = sum([x['true_pos_up'] for x in self.test_step_outputs])
        false_pos_up = sum([x['false_pos_up'] for x in self.test_step_outputs])
        false_neg_up = sum([x['false_neg_up'] for x in self.test_step_outputs])
        true_pos_dwn = sum([x['true_pos_dwn'] for x in self.test_step_outputs])
        false_pos_dwn = sum([x['false_pos_dwn'] for x in self.test_step_outputs])
        false_neg_dwn = sum([x['false_neg_dwn'] for x in self.test_step_outputs])
        true_pos_stag = sum([x['true_pos_stag'] for x in self.test_step_outputs])
        false_pos_stag = sum([x['false_pos_stag'] for x in self.test_step_outputs])
        false_neg_stag = sum([x['false_neg_stag'] for x in self.test_step_outputs])
        up = sum([x['up'] for x in self.test_step_outputs])
        dwn = sum([x['dwn'] for x in self.test_step_outputs])
        stag = sum([x['stag'] for x in self.test_step_outputs])

        with open('lstm_metrics_results.txt', 'w') as f:
            f.write(f'total: {total}\n')
            f.write(f'correct: {correct}\n')
            f.write(f'true_pos_up: {true_pos_up}\n')
            f.write(f'true_pos_dwn: {true_pos_dwn}\n')
            f.write(f'true_pos_stag: {true_pos_stag}\n')
            f.write(f'false_pos_up: {false_pos_up}\n')
            f.write(f'false_pos_dwn: {false_pos_dwn}\n')
            f.write(f'false_pos_stag: {false_pos_stag}\n')
            f.write(f'false_neg_up: {false_neg_up}\n')
            f.write(f'false_neg_dwn: {false_neg_dwn}\n')
            f.write(f'false_neg_stag: {false_neg_stag}\n')
            f.write(f'up: {up}\n')
            f.write(f'dwn: {dwn}\n')
            f.write(f'stag: {stag}\n')

        self.test_step_outputs.clear()

        res = 100 * correct / total
        try:
            precision_up = true_pos_up / (true_pos_up + false_pos_up)
        except ZeroDivisionError:
            precision_up = 0
        try:
            precision_dwn = true_pos_dwn / (true_pos_dwn + false_pos_dwn)
        except ZeroDivisionError:
            precision_dwn = 0
        try:
            precision_stag = true_pos_stag / (true_pos_stag + false_pos_stag) 
        except ZeroDivisionError:
            precision_stag = 0
        try:
            recall_up = true_pos_up / (true_pos_up + false_neg_up)
        except ZeroDivisionError:
            recall_up = 0
        try:
            recall_dwn = true_pos_dwn / (true_pos_dwn + false_neg_dwn)
        except ZeroDivisionError:
            recall_dwn = 0
        try:
            recall_stag = true_pos_stag / (true_pos_stag + false_neg_stag) 
        except ZeroDivisionError:
            recall_stag = 0 
        precision_macro_avg = (precision_up + precision_dwn + precision_stag) / 3
        recall_macro_avg = (recall_up + recall_dwn + recall_stag) / 3
        f1_score_macro_avg = 2 * (precision_macro_avg * recall_macro_avg) / (precision_macro_avg + recall_macro_avg)

        precision_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + false_pos_up + false_pos_dwn + false_pos_stag)
        recall_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + false_neg_up + false_neg_dwn + false_neg_stag)
        f1_score_micro_avg = (true_pos_up + true_pos_dwn + true_pos_stag) / (true_pos_up + true_pos_dwn + true_pos_stag + (false_pos_up + false_pos_dwn + false_pos_stag + false_neg_up + false_neg_dwn + false_neg_stag) / 2)

        with open('lstm_results.txt', 'w') as f:
            f.write(f'Accuracy: {res}\n')
            f.write(f'Precision macro average: {100 *precision_macro_avg}\n')
            f.write(f'Recall macro average: {100 * recall_macro_avg}\n')
            f.write(f'F1-score macro average: {100 * f1_score_macro_avg}\n')
            f.write(f'Precision micro average: {100 * precision_micro_avg}\n')
            f.write(f'Recall micro average: {100 * recall_micro_avg}\n')
            f.write(f'F1-score micro average: {100 * f1_score_micro_avg}\n')
            f.write(f'Precision up: {100 * precision_up}\n')
            f.write(f'Precision down: {100 * precision_dwn}\n')
            f.write(f'Precision stag: {100 * precision_stag}\n')
            f.write(f'Recall up: {100 * recall_up}\n')
            f.write(f'Recall down: {100 * recall_dwn}\n')
            f.write(f'Recall stag: {100 * recall_stag}\n')
            
        return {'overall_accuracy': res}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    

if __name__ == '__main__':
    pl.seed_everything(42)

    data_module = DataModule(data_type='BTCUSDT', window=50, batch_size=64, pred_horizon=20, alpha=0.0002)

    model = ANN(input_size=7, hidden_size=100)
    path = f'/../checkpoints/ann.ckpt'

    # model = RNN(input_size=7, hidden_size=100, num_layers=10)
    # path = f'/../checkpoints/rnn.ckpt'

    # model = LSTM(input_size=7, hidden_size=100, seq_len=50, num_layers=10)
    # path = f'/../checkpoints/lstm.ckpt'

    logger = TensorBoardLogger('tb_logs', name='ode_logs')
    trainer = pl.Trainer(max_epochs=40, logger=logger, deterministic=True)
    trainer.fit(model, data_module)
    trainer.save_checkpoint(os.path.abspath(__file__ + path))
    trainer.test(model, data_module.test_dataloader())