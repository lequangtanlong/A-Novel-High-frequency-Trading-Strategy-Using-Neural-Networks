import numpy as np
import pandas as pd

class TradingDataLoader():
    def __init__(self, path):
        self.df, self.timestamp = self.__load_data__(path)
        self.index = 0

    def step(self):
        self.index += 1
        if (self.index >= len(self.df)):
            return (self.df.iloc[self.index-1], None, self.timestamp.iloc[self.index-1])
        
        return (self.df.iloc[self.index-1], self.df.iloc[self.index][-1], self.timestamp.iloc[self.index-1])

    def __load_data__(self, path):
        df = pd.read_csv(path)
        timestamp = df['Local time']
        return self.__convert_to_seconds__(df), timestamp

    def __convert_to_seconds__(self, df):
        delta_t = []
        time_in_seconds = []
        prev_time = 0.0
        start = 0.0
        for index, row in df.iterrows():
            time = float(row['Local time'])

            if (index == 0):
                delta_t.append(0)
                start = time
            else:
                delta_t.append(time - prev_time)
                
            time_in_seconds.append(time - start)
            prev_time = time
        time_in_seconds = np.array(time_in_seconds)
        df['Local time'] = time_in_seconds
        df['DeltaT'] = delta_t
        
        return df