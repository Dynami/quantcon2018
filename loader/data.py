import pandas as pd
import datetime as dt

class DataLoader(object):
    def __init__(self, file_name):
        dateparse = lambda x, y: dt.datetime.strptime(x + ' ' + y, '%m/%d/%Y %H:%M')
        self.df = pd.read_csv(file_name, parse_dates=[[0, 1]], index_col=0, skiprows=0, date_parser=dateparse)
        self.df.index.rename('Time', inplace=True)
        self.df.columns = ['open', 'high', 'low', 'close', 'volume']
        pass

    def data(self):
        return self.df

    def resample(self, freq='1H'):
        return self.df['close'].resample(freq, label='right', closed='right').ohlc().dropna()

    def preprocess(self, start_hour=9, end_hour=17):
        df_full_lite = self.df.loc[self.df.index.hour >= start_hour]
        df_full_lite = df_full_lite.loc[df_full_lite.index.hour <= end_hour]
        # time = df_full_lite.index.hour * 100 + df_full_lite.index.minute
        return df_full_lite
