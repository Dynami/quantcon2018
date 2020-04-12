import pandas as pd


class DataLoader(object):
    def __init__(self, file_name):
        dateparse = lambda x, y: pd.datetime.strptime(x + ' ' + y, '%m/%d/%Y %H:%M')
        self.df = pd.read_csv(file_name, parse_dates=[[0, 1]], index_col=0, skiprows=0, date_parser=dateparse)
        self.df.index.rename('Time', inplace=True)
        self.df.columns = ['open', 'high', 'low', 'close', 'volume']
        pass

    def data(self):
        return self.df

    def preprocess(self, start_hour=9, end_hour=17):
        df_full_lite = self.df.loc[self.df.index.hour >= start_hour]
        df_full_lite = df_full_lite.loc[df_full_lite.index.hour <= end_hour]
        # time = df_full_lite.index.hour * 100 + df_full_lite.index.minute
        return df_full_lite
