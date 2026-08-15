from src.configs import DataParameters
import yfinance as yf
import pandas as pd


class DataRetriever:
    def __init__(self, data_params:DataParameters):

        if not data_params.inference:
            self.reference = data_params.target
            self.variables = [data_params.var1, data_params.var2, data_params.var3]
        else:
            self.reference=data_params.var1
            self.variables = [data_params.var2, data_params.var3]

        self.start_date = data_params.start_date
        self.end_date = data_params.end_date


    def retrieve(self, index):
        name, ticker = index.split('/')
        df = yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)

        if df.index.dtype=='datetime64[s, Asia/Seoul]':
            s = df['Open']
        else:
            s = df['Close']

        s = s.rename(name)
        s.index = s.index.date
        return s


    def retrieve_all(self):
        reference_series = self.retrieve(self.reference)
        variables_series = [self.retrieve(var) for var in self.variables]

        df = pd.DataFrame(reference_series)

        for series in variables_series:
            df = df.join(series, how='left')

        return df
