import time
import pandas as pd
import numpy as np
from utils.config import BINANCE_CLIENT, DATA_DB

def time_to_utc_timestamp(t):
    if type(t) == int or type(t) == float:
        return t
    return pd.to_datetime(t).timestamp()

def time_to_utc_datetime(t):
    if type(t) == int or type(t) == float:
        return t
    return pd.to_datetime(t)

def candle_to_seconds(candle):
    if candle.endswith("s"):
        return int(candle[:-1])
    if candle.endswith("m"):
        return int(candle[:-1]) * 60
    if candle.endswith("h"):
        return int(candle[:-1]) * 3600
    if candle.endswith("d"):
        return int(candle[:-1]) * 86400
    if candle.endswith("w"):
        return int(candle[:-1]) * 86400 * 7
    if candle.endswith("mo"):
        return int(candle[:-2]) * 86400 * 30
    if candle.endswith("y"):
        return int(candle[:-1]) * 86400 * 365
    raise NotImplementedError


def convert_grouper_freq(candle):
    if candle[-1] == "m":
        return candle.replace("m", "min")
    elif candle[-1] == "h":
        return candle.replace("h", "H")
    elif candle[-1] == "d":
        return candle.replace("h", "D")

def group(df, candle):
    data = df.groupby(pd.Grouper(freq=convert_grouper_freq(candle), closed='right', label='right')).agg({
        "open": "first",
        "close": "last",
        "high": "max",
        "low": "min",
        "volume": "sum"
    })
    # if data.index[0] != df.index[0]:
        # return data.iloc[1:]
    return data

class Timeseries: 
    def __init__( self, ticker, start, end, candle, is_volatile = False, ts = None, columns = [] ): 
        self.ticker = ticker 
        self.start = start 
        self.end = end 
        self.candle = candle 
        self.is_volatile = is_volatile 
        if ts is not None: self.series = ts 
        else: self.series = pd.DataFrame(columns=columns) 
        
    def query_klines(self, ticker = None, force = False): 
        if ticker is None: ticker = self.ticker 
        self.series = get_klines(ticker, self.candle, self.start, self.end, force = force) 
    
    def append(self, row):
        self.series = self.series.append(row, ignore_index = True)

    def at(self, t):
        return self.series.loc[t]

class ConstantTimeseries(Timeseries):
    def __init__(
        self,
        ticker,
        start,
        end,
        candle,
        values = {"price": 1.0, "high": 1.0, "low": 1.0, "open": 1.0, "close": 1.0}
    ):
        s, e = time_to_utc_timestamp(start), time_to_utc_timestamp(end)
        c = candle_to_seconds(candle)
        assert int((e-s)/c) * c == e-s
        index = np.linspace(s + c, e, int((e-s)/c), dtype=np.int64)
        ts = pd.DataFrame([values] * len(index), index = index)
        super().__init__(ticker, start, end, candle, ts = ts, is_volatile=False)

def get_klines_from_db(ticker, interval, start, end):
    start_candle = DATA_DB.get_data_from_table("klines", where = {"ticker": ticker, "candle": interval, "open_time": start})
    end_candle = DATA_DB.get_data_from_table("klines", where = {"ticker": ticker, "candle": interval, "close_time": end})
    if len(start_candle) == 1 and len(end_candle) == 1:
        return DATA_DB.query_from_db(
            f'SELECT * FROM klines WHERE ticker = "{ticker}" AND candle = "{interval}" AND open_time >= {start} AND close_time <= {end} ORDER BY open_time ASC'
        )
    return None


def get_klines(ticker, interval, start, end, limit = 900, sleep = 1, index_col = 'close_time', price_col = 'close', force = False):
    data_cols = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
    start = time_to_utc_timestamp(start)
    end = time_to_utc_timestamp(end)
    print(start, end, interval, ticker)

    data = None
    if not force: data = get_klines_from_db(ticker, interval, start, end)
    if data is None:
        start *= 1000
        end *= 1000
        close_time_idx = data_cols.index("close_time")
        candle_ms = candle_to_seconds(interval) * 1000
        data = []

        while start < end:
            print(f"start={time.ctime(start/1000)}")
            data += BINANCE_CLIENT.get_historical_klines(ticker, interval, int(start), int(min(start + limit * candle_ms, end)))
            if len(data) and data[-1][close_time_idx] + 1 == start:
                break
            if len(data): start = data[-1][close_time_idx] + 1
            else: start = int(min(start + limit * candle_ms, end))
            time.sleep(sleep)
        assert data[-1][close_time_idx] + 1 >= end
        data = pd.DataFrame(data, columns=data_cols, dtype=np.float64)   
        data['open_time'] = (data['open_time'] / 1000).astype(int)
        data['close_time'] = ((data['close_time'] + 1)/1000).astype(int)
        data['ticker'] = ticker
        data['candle'] = interval
        DATA_DB.insert_into_table("klines", data.to_dict('records'))
    else:
        data = pd.DataFrame(data)
        data['open_time'] = data['open_time'].astype(int)
        data['close_time'] = data['close_time'].astype(int)
    
    if price_col is not None: data['price'] = data[price_col]
    if index_col is not None: data.set_index(index_col, inplace=True)
    return data
