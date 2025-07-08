from dotenv import dotenv_values
from binance import Client
from utils.SQL import SQL

tables = [
    {
        "name": "klines",
        "fields": ['ticker', 'candle', 'open_time','open', 'high', 'low', 'close', 'volume', 'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore'],
        "types": [str, str] + [float] * 12,
        "primary": ['ticker', 'candle', 'close_time']
    }
]

# Add binance key, secret to .env and set the following to True to query data
SET_BINANCE_CLIENT = True
DATA_DIR = "data"
DATA_DB = SQL("klines.db", DATA_DIR, tables)
if SET_BINANCE_CLIENT:
    BINANCE_CLIENT = Client(dotenv_values()['BINANCE_KEY'], dotenv_values()['BINANCE_SECRET'])
else: 
    BINANCE_CLIENT = None
