from utils.utils import Timeseries
from typing import List
import pandas as pd
import numpy as np
import pickle

class BacktestEnv:
    def __init__(
        self,
        index: int,
        init_capital: float,
        asset_ts: List[Timeseries],
        strategy, 
        start: str, 
        end: str, 
        candle: str,
        tx_cost: float = 0.0,
        graphs: dict = {}
    ):
        """Backtest environment

        General flow:

        * Create a strategy class with a .trigger(env, timestamp, prices, portfolio) method

        * interact with the env (in this env only a .reset_portfolio method is available)

        * keep track of portfolio and return it -- any interaction with the env returns 
        the modified portfolio, which is then returned back to the env

        * the env does not check if the returned portfolio is a consequence of the interactions, e.g.,
        the next portfolio might be returned without calling .reset_portfolio(), which may not lead to errors

        Parameters
        ----------
        index : int
            stores stats in data/runs/{index}
        init_capital : float
            initial cash
        asset_ts : List[Timeseries]
            list of asset timeseries
        strategy : 
            strategy class

            must implement a .trigger(env, timestamp, prices, portfolio) method which is
            called by the environment on every candle
        start : str
            start date
        end : str
            end date
        candle : str
            candle 

            Note: at the moment it is just for information. .trigger is called based on the 
            indices of the timeseries
        tx_cost : float, optional
            tx cost in absolute terms
        graphs : dict, optional
            store custom statistics 

            format: 
            graphs = {"graph1": None, "graph2": ["graph2_1", "graph2_2"]}

            to store values, call BacktestEnv.register_graph_point()

            e.g.,
            env.register_graph_point(time, "graph1", float_val) 
            env.register_graph_point(time, "graph2", {"graph2_1": val1, "graph2_2": val2})
            
            vals can be None, need not call on every candle.
            
            by default {}
        """
        self.index = index
        self.init_capital = init_capital
        self.asset_ts = asset_ts
        self.strategy = strategy
        self.tx_cost = tx_cost
        self.graphs = graphs
        self.graphs_data = []
        self.n_assets = len(asset_ts)
        self.columns = ['time', 'value', 'pos0', 'volume', 'tx_cost', 'invested']
        self.start, self.end, self.candle = start, end, candle
        self.columns += [f"{x.ticker}_pos" for x in asset_ts]
        self.columns += [f"{x.ticker}_price" for x in asset_ts]
        self.ts = Timeseries("backtest", start, end, candle, columns=self.columns)
    
    def get_prices(self, tstamp):
        """get prices at timestamp
        """
        return np.array([x.at(tstamp)['price'] for x in self.asset_ts])
    
    def get_intersect_times(self, start, end):
        """get intersection of indices of asset_ts
        """
        times = self.asset_ts[0].series.index
        for i in range(1, len(self.asset_ts)):
            times = np.intersect1d(times, self.asset_ts[i].series.index)
        return [t for t in times if start <= t <= end]
    
    def progress(self, value, i, N, pos0, n_sticks = 50):
        sticks = int((i+1) * n_sticks/N)
        ret = value / self.init_capital - 1
        print(f"\r[{'|' * sticks}{'.' * (n_sticks-sticks)}] {'↑' if ret >= 0 else '↓'} {ret*100:.2f}%, pos0={pos0:.2f}", end="")

    def run(self, verbose=True):
        """entry point

        Parameters
        ----------
        verbose : bool, optional
            will print progress bar, by default True
        """
        start = pd.to_datetime(self.start).timestamp()
        end = pd.to_datetime(self.end).timestamp()

        last_data = {k: 0 for k in self.columns}
        last_data['time'] = start - 1
        last_data['value'] = self.init_capital
        last_data['pos0'] = self.init_capital
        prices = self.get_prices(start)
        last_data.update({f'{x.ticker}_price': prices[i] for i, x in enumerate(self.asset_ts)})
        self.ts.append(last_data)
        self.pos0 = self.init_capital

        times = self.get_intersect_times(start, end)
        for i, current_time in enumerate(times):
            if verbose: self.progress(last_data['value'], i, len(times), self.pos0)

            self.current_data = {
                "time": current_time, 
                "volume": last_data['volume'], 
                "tx_cost": last_data['tx_cost'],
            }
            prices = self.get_prices(current_time)
            portfolio = [last_data[f'{x.ticker}_pos'] for x in self.asset_ts]

            # Run strategy
            portfolio = self.strategy.trigger(self, current_time, prices, portfolio)

            # process backtest timeseries
            self.process_portfolio(current_time, prices, portfolio)   

            last_data = self.current_data

        self.statistics = self.compute_statistics()
        for g, c in self.graphs.items():
            points = [x for x in self.graphs_data if x[1] == g]
            if len(points) == 0: continue
            if type(c) == list and len(c):
                for l in c:
                    graph_name = f"graph_{g}+{l}"
                    self.ts.series[graph_name] = None
                    for p in points: 
                        self.ts.series.loc[self.ts.series.time == p[0], graph_name] = p[2][l]
            else:
                graph_name = f'graph_{g}'
                self.ts.series[graph_name] = None
                for p in points: 
                    self.ts.series.loc[self.ts.series.time == p[0], graph_name] = p[2]

        with open(f"data/runs/{self.index}", "wb") as file:
            pickle.dump([self.ts, self.statistics], file)

    def _max_drawdown(self, value):
        M = value[0]
        MD = float('inf')
        for i in range(1, len(value)):
            M = max(M, value[i])
            MD = min(MD, value[i] - M)
        return MD

    def compute_statistics(self):
        """Compute strategy statistics at the end of the run
        """
        value = self.ts.series['value'].to_numpy()
        rets = np.log(value[1:]/value[:-1])
        return {
            "sharpe": np.mean(rets) / np.std(rets),
            "max_drawdown": self._max_drawdown(value)
        }

    def reset_portfolio(self, current_time, prices, old_portfolio, new_portfolio):
        """Call this method to reset portfolio (interaction method)
        """
        old_portfolio = np.array(old_portfolio)
        new_portfolio = np.array(new_portfolio)
        opv, npv = old_portfolio * prices, new_portfolio * prices
        volume = np.sum(np.abs(npv - opv))
        diff = np.sum(npv - opv)
        tx_cost = volume * self.tx_cost
        self.current_data['volume'] += volume
        self.current_data['tx_cost'] += tx_cost
        
        self.pos0 -= tx_cost + diff
        return new_portfolio
    
    def process_portfolio(self, current_time, prices, portfolio):
        """Called every candle to store data in the env timeseries"""
        self.current_data['pos0'] = self.pos0
        self.current_data.update({
            f'{x.ticker}_pos': portfolio[i] for i, x in enumerate(self.asset_ts)
        })
        self.current_data.update({f'{x.ticker}_price': prices[i] for i, x in enumerate(self.asset_ts)})
        self.current_data['value'] = self.pos0 + np.sum(portfolio * prices)
        self.current_data['invested'] = np.sum(np.abs(portfolio * prices))

        self.ts.append(self.current_data)

    def register_graph_point(self, time, graph_name, data):
        """register external graphs"""
        self.graphs_data.append((time, graph_name, data))

    def value(self, prices, portfolio, pos0=None):
        """compute value of the portfolio"""
        if pos0 is None: pos0 = self.pos0
        portfolio = np.array(portfolio)
        return pos0 + np.sum(portfolio * prices)

        
class BaseStrategy:
    """Base strategy for the strategies tested in this project
    """
    def get_price_history(self, env:BacktestEnv, timestamp):
        dfs = []
        for ts in env.asset_ts:
            df = ts.series[ts.series.index <= timestamp][['price']]
            dfs.append(df.rename(columns={"price": ts.ticker}))

        df = dfs[0].join(dfs[1:]).to_numpy().T
        return df
    
    def get_log_price_history(self, env:BacktestEnv, timestamp):
        return np.log(self.get_price_history(env, timestamp))
    
    def trigger(self, env, current_time, prices, portfolio):
        raise NotImplementedError


