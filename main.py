from utils.utils import Timeseries, candle_to_seconds
import numpy as np
from utils.param_est import param_estimation
from utils.solver import Solver, Box, SolverLinear, SolverLinearFees
from utils.backtest import BacktestEnv, BaseStrategy
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.vector_ar.vecm import VECM
import warnings 
warnings.filterwarnings('ignore')


class Strategy(BaseStrategy):
    def __init__(
        self, 
        gamma = 1.0,
        inv_ratio = 0.9,
        inv_const = None, 
        reset_model_every = '3d', 
        lookback='1mo', 
        factor = 5.0
    ):
        """Strategy for utility maximizer

        Parameters
        ----------
        gamma : float, optional
            gamma value, by default 1.0
        inv_ratio : float, optional
            ratio of current value invested, by default 0.9
        inv_const : _type_, optional
            constant value invested, by default None

            overrides inv_ratio
        reset_model_every : str, optional
            recalculate model every, by default '3d'
        lookback : str, optional
            lookback for the model, by default '1mo'
        factor : float, optional
            leverage factor, by default 5.0
        """
        assert inv_ratio is None or inv_const is None
        self.gamma = gamma
        self.inv_ratio = inv_ratio
        self.inv_const = inv_const
        self.model = None
        self.reset_model_every = candle_to_seconds(reset_model_every)
        self.lookback = candle_to_seconds(lookback)
        self.factor = factor

    def set_model(self, env, current_time, prices, portfolio):
        """Calculate model"""
        logprices = self.get_log_price_history(env, current_time)
        logprices = logprices[:, -self.lookback//candle_to_seconds(env.candle):]
        a0, A, delta, omega = param_estimation(
            logprices, 
            1
        )
        a = a0.T + A.T @ logprices
        T = self.reset_model_every // candle_to_seconds(env.candle)

        solver = Solver(
            np.linalg.cholesky(omega), 
            delta, 
            A, 
            Box(0, T, 200), 
            [
                Box(5 * a[0].min(), 5 * a[0].max(), 200),
            ],
            self.gamma, 
            True
        )
        solver.solve()
        self.model = solver
        self.params = [
            a0, A, delta, omega
        ]
        if self.inv_ratio is not None:
            self.v0 = env.value(prices, portfolio) * self.inv_ratio * self.factor
        else:
            self.v0 = self.inv_const * self.factor

    def get_params(self, env, current_time, prices):
        """get current variable values"""
        t=(current_time % self.reset_model_every) / candle_to_seconds(env.candle)
        a=self.params[0].T + self.params[1].T @ np.array([np.log(prices)]).T
        return [t, a[0, 0]]

    def trigger(self, env:BacktestEnv, current_time, prices, portfolio):
        """Main method
        
        get current parameters, calculate policy and reset portfolio
        """
        if current_time % self.reset_model_every == 0:
            self.set_model(env, current_time, prices, portfolio)

        if not self.model: return portfolio

        params = self.get_params(env, current_time, prices)
        policy = self.model.get_policy(params) 
        policy = policy / prices * self.v0
        portfolio = env.reset_portfolio(current_time, prices, portfolio, policy)
        env.register_graph_point(current_time, 'alpha', params[1])
        return policy
    

class StrategyLinear(Strategy):
    def __init__(self, divers, inv_ratio=1.0, inv_const=None, reset_model_every='3d', lookback='1mo'):
        """Strategy for the linear model"""
        super().__init__(None, inv_ratio, inv_const, reset_model_every, lookback, 1)
        self.divers = divers

    def set_model(self, env, current_time, prices, portfolio):
        logprices = self.get_log_price_history(env, current_time)
        logprices = logprices[:, -self.lookback//candle_to_seconds(env.candle):]
        a0, A, delta, omega = param_estimation(
            logprices, 
            2
        )
        self.params = [
            a0, A, delta, omega
        ]
        if self.inv_ratio is not None:
            self.v0 = env.value(prices, portfolio) * self.inv_ratio * self.factor
        else:
            self.v0 = self.inv_const * self.factor

    def get_policy(self, env, current_time, prices):
        a=self.params[0].T + self.params[1].T @ np.array([np.log(prices)]).T
        vals = self.params[2] @ a
        vals = vals.reshape(-1)
        vals = sorted(enumerate(vals), key=lambda l: abs(l[1]), reverse=True)

        policy = np.zeros(len(vals))
        for i in range(self.divers):
            policy[vals[i][0]] = np.sign(vals[i][1]) / self.divers

        return policy        

    def trigger(self, env, current_time, prices, portfolio):
        if current_time % self.reset_model_every == 0:
            self.set_model(env, current_time, prices, portfolio)

        if not hasattr(self, 'params'): return portfolio
        policy = self.get_policy(env, current_time, prices) * self.v0
        policy = policy / prices
        portfolio = env.reset_portfolio(current_time, prices, portfolio, policy)

        return portfolio

class StrategyFeesSimple(Strategy):
    def __init__(self, gamma=1, inv_ratio=0.9, inv_const=None, reset_model_every='3d', lookback='1mo', factor=5, quantum = 0.02):
        """Simple strategy with fees
        
        resets portfolio if the absolute change in portfolio crosses a threshold
        """
        super().__init__(gamma, inv_ratio, inv_const, reset_model_every, lookback, factor)
        self.quantum = quantum

    def trigger(self, env:BacktestEnv, current_time, prices, portfolio):
        if current_time % self.reset_model_every == 0:
            self.set_model(env, current_time, prices, portfolio)

        if not self.model: return portfolio

        params = self.get_params(env, current_time, prices)
        policy = self.model.get_policy(params) * self.v0

        if np.sum(np.abs(policy - portfolio * prices)) >= self.quantum * self.v0 :
            policy = policy / prices
            portfolio = env.reset_portfolio(current_time, prices, portfolio, policy)
        else:
            policy = portfolio
        env.register_graph_point(current_time, 'alpha', params[1])
        return policy


class FeesStrategy(Strategy):
    def __init__(self, gamma=1, inv_ratio=0.9, inv_const=None, reset_model_every='1d', lookback='1mo', qmax = 2, fees=0.0004):
        """Strategy for linear model with fees 
        """
        super().__init__(gamma, inv_ratio, inv_const, reset_model_every, lookback, 1.0)
        self.fees = fees
        self.qmax = qmax

    def set_model(self, env:BacktestEnv, current_time, prices, portfolio):
        logprices = self.get_log_price_history(env, current_time)
        logprices = logprices[:, -self.lookback//candle_to_seconds(env.candle):]
        a0, A, delta, omega = param_estimation(
            logprices, 
            1
        )
        a = a0.T + A.T @ logprices
        T = self.reset_model_every // candle_to_seconds(env.candle)

        if self.inv_ratio is not None:
            self.v0 = env.value(prices, portfolio) * self.inv_ratio * self.factor 
        else:
            self.v0 = self.inv_const * self.factor

        self.v0 /= self.qmax * len(omega)


        solver = SolverLinearFees(
            np.linalg.cholesky(omega), 
            delta, 
            A, 
            Box(0, T, 49), 
            Box(5 * a[0].min(), 5 * a[0].max(), 50),
            self.gamma, 
            self.qmax, 
            self.fees
        )
        solver.solve()
        self.model = solver
        self.params = [
            a0, A, delta, omega
        ]

        if not hasattr(self, 'q'):
            self.q = np.zeros(len(omega))

    def trigger(self, env:BacktestEnv, current_time, prices, portfolio):
        if current_time % self.reset_model_every == 0:
            self.set_model(env, current_time, prices, portfolio)

        if not self.model: return portfolio

        params = self.get_params(env, current_time, prices)
        policy = self.model.get_policy(params)[self.model.qvec_to_qidx(self.q)]
        self.q = self.q + self.model.Q[policy][0]
        policy = portfolio + self.model.Q[policy][0] * self.v0 / prices
        portfolio = env.reset_portfolio(current_time, prices, portfolio, policy)
        env.register_graph_point(current_time, 'alpha', params[1])
        return policy
        

if __name__ == "__main__":
    def get_asset_ts(ticker, start, end, candle):
        ts = Timeseries(ticker, start, end, candle)
        ts.query_klines(ticker)
        return ts

    startwlookback, start, end, candle = '2024-01-01', '2024-02-01', '2025-02-01', '1h'
    tickers = ["ETHUSDT", "BTCUSDT", "LINKUSDT", "BNBUSDT", "SOLUSDT"]
    env = BacktestEnv(
        1, 
        1000,
        [get_asset_ts(x, startwlookback, end, candle) for x in tickers],
        Strategy(gamma=0.5, inv_ratio=0.9),
        # StrategyLinear(3),
        start, end, candle, 
        0.0, 
        {"alpha": None}
    )
    env.run(True)
    print(env.statistics)
