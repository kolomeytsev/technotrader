import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils
import cvxopt as opt
from cvxopt import solvers


class MarkowitzAgent(Agent):
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.exchange = config['exchange']
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                timetable, config, config["fit_window"] + 1, False)
        self.instruments_list = config["instruments_list"]
        self.n_steps = 0
        self.fit_frequency = config["fit_frequency"]
        self.rebalance_frequency = config["rebalance_frequency"]
        self.min_return_level = config["min_return_level"]
        self.allow_short = config["allow_short"]
        self.use_risk_free = config["use_risk_free"]
        self.n_inst = len(self.instruments_list)
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst

    def markowitz_portfolio(self, cov_mat, exp_rets, target_ret,
                        allow_short=False, market_neutral=False):
        """
        Computes a Markowitz portfolio.
        Based on portfolioopt implementation:
        https://github.com/czielinski/portfolioopt/tree/master/portfolioopt

        Parameters
        ----------
        cov_mat: pandas.DataFrame
            Covariance matrix of asset returns.
        exp_rets: pandas.Series
            Expected asset returns (often historical returns).
        target_ret: float
            Target return of portfolio.
        allow_short: bool, self.optional
            If 'False' construct a long-only portfolio.
            If 'True' allow shorting, i.e. negative weights.
        market_neutral: bool, self.optional
            If 'False' sum of weights equals one.
            If 'True' sum of weights equal zero, i.e. create a
                market neutral portfolio (implies allow_short=True).
                
        Returns
        -------
        weights: pandas.Series
            self.optimal asset weights.
        """
        if not isinstance(cov_mat, pd.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")

        if not isinstance(exp_rets, pd.Series):
            raise ValueError("Expected returns is not a Series")

        if not isinstance(target_ret, float):
            raise ValueError("Target return is not a float")

        if not cov_mat.index.equals(exp_rets.index):
            raise ValueError("Indices do not match")

        if market_neutral and not allow_short:
            warnings.warn("A market neutral portfolio implies shorting")
            allow_short=True

        n = len(cov_mat)

        P = opt.matrix(cov_mat.values)
        q = opt.matrix(0.0, (n, 1))

        # Constraints Gx <= h
        if not allow_short:
            # exp_rets*x >= target_ret and x >= 0
            G = opt.matrix(np.vstack((-exp_rets.values,
                                      -np.identity(n))))
            h = opt.matrix(np.vstack((-target_ret,
                                      +np.zeros((n, 1)))))
        else:
            # exp_rets*x >= target_ret
            G = opt.matrix(-exp_rets.values).T
            h = opt.matrix(-target_ret)

        # Constraints Ax = b
        # sum(x) = 1
        A = opt.matrix(1.0, (1, n))

        if not market_neutral:
            b = opt.matrix(1.0)
        else:
            b = opt.matrix(0.0)

        # Solve
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)

        if sol['status'] != 'optimal':
            warnings.warn("Convergence problem")

        # Put weights into a labeled series
        weights = pd.Series(sol['x'], index=cov_mat.index)
        return weights

    def fit_markowitz(self, epoch):
        '''
        data_prices = self.storage_driver.last_candles(
            {self.exchange.name: self.instruments_list},
            self.fit_window + 1,
            fields=[self.price_label], out_format='pandas_wide', 
            candles_resolution=self.candles_res
        )
        data_prices.fillna(method="backfill", inplace=True)
        data_prices.fillna(method="pad", inplace=True)
        data_prices.reset_index(inplace=True, drop=True)
        data_prices = data_prices[self.relevant_columns]
        data_prices = nparray(data_prices)
        if self.use_risk_free:
            data_prices = nphstack(
                [data_prices, npones((data_prices.shape[0], 1))]
            )
        '''
        data_prices = self.data_extractor(epoch)
        returns = data_prices[1:] / data_prices[:-1] - 1
        returns_train = pd.DataFrame(returns)
        cov_mat = returns_train.cov()
        avg_rets = returns_train.mean(axis = 0)
        target_ret = avg_rets.quantile(self.min_return_level)
        self.weights = self.markowitz_portfolio(
            cov_mat,
            avg_rets,
            target_ret,
            allow_short=self.allow_short
        )
        self.weights = np.array(self.weights)

    def markowitz_next_weight(self):
        if self.n_steps % self.rebalance_frequency == 0 or \
                        self.n_steps % self.fit_frequency == 0:
            return self.weights
        else:
            return None

    def compute_portfolio(self, epoch):
        if self.n_steps % self.fit_frequency == 0:
            self.fit_markowitz(epoch)
        self.n_steps += 1
        day_weight = self.markowitz_next_weight()
        print("markowitz weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
