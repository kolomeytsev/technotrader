import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils
import cvxopt


class MinVarianceAgent(Agent):
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)        
        self.exchange = config['exchange']
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                timetable, config, config["fit_window"] + 1, False)
        self.instruments_list = config["instruments_list"]
        self.n_steps = 0
        self.n_inst = len(self.instruments_list)
        self.fit_frequency = config["fit_frequency"]
        self.fit_window = config["fit_window"]
        self.rebalance_frequency = config["rebalance_frequency"]
        self.allow_short = config["allow_short"]
        self.use_risk_free = config["use_risk_free"]
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst

    def min_var_portfolio(self, cov_mat, allow_short=False):
        """
        Computes the minimum variance portfolio.
        Based on portfolioopt implementation:
        https://github.com/czielinski/portfolioopt/tree/master/portfolioopt

        Note: As the variance is not invariant with respect
        to leverage, it is not possible to construct non-trivial
        market neutral minimum variance portfolios. This is because
        the variance approaches zero with decreasing leverage,
        i.e. the market neutral portfolio with minimum variance
        is not invested at all.
        
        Parameters
        ----------
        cov_mat: pandas.DataFrame
            Covariance matrix of asset returns.
        allow_short: bool, optional
            If 'False' construct a long-only portfolio.
            If 'True' allow shorting, i.e. negative weights.
        Returns
        -------
        weights: pandas.Series
            Optimal asset weights.
        """
        if not isinstance(cov_mat, pd.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")

        n = len(cov_mat)

        P = cvxopt.matrix(cov_mat.values)
        q = cvxopt.matrix(0.0, (n, 1))

        # Constraints Gx <= h
        if not allow_short:
            # x >= 0
            G = cvxopt.matrix(-np.identity(n))
            h = cvxopt.matrix(0.0, (n, 1))
        else:
            G = None
            h = None

        # Constraints Ax = b
        # sum(x) = 1
        A = cvxopt.matrix(1.0, (1, n))
        b = cvxopt.matrix(1.0)

        # Solve
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        if sol['status'] != 'optimal':
            warnings.warn("Convergence problem")

        # Put weights into a labeled series
        weights = pd.Series(sol['x'], index=cov_mat.index)
        return weights

    def fit_min_variance(self, epoch):
        data_prices = self.data_extractor(epoch)
        returns = data_prices[1:] / data_prices[:-1] - 1
        returns_train = pd.DataFrame(returns)
        cov_mat = returns_train.cov()
        self.weights = self.min_var_portfolio(
            cov_mat,
            allow_short=self.allow_short
        )
        self.weights = np.array(self.weights)

    def min_variance_next_weight(self):
        if self.n_steps % self.rebalance_frequency == 0 or \
                        self.n_steps % self.fit_frequency == 0:
            return self.weights
        else:
            return None

    def compute_portfolio(self, epoch):
        if self.n_steps % self.fit_frequency == 0:
            self.fit_min_variance(epoch)
        self.n_steps += 1
        day_weight = self.min_variance_next_weight()
        print("weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
