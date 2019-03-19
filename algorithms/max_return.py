import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class MaxReturnAgent(Agent):
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config["instruments_list"]
        self.n_steps = 0
        self.exchange = config['exchange']
        self.n_inst = len(self.instruments_list)
        self.fit_frequency = config["fit_frequency"]
        self.fit_window = config["fit_window"]
        self.rebalance_frequency = config["rebalance_frequency"]
        self.use_risk_free = config["use_risk_free"]
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                timetable, config, config["fit_window"] + 1, False)
        if config.get("short_flag") is not None:
            self.short_flag = config["short_flag"]
        else:
            self.short_flag = False

    def max_ret_portfolio(self, exp_rets):
        """
        Computes a long-only maximum return portfolio, i.e. selects
        the assets with maximal return. If there is more than one
        asset with maximal return, equally weight all of them.

        Based on portfolioopt implementation:
        https://github.com/czielinski/portfolioopt/tree/master/portfolioopt
        
        Parameters
        ----------
        exp_rets: pandas.Series
            Expected asset returns (often historical returns).
        Returns
        -------
        weights: pandas.Series
            Optimal asset weights.
        """
        if not isinstance(exp_rets, pd.Series):
            raise ValueError("Expected returns is not a Series")
        exp_rets = np.array(exp_rets)
        weights = np.zeros(exp_rets.shape[0])
        weights[exp_rets == exp_rets.max()] = 1.0
        weights[exp_rets != exp_rets.max()] = 0.0
        if self.short_flag:
            weights[exp_rets == exp_rets.min()] = -1.0
        return weights / np.abs(weights).sum()

    def fit_max_return(self, epoch):
        data_prices = self.data_extractor(epoch)
        data_prices = data_prices[-self.fit_window - 2:, :]
        returns = data_prices[1:] / data_prices[:-1] - 1
        returns_train = pd.DataFrame(returns)
        avg_rets = returns_train.mean(axis = 0)
        self.weights = self.max_ret_portfolio(avg_rets)

    def max_return_next_weight(self):
        if self.n_steps % self.rebalance_frequency == 0 or \
                        self.n_steps % self.fit_frequency == 0:
            return self.weights
        else:
            return None

    def compute_portfolio(self, epoch):
        if self.n_steps % self.fit_frequency == 0:
            self.fit_max_return(epoch)
        self.n_steps += 1
        day_weight = self.max_return_next_weight()
        print("max_return weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
