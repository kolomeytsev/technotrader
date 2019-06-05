import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class PamrShortsAgent(Agent):
    """
    Passive Aggressive Mean Reversion strategy.

    https://link.springer.com/article/10.1007/s10994-012-5281-z
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)        
        self.simplex_projection = agent_utils.simplex_projection
        self.instruments_list = config["instruments_list"]
        self.instruments_number = len(self.instruments_list)
        self.price_label = config["price_label"]
        self.step = config['step']
        self.exchange = config['exchange']
        self.use_risk_free = config["use_risk_free"]
        if self.use_risk_free:
            self.instruments_number += 1
        self.last_portfolio = np.ones(self.instruments_number) / self.instruments_number,
        self.eps = config['mean_reversion_threshold']
        self.C = config['aggressive_param']
        self.variant = config['variant']
        self.eta = np.inf
        self.daily_ret = 1.
        self.n_steps = 0
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                            timetable, config, 2, True)

    def pamr_expert(self, x):
        weight = self.last_portfolio - self.eta * (x - np.mean(x))
        weight = self.weights_projection(weight)
        if (weight < -1.00001).any() or (weight > 1.00001).any():
            str_print = 'pamr_expert: t=%d, sum(weight)=%f, returning uniform weights'
            print(str_print % (t, weight.sum()))
            return np.ones(len(self.last_portfolio)) / len(self.last_portfolio)
        return weight / np.abs(weight).sum()

    def update_lagrange_multiplier(self, price_ratios):
        numerator = np.maximum(0., np.dot(self.last_portfolio, price_ratios - 1) + 1 - self.eps)
        denom_part = price_ratios - np.mean(price_ratios)
        denominator = np.dot(denom_part, denom_part) + 1e-8
        if self.variant == 0:
            self.eta = numerator / denominator
        elif self.variant == 1:
            self.eta = np.minimum(self.C, numerator / denominator)
        elif self.variant == 2:
            self.eta = numerator / (denominator + 0.5 / self.C)
        else:
            print("Wrong variant parameter: must be 0, 1 or 2. Exiting")
            return None
        self.eta = min(1e10, max(0, self.eta))

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch)
        self.update_lagrange_multiplier(data_price_relatives[-1])
        if self.n_steps == 1:
            day_weight = np.ones(self.instruments_number) / self.instruments_number
        else:
            day_weight = self.pamr_expert(data_price_relatives[-1])
        if self.verbose: 
            print("pamr weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict

