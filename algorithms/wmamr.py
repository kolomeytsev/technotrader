import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class WmamrBlock(Block):
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

        self.last_portfolio = self.np.ones(self.instruments_number) / self.instruments_number,
        self.eps = config['mean_reversion_threshold']
        self.C = config['aggressive_param']
        self.variant = config['variant']
        self.window = config['window']
        self.eta = self.np.inf
        self.daily_ret = 1.
        self.n_steps = 0
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                timetable, config, self.window + 1, True)

    def pamr_expert(self, x):
        weight = self.last_portfolio - self.eta * (x - self.np.mean(x))
        print(weight)
        weight = self.simplex_projection(weight)
        if (weight < -0.00001).any() or (weight > 1.00001).any():
            str_print = 'pamr_expert: t=%d, sum(weight)=%f, returning uniform weights'
            print(str_print % (t, weight.sum()))
            return self.np.ones(len(self.last_portfolio)) / len(self.last_portfolio)
        return weight / sum(weight)

    def update_lagrange_multiplier(self, price_ratios):
        denom_part = price_ratios - self.np.mean(price_ratios)
        if self.variant == 0:
            denominator = self.np.dot(denom_part, denom_part)
        elif self.variant == 1:
            # not yet implemented
            print("Variant 1 is not implemented, using 0 variant...")
            denominator = self.np.dot(denom_part, denom_part)
        elif self.variant == 2:
            # not yet implemented
            print("Variant 2 is not implemented, using 0 variant...")
            denominator = self.np.dot(denom_part, denom_part)
        else:
            print("Wrong variant parameter: must be 0, 1 or 2. Exiting.")
            return None
        if denominator != 0:
            self.eta = (self.daily_ret - self.eps) / denominator
        self.eta = min(1e10, max(0, self.eta))

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch)
        self.update_lagrange_multiplier(data_price_relatives.mean(0))

        if self.n_steps == 1:
            day_weight = self.np.ones(self.instruments_number) / self.instruments_number
        else:   
            day_weight = self.pamr_expert(data_price_relatives[-1])
            bt_return = results('backtest', [epoch - self.step])
            daily_ret = bt_return['return_fee'].values[0]
            
        print("weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
